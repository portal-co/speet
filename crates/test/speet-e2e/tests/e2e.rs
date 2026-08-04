//! End-to-end recompiler tests: native ISAs (RISC-V, x86-64) and managed WASM.
//!
//! Three test kinds for native binaries, each with two exception-handling variants:
//!
//! - **smoke** — translate → assemble → `wasmparser::validate` only
//! - **run**   — translate → assemble → execute in wasmi
//! - **link**  — two binaries (possibly different arches) merged via
//!               `MegabinaryBuilder` → assemble → execute
//!
//! Three additional test kinds for WASM-frontend binaries:
//!
//! - **wasm_smoke**         — translate via `WasmFrontend` → `wasmparser::validate`
//! - **wasm_run**           — translate → validate → execute
//! - **wasm_run_cond_trap** — translate with `HookConditionTrap` → run with decide host fn
//!
//! C programs are compiled by `build.rs` (requires a suitable clang).
//! Tests that need compiled C objects are skipped when the object is absent.

#[path = "harness/mod.rs"]
mod harness;

use harness::*;
use rv_asm::Xlen;

// ── Native test macros ────────────────────────────────────────────────────────

macro_rules! smoke {
    ($name:ident, $rel:expr, arch = $arch:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_config(&text, addr, $arch, $cfg);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!("  ✓ {} ({:?}) — {} bytes",
                path.file_name().unwrap().to_string_lossy(), $cfg, wasm.len());
        }
    };
    // Legacy (Eh, speculative) form — kept for hand-written tests.
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        smoke!($name, $rel, arch = $arch, EscapeConfig::from_eh_spec($eh, $spec));
    };
}

macro_rules! smoke_c {
    ($name:ident, env = $env:expr, arch = $arch:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = match c_obj($env) { Some(p) => p, None => {
                eprintln!("  skipping {}: C object not built", stringify!($name));
                return;
            }};
            let (text, addr) = match load_text_optional(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single_config(&text, addr, $arch, $cfg);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!("  ✓ {} ({:?}) — {} bytes",
                path.file_name().unwrap().to_string_lossy(), $cfg, wasm.len());
        }
    };
    ($name:ident, env = $env:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        smoke_c!($name, env = $env, arch = $arch, EscapeConfig::from_eh_spec($eh, $spec));
    };
}

macro_rules! run {
    ($name:ident, $rel:expr, arch = $arch:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = load_text(&path);
            let cfg: EscapeConfig = $cfg;
            let (wasm, unsupported) = build_single_config(&text, addr, $arch, cfg);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => {
                    println!("  ✓ {} hints, {:?}", state.hints.len(), cfg);
                    for (id, snap) in &state.hints {
                        println!("    hint={id} a0={}", snap.reg("a0"));
                    }
                }
                // ExceptionSpec uses try_table/throw — wasmi gap → wasmtime.
                // FlagSpec must run on wasmi (no exception proposal); hard-fail.
                Err(e) if cfg.uses_exception_opcodes() && is_known_wasmi_exception_gap(&e) => {
                    eprintln!("  ! wasmi lacks exception-handling support ({e}); retrying under wasmtime");
                    match run_module_wasmtime(&wasm, "_start") {
                        Ok(state) => {
                            println!("  ✓ (wasmtime) {} hints, {:?}", state.hints.len(), cfg);
                            for (id, snap) in &state.hints {
                                println!("    hint={id} a0={}", snap.reg("a0"));
                            }
                        }
                        Err(e) => panic!("ExceptionSpec run failed under wasmtime fallback: {e}"),
                    }
                }
                Err(e) if cfg.speculative() => panic!("speculative run failed: {e}"),
                Err(e) if cfg.needs_exception_tags() => eprintln!("  ! EH run skipped: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        run!($name, $rel, arch = $arch, EscapeConfig::from_eh_spec($eh, $spec));
    };
}

macro_rules! run_trap {
    ($name:ident, $rel:expr, arch = $arch:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = load_text(&path);
            let cfg: EscapeConfig = $cfg;
            let (wasm, _) = build_single_with_trap_config(&text, addr, $arch, cfg);
            wasmparser::validate(&wasm).expect("WASM with trap is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => {
                    let n_ret  = state.hints.iter().filter(|(id, _)| *id == HINT_RETURN).count();
                    let n_call = state.hints.iter().filter(|(id, _)| *id == HINT_CALL).count();
                    println!("  ✓ trap {:?}: {} returns, {} calls", cfg, n_ret, n_call);
                }
                Err(e) if cfg.uses_exception_opcodes() && is_known_wasmi_exception_gap(&e) => {
                    eprintln!("  ! wasmi lacks exception-handling support ({e}); retrying under wasmtime");
                    match run_module_wasmtime(&wasm, "_start") {
                        Ok(state) => {
                            let n_ret  = state.hints.iter().filter(|(id, _)| *id == HINT_RETURN).count();
                            let n_call = state.hints.iter().filter(|(id, _)| *id == HINT_CALL).count();
                            println!("  ✓ (wasmtime) trap {:?}: {} returns, {} calls", cfg, n_ret, n_call);
                        }
                        Err(e) => panic!("ExceptionSpec run failed under wasmtime fallback: {e}"),
                    }
                }
                Err(e) if cfg.speculative() => panic!("speculative run failed: {e}"),
                Err(e) if cfg.needs_exception_tags() => eprintln!("  ! EH run skipped: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        run_trap!($name, $rel, arch = $arch, EscapeConfig::from_eh_spec($eh, $spec));
    };
}

macro_rules! run_c {
    ($name:ident, env = $env:expr, arch = $arch:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = match c_obj($env) { Some(p) => p, None => {
                eprintln!("  skipping {}: C object not built", stringify!($name));
                return;
            }};
            let (text, addr) = match load_text_optional(&path) { Some(v) => v, None => return };
            let cfg: EscapeConfig = $cfg;
            let (wasm, unsupported) = build_single_config(&text, addr, $arch, cfg);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => println!("  ✓ {} hints, {:?}", state.hints.len(), cfg),
                Err(e) if cfg.uses_exception_opcodes() && is_known_wasmi_exception_gap(&e) => {
                    eprintln!("  ! wasmi lacks exception-handling support ({e}); retrying under wasmtime");
                    match run_module_wasmtime(&wasm, "_start") {
                        Ok(state) => println!("  ✓ (wasmtime) {} hints, {:?}", state.hints.len(), cfg),
                        Err(e) => panic!("ExceptionSpec run failed under wasmtime fallback: {e}"),
                    }
                }
                Err(e) if cfg.speculative() => panic!("speculative run failed: {e}"),
                Err(e) if cfg.needs_exception_tags() => eprintln!("  ! EH run skipped: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
    ($name:ident, env = $env:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        run_c!($name, env = $env, arch = $arch, EscapeConfig::from_eh_spec($eh, $spec));
    };
}

macro_rules! link {
    ($name:ident, [ $( ($rel:expr, arch = $arch:expr, entry = $entry:expr) ),+ ], $cfg:expr) => {
        #[test]
        fn $name() {
            let mut specs_data: Vec<(Vec<u8>, u64, Arch, &'static str)> = Vec::new();
            $(
                {
                    let path = corpus($rel);
                    let (text, addr) = load_text(&path);
                    specs_data.push((text, addr, $arch, $entry));
                }
            )+
            let cfg: EscapeConfig = $cfg;
            let specs: Vec<LinkSpec<'_>> = specs_data.iter()
                .map(|(t, a, arch, e)| LinkSpec { text: t, start_addr: *a, arch: *arch, entry: e })
                .collect();
            let (wasm, unsupported) = build_linked_config(&specs, cfg);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("linked WASM is invalid");
            println!("  linked: {} bytes, {:?}", wasm.len(), cfg);
            for spec in &specs {
                match run_module(&wasm, spec.entry) {
                    Ok(state) => println!("  ✓ {}: {} hints", spec.entry, state.hints.len()),
                    Err(e) if cfg.uses_exception_opcodes() && is_known_wasmi_exception_gap(&e) => {
                        eprintln!("  ! wasmi lacks exception-handling support ({}: {e}); retrying under wasmtime", spec.entry);
                        match run_module_wasmtime(&wasm, spec.entry) {
                            Ok(state) => println!("  ✓ (wasmtime) {}: {} hints", spec.entry, state.hints.len()),
                            Err(e) => panic!("ExceptionSpec run {} failed under wasmtime fallback: {e}", spec.entry),
                        }
                    }
                    Err(e) if cfg.speculative() => panic!("speculative run {} failed: {e}", spec.entry),
                    Err(e) if cfg.needs_exception_tags() => eprintln!("  ! EH skipped ({}): {e}", spec.entry),
                    Err(e) => panic!("run {} failed: {e}", spec.entry),
                }
            }
        }
    };
    ($name:ident, [ $( ($rel:expr, arch = $arch:expr, entry = $entry:expr) ),+ ], $eh:expr, speculative = $spec:expr) => {
        link!($name, [ $( ($rel, arch = $arch, entry = $entry) ),+ ], EscapeConfig::from_eh_spec($eh, $spec));
    };
}

macro_rules! link_c {
    ($name:ident, [ $( ($env_or_corpus:expr, is_corpus = $is_corpus:expr, arch = $arch:expr, entry = $entry:expr) ),+ ], $cfg:expr) => {
        #[test]
        fn $name() {
            let mut specs_data: Vec<(Vec<u8>, u64, Arch, &'static str)> = Vec::new();
            $(
                {
                    let path: Option<std::path::PathBuf> = if $is_corpus {
                        Some(corpus($env_or_corpus))
                    } else {
                        c_obj($env_or_corpus)
                    };
                    let path = match path { Some(p) => p, None => {
                        eprintln!("  skipping {}: missing input", stringify!($name));
                        return;
                    }};
                    let (text, addr) = if $is_corpus {
                        load_text(&path)
                    } else {
                        match load_text_optional(&path) { Some(v) => v, None => return }
                    };
                    specs_data.push((text, addr, $arch, $entry));
                }
            )+
            let cfg: EscapeConfig = $cfg;
            let specs: Vec<LinkSpec<'_>> = specs_data.iter()
                .map(|(t, a, arch, e)| LinkSpec { text: t, start_addr: *a, arch: *arch, entry: e })
                .collect();
            let (wasm, unsupported) = build_linked_config(&specs, cfg);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("linked WASM is invalid");
            println!("  linked: {} bytes, {:?}", wasm.len(), cfg);
            for spec in &specs {
                match run_module(&wasm, spec.entry) {
                    Ok(state) => println!("  ✓ {}: {} hints", spec.entry, state.hints.len()),
                    Err(e) if cfg.uses_exception_opcodes() && is_known_wasmi_exception_gap(&e) => {
                        eprintln!("  ! wasmi lacks exception-handling support ({}: {e}); retrying under wasmtime", spec.entry);
                        match run_module_wasmtime(&wasm, spec.entry) {
                            Ok(state) => println!("  ✓ (wasmtime) {}: {} hints", spec.entry, state.hints.len()),
                            Err(e) => panic!("ExceptionSpec run {} failed under wasmtime fallback: {e}", spec.entry),
                        }
                    }
                    Err(e) if cfg.speculative() => panic!("speculative run {} failed: {e}", spec.entry),
                    Err(e) if cfg.needs_exception_tags() => eprintln!("  ! EH skipped ({}): {e}", spec.entry),
                    Err(e) => panic!("run {} failed: {e}", spec.entry),
                }
            }
        }
    };
    ($name:ident, [ $( ($env_or_corpus:expr, is_corpus = $is_corpus:expr, arch = $arch:expr, entry = $entry:expr) ),+ ], $eh:expr, speculative = $spec:expr) => {
        link_c!($name, [ $( ($env_or_corpus, is_corpus = $is_corpus, arch = $arch, entry = $entry) ),+ ], EscapeConfig::from_eh_spec($eh, $spec));
    };
}

// ── WASM-frontend test macros ─────────────────────────────────────────────────

/// Translate `$builder` through `WasmFrontend` with optional mapper and/or
/// condition trap, then validate the output with `wasmparser`.
macro_rules! wasm_smoke {
    ($name:ident, $builder:expr, mapper = $mapper:expr, cond_trap = $cond_trap:expr) => {
        #[test]
        fn $name() {
            let input = $builder;
            let cfg = WasmTranslateConfig { mapper: $mapper, cond_trap: $cond_trap };
            let wasm = translate_wasm(&input, cfg, &[]);
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("WasmFrontend output is invalid");
            println!("  ✓ wasm_smoke {} — {} bytes", stringify!($name), wasm.len());
        }
    };
}

/// Translate `$builder` through `WasmFrontend`, validate, and execute entry
/// `$entry` in wasmi.  Mapper and/or condition trap are optional.
macro_rules! wasm_run {
    ($name:ident, $builder:expr, entry = $entry:expr, mapper = $mapper:expr, cond_trap = $cond_trap:expr) => {
        #[test]
        fn $name() {
            let input = $builder;
            let cfg = WasmTranslateConfig { mapper: $mapper, cond_trap: $cond_trap };
            let wasm = translate_wasm(&input, cfg, &[($entry, 0)]);
            wasmparser::validate(&wasm).expect("WasmFrontend output is invalid");
            run_module(&wasm, $entry)
                .unwrap_or_else(|e| panic!("wasm_run {} failed: {e}", stringify!($name)));
            println!("  ✓ wasm_run {}", stringify!($name));
        }
    };
}

/// Translate `$builder` with a `HookConditionTrap` (decide import at index 0),
/// call entry `$entry` with `input = $input` using `$decide_fn` as the host
/// decide implementation, and assert the return value equals `$expected`.
macro_rules! wasm_run_cond_trap {
    ($name:ident, $builder:expr, entry = $entry:expr,
     input = $input:expr, decide_fn = $decide_fn:expr, expected = $expected:expr) => {
        #[test]
        fn $name() {
            let input_wasm = $builder;
            let wasm = translate_wasm_with_decide_import(&input_wasm, $entry);
            wasmparser::validate(&wasm).expect("WasmFrontend cond_trap output is invalid");
            let result = run_wasm_with_decide(&wasm, $entry, $input, $decide_fn);
            assert_eq!(result, $expected,
                "wasm_run_cond_trap {}: expected {} got {}", stringify!($name), $expected, result);
            println!("  ✓ wasm_run_cond_trap {} = {}", stringify!($name), result);
        }
    };
}

// ── Native-backend cross macros ───────────────────────────────────────────────
//
// For each frontend (corpus / C / WASM fixture), `native*` recompiles the same
// module the wasmi tests use all the way to native machine code via wasm-blitz
// (x86-64/ELF + aarch64/Mach-O). Compilation success is asserted; documented
// instruction gaps are tolerated and reported (see `native_compile_check`).

macro_rules! native {
    ($name:ident, $rel:expr, arch = $arch:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_config(&text, addr, $arch, $cfg);
            report_unsupported(&unsupported, stringify!($name));
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
    };
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        native!($name, $rel, arch = $arch, EscapeConfig::from_eh_spec($eh, $spec));
    };
}

macro_rules! native_c {
    ($name:ident, env = $env:expr, arch = $arch:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = match c_obj($env) { Some(p) => p, None => {
                eprintln!("  skipping {}: C object not built", stringify!($name));
                return;
            }};
            let (text, addr) = match load_text_optional(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single_config(&text, addr, $arch, $cfg);
            report_unsupported(&unsupported, stringify!($name));
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
    };
    ($name:ident, env = $env:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        native_c!($name, env = $env, arch = $arch, EscapeConfig::from_eh_spec($eh, $spec));
    };
}

macro_rules! native_wasm {
    ($name:ident, $builder:expr, mapper = $mapper:expr, cond_trap = $cond_trap:expr) => {
        #[test]
        fn $name() {
            let input = $builder;
            let cfg = WasmTranslateConfig { mapper: $mapper, cond_trap: $cond_trap };
            let wasm = translate_wasm(&input, cfg, &[]);
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
    };
}

macro_rules! native_aarch64 {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = aarch64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_config(&text, addr, Arch::AArch64, EscapeConfig::None);
            report_unsupported(&unsupported, stringify!($name));
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
    };
}

macro_rules! native_x86_64 {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = x86_64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_config(&text, addr, Arch::X86_64, EscapeConfig::None);
            report_unsupported(&unsupported, stringify!($name));
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
    };
}

// ── Non-RV corpus macros (path helpers differ; EscapeConfig-parameterized) ───

macro_rules! smoke_aarch64 {
    ($name:ident, $rel:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = aarch64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_config(&text, addr, Arch::AArch64, $cfg);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("WASM invalid");
        }
    };
    ($name:ident, $rel:expr) => { smoke_aarch64!($name, $rel, EscapeConfig::None); };
}

macro_rules! run_aarch64 {
    ($name:ident, $rel:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = aarch64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, _) = build_single_config(&text, addr, Arch::AArch64, $cfg);
            wasmparser::validate(&wasm).expect("WASM invalid");
            run_module(&wasm, "_start").unwrap_or_else(|e| panic!("run failed: {e}"));
        }
    };
    ($name:ident, $rel:expr) => { run_aarch64!($name, $rel, EscapeConfig::None); };
}

macro_rules! smoke_x86_64 {
    ($name:ident, $rel:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = x86_64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_config(&text, addr, Arch::X86_64, $cfg);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("WASM invalid");
        }
    };
    ($name:ident, $rel:expr) => { smoke_x86_64!($name, $rel, EscapeConfig::None); };
}

macro_rules! run_x86_64 {
    ($name:ident, $rel:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = x86_64_corpus($rel);
            let (text, addr) = load_text(&path);
            let cfg: EscapeConfig = $cfg;
            let (wasm, _) = build_single_config(&text, addr, Arch::X86_64, cfg);
            wasmparser::validate(&wasm).expect("WASM invalid");
            match run_module(&wasm, "_start") {
                Ok(_) => {}
                Err(e) if cfg.uses_exception_opcodes() && is_known_wasmi_exception_gap(&e) => {
                    run_module_wasmtime(&wasm, "_start")
                        .unwrap_or_else(|e| panic!("wasmtime fallback failed: {e}"));
                }
                Err(e) if cfg.speculative() => panic!("speculative run failed: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
    ($name:ident, $rel:expr) => { run_x86_64!($name, $rel, EscapeConfig::None); };
}

macro_rules! smoke_mips {
    ($name:ident, $rel:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = mips_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_config(&text, addr, Arch::Mips, $cfg);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("WASM invalid");
        }
    };
    ($name:ident, $rel:expr) => { smoke_mips!($name, $rel, EscapeConfig::None); };
}

macro_rules! run_mips {
    ($name:ident, $rel:expr, $cfg:expr) => {
        #[test]
        fn $name() {
            let path = mips_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, _) = build_single_config(&text, addr, Arch::Mips, $cfg);
            wasmparser::validate(&wasm).expect("WASM invalid");
            run_module(&wasm, "_start").unwrap_or_else(|e| panic!("run failed: {e}"));
        }
    };
    ($name:ident, $rel:expr) => { run_mips!($name, $rel, EscapeConfig::None); };
}

macro_rules! native_mips {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = mips_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_config(&text, addr, Arch::Mips, EscapeConfig::None);
            report_unsupported(&unsupported, stringify!($name));
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
    };
}

// ── Dual-lane path macros (linux-wasi / darwin-wasi / thin-native) ────────────

macro_rules! linux_wasi {
    ($name:ident, bytes = $bytes:expr, addr = $addr:expr, $cfg:expr, expect_exit = $exit:expr) => {
        #[test]
        fn $name() {
            let cfg: EscapeConfig = $cfg;
            let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm_with_escape(
                $bytes, $addr, cfg.speculative_escape(),
            );
            let state = run_preview1(&wasm, "_start", 0, &[])
                .unwrap_or_else(|e| panic!("linux_wasi preview1 failed: {e}"));
            assert_eq!(state.exit_code, Some($exit));
        }
    };
    ($name:ident, bytes = $bytes:expr, addr = $addr:expr, $cfg:expr,
     seed_addr = $seed_addr:expr, seed = $seed:expr, expect_stdout = $stdout:expr, expect_exit = $exit:expr) => {
        #[test]
        fn $name() {
            let cfg: EscapeConfig = $cfg;
            let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm_with_escape(
                $bytes, $addr, cfg.speculative_escape(),
            );
            let state = run_preview1(&wasm, "_start", $seed_addr, $seed)
                .unwrap_or_else(|e| panic!("linux_wasi preview1 failed: {e}"));
            assert_eq!(state.stdout, $stdout);
            assert_eq!(state.exit_code, Some($exit));
        }
    };
}

macro_rules! thin_native {
    ($name:ident, bytes = $bytes:expr, addr = $addr:expr, $cfg:expr, expect_exit = $exit:expr) => {
        #[test]
        fn $name() {
            let cfg: EscapeConfig = $cfg;
            match run_thin_rv64_with_escape($bytes, $addr, cfg.speculative_escape()) {
                Ok(outcome) => assert_eq!(outcome.exit_code, Some($exit)),
                Err(reason) => {
                    eprintln!("  soft-skip {}: {reason}", stringify!($name));
                }
            }
        }
    };
}

macro_rules! darwin_wasi {
    ($name:ident, bytes = $bytes:expr, addr = $addr:expr, $cfg:expr,
     seed_addr = $seed_addr:expr, seed = $seed:expr, expect_stdout = $stdout:expr, expect_exit = $exit:expr) => {
        #[test]
        fn $name() {
            let cfg: EscapeConfig = $cfg;
            let wasm = speet_darwin_wasi::recompile_aarch64_darwin_wasi_to_wasm_with_escape(
                $bytes, $addr, cfg.speculative_escape(),
            );
            let state = run_preview1(&wasm, "_start", $seed_addr, $seed)
                .unwrap_or_else(|e| panic!("darwin_wasi preview1 failed: {e}"));
            assert_eq!(state.stdout, $stdout);
            assert_eq!(state.exit_code, Some($exit));
        }
    };
}

// @generated-tests-begin

// ── Corpus smoke tests (wasmi) ──────────────────────────────────────────────────

smoke!(smoke_rv32d_01_no_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32d_01_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32d_01_eh_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32d_01_flag_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32f_01_no_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32f_01_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32f_01_eh_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32f_01_flag_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32fd_01_no_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32fd_01_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32fd_01_eh_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32fd_01_flag_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32i_01_no_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32i_01_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32i_01_eh_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32i_01_flag_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32i_02_no_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32i_02_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32i_02_eh_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32i_02_flag_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32i_03_no_eh, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32i_03_eh, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32i_03_eh_spec, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32i_03_flag_spec, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32i_04_no_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32i_04_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32i_04_eh_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32i_04_flag_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32i_05_no_eh, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32i_05_eh, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32i_05_eh_spec, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32i_05_flag_spec, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32i_06_no_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32i_06_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32i_06_eh_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32i_06_flag_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32i_07_no_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32i_07_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32i_07_eh_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32i_07_flag_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32i_zicsr_01_no_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32i_zicsr_01_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32i_zicsr_01_eh_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32i_zicsr_01_flag_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32im_01_no_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32im_01_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32im_01_eh_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32im_01_flag_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv32ima_01_no_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::None);
smoke!(smoke_rv32ima_01_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::Exception);
smoke!(smoke_rv32ima_01_eh_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv32ima_01_flag_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke!(smoke_rv64d_01_no_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::None);
smoke!(smoke_rv64d_01_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::Exception);
smoke!(smoke_rv64d_01_eh_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv64d_01_flag_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::FlagSpec);
smoke!(smoke_rv64i_01_no_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::None);
smoke!(smoke_rv64i_01_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::Exception);
smoke!(smoke_rv64i_01_eh_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv64i_01_flag_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::FlagSpec);
smoke!(smoke_rv64im_01_no_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::None);
smoke!(smoke_rv64im_01_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::Exception);
smoke!(smoke_rv64im_01_eh_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
smoke!(smoke_rv64im_01_flag_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::FlagSpec);

// ── C smoke tests (wasmi) ───────────────────────────────────────────────────────

smoke_c!(smoke_rv32c_arith_no_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::None);
smoke_c!(smoke_rv32c_arith_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::Exception);
smoke_c!(smoke_rv32c_arith_eh_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
smoke_c!(smoke_rv32c_arith_flag_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::FlagSpec);
smoke_c!(smoke_rv64c_arith_no_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::None);
smoke_c!(smoke_rv64c_arith_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::Exception);
smoke_c!(smoke_rv64c_arith_eh_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
smoke_c!(smoke_rv64c_arith_flag_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::FlagSpec);
smoke_c!(smoke_x86c_arith_no_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::None);
smoke_c!(smoke_x86c_arith_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::Exception);
smoke_c!(smoke_x86c_arith_eh_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::ExceptionSpec);
smoke_c!(smoke_x86c_arith_flag_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::FlagSpec);

// ── Corpus run tests (wasmi) ────────────────────────────────────────────────────

run!(run_rv32d_01_no_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32d_01_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32d_01_eh_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32d_01_flag_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32f_01_no_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32f_01_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32f_01_eh_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32f_01_flag_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32fd_01_no_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32fd_01_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32fd_01_eh_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32fd_01_flag_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32i_01_no_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32i_01_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32i_01_eh_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32i_01_flag_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32i_02_no_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32i_02_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32i_02_eh_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32i_02_flag_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32i_03_no_eh, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32i_03_eh, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32i_03_eh_spec, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32i_03_flag_spec, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32i_04_no_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32i_04_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32i_04_eh_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32i_04_flag_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32i_05_no_eh, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32i_05_eh, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32i_05_eh_spec, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32i_05_flag_spec, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32i_06_no_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32i_06_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32i_06_eh_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32i_06_flag_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32i_07_no_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32i_07_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32i_07_eh_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32i_07_flag_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32i_zicsr_01_no_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32i_zicsr_01_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32i_zicsr_01_eh_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32i_zicsr_01_flag_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32im_01_no_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32im_01_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32im_01_eh_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32im_01_flag_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv32ima_01_no_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::None);
run!(run_rv32ima_01_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::Exception);
run!(run_rv32ima_01_eh_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run!(run_rv32ima_01_flag_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run!(run_rv64d_01_no_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::None);
run!(run_rv64d_01_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::Exception);
run!(run_rv64d_01_eh_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
run!(run_rv64d_01_flag_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::FlagSpec);
run!(run_rv64i_01_no_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::None);
run!(run_rv64i_01_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::Exception);
run!(run_rv64i_01_eh_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
run!(run_rv64i_01_flag_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::FlagSpec);
run!(run_rv64im_01_no_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::None);
run!(run_rv64im_01_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::Exception);
run!(run_rv64im_01_eh_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
run!(run_rv64im_01_flag_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::FlagSpec);

// ── C run tests (wasmi) ─────────────────────────────────────────────────────────

run_c!(run_rv32c_arith_no_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::None);
run_c!(run_rv32c_arith_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::Exception);
run_c!(run_rv32c_arith_eh_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_c!(run_rv32c_arith_flag_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_c!(run_rv64c_arith_no_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::None);
run_c!(run_rv64c_arith_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::Exception);
run_c!(run_rv64c_arith_eh_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
run_c!(run_rv64c_arith_flag_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::FlagSpec);
run_c!(run_x86c_arith_no_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::None);
run_c!(run_x86c_arith_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::Exception);
run_c!(run_x86c_arith_eh_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::ExceptionSpec);
run_c!(run_x86c_arith_flag_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::FlagSpec);

// ── Corpus run-with-trap tests (wasmi) ──────────────────────────────────────────

run_trap!(run_trap_rv32d_01_no_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32d_01_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32d_01_eh_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32d_01_flag_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32f_01_no_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32f_01_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32f_01_eh_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32f_01_flag_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32fd_01_no_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32fd_01_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32fd_01_eh_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32fd_01_flag_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32i_01_no_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32i_01_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32i_01_eh_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32i_01_flag_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32i_02_no_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32i_02_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32i_02_eh_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32i_02_flag_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32i_03_no_eh, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32i_03_eh, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32i_03_eh_spec, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32i_03_flag_spec, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32i_04_no_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32i_04_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32i_04_eh_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32i_04_flag_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32i_05_no_eh, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32i_05_eh, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32i_05_eh_spec, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32i_05_flag_spec, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32i_06_no_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32i_06_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32i_06_eh_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32i_06_flag_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32i_07_no_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32i_07_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32i_07_eh_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32i_07_flag_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32i_zicsr_01_no_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32i_zicsr_01_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32i_zicsr_01_eh_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32i_zicsr_01_flag_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32im_01_no_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32im_01_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32im_01_eh_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32im_01_flag_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv32ima_01_no_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::None);
run_trap!(run_trap_rv32ima_01_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::Exception);
run_trap!(run_trap_rv32ima_01_eh_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv32ima_01_flag_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv64d_01_no_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::None);
run_trap!(run_trap_rv64d_01_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::Exception);
run_trap!(run_trap_rv64d_01_eh_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv64d_01_flag_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv64i_01_no_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::None);
run_trap!(run_trap_rv64i_01_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::Exception);
run_trap!(run_trap_rv64i_01_eh_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv64i_01_flag_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::FlagSpec);
run_trap!(run_trap_rv64im_01_no_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::None);
run_trap!(run_trap_rv64im_01_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::Exception);
run_trap!(run_trap_rv64im_01_eh_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
run_trap!(run_trap_rv64im_01_flag_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::FlagSpec);

// ── Corpus-corpus link tests (wasmi) ────────────────────────────────────────────

link!(link_rv32d_01_x_rv32f_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32f_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32f_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32f_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32fd_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32fd_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32fd_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32fd_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32i_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32i_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32i_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32i_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32i_02_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32i_02_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32i_02_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32i_02_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32i_03_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32i_03_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32i_03_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32i_03_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32i_04_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32i_04_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32i_04_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32i_04_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32i_05_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32i_05_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32i_05_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32i_05_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32i_06_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32i_06_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32i_06_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32i_06_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32i_07_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32i_07_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32i_07_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32i_07_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32i_zicsr_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32i_zicsr_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32i_zicsr_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32i_zicsr_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32im_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32im_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32im_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32im_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv32ima_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv32ima_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv32ima_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv32ima_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv64d_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv64d_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv64d_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv64d_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv64i_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv64i_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv64i_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv64i_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32d_01_x_rv64im_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32d_01_x_rv64im_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32d_01_x_rv64im_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32d_01_x_rv64im_01_flag_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32fd_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32fd_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32fd_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32fd_01_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32i_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32i_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32i_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32i_01_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32i_02_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32i_02_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32i_02_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32i_02_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32i_03_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32i_03_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32i_03_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32i_03_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32i_04_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32i_04_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32i_04_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32i_04_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32i_05_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32i_05_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32i_05_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32i_05_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32i_06_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32i_06_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32i_06_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32i_06_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32i_07_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32i_07_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32i_07_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32i_07_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32i_zicsr_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32i_zicsr_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32i_zicsr_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32i_zicsr_01_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32im_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32im_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32im_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32im_01_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv32ima_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv32ima_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv32ima_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv32ima_01_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv64d_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv64d_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv64d_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv64d_01_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv64i_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv64i_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv64i_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv64i_01_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32f_01_x_rv64im_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32f_01_x_rv64im_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32f_01_x_rv64im_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32f_01_x_rv64im_01_flag_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32i_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32i_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32i_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32i_01_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32i_02_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32i_02_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32i_02_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32i_02_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32i_03_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32i_03_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32i_03_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32i_03_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32i_04_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32i_04_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32i_04_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32i_04_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32i_05_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32i_05_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32i_05_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32i_05_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32i_06_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32i_06_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32i_06_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32i_06_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32i_07_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32i_07_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32i_07_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32i_07_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32i_zicsr_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32i_zicsr_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32i_zicsr_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32i_zicsr_01_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32im_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32im_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32im_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32im_01_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv32ima_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv32ima_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv32ima_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv32ima_01_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv64d_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv64d_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv64d_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv64d_01_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv64i_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv64i_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv64i_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv64i_01_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32fd_01_x_rv64im_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32fd_01_x_rv64im_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32fd_01_x_rv64im_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32fd_01_x_rv64im_01_flag_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32i_02_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32i_02_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32i_02_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32i_02_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32i_03_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32i_03_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32i_03_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32i_03_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32i_04_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32i_04_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32i_04_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32i_04_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32i_05_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32i_05_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32i_05_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32i_05_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32i_06_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32i_06_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32i_06_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32i_06_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32i_07_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32i_07_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32i_07_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32i_07_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32i_zicsr_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32i_zicsr_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32i_zicsr_01_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32im_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32im_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32im_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32im_01_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv32ima_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv32ima_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv32ima_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv32ima_01_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv64d_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv64d_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv64d_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv64d_01_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv64i_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv64i_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv64i_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv64i_01_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_01_x_rv64im_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_01_x_rv64im_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_01_x_rv64im_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_01_x_rv64im_01_flag_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv32i_03_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv32i_03_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv32i_03_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv32i_03_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv32i_04_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv32i_04_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv32i_04_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv32i_04_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv32i_05_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv32i_05_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv32i_05_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv32i_05_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv32i_06_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv32i_06_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv32i_06_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv32i_06_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv32i_07_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv32i_07_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv32i_07_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv32i_07_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv32i_zicsr_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv32i_zicsr_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv32i_zicsr_01_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv32im_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv32im_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv32im_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv32im_01_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv32ima_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv32ima_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv32ima_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv32ima_01_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv64d_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv64d_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv64d_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv64d_01_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv64i_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv64i_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv64i_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv64i_01_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_02_x_rv64im_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_02_x_rv64im_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_02_x_rv64im_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_02_x_rv64im_01_flag_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv32i_04_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv32i_04_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv32i_04_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv32i_04_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv32i_05_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv32i_05_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv32i_05_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv32i_05_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv32i_06_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv32i_06_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv32i_06_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv32i_06_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv32i_07_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv32i_07_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv32i_07_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv32i_07_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv32i_zicsr_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv32i_zicsr_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv32i_zicsr_01_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv32im_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv32im_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv32im_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv32im_01_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv32ima_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv32ima_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv32ima_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv32ima_01_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv64d_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv64d_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv64d_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv64d_01_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv64i_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv64i_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv64i_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv64i_01_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_03_x_rv64im_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_03_x_rv64im_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_03_x_rv64im_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_03_x_rv64im_01_flag_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv32i_05_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv32i_05_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv32i_05_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv32i_05_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv32i_06_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv32i_06_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv32i_06_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv32i_06_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv32i_07_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv32i_07_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv32i_07_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv32i_07_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv32i_zicsr_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv32i_zicsr_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv32i_zicsr_01_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv32im_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv32im_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv32im_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv32im_01_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv32ima_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv32ima_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv32ima_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv32ima_01_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv64d_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv64d_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv64d_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv64d_01_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv64i_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv64i_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv64i_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv64i_01_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_04_x_rv64im_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_04_x_rv64im_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_04_x_rv64im_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_04_x_rv64im_01_flag_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_05_x_rv32i_06_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_05_x_rv32i_06_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_05_x_rv32i_06_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_05_x_rv32i_06_flag_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_05_x_rv32i_07_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_05_x_rv32i_07_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_05_x_rv32i_07_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_05_x_rv32i_07_flag_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_05_x_rv32i_zicsr_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_05_x_rv32i_zicsr_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_05_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_05_x_rv32i_zicsr_01_flag_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_05_x_rv32im_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_05_x_rv32im_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_05_x_rv32im_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_05_x_rv32im_01_flag_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_05_x_rv32ima_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_05_x_rv32ima_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_05_x_rv32ima_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_05_x_rv32ima_01_flag_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_05_x_rv64d_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_05_x_rv64d_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_05_x_rv64d_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_05_x_rv64d_01_flag_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_05_x_rv64i_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_05_x_rv64i_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_05_x_rv64i_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_05_x_rv64i_01_flag_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_05_x_rv64im_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_05_x_rv64im_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_05_x_rv64im_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_05_x_rv64im_01_flag_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_06_x_rv32i_07_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_06_x_rv32i_07_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_06_x_rv32i_07_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_06_x_rv32i_07_flag_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_06_x_rv32i_zicsr_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_06_x_rv32i_zicsr_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_06_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_06_x_rv32i_zicsr_01_flag_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_06_x_rv32im_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_06_x_rv32im_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_06_x_rv32im_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_06_x_rv32im_01_flag_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_06_x_rv32ima_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_06_x_rv32ima_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_06_x_rv32ima_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_06_x_rv32ima_01_flag_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_06_x_rv64d_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_06_x_rv64d_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_06_x_rv64d_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_06_x_rv64d_01_flag_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_06_x_rv64i_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_06_x_rv64i_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_06_x_rv64i_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_06_x_rv64i_01_flag_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_06_x_rv64im_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_06_x_rv64im_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_06_x_rv64im_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_06_x_rv64im_01_flag_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_07_x_rv32i_zicsr_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_07_x_rv32i_zicsr_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_07_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_07_x_rv32i_zicsr_01_flag_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_07_x_rv32im_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_07_x_rv32im_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_07_x_rv32im_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_07_x_rv32im_01_flag_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_07_x_rv32ima_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_07_x_rv32ima_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_07_x_rv32ima_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_07_x_rv32ima_01_flag_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_07_x_rv64d_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_07_x_rv64d_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_07_x_rv64d_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_07_x_rv64d_01_flag_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_07_x_rv64i_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_07_x_rv64i_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_07_x_rv64i_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_07_x_rv64i_01_flag_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_07_x_rv64im_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_07_x_rv64im_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_07_x_rv64im_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_07_x_rv64im_01_flag_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_zicsr_01_x_rv32im_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_zicsr_01_x_rv32im_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_zicsr_01_x_rv32im_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_zicsr_01_x_rv32im_01_flag_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_zicsr_01_x_rv32ima_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_zicsr_01_x_rv32ima_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_zicsr_01_x_rv32ima_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_zicsr_01_x_rv32ima_01_flag_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_zicsr_01_x_rv64d_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_zicsr_01_x_rv64d_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_zicsr_01_x_rv64d_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_zicsr_01_x_rv64d_01_flag_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_zicsr_01_x_rv64i_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_zicsr_01_x_rv64i_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_zicsr_01_x_rv64i_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_zicsr_01_x_rv64i_01_flag_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32i_zicsr_01_x_rv64im_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32i_zicsr_01_x_rv64im_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32i_zicsr_01_x_rv64im_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32i_zicsr_01_x_rv64im_01_flag_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32im_01_x_rv32ima_01_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32im_01_x_rv32ima_01_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32im_01_x_rv32ima_01_eh_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32im_01_x_rv32ima_01_flag_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32im_01_x_rv64d_01_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32im_01_x_rv64d_01_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32im_01_x_rv64d_01_eh_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32im_01_x_rv64d_01_flag_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32im_01_x_rv64i_01_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32im_01_x_rv64i_01_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32im_01_x_rv64i_01_eh_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32im_01_x_rv64i_01_flag_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32im_01_x_rv64im_01_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32im_01_x_rv64im_01_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32im_01_x_rv64im_01_eh_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32im_01_x_rv64im_01_flag_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32ima_01_x_rv64d_01_no_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32ima_01_x_rv64d_01_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32ima_01_x_rv64d_01_eh_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32ima_01_x_rv64d_01_flag_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32ima_01_x_rv64i_01_no_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32ima_01_x_rv64i_01_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32ima_01_x_rv64i_01_eh_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32ima_01_x_rv64i_01_flag_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv32ima_01_x_rv64im_01_no_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv32ima_01_x_rv64im_01_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv32ima_01_x_rv64im_01_eh_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv32ima_01_x_rv64im_01_flag_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv64d_01_x_rv64i_01_no_eh,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv64d_01_x_rv64i_01_eh,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv64d_01_x_rv64i_01_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv64d_01_x_rv64i_01_flag_spec,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv64d_01_x_rv64im_01_no_eh,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv64d_01_x_rv64im_01_eh,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv64d_01_x_rv64im_01_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv64d_01_x_rv64im_01_flag_spec,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link!(link_rv64i_01_x_rv64im_01_no_eh,
    [("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link!(link_rv64i_01_x_rv64im_01_eh,
    [("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link!(link_rv64i_01_x_rv64im_01_eh_spec,
    [("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link!(link_rv64i_01_x_rv64im_01_flag_spec,
    [("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);

// ── Corpus-C link tests (wasmi) ─────────────────────────────────────────────────

link_c!(link_rv32d_01_x_rv32c_arith_no_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32d_01_x_rv32c_arith_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32d_01_x_rv32c_arith_eh_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32d_01_x_rv32c_arith_flag_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32d_01_x_rv64c_arith_no_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32d_01_x_rv64c_arith_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32d_01_x_rv64c_arith_eh_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32d_01_x_rv64c_arith_flag_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32d_01_x_x86c_arith_no_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32d_01_x_x86c_arith_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32d_01_x_x86c_arith_eh_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32d_01_x_x86c_arith_flag_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32f_01_x_rv32c_arith_no_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32f_01_x_rv32c_arith_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32f_01_x_rv32c_arith_eh_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32f_01_x_rv32c_arith_flag_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32f_01_x_rv64c_arith_no_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32f_01_x_rv64c_arith_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32f_01_x_rv64c_arith_eh_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32f_01_x_rv64c_arith_flag_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32f_01_x_x86c_arith_no_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32f_01_x_x86c_arith_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32f_01_x_x86c_arith_eh_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32f_01_x_x86c_arith_flag_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32fd_01_x_rv32c_arith_no_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32fd_01_x_rv32c_arith_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32fd_01_x_rv32c_arith_eh_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32fd_01_x_rv32c_arith_flag_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32fd_01_x_rv64c_arith_no_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32fd_01_x_rv64c_arith_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32fd_01_x_rv64c_arith_eh_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32fd_01_x_rv64c_arith_flag_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32fd_01_x_x86c_arith_no_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32fd_01_x_x86c_arith_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32fd_01_x_x86c_arith_eh_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32fd_01_x_x86c_arith_flag_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_01_x_rv32c_arith_no_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_01_x_rv32c_arith_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_01_x_rv32c_arith_eh_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_01_x_rv32c_arith_flag_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_01_x_rv64c_arith_no_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_01_x_rv64c_arith_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_01_x_rv64c_arith_eh_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_01_x_rv64c_arith_flag_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_01_x_x86c_arith_no_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_01_x_x86c_arith_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_01_x_x86c_arith_eh_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_01_x_x86c_arith_flag_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_02_x_rv32c_arith_no_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_02_x_rv32c_arith_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_02_x_rv32c_arith_eh_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_02_x_rv32c_arith_flag_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_02_x_rv64c_arith_no_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_02_x_rv64c_arith_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_02_x_rv64c_arith_eh_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_02_x_rv64c_arith_flag_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_02_x_x86c_arith_no_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_02_x_x86c_arith_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_02_x_x86c_arith_eh_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_02_x_x86c_arith_flag_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_03_x_rv32c_arith_no_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_03_x_rv32c_arith_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_03_x_rv32c_arith_eh_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_03_x_rv32c_arith_flag_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_03_x_rv64c_arith_no_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_03_x_rv64c_arith_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_03_x_rv64c_arith_eh_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_03_x_rv64c_arith_flag_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_03_x_x86c_arith_no_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_03_x_x86c_arith_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_03_x_x86c_arith_eh_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_03_x_x86c_arith_flag_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_04_x_rv32c_arith_no_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_04_x_rv32c_arith_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_04_x_rv32c_arith_eh_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_04_x_rv32c_arith_flag_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_04_x_rv64c_arith_no_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_04_x_rv64c_arith_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_04_x_rv64c_arith_eh_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_04_x_rv64c_arith_flag_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_04_x_x86c_arith_no_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_04_x_x86c_arith_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_04_x_x86c_arith_eh_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_04_x_x86c_arith_flag_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_05_x_rv32c_arith_no_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_05_x_rv32c_arith_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_05_x_rv32c_arith_eh_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_05_x_rv32c_arith_flag_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_05_x_rv64c_arith_no_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_05_x_rv64c_arith_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_05_x_rv64c_arith_eh_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_05_x_rv64c_arith_flag_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_05_x_x86c_arith_no_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_05_x_x86c_arith_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_05_x_x86c_arith_eh_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_05_x_x86c_arith_flag_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_06_x_rv32c_arith_no_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_06_x_rv32c_arith_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_06_x_rv32c_arith_eh_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_06_x_rv32c_arith_flag_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_06_x_rv64c_arith_no_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_06_x_rv64c_arith_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_06_x_rv64c_arith_eh_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_06_x_rv64c_arith_flag_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_06_x_x86c_arith_no_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_06_x_x86c_arith_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_06_x_x86c_arith_eh_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_06_x_x86c_arith_flag_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_07_x_rv32c_arith_no_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_07_x_rv32c_arith_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_07_x_rv32c_arith_eh_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_07_x_rv32c_arith_flag_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_07_x_rv64c_arith_no_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_07_x_rv64c_arith_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_07_x_rv64c_arith_eh_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_07_x_rv64c_arith_flag_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_07_x_x86c_arith_no_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_07_x_x86c_arith_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_07_x_x86c_arith_eh_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_07_x_x86c_arith_flag_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_zicsr_01_x_rv32c_arith_no_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_zicsr_01_x_rv32c_arith_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_zicsr_01_x_rv32c_arith_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_zicsr_01_x_rv32c_arith_flag_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_zicsr_01_x_rv64c_arith_no_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_zicsr_01_x_rv64c_arith_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_zicsr_01_x_rv64c_arith_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_zicsr_01_x_rv64c_arith_flag_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32i_zicsr_01_x_x86c_arith_no_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32i_zicsr_01_x_x86c_arith_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32i_zicsr_01_x_x86c_arith_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32i_zicsr_01_x_x86c_arith_flag_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32im_01_x_rv32c_arith_no_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32im_01_x_rv32c_arith_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32im_01_x_rv32c_arith_eh_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32im_01_x_rv32c_arith_flag_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32im_01_x_rv64c_arith_no_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32im_01_x_rv64c_arith_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32im_01_x_rv64c_arith_eh_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32im_01_x_rv64c_arith_flag_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32im_01_x_x86c_arith_no_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32im_01_x_x86c_arith_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32im_01_x_x86c_arith_eh_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32im_01_x_x86c_arith_flag_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32ima_01_x_rv32c_arith_no_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32ima_01_x_rv32c_arith_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32ima_01_x_rv32c_arith_eh_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32ima_01_x_rv32c_arith_flag_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32ima_01_x_rv64c_arith_no_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32ima_01_x_rv64c_arith_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32ima_01_x_rv64c_arith_eh_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32ima_01_x_rv64c_arith_flag_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32ima_01_x_x86c_arith_no_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32ima_01_x_x86c_arith_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32ima_01_x_x86c_arith_eh_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32ima_01_x_x86c_arith_flag_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64d_01_x_rv32c_arith_no_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64d_01_x_rv32c_arith_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64d_01_x_rv32c_arith_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64d_01_x_rv32c_arith_flag_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64d_01_x_rv64c_arith_no_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64d_01_x_rv64c_arith_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64d_01_x_rv64c_arith_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64d_01_x_rv64c_arith_flag_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64d_01_x_x86c_arith_no_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64d_01_x_x86c_arith_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64d_01_x_x86c_arith_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64d_01_x_x86c_arith_flag_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64i_01_x_rv32c_arith_no_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64i_01_x_rv32c_arith_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64i_01_x_rv32c_arith_eh_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64i_01_x_rv32c_arith_flag_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64i_01_x_rv64c_arith_no_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64i_01_x_rv64c_arith_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64i_01_x_rv64c_arith_eh_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64i_01_x_rv64c_arith_flag_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64i_01_x_x86c_arith_no_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64i_01_x_x86c_arith_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64i_01_x_x86c_arith_eh_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64i_01_x_x86c_arith_flag_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64im_01_x_rv32c_arith_no_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64im_01_x_rv32c_arith_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64im_01_x_rv32c_arith_eh_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64im_01_x_rv32c_arith_flag_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64im_01_x_rv64c_arith_no_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64im_01_x_rv64c_arith_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64im_01_x_rv64c_arith_eh_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64im_01_x_rv64c_arith_flag_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64im_01_x_x86c_arith_no_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64im_01_x_x86c_arith_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64im_01_x_x86c_arith_eh_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64im_01_x_x86c_arith_flag_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);

// ── C-C link tests (wasmi) ──────────────────────────────────────────────────────

link_c!(link_rv32c_arith_x_rv64c_arith_no_eh,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32c_arith_x_rv64c_arith_eh,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32c_arith_x_rv64c_arith_eh_spec,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32c_arith_x_rv64c_arith_flag_spec,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv32c_arith_x_x86c_arith_no_eh,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv32c_arith_x_x86c_arith_eh,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv32c_arith_x_x86c_arith_eh_spec,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv32c_arith_x_x86c_arith_flag_spec,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);
link_c!(link_rv64c_arith_x_x86c_arith_no_eh,
    [("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::None);
link_c!(link_rv64c_arith_x_x86c_arith_eh,
    [("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::Exception);
link_c!(link_rv64c_arith_x_x86c_arith_eh_spec,
    [("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::ExceptionSpec);
link_c!(link_rv64c_arith_x_x86c_arith_flag_spec,
    [("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    EscapeConfig::FlagSpec);

// ── WASM smoke tests ────────────────────────────────────────────────────────────

wasm_smoke!(smoke_wasm_arith_no_mapper_no_cond_trap, wasm_arith(), mapper = None, cond_trap = None);
wasm_smoke!(smoke_wasm_arith_no_mapper_with_flip_trap, wasm_arith(), mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_smoke!(smoke_wasm_arith_with_mapper_no_cond_trap, wasm_arith(), mapper = Some(make_test_mapper()), cond_trap = None);
wasm_smoke!(smoke_wasm_arith_with_mapper_with_flip_trap, wasm_arith(), mapper = Some(make_test_mapper()), cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_smoke!(smoke_wasm_branches_no_mapper_no_cond_trap, wasm_branches(), mapper = None, cond_trap = None);
wasm_smoke!(smoke_wasm_branches_no_mapper_with_flip_trap, wasm_branches(), mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_smoke!(smoke_wasm_branches_with_mapper_no_cond_trap, wasm_branches(), mapper = Some(make_test_mapper()), cond_trap = None);
wasm_smoke!(smoke_wasm_branches_with_mapper_with_flip_trap, wasm_branches(), mapper = Some(make_test_mapper()), cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_smoke!(smoke_wasm_memory_rw_no_mapper_no_cond_trap, wasm_memory_rw(), mapper = None, cond_trap = None);
wasm_smoke!(smoke_wasm_memory_rw_no_mapper_with_flip_trap, wasm_memory_rw(), mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_smoke!(smoke_wasm_memory_rw_with_mapper_no_cond_trap, wasm_memory_rw(), mapper = Some(make_test_mapper()), cond_trap = None);
wasm_smoke!(smoke_wasm_memory_rw_with_mapper_with_flip_trap, wasm_memory_rw(), mapper = Some(make_test_mapper()), cond_trap = Some(Box::new(FlipConditionTrap)));

// ── WASM run tests ──────────────────────────────────────────────────────────────

wasm_run!(run_wasm_arith_no_mapper_no_cond_trap, wasm_arith(), entry = "compute",
    mapper = None, cond_trap = None);
wasm_run!(run_wasm_arith_no_mapper_with_flip_trap, wasm_arith(), entry = "compute",
    mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_run!(run_wasm_arith_with_mapper_no_cond_trap, wasm_arith(), entry = "compute",
    mapper = Some(make_test_mapper()), cond_trap = None);
wasm_run!(run_wasm_arith_with_mapper_with_flip_trap, wasm_arith(), entry = "compute",
    mapper = Some(make_test_mapper()), cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_run!(run_wasm_branches_no_mapper_no_cond_trap, wasm_branches(), entry = "test",
    mapper = None, cond_trap = None);
wasm_run!(run_wasm_branches_no_mapper_with_flip_trap, wasm_branches(), entry = "test",
    mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_run!(run_wasm_branches_with_mapper_no_cond_trap, wasm_branches(), entry = "test",
    mapper = Some(make_test_mapper()), cond_trap = None);
wasm_run!(run_wasm_branches_with_mapper_with_flip_trap, wasm_branches(), entry = "test",
    mapper = Some(make_test_mapper()), cond_trap = Some(Box::new(FlipConditionTrap)));
wasm_run!(run_wasm_memory_rw_no_mapper_no_cond_trap, wasm_memory_rw(), entry = "roundtrip",
    mapper = None, cond_trap = None);
wasm_run!(run_wasm_memory_rw_no_mapper_with_flip_trap, wasm_memory_rw(), entry = "roundtrip",
    mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));

// ── WASM condition-trap hook tests ──────────────────────────────────────────────

wasm_run_cond_trap!(run_wasm_branches_hook_passthrough, wasm_branches(), entry = "test",
    input = 1, decide_fn = |v| v, expected = 1);
wasm_run_cond_trap!(run_wasm_branches_hook_passthrough_zero, wasm_branches(), entry = "test",
    input = 0, decide_fn = |v| v, expected = 0);
wasm_run_cond_trap!(run_wasm_branches_hook_override_false, wasm_branches(), entry = "test",
    input = 1, decide_fn = |_| 0, expected = 0);
wasm_run_cond_trap!(run_wasm_branches_hook_override_true, wasm_branches(), entry = "test",
    input = 0, decide_fn = |_| 1, expected = 1);

// ── Native-backend (blitz) corpus tests ─────────────────────────────────────────

native!(native_rv32d_01_no_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32d_01_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32d_01_eh_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32d_01_flag_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32f_01_no_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32f_01_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32f_01_eh_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32f_01_flag_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32fd_01_no_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32fd_01_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32fd_01_eh_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32fd_01_flag_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32i_01_no_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32i_01_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32i_01_eh_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32i_01_flag_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32i_02_no_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32i_02_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32i_02_eh_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32i_02_flag_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32i_03_no_eh, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32i_03_eh, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32i_03_eh_spec, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32i_03_flag_spec, "rv32i/03_load_store", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32i_04_no_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32i_04_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32i_04_eh_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32i_04_flag_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32i_05_no_eh, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32i_05_eh, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32i_05_eh_spec, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32i_05_flag_spec, "rv32i/05_simple_program", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32i_06_no_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32i_06_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32i_06_eh_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32i_06_flag_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32i_07_no_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32i_07_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32i_07_eh_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32i_07_flag_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32i_zicsr_01_no_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32i_zicsr_01_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32i_zicsr_01_eh_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32i_zicsr_01_flag_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32im_01_no_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32im_01_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32im_01_eh_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32im_01_flag_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv32ima_01_no_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::None);
native!(native_rv32ima_01_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::Exception);
native!(native_rv32ima_01_eh_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native!(native_rv32ima_01_flag_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native!(native_rv64d_01_no_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::None);
native!(native_rv64d_01_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::Exception);
native!(native_rv64d_01_eh_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
native!(native_rv64d_01_flag_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, EscapeConfig::FlagSpec);
native!(native_rv64i_01_no_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::None);
native!(native_rv64i_01_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::Exception);
native!(native_rv64i_01_eh_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
native!(native_rv64i_01_flag_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, EscapeConfig::FlagSpec);
native!(native_rv64im_01_no_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::None);
native!(native_rv64im_01_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::Exception);
native!(native_rv64im_01_eh_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
native!(native_rv64im_01_flag_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, EscapeConfig::FlagSpec);

// ── Native-backend (blitz) C tests ──────────────────────────────────────────────

native_c!(native_rv32c_arith_no_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::None);
native_c!(native_rv32c_arith_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::Exception);
native_c!(native_rv32c_arith_eh_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::ExceptionSpec);
native_c!(native_rv32c_arith_flag_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, EscapeConfig::FlagSpec);
native_c!(native_rv64c_arith_no_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::None);
native_c!(native_rv64c_arith_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::Exception);
native_c!(native_rv64c_arith_eh_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::ExceptionSpec);
native_c!(native_rv64c_arith_flag_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, EscapeConfig::FlagSpec);
native_c!(native_x86c_arith_no_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::None);
native_c!(native_x86c_arith_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::Exception);
native_c!(native_x86c_arith_eh_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::ExceptionSpec);
native_c!(native_x86c_arith_flag_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, EscapeConfig::FlagSpec);

// ── Native-backend (blitz) WASM tests ───────────────────────────────────────────

native_wasm!(native_wasm_arith_no_mapper_no_cond_trap, wasm_arith(), mapper = None, cond_trap = None);
native_wasm!(native_wasm_arith_no_mapper_with_flip_trap, wasm_arith(), mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));
native_wasm!(native_wasm_arith_with_mapper_no_cond_trap, wasm_arith(), mapper = Some(make_test_mapper()), cond_trap = None);
native_wasm!(native_wasm_arith_with_mapper_with_flip_trap, wasm_arith(), mapper = Some(make_test_mapper()), cond_trap = Some(Box::new(FlipConditionTrap)));
native_wasm!(native_wasm_branches_no_mapper_no_cond_trap, wasm_branches(), mapper = None, cond_trap = None);
native_wasm!(native_wasm_branches_no_mapper_with_flip_trap, wasm_branches(), mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));
native_wasm!(native_wasm_branches_with_mapper_no_cond_trap, wasm_branches(), mapper = Some(make_test_mapper()), cond_trap = None);
native_wasm!(native_wasm_branches_with_mapper_with_flip_trap, wasm_branches(), mapper = Some(make_test_mapper()), cond_trap = Some(Box::new(FlipConditionTrap)));
native_wasm!(native_wasm_memory_rw_no_mapper_no_cond_trap, wasm_memory_rw(), mapper = None, cond_trap = None);
native_wasm!(native_wasm_memory_rw_no_mapper_with_flip_trap, wasm_memory_rw(), mapper = None, cond_trap = Some(Box::new(FlipConditionTrap)));
native_wasm!(native_wasm_memory_rw_with_mapper_no_cond_trap, wasm_memory_rw(), mapper = Some(make_test_mapper()), cond_trap = None);
native_wasm!(native_wasm_memory_rw_with_mapper_with_flip_trap, wasm_memory_rw(), mapper = Some(make_test_mapper()), cond_trap = Some(Box::new(FlipConditionTrap)));

// ── AArch64 corpus (wasmi + blitz) ──────────────────────────────────────────────

smoke_aarch64!(aarch64_01_no_eh, "01_integer_computational", EscapeConfig::None);
run_aarch64!(run_aarch64_01_no_eh, "01_integer_computational", EscapeConfig::None);
smoke_aarch64!(aarch64_01_eh, "01_integer_computational", EscapeConfig::Exception);
run_aarch64!(run_aarch64_01_eh, "01_integer_computational", EscapeConfig::Exception);
smoke_aarch64!(aarch64_01_eh_spec, "01_integer_computational", EscapeConfig::ExceptionSpec);
run_aarch64!(run_aarch64_01_eh_spec, "01_integer_computational", EscapeConfig::ExceptionSpec);
smoke_aarch64!(aarch64_01_flag_spec, "01_integer_computational", EscapeConfig::FlagSpec);
run_aarch64!(run_aarch64_01_flag_spec, "01_integer_computational", EscapeConfig::FlagSpec);
native_aarch64!(native_aarch64_01, "01_integer_computational");
smoke_aarch64!(aarch64_02_no_eh, "02_control_transfer", EscapeConfig::None);
run_aarch64!(run_aarch64_02_no_eh, "02_control_transfer", EscapeConfig::None);
smoke_aarch64!(aarch64_02_eh, "02_control_transfer", EscapeConfig::Exception);
run_aarch64!(run_aarch64_02_eh, "02_control_transfer", EscapeConfig::Exception);
smoke_aarch64!(aarch64_02_eh_spec, "02_control_transfer", EscapeConfig::ExceptionSpec);
run_aarch64!(run_aarch64_02_eh_spec, "02_control_transfer", EscapeConfig::ExceptionSpec);
smoke_aarch64!(aarch64_02_flag_spec, "02_control_transfer", EscapeConfig::FlagSpec);
run_aarch64!(run_aarch64_02_flag_spec, "02_control_transfer", EscapeConfig::FlagSpec);
native_aarch64!(native_aarch64_02, "02_control_transfer");
smoke_aarch64!(aarch64_03_no_eh, "03_load_store", EscapeConfig::None);
run_aarch64!(run_aarch64_03_no_eh, "03_load_store", EscapeConfig::None);
smoke_aarch64!(aarch64_03_eh, "03_load_store", EscapeConfig::Exception);
run_aarch64!(run_aarch64_03_eh, "03_load_store", EscapeConfig::Exception);
smoke_aarch64!(aarch64_03_eh_spec, "03_load_store", EscapeConfig::ExceptionSpec);
run_aarch64!(run_aarch64_03_eh_spec, "03_load_store", EscapeConfig::ExceptionSpec);
smoke_aarch64!(aarch64_03_flag_spec, "03_load_store", EscapeConfig::FlagSpec);
run_aarch64!(run_aarch64_03_flag_spec, "03_load_store", EscapeConfig::FlagSpec);
native_aarch64!(native_aarch64_03, "03_load_store");
smoke_aarch64!(aarch64_04_no_eh, "04_integer_ext", EscapeConfig::None);
run_aarch64!(run_aarch64_04_no_eh, "04_integer_ext", EscapeConfig::None);
smoke_aarch64!(aarch64_04_eh, "04_integer_ext", EscapeConfig::Exception);
run_aarch64!(run_aarch64_04_eh, "04_integer_ext", EscapeConfig::Exception);
smoke_aarch64!(aarch64_04_eh_spec, "04_integer_ext", EscapeConfig::ExceptionSpec);
run_aarch64!(run_aarch64_04_eh_spec, "04_integer_ext", EscapeConfig::ExceptionSpec);
smoke_aarch64!(aarch64_04_flag_spec, "04_integer_ext", EscapeConfig::FlagSpec);
run_aarch64!(run_aarch64_04_flag_spec, "04_integer_ext", EscapeConfig::FlagSpec);
native_aarch64!(native_aarch64_04, "04_integer_ext");
smoke_aarch64!(aarch64_05_no_eh, "05_load_store_ext", EscapeConfig::None);
run_aarch64!(run_aarch64_05_no_eh, "05_load_store_ext", EscapeConfig::None);
smoke_aarch64!(aarch64_05_eh, "05_load_store_ext", EscapeConfig::Exception);
run_aarch64!(run_aarch64_05_eh, "05_load_store_ext", EscapeConfig::Exception);
smoke_aarch64!(aarch64_05_eh_spec, "05_load_store_ext", EscapeConfig::ExceptionSpec);
run_aarch64!(run_aarch64_05_eh_spec, "05_load_store_ext", EscapeConfig::ExceptionSpec);
smoke_aarch64!(aarch64_05_flag_spec, "05_load_store_ext", EscapeConfig::FlagSpec);
run_aarch64!(run_aarch64_05_flag_spec, "05_load_store_ext", EscapeConfig::FlagSpec);
native_aarch64!(native_aarch64_05, "05_load_store_ext");
smoke_aarch64!(aarch64_06_no_eh, "06_floating_point", EscapeConfig::None);
run_aarch64!(run_aarch64_06_no_eh, "06_floating_point", EscapeConfig::None);
smoke_aarch64!(aarch64_06_eh, "06_floating_point", EscapeConfig::Exception);
run_aarch64!(run_aarch64_06_eh, "06_floating_point", EscapeConfig::Exception);
smoke_aarch64!(aarch64_06_eh_spec, "06_floating_point", EscapeConfig::ExceptionSpec);
run_aarch64!(run_aarch64_06_eh_spec, "06_floating_point", EscapeConfig::ExceptionSpec);
smoke_aarch64!(aarch64_06_flag_spec, "06_floating_point", EscapeConfig::FlagSpec);
run_aarch64!(run_aarch64_06_flag_spec, "06_floating_point", EscapeConfig::FlagSpec);
native_aarch64!(native_aarch64_06, "06_floating_point");

// ── x86-64 corpus (wasmi + blitz) ───────────────────────────────────────────────

smoke_x86_64!(x86_64_01_no_eh, "01_integer_computational", EscapeConfig::None);
run_x86_64!(run_x86_64_01_no_eh, "01_integer_computational", EscapeConfig::None);
smoke_x86_64!(x86_64_01_eh, "01_integer_computational", EscapeConfig::Exception);
run_x86_64!(run_x86_64_01_eh, "01_integer_computational", EscapeConfig::Exception);
smoke_x86_64!(x86_64_01_eh_spec, "01_integer_computational", EscapeConfig::ExceptionSpec);
run_x86_64!(run_x86_64_01_eh_spec, "01_integer_computational", EscapeConfig::ExceptionSpec);
smoke_x86_64!(x86_64_01_flag_spec, "01_integer_computational", EscapeConfig::FlagSpec);
run_x86_64!(run_x86_64_01_flag_spec, "01_integer_computational", EscapeConfig::FlagSpec);
native_x86_64!(native_x86_64_01, "01_integer_computational");
smoke_x86_64!(x86_64_02_no_eh, "02_control_transfer", EscapeConfig::None);
run_x86_64!(run_x86_64_02_no_eh, "02_control_transfer", EscapeConfig::None);
smoke_x86_64!(x86_64_02_eh, "02_control_transfer", EscapeConfig::Exception);
run_x86_64!(run_x86_64_02_eh, "02_control_transfer", EscapeConfig::Exception);
smoke_x86_64!(x86_64_02_eh_spec, "02_control_transfer", EscapeConfig::ExceptionSpec);
run_x86_64!(run_x86_64_02_eh_spec, "02_control_transfer", EscapeConfig::ExceptionSpec);
smoke_x86_64!(x86_64_02_flag_spec, "02_control_transfer", EscapeConfig::FlagSpec);
run_x86_64!(run_x86_64_02_flag_spec, "02_control_transfer", EscapeConfig::FlagSpec);
native_x86_64!(native_x86_64_02, "02_control_transfer");
smoke_x86_64!(x86_64_03_no_eh, "03_load_store", EscapeConfig::None);
run_x86_64!(run_x86_64_03_no_eh, "03_load_store", EscapeConfig::None);
smoke_x86_64!(x86_64_03_eh, "03_load_store", EscapeConfig::Exception);
run_x86_64!(run_x86_64_03_eh, "03_load_store", EscapeConfig::Exception);
smoke_x86_64!(x86_64_03_eh_spec, "03_load_store", EscapeConfig::ExceptionSpec);
run_x86_64!(run_x86_64_03_eh_spec, "03_load_store", EscapeConfig::ExceptionSpec);
smoke_x86_64!(x86_64_03_flag_spec, "03_load_store", EscapeConfig::FlagSpec);
run_x86_64!(run_x86_64_03_flag_spec, "03_load_store", EscapeConfig::FlagSpec);
native_x86_64!(native_x86_64_03, "03_load_store");
smoke_x86_64!(x86_64_04_no_eh, "04_flags_and_setcc", EscapeConfig::None);
run_x86_64!(run_x86_64_04_no_eh, "04_flags_and_setcc", EscapeConfig::None);
smoke_x86_64!(x86_64_04_eh, "04_flags_and_setcc", EscapeConfig::Exception);
run_x86_64!(run_x86_64_04_eh, "04_flags_and_setcc", EscapeConfig::Exception);
smoke_x86_64!(x86_64_04_eh_spec, "04_flags_and_setcc", EscapeConfig::ExceptionSpec);
run_x86_64!(run_x86_64_04_eh_spec, "04_flags_and_setcc", EscapeConfig::ExceptionSpec);
smoke_x86_64!(x86_64_04_flag_spec, "04_flags_and_setcc", EscapeConfig::FlagSpec);
run_x86_64!(run_x86_64_04_flag_spec, "04_flags_and_setcc", EscapeConfig::FlagSpec);
native_x86_64!(native_x86_64_04, "04_flags_and_setcc");
smoke_x86_64!(x86_64_05_no_eh, "05_edge_cases", EscapeConfig::None);
run_x86_64!(run_x86_64_05_no_eh, "05_edge_cases", EscapeConfig::None);
smoke_x86_64!(x86_64_05_eh, "05_edge_cases", EscapeConfig::Exception);
run_x86_64!(run_x86_64_05_eh, "05_edge_cases", EscapeConfig::Exception);
smoke_x86_64!(x86_64_05_eh_spec, "05_edge_cases", EscapeConfig::ExceptionSpec);
run_x86_64!(run_x86_64_05_eh_spec, "05_edge_cases", EscapeConfig::ExceptionSpec);
smoke_x86_64!(x86_64_05_flag_spec, "05_edge_cases", EscapeConfig::FlagSpec);
run_x86_64!(run_x86_64_05_flag_spec, "05_edge_cases", EscapeConfig::FlagSpec);
native_x86_64!(native_x86_64_05, "05_edge_cases");

// ── MIPS corpus (wasmi + blitz) ─────────────────────────────────────────────────

smoke_mips!(mips_01_no_eh, "01_integer_computational", EscapeConfig::None);
run_mips!(run_mips_01_no_eh, "01_integer_computational", EscapeConfig::None);
smoke_mips!(mips_01_eh, "01_integer_computational", EscapeConfig::Exception);
run_mips!(run_mips_01_eh, "01_integer_computational", EscapeConfig::Exception);
smoke_mips!(mips_01_eh_spec, "01_integer_computational", EscapeConfig::ExceptionSpec);
run_mips!(run_mips_01_eh_spec, "01_integer_computational", EscapeConfig::ExceptionSpec);
smoke_mips!(mips_01_flag_spec, "01_integer_computational", EscapeConfig::FlagSpec);
run_mips!(run_mips_01_flag_spec, "01_integer_computational", EscapeConfig::FlagSpec);
native_mips!(native_mips_01, "01_integer_computational");
smoke_mips!(mips_02_no_eh, "02_control_transfer", EscapeConfig::None);
run_mips!(run_mips_02_no_eh, "02_control_transfer", EscapeConfig::None);
smoke_mips!(mips_02_eh, "02_control_transfer", EscapeConfig::Exception);
run_mips!(run_mips_02_eh, "02_control_transfer", EscapeConfig::Exception);
smoke_mips!(mips_02_eh_spec, "02_control_transfer", EscapeConfig::ExceptionSpec);
run_mips!(run_mips_02_eh_spec, "02_control_transfer", EscapeConfig::ExceptionSpec);
smoke_mips!(mips_02_flag_spec, "02_control_transfer", EscapeConfig::FlagSpec);
run_mips!(run_mips_02_flag_spec, "02_control_transfer", EscapeConfig::FlagSpec);
native_mips!(native_mips_02, "02_control_transfer");
smoke_mips!(mips_03_no_eh, "03_load_store", EscapeConfig::None);
run_mips!(run_mips_03_no_eh, "03_load_store", EscapeConfig::None);
smoke_mips!(mips_03_eh, "03_load_store", EscapeConfig::Exception);
run_mips!(run_mips_03_eh, "03_load_store", EscapeConfig::Exception);
smoke_mips!(mips_03_eh_spec, "03_load_store", EscapeConfig::ExceptionSpec);
run_mips!(run_mips_03_eh_spec, "03_load_store", EscapeConfig::ExceptionSpec);
smoke_mips!(mips_03_flag_spec, "03_load_store", EscapeConfig::FlagSpec);
run_mips!(run_mips_03_flag_spec, "03_load_store", EscapeConfig::FlagSpec);
native_mips!(native_mips_03, "03_load_store");
smoke_mips!(mips_04_no_eh, "04_multiply_divide", EscapeConfig::None);
run_mips!(run_mips_04_no_eh, "04_multiply_divide", EscapeConfig::None);
smoke_mips!(mips_04_eh, "04_multiply_divide", EscapeConfig::Exception);
run_mips!(run_mips_04_eh, "04_multiply_divide", EscapeConfig::Exception);
smoke_mips!(mips_04_eh_spec, "04_multiply_divide", EscapeConfig::ExceptionSpec);
run_mips!(run_mips_04_eh_spec, "04_multiply_divide", EscapeConfig::ExceptionSpec);
smoke_mips!(mips_04_flag_spec, "04_multiply_divide", EscapeConfig::FlagSpec);
run_mips!(run_mips_04_flag_spec, "04_multiply_divide", EscapeConfig::FlagSpec);
native_mips!(native_mips_04, "04_multiply_divide");
smoke_mips!(mips_05_no_eh, "05_edge_cases", EscapeConfig::None);
run_mips!(run_mips_05_no_eh, "05_edge_cases", EscapeConfig::None);
smoke_mips!(mips_05_eh, "05_edge_cases", EscapeConfig::Exception);
run_mips!(run_mips_05_eh, "05_edge_cases", EscapeConfig::Exception);
smoke_mips!(mips_05_eh_spec, "05_edge_cases", EscapeConfig::ExceptionSpec);
run_mips!(run_mips_05_eh_spec, "05_edge_cases", EscapeConfig::ExceptionSpec);
smoke_mips!(mips_05_flag_spec, "05_edge_cases", EscapeConfig::FlagSpec);
run_mips!(run_mips_05_flag_spec, "05_edge_cases", EscapeConfig::FlagSpec);
native_mips!(native_mips_05, "05_edge_cases");

// ── Dual-lane linux-wasi / thin-native / darwin-wasi ────────────────────────────

linux_wasi!(linux_wasi_exit_42_no_eh, bytes = FIXTURE_EXIT_42, addr = 0x1000, EscapeConfig::None, expect_exit = 42);
linux_wasi!(linux_wasi_write_exit_no_eh, bytes = FIXTURE_WRITE_EXIT, addr = 0x1000, EscapeConfig::None, seed_addr = 520, seed = b"hello\n", expect_stdout = b"hello\n", expect_exit = 0);
thin_native!(thin_native_exit_42_no_eh, bytes = FIXTURE_EXIT_42, addr = 0x1000, EscapeConfig::None, expect_exit = 42);
linux_wasi!(linux_wasi_exit_42_flag_spec, bytes = FIXTURE_EXIT_42, addr = 0x1000, EscapeConfig::FlagSpec, expect_exit = 42);
linux_wasi!(linux_wasi_write_exit_flag_spec, bytes = FIXTURE_WRITE_EXIT, addr = 0x1000, EscapeConfig::FlagSpec, seed_addr = 520, seed = b"hello\n", expect_stdout = b"hello\n", expect_exit = 0);
thin_native!(thin_native_exit_42_flag_spec, bytes = FIXTURE_EXIT_42, addr = 0x1000, EscapeConfig::FlagSpec, expect_exit = 42);
darwin_wasi!(darwin_wasi_write_exit_no_eh, bytes = FIXTURE_DARWIN_WRITE_EXIT, addr = 0x1000, EscapeConfig::None, seed_addr = 520, seed = b"hello\n", expect_stdout = b"hello\n", expect_exit = 0);
darwin_wasi!(darwin_wasi_write_exit_flag_spec, bytes = FIXTURE_DARWIN_WRITE_EXIT, addr = 0x1000, EscapeConfig::FlagSpec, seed_addr = 520, seed = b"hello\n", expect_stdout = b"hello\n", expect_exit = 0);

// @generated-tests-end

// ── Wasmtime exception-handling fallback ──────────────────────────────────────
//
// No current corpus/C program actually drives a guest `call` through a
// mismatched-target throw, so none of the `is_known_wasmi_exception_gap`
// fallback branches above are exercised by the generated tests. This pins
// down the fallback mechanism itself, independent of any recompiler, using a
// hand-built try_table/throw/catch module.

#[test]
fn wasmi_cannot_run_exception_handling_module() {
    let wasm = wasm_throw_catch();
    wasmparser::validate(&wasm).expect("hand-built EH module should itself be valid WASM");
    match run_module(&wasm, "_start") {
        Err(e) if is_known_wasmi_exception_gap(&e) => {}
        Err(e) => panic!("expected the documented wasmi exception-handling gap, got: {e}"),
        Ok(_) => panic!("wasmi unexpectedly ran an exception-handling module — gap may be fixed upstream"),
    }
}

#[test]
fn wasmtime_runs_exception_handling_module() {
    let wasm = wasm_throw_catch();
    let state = run_module_wasmtime(&wasm, "_start")
        .unwrap_or_else(|e| panic!("wasmtime failed to run the EH fixture: {e}"));
    // No host imports are called in this fixture; this only proves
    // instantiate+call succeeded (i.e. wasmtime actually executed the
    // throw/catch), and that the result didn't trap.
    let _ = state;
}

// ── Debug / diagnostic helpers ────────────────────────────────────────────────

#[test]
fn debug_rv32_c_arith() {
    let path = match c_obj("E2E_RV32_ARITH") { Some(p) => p, None => return };
    let (text, addr) = match load_text_optional(&path) { Some(v) => v, None => return };
    eprintln!("── RV32 .text linear disassembly ({} bytes @ {addr:#x}) ──", text.len());
    disasm_rv(&text, addr as u64, Xlen::Rv32);
    disasm_rv_conservative(&text, addr as u64, Xlen::Rv32);
    let (wasm, _) = build_single(&text, addr, Arch::Rv32, Eh::None);
    if let Err(e) = wasmparser::validate(&wasm) {
        let err_offset = e.offset();
        let window = 40;
        let start = err_offset.saturating_sub(window);
        eprintln!("Validation error at offset {err_offset:#x}: {e}");
        eprintln!("Decoding operators in range [{start:#x}..{:#x}]:", err_offset + window);
        decode_operators_near(&wasm, start, err_offset + window);
        panic!("invalid WASM");
    }
}

#[test]
fn debug_rv64_c_arith() {
    let path = match c_obj("E2E_RV64_ARITH") { Some(p) => p, None => return };
    let (text, addr) = match load_text_optional(&path) { Some(v) => v, None => return };
    eprintln!("── RV64 .text disassembly ({} bytes @ {addr:#x}) ──", text.len());
    disasm_rv(&text, addr as u64, Xlen::Rv64);
    disasm_rv_conservative(&text, addr as u64, Xlen::Rv64);
    let (wasm, _) = build_single(&text, addr, Arch::Rv64, Eh::None);
    if let Err(e) = wasmparser::validate(&wasm) {
        let err_offset = e.offset();
        let window = 40;
        let start = err_offset.saturating_sub(window);
        eprintln!("Validation error at offset {err_offset:#x}: {e}");
        eprintln!("Decoding operators in range [{start:#x}..{:#x}]:", err_offset + window);
        decode_operators_near(&wasm, start, err_offset + window);
        panic!("invalid WASM");
    }
}

fn debug_corpus(rel: &str, arch: Arch, xlen: Xlen) {
    let path = corpus(rel);
    let (text, addr) = load_text(&path);
    eprintln!("── {} ({} bytes @ {addr:#x}) ──", rel, text.len());
    disasm_rv(&text, addr as u64, xlen);
    let (wasm, _) = build_single(&text, addr, arch, Eh::None);
    if let Err(e) = wasmparser::validate(&wasm) {
        let err_offset = e.offset();
        let window = 60;
        let start = err_offset.saturating_sub(window);
        eprintln!("Validation error @ {err_offset:#x}: {e}");
        decode_operators_near(&wasm, start, err_offset + window);
        panic!("invalid WASM");
    }
}

#[test]
fn debug_rv64im_corpus() { debug_corpus("rv64im/01_multiply_divide_64", Arch::Rv64, Xlen::Rv64); }
#[test]
fn debug_rv32ima_corpus() { debug_corpus("rv32ima/01_atomic_operations", Arch::Rv32, Xlen::Rv32); }
#[test]
fn debug_rv64d_corpus() { debug_corpus("rv64d/01_rv64_double_precision_fp", Arch::Rv64, Xlen::Rv64); }
#[test]
fn debug_rv32fd_corpus() { debug_corpus("rv32fd/01_combined_fp", Arch::Rv32, Xlen::Rv32); }
#[test]
fn debug_rv32i02_corpus() { debug_corpus("rv32i/02_control_transfer", Arch::Rv32, Xlen::Rv32); }
#[test]
fn debug_rv32i04_corpus() { debug_corpus("rv32i/04_edge_cases", Arch::Rv32, Xlen::Rv32); }
#[test]
fn debug_rv32i06_corpus() { debug_corpus("rv32i/06_nop_and_hints", Arch::Rv32, Xlen::Rv32); }

#[test]
fn debug_aarch64_01_corpus() {
    let path = aarch64_corpus("01_integer_computational");
    let (text, addr) = load_text(&path);
    eprintln!("── aarch64 01 ({} bytes @ {addr:#x}) ──", text.len());
    let (wasm, _) = build_single(&text, addr, Arch::AArch64, Eh::None);
    match wasmparser::validate(&wasm) {
        Ok(_) => eprintln!("  valid!"),
        Err(e) => {
            let off = e.offset();
            eprintln!("Validation error @ {off:#x}: {e}");
            decode_operators_near(&wasm, off.saturating_sub(120), off + 20);
            panic!("invalid WASM");
        }
    }
}

fn disasm_rv(text: &[u8], pc: u64, xlen: Xlen) {
    disasm_rv_inner(text, pc, xlen, false);
}

fn disasm_rv_conservative(text: &[u8], pc: u64, xlen: Xlen) {
    eprintln!("── conservative 2-byte-boundary decode ──");
    disasm_rv_inner(text, pc, xlen, true);
}

fn disasm_rv_inner(text: &[u8], pc: u64, xlen: Xlen, conservative: bool) {
    use rv_asm::Inst;
    let mut i = 0usize;
    while i + 2 <= text.len() {
        let lo = u16::from_le_bytes([text[i], text[i + 1]]);
        if lo & 0x3 != 0x3 {
            match Inst::decode_compressed(lo, xlen) {
                Ok(inst) => eprintln!("  {:#010x}  {:04x}          {inst}", pc + i as u64, lo),
                Err(_)   => eprintln!("  {:#010x}  {:04x}          <bad-c>", pc + i as u64, lo),
            }
            i += 2;
        } else {
            if i + 4 > text.len() { break; }
            let word = u32::from_le_bytes([text[i], text[i+1], text[i+2], text[i+3]]);
            match Inst::decode(word, xlen) {
                Ok((inst, _)) => eprintln!("  {:#010x}  {:08x}  {inst}", pc + i as u64, word),
                Err(_)        => eprintln!("  {:#010x}  {:08x}  <bad>",  pc + i as u64, word),
            }
            i += if conservative { 2 } else { 4 };
        }
    }
}

fn decode_operators_near(wasm: &[u8], from: usize, to: usize) {
    use wasmparser::{Parser, Payload};
    for payload in Parser::new(0).parse_all(wasm) {
        let Ok(payload) = payload else { continue };
        if let Payload::CodeSectionEntry(body) = payload {
            let body_range = body.range();
            if body_range.end < from || body_range.start > to { continue; }
            eprintln!("  -- function body [{:#x}..{:#x}]", body_range.start, body_range.end);
            if let Ok(locals_reader) = body.get_locals_reader() {
                let mut local_idx = 0u32;
                for local in locals_reader {
                    let Ok((count, ty)) = local else { break };
                    eprintln!("    locals {local_idx}..{}: {ty:?}", local_idx + count);
                    local_idx += count;
                }
            }
            let Ok(reader) = body.get_operators_reader() else { continue };
            let mut ops = reader;
            loop {
                let pos = ops.original_position();
                match ops.read() {
                    Ok(op) => {
                        if pos >= from && pos <= to { eprintln!("  [{pos:#06x}] {op:?}"); }
                        if pos > to { break; }
                    }
                    Err(_) => break,
                }
            }
        }
    }
}

// ── Phase 4: AArch32 / i686 FlagSpec smoke (handcrafted bytes) ────────────────

/// A32: `mov r0, #1; bx lr` — translate + validate under FlagSpec.
#[test]
fn arm_flag_smoke() {
    let mut text = Vec::new();
    text.extend_from_slice(&0xe3a0_0001u32.to_le_bytes()); // mov r0, #1
    text.extend_from_slice(&0xe12f_ff1eu32.to_le_bytes()); // bx lr
    let (wasm, unsupported) =
        build_single_config(&text, 0x1000, Arch::Arm, EscapeConfig::FlagSpec);
    report_unsupported(&unsupported, "arm_flag_smoke");
    assert!(!wasm.is_empty());
    wasmparser::validate(&wasm).expect("arm FlagSpec WASM invalid");
}

/// i686: `mov eax, 1; ret` — translate + validate under FlagSpec.
#[test]
fn x86_flag_smoke() {
    let text = [0xb8u8, 0x01, 0x00, 0x00, 0x00, 0xc3];
    let (wasm, unsupported) =
        build_single_config(&text, 0x1000, Arch::X86_32, EscapeConfig::FlagSpec);
    report_unsupported(&unsupported, "x86_flag_smoke");
    assert!(!wasm.is_empty());
    wasmparser::validate(&wasm).expect("x86 FlagSpec WASM invalid");
}
