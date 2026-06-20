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
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_speculative(&text, addr, $arch, $eh, $spec);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!("  ✓ {} ({:?}) — {} bytes",
                path.file_name().unwrap().to_string_lossy(), $eh, wasm.len());
        }
    };
}

macro_rules! smoke_c {
    ($name:ident, env = $env:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        #[test]
        fn $name() {
            let path = match c_obj($env) { Some(p) => p, None => {
                eprintln!("  skipping {}: C object not built", stringify!($name));
                return;
            }};
            let (text, addr) = match load_text_optional(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single_speculative(&text, addr, $arch, $eh, $spec);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!("  ✓ {} ({:?}) — {} bytes",
                path.file_name().unwrap().to_string_lossy(), $eh, wasm.len());
        }
    };
}

macro_rules! run {
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_speculative(&text, addr, $arch, $eh, $spec);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => {
                    println!("  ✓ {} hints, {:?}", state.hints.len(), $eh);
                    for (id, snap) in &state.hints {
                        println!("    hint={id} a0={}", snap.reg("a0"));
                    }
                }
                // Speculative runs are a deliberate, asserted feature — never
                // soft-skip a failure here, or a real regression would
                // silently downgrade to a log line instead of failing CI.
                Err(e) if $spec && is_known_wasmi_exception_gap(&e) => eprintln!("  ! known wasmi limitation (no exception-handling support): {e}"),
                Err(e) if $spec => panic!("speculative run failed: {e}"),
                Err(e) if $eh == Eh::With => eprintln!("  ! EH run skipped: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
}

macro_rules! run_trap {
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, _) = build_single_with_trap_speculative(&text, addr, $arch, $eh, $spec);
            wasmparser::validate(&wasm).expect("WASM with trap is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => {
                    let n_ret  = state.hints.iter().filter(|(id, _)| *id == HINT_RETURN).count();
                    let n_call = state.hints.iter().filter(|(id, _)| *id == HINT_CALL).count();
                    println!("  ✓ trap {:?}: {} returns, {} calls", $eh, n_ret, n_call);
                }
                Err(e) if $spec && is_known_wasmi_exception_gap(&e) => eprintln!("  ! known wasmi limitation (no exception-handling support): {e}"),
                Err(e) if $spec => panic!("speculative run failed: {e}"),
                Err(e) if $eh == Eh::With => eprintln!("  ! EH run skipped: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
}

macro_rules! run_c {
    ($name:ident, env = $env:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        #[test]
        fn $name() {
            let path = match c_obj($env) { Some(p) => p, None => {
                eprintln!("  skipping {}: C object not built", stringify!($name));
                return;
            }};
            let (text, addr) = match load_text_optional(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single_speculative(&text, addr, $arch, $eh, $spec);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => println!("  ✓ {} hints, {:?}", state.hints.len(), $eh),
                Err(e) if $spec && is_known_wasmi_exception_gap(&e) => eprintln!("  ! known wasmi limitation (no exception-handling support): {e}"),
                Err(e) if $spec => panic!("speculative run failed: {e}"),
                Err(e) if $eh == Eh::With => eprintln!("  ! EH run skipped: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
}

macro_rules! link {
    ($name:ident, [ $( ($rel:expr, arch = $arch:expr, entry = $entry:expr) ),+ ], $eh:expr, speculative = $spec:expr) => {
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
            let specs: Vec<LinkSpec<'_>> = specs_data.iter()
                .map(|(t, a, arch, e)| LinkSpec { text: t, start_addr: *a, arch: *arch, entry: e })
                .collect();
            let (wasm, unsupported) = build_linked_speculative(&specs, $eh, $spec);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("linked WASM is invalid");
            println!("  linked: {} bytes, {:?}", wasm.len(), $eh);
            for spec in &specs {
                match run_module(&wasm, spec.entry) {
                    Ok(state) => println!("  ✓ {}: {} hints", spec.entry, state.hints.len()),
                    Err(e) if $spec && is_known_wasmi_exception_gap(&e) => eprintln!("  ! known wasmi limitation ({}): {e}", spec.entry),
                    Err(e) if $spec => panic!("speculative run {} failed: {e}", spec.entry),
                    Err(e) if $eh == Eh::With => eprintln!("  ! EH skipped ({}): {e}", spec.entry),
                    Err(e) => panic!("run {} failed: {e}", spec.entry),
                }
            }
        }
    };
}

macro_rules! link_c {
    ($name:ident, [ $( ($env_or_corpus:expr, is_corpus = $is_corpus:expr, arch = $arch:expr, entry = $entry:expr) ),+ ], $eh:expr, speculative = $spec:expr) => {
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
            let specs: Vec<LinkSpec<'_>> = specs_data.iter()
                .map(|(t, a, arch, e)| LinkSpec { text: t, start_addr: *a, arch: *arch, entry: e })
                .collect();
            let (wasm, unsupported) = build_linked_speculative(&specs, $eh, $spec);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("linked WASM is invalid");
            println!("  linked: {} bytes, {:?}", wasm.len(), $eh);
            for spec in &specs {
                match run_module(&wasm, spec.entry) {
                    Ok(state) => println!("  ✓ {}: {} hints", spec.entry, state.hints.len()),
                    Err(e) if $spec && is_known_wasmi_exception_gap(&e) => eprintln!("  ! known wasmi limitation ({}): {e}", spec.entry),
                    Err(e) if $spec => panic!("speculative run {} failed: {e}", spec.entry),
                    Err(e) if $eh == Eh::With => eprintln!("  ! EH skipped ({}): {e}", spec.entry),
                    Err(e) => panic!("run {} failed: {e}", spec.entry),
                }
            }
        }
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
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single_speculative(&text, addr, $arch, $eh, $spec);
            report_unsupported(&unsupported, stringify!($name));
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
    };
}

macro_rules! native_c {
    ($name:ident, env = $env:expr, arch = $arch:expr, $eh:expr, speculative = $spec:expr) => {
        #[test]
        fn $name() {
            let path = match c_obj($env) { Some(p) => p, None => {
                eprintln!("  skipping {}: C object not built", stringify!($name));
                return;
            }};
            let (text, addr) = match load_text_optional(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single_speculative(&text, addr, $arch, $eh, $spec);
            report_unsupported(&unsupported, stringify!($name));
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
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
            let (wasm, unsupported) = build_single(&text, addr, Arch::AArch64, Eh::None);
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
            let (wasm, unsupported) = build_single(&text, addr, Arch::X86_64, Eh::None);
            report_unsupported(&unsupported, stringify!($name));
            if wasm.is_empty() || wasmparser::validate(&wasm).is_err() { return; }
            native_compile_check(&wasm, stringify!($name));
        }
    };
}

// ── AArch64 corpus tests ──────────────────────────────────────────────────────

macro_rules! smoke_aarch64 {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = aarch64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single(&text, addr, Arch::AArch64, Eh::None);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("WASM invalid");
        }
    };
}

macro_rules! run_aarch64 {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = aarch64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, _) = build_single(&text, addr, Arch::AArch64, Eh::None);
            wasmparser::validate(&wasm).expect("WASM invalid");
            run_module(&wasm, "_start").unwrap_or_else(|e| panic!("run failed: {e}"));
        }
    };
}

smoke_aarch64!(smoke_aarch64_01_arith_no_eh,        "01_integer_computational");
run_aarch64!(  run_aarch64_01_arith_no_eh,           "01_integer_computational");
smoke_aarch64!(smoke_aarch64_02_control_no_eh,       "02_control_transfer");
run_aarch64!(  run_aarch64_02_control_no_eh,         "02_control_transfer");
smoke_aarch64!(smoke_aarch64_03_load_store_no_eh,    "03_load_store");
run_aarch64!(  run_aarch64_03_load_store_no_eh,      "03_load_store");
smoke_aarch64!(smoke_aarch64_04_integer_ext_no_eh,   "04_integer_ext");
run_aarch64!(  run_aarch64_04_integer_ext_no_eh,     "04_integer_ext");
smoke_aarch64!(smoke_aarch64_05_load_store_ext_no_eh,"05_load_store_ext");
run_aarch64!(  run_aarch64_05_load_store_ext_no_eh,  "05_load_store_ext");
smoke_aarch64!(smoke_aarch64_06_fp_no_eh,            "06_floating_point");
run_aarch64!(  run_aarch64_06_fp_no_eh,              "06_floating_point");

// Native-backend cross for the AArch64 frontend (guest aarch64 → WASM → native).
native_aarch64!(native_aarch64_01_arith,        "01_integer_computational");
native_aarch64!(native_aarch64_02_control,      "02_control_transfer");
native_aarch64!(native_aarch64_03_load_store,   "03_load_store");
native_aarch64!(native_aarch64_04_integer_ext,  "04_integer_ext");
native_aarch64!(native_aarch64_05_load_store_ext,"05_load_store_ext");
native_aarch64!(native_aarch64_06_fp,           "06_floating_point");

// ── x86-64 corpus tests ───────────────────────────────────────────────────────

macro_rules! smoke_x86_64 {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = x86_64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, unsupported) = build_single(&text, addr, Arch::X86_64, Eh::None);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("WASM invalid");
        }
    };
}

macro_rules! run_x86_64 {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = x86_64_corpus($rel);
            let (text, addr) = load_text(&path);
            let (wasm, _) = build_single(&text, addr, Arch::X86_64, Eh::None);
            wasmparser::validate(&wasm).expect("WASM invalid");
            run_module(&wasm, "_start").unwrap_or_else(|e| panic!("run failed: {e}"));
        }
    };
}

// 01: arithmetic only — full smoke+run coverage.
// 02-05: the x86-64 recompiler has pre-existing issues with control-flow
//        corpora (unsupported instructions, branch-target index OOB); these
//        are tracked separately and excluded here until fixed.
smoke_x86_64!(smoke_x86_64_01_arith_no_eh,  "01_integer_computational");
run_x86_64!(  run_x86_64_01_arith_no_eh,    "01_integer_computational");
native_x86_64!(native_x86_64_01_arith,      "01_integer_computational");

// @generated-tests-begin

// ── Corpus smoke tests ──────────────────────────────────────────────────────────

smoke!(smoke_rv32d_01_no_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32d_01_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32d_01_eh_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32f_01_no_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32f_01_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32f_01_eh_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32fd_01_no_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32fd_01_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32fd_01_eh_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32i_01_no_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32i_01_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32i_01_eh_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32i_02_no_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32i_02_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32i_02_eh_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32i_03_no_eh, "rv32i/03_load_store", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32i_03_eh, "rv32i/03_load_store", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32i_03_eh_spec, "rv32i/03_load_store", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32i_04_no_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32i_04_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32i_04_eh_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32i_05_no_eh, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32i_05_eh, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32i_05_eh_spec, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32i_06_no_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32i_06_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32i_06_eh_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32i_07_no_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32i_07_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32i_07_eh_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32i_zicsr_01_no_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32i_zicsr_01_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32i_zicsr_01_eh_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32im_01_no_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32im_01_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32im_01_eh_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv32ima_01_no_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::None, speculative=false);
smoke!(smoke_rv32ima_01_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::With, speculative=false);
smoke!(smoke_rv32ima_01_eh_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::With, speculative=true);
smoke!(smoke_rv64d_01_no_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::None, speculative=false);
smoke!(smoke_rv64d_01_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::With, speculative=false);
smoke!(smoke_rv64d_01_eh_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::With, speculative=true);
smoke!(smoke_rv64i_01_no_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::None, speculative=false);
smoke!(smoke_rv64i_01_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::With, speculative=false);
smoke!(smoke_rv64i_01_eh_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::With, speculative=true);
smoke!(smoke_rv64im_01_no_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::None, speculative=false);
smoke!(smoke_rv64im_01_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::With, speculative=false);
smoke!(smoke_rv64im_01_eh_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::With, speculative=true);

// ── C smoke tests ───────────────────────────────────────────────────────────────

smoke_c!(smoke_rv32c_arith_no_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::None, speculative=false);
smoke_c!(smoke_rv32c_arith_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::With, speculative=false);
smoke_c!(smoke_rv32c_arith_eh_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::With, speculative=true);
smoke_c!(smoke_rv64c_arith_no_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::None, speculative=false);
smoke_c!(smoke_rv64c_arith_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::With, speculative=false);
smoke_c!(smoke_rv64c_arith_eh_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::With, speculative=true);
smoke_c!(smoke_x86c_arith_no_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::None, speculative=false);
smoke_c!(smoke_x86c_arith_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::With, speculative=false);
smoke_c!(smoke_x86c_arith_eh_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::With, speculative=true);

// ── Corpus run tests ────────────────────────────────────────────────────────────

run!(run_rv32d_01_no_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32d_01_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32d_01_eh_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32f_01_no_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32f_01_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32f_01_eh_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32fd_01_no_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32fd_01_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32fd_01_eh_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32i_01_no_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32i_01_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32i_01_eh_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32i_02_no_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32i_02_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32i_02_eh_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32i_03_no_eh, "rv32i/03_load_store", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32i_03_eh, "rv32i/03_load_store", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32i_03_eh_spec, "rv32i/03_load_store", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32i_04_no_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32i_04_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32i_04_eh_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32i_05_no_eh, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32i_05_eh, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32i_05_eh_spec, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32i_06_no_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32i_06_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32i_06_eh_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32i_07_no_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32i_07_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32i_07_eh_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32i_zicsr_01_no_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32i_zicsr_01_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32i_zicsr_01_eh_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32im_01_no_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32im_01_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32im_01_eh_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv32ima_01_no_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::None, speculative=false);
run!(run_rv32ima_01_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::With, speculative=false);
run!(run_rv32ima_01_eh_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::With, speculative=true);
run!(run_rv64d_01_no_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::None, speculative=false);
run!(run_rv64d_01_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::With, speculative=false);
run!(run_rv64d_01_eh_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::With, speculative=true);
run!(run_rv64i_01_no_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::None, speculative=false);
run!(run_rv64i_01_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::With, speculative=false);
run!(run_rv64i_01_eh_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::With, speculative=true);
run!(run_rv64im_01_no_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::None, speculative=false);
run!(run_rv64im_01_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::With, speculative=false);
run!(run_rv64im_01_eh_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::With, speculative=true);

// ── C run tests ─────────────────────────────────────────────────────────────────

run_c!(run_rv32c_arith_no_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::None, speculative=false);
run_c!(run_rv32c_arith_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::With, speculative=false);
run_c!(run_rv32c_arith_eh_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::With, speculative=true);
run_c!(run_rv64c_arith_no_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::None, speculative=false);
run_c!(run_rv64c_arith_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::With, speculative=false);
run_c!(run_rv64c_arith_eh_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::With, speculative=true);
run_c!(run_x86c_arith_no_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::None, speculative=false);
run_c!(run_x86c_arith_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::With, speculative=false);
run_c!(run_x86c_arith_eh_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::With, speculative=true);

// ── Corpus run-with-trap tests ──────────────────────────────────────────────────

run_trap!(run_trap_rv32d_01_no_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32d_01_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32d_01_eh_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32f_01_no_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32f_01_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32f_01_eh_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32fd_01_no_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32fd_01_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32fd_01_eh_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32i_01_no_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32i_01_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32i_01_eh_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32i_02_no_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32i_02_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32i_02_eh_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32i_03_no_eh, "rv32i/03_load_store", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32i_03_eh, "rv32i/03_load_store", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32i_03_eh_spec, "rv32i/03_load_store", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32i_04_no_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32i_04_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32i_04_eh_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32i_05_no_eh, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32i_05_eh, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32i_05_eh_spec, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32i_06_no_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32i_06_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32i_06_eh_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32i_07_no_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32i_07_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32i_07_eh_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32i_zicsr_01_no_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32i_zicsr_01_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32i_zicsr_01_eh_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32im_01_no_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32im_01_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32im_01_eh_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv32ima_01_no_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::None, speculative=false);
run_trap!(run_trap_rv32ima_01_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::With, speculative=false);
run_trap!(run_trap_rv32ima_01_eh_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::With, speculative=true);
run_trap!(run_trap_rv64d_01_no_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::None, speculative=false);
run_trap!(run_trap_rv64d_01_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::With, speculative=false);
run_trap!(run_trap_rv64d_01_eh_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::With, speculative=true);
run_trap!(run_trap_rv64i_01_no_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::None, speculative=false);
run_trap!(run_trap_rv64i_01_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::With, speculative=false);
run_trap!(run_trap_rv64i_01_eh_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::With, speculative=true);
run_trap!(run_trap_rv64im_01_no_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::None, speculative=false);
run_trap!(run_trap_rv64im_01_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::With, speculative=false);
run_trap!(run_trap_rv64im_01_eh_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::With, speculative=true);

// ── Corpus-corpus link tests ────────────────────────────────────────────────────

link!(link_rv32d_01_x_rv32f_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32f_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32f_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32fd_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32fd_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32fd_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32i_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32i_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32i_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32i_02_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32i_02_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32i_02_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32i_03_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32i_03_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32i_03_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32i_04_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32i_04_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32i_04_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32i_05_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32i_05_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32i_05_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32i_06_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32i_06_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32i_06_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32i_07_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32i_07_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32i_07_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32i_zicsr_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32i_zicsr_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32i_zicsr_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32im_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32im_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32im_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv32ima_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv32ima_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv32ima_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv64d_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv64d_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv64d_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv64i_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv64i_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv64i_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32d_01_x_rv64im_01_no_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32d_01_x_rv64im_01_eh,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32d_01_x_rv64im_01_eh_spec,
    [("rv32d/01_double_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32fd_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32fd_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32fd_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32i_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32i_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32i_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32i_02_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32i_02_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32i_02_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32i_03_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32i_03_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32i_03_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32i_04_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32i_04_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32i_04_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32i_05_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32i_05_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32i_05_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32i_06_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32i_06_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32i_06_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32i_07_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32i_07_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32i_07_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32i_zicsr_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32i_zicsr_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32i_zicsr_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32im_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32im_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32im_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv32ima_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv32ima_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv32ima_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv64d_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv64d_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv64d_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv64i_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv64i_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv64i_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32f_01_x_rv64im_01_no_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32f_01_x_rv64im_01_eh,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32f_01_x_rv64im_01_eh_spec,
    [("rv32f/01_single_precision_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32i_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32i_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32i_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32i_02_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32i_02_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32i_02_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32i_03_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32i_03_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32i_03_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32i_04_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32i_04_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32i_04_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32i_05_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32i_05_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32i_05_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32i_06_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32i_06_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32i_06_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32i_07_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32i_07_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32i_07_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32i_zicsr_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32i_zicsr_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32i_zicsr_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32im_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32im_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32im_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv32ima_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv32ima_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv32ima_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv64d_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv64d_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv64d_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv64i_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv64i_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv64i_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32fd_01_x_rv64im_01_no_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32fd_01_x_rv64im_01_eh,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32fd_01_x_rv64im_01_eh_spec,
    [("rv32fd/01_combined_fp", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32i_02_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32i_02_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32i_02_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32i_03_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32i_03_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32i_03_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32i_04_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32i_04_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32i_04_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32i_05_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32i_05_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32i_05_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32i_06_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32i_06_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32i_06_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32i_07_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32i_07_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32i_07_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32i_zicsr_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32i_zicsr_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32im_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32im_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32im_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv32ima_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv32ima_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv32ima_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv64d_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv64d_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv64d_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv64i_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv64i_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv64i_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_01_x_rv64im_01_no_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_01_x_rv64im_01_eh,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_01_x_rv64im_01_eh_spec,
    [("rv32i/01_integer_computational", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv32i_03_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv32i_03_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv32i_03_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv32i_04_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv32i_04_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv32i_04_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv32i_05_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv32i_05_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv32i_05_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv32i_06_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv32i_06_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv32i_06_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv32i_07_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv32i_07_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv32i_07_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv32i_zicsr_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv32i_zicsr_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv32im_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv32im_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv32im_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv32ima_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv32ima_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv32ima_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv64d_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv64d_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv64d_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv64i_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv64i_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv64i_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_02_x_rv64im_01_no_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_02_x_rv64im_01_eh,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_02_x_rv64im_01_eh_spec,
    [("rv32i/02_control_transfer", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv32i_04_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv32i_04_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv32i_04_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv32i_05_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv32i_05_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv32i_05_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv32i_06_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv32i_06_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv32i_06_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv32i_07_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv32i_07_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv32i_07_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv32i_zicsr_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv32i_zicsr_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv32im_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv32im_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv32im_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv32ima_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv32ima_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv32ima_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv64d_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv64d_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv64d_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv64i_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv64i_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv64i_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_03_x_rv64im_01_no_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_03_x_rv64im_01_eh,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_03_x_rv64im_01_eh_spec,
    [("rv32i/03_load_store", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv32i_05_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv32i_05_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv32i_05_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv32i_06_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv32i_06_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv32i_06_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv32i_07_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv32i_07_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv32i_07_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv32i_zicsr_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv32i_zicsr_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv32im_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv32im_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv32im_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv32ima_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv32ima_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv32ima_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv64d_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv64d_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv64d_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv64i_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv64i_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv64i_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_04_x_rv64im_01_no_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_04_x_rv64im_01_eh,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_04_x_rv64im_01_eh_spec,
    [("rv32i/04_edge_cases", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_05_x_rv32i_06_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_05_x_rv32i_06_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_05_x_rv32i_06_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_05_x_rv32i_07_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_05_x_rv32i_07_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_05_x_rv32i_07_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_05_x_rv32i_zicsr_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_05_x_rv32i_zicsr_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_05_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_05_x_rv32im_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_05_x_rv32im_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_05_x_rv32im_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_05_x_rv32ima_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_05_x_rv32ima_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_05_x_rv32ima_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_05_x_rv64d_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_05_x_rv64d_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_05_x_rv64d_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_05_x_rv64i_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_05_x_rv64i_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_05_x_rv64i_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_05_x_rv64im_01_no_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_05_x_rv64im_01_eh,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_05_x_rv64im_01_eh_spec,
    [("rv32i/05_simple_program", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_06_x_rv32i_07_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_06_x_rv32i_07_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_06_x_rv32i_07_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_06_x_rv32i_zicsr_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_06_x_rv32i_zicsr_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_06_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_06_x_rv32im_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_06_x_rv32im_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_06_x_rv32im_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_06_x_rv32ima_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_06_x_rv32ima_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_06_x_rv32ima_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_06_x_rv64d_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_06_x_rv64d_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_06_x_rv64d_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_06_x_rv64i_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_06_x_rv64i_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_06_x_rv64i_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_06_x_rv64im_01_no_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_06_x_rv64im_01_eh,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_06_x_rv64im_01_eh_spec,
    [("rv32i/06_nop_and_hints", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_07_x_rv32i_zicsr_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_07_x_rv32i_zicsr_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_07_x_rv32i_zicsr_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_07_x_rv32im_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_07_x_rv32im_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_07_x_rv32im_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_07_x_rv32ima_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_07_x_rv32ima_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_07_x_rv32ima_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_07_x_rv64d_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_07_x_rv64d_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_07_x_rv64d_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_07_x_rv64i_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_07_x_rv64i_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_07_x_rv64i_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_07_x_rv64im_01_no_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_07_x_rv64im_01_eh,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_07_x_rv64im_01_eh_spec,
    [("rv32i/07_pseudo_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_zicsr_01_x_rv32im_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_zicsr_01_x_rv32im_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_zicsr_01_x_rv32im_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_zicsr_01_x_rv32ima_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_zicsr_01_x_rv32ima_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_zicsr_01_x_rv32ima_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_zicsr_01_x_rv64d_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_zicsr_01_x_rv64d_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_zicsr_01_x_rv64d_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_zicsr_01_x_rv64i_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_zicsr_01_x_rv64i_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_zicsr_01_x_rv64i_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32i_zicsr_01_x_rv64im_01_no_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32i_zicsr_01_x_rv64im_01_eh,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32i_zicsr_01_x_rv64im_01_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32im_01_x_rv32ima_01_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32im_01_x_rv32ima_01_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32im_01_x_rv32ima_01_eh_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32im_01_x_rv64d_01_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32im_01_x_rv64d_01_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32im_01_x_rv64d_01_eh_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32im_01_x_rv64i_01_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32im_01_x_rv64i_01_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32im_01_x_rv64i_01_eh_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32im_01_x_rv64im_01_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32im_01_x_rv64im_01_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32im_01_x_rv64im_01_eh_spec,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32ima_01_x_rv64d_01_no_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32ima_01_x_rv64d_01_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32ima_01_x_rv64d_01_eh_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32ima_01_x_rv64i_01_no_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32ima_01_x_rv64i_01_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32ima_01_x_rv64i_01_eh_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv32ima_01_x_rv64im_01_no_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv32ima_01_x_rv64im_01_eh,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv32ima_01_x_rv64im_01_eh_spec,
    [("rv32ima/01_atomic_operations", arch=Arch::Rv32, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv64d_01_x_rv64i_01_no_eh,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv64d_01_x_rv64i_01_eh,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv64d_01_x_rv64i_01_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv64d_01_x_rv64im_01_no_eh,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv64d_01_x_rv64im_01_eh,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv64d_01_x_rv64im_01_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link!(link_rv64i_01_x_rv64im_01_no_eh,
    [("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link!(link_rv64i_01_x_rv64im_01_eh,
    [("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link!(link_rv64i_01_x_rv64im_01_eh_spec,
    [("rv64i/01_basic_64bit", arch=Arch::Rv64, entry="entry_0"),
     ("rv64im/01_multiply_divide_64", arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);

// ── Corpus-C link tests ─────────────────────────────────────────────────────────

link_c!(link_rv32d_01_x_rv32c_arith_no_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32d_01_x_rv32c_arith_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32d_01_x_rv32c_arith_eh_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32d_01_x_rv64c_arith_no_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32d_01_x_rv64c_arith_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32d_01_x_rv64c_arith_eh_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32d_01_x_x86c_arith_no_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32d_01_x_x86c_arith_eh,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32d_01_x_x86c_arith_eh_spec,
    [("rv32d/01_double_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32f_01_x_rv32c_arith_no_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32f_01_x_rv32c_arith_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32f_01_x_rv32c_arith_eh_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32f_01_x_rv64c_arith_no_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32f_01_x_rv64c_arith_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32f_01_x_rv64c_arith_eh_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32f_01_x_x86c_arith_no_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32f_01_x_x86c_arith_eh,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32f_01_x_x86c_arith_eh_spec,
    [("rv32f/01_single_precision_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32fd_01_x_rv32c_arith_no_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32fd_01_x_rv32c_arith_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32fd_01_x_rv32c_arith_eh_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32fd_01_x_rv64c_arith_no_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32fd_01_x_rv64c_arith_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32fd_01_x_rv64c_arith_eh_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32fd_01_x_x86c_arith_no_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32fd_01_x_x86c_arith_eh,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32fd_01_x_x86c_arith_eh_spec,
    [("rv32fd/01_combined_fp", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_01_x_rv32c_arith_no_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_01_x_rv32c_arith_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_01_x_rv32c_arith_eh_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_01_x_rv64c_arith_no_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_01_x_rv64c_arith_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_01_x_rv64c_arith_eh_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_01_x_x86c_arith_no_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_01_x_x86c_arith_eh,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_01_x_x86c_arith_eh_spec,
    [("rv32i/01_integer_computational", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_02_x_rv32c_arith_no_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_02_x_rv32c_arith_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_02_x_rv32c_arith_eh_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_02_x_rv64c_arith_no_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_02_x_rv64c_arith_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_02_x_rv64c_arith_eh_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_02_x_x86c_arith_no_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_02_x_x86c_arith_eh,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_02_x_x86c_arith_eh_spec,
    [("rv32i/02_control_transfer", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_03_x_rv32c_arith_no_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_03_x_rv32c_arith_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_03_x_rv32c_arith_eh_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_03_x_rv64c_arith_no_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_03_x_rv64c_arith_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_03_x_rv64c_arith_eh_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_03_x_x86c_arith_no_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_03_x_x86c_arith_eh,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_03_x_x86c_arith_eh_spec,
    [("rv32i/03_load_store", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_04_x_rv32c_arith_no_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_04_x_rv32c_arith_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_04_x_rv32c_arith_eh_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_04_x_rv64c_arith_no_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_04_x_rv64c_arith_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_04_x_rv64c_arith_eh_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_04_x_x86c_arith_no_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_04_x_x86c_arith_eh,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_04_x_x86c_arith_eh_spec,
    [("rv32i/04_edge_cases", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_05_x_rv32c_arith_no_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_05_x_rv32c_arith_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_05_x_rv32c_arith_eh_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_05_x_rv64c_arith_no_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_05_x_rv64c_arith_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_05_x_rv64c_arith_eh_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_05_x_x86c_arith_no_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_05_x_x86c_arith_eh,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_05_x_x86c_arith_eh_spec,
    [("rv32i/05_simple_program", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_06_x_rv32c_arith_no_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_06_x_rv32c_arith_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_06_x_rv32c_arith_eh_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_06_x_rv64c_arith_no_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_06_x_rv64c_arith_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_06_x_rv64c_arith_eh_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_06_x_x86c_arith_no_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_06_x_x86c_arith_eh,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_06_x_x86c_arith_eh_spec,
    [("rv32i/06_nop_and_hints", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_07_x_rv32c_arith_no_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_07_x_rv32c_arith_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_07_x_rv32c_arith_eh_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_07_x_rv64c_arith_no_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_07_x_rv64c_arith_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_07_x_rv64c_arith_eh_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_07_x_x86c_arith_no_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_07_x_x86c_arith_eh,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_07_x_x86c_arith_eh_spec,
    [("rv32i/07_pseudo_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_zicsr_01_x_rv32c_arith_no_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_zicsr_01_x_rv32c_arith_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_zicsr_01_x_rv32c_arith_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_zicsr_01_x_rv64c_arith_no_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_zicsr_01_x_rv64c_arith_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_zicsr_01_x_rv64c_arith_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32i_zicsr_01_x_x86c_arith_no_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32i_zicsr_01_x_x86c_arith_eh,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32i_zicsr_01_x_x86c_arith_eh_spec,
    [("rv32i_zicsr/01_csr_instructions", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32im_01_x_rv32c_arith_no_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32im_01_x_rv32c_arith_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32im_01_x_rv32c_arith_eh_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32im_01_x_rv64c_arith_no_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32im_01_x_rv64c_arith_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32im_01_x_rv64c_arith_eh_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32im_01_x_x86c_arith_no_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32im_01_x_x86c_arith_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32im_01_x_x86c_arith_eh_spec,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32ima_01_x_rv32c_arith_no_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32ima_01_x_rv32c_arith_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32ima_01_x_rv32c_arith_eh_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32ima_01_x_rv64c_arith_no_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32ima_01_x_rv64c_arith_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32ima_01_x_rv64c_arith_eh_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32ima_01_x_x86c_arith_no_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32ima_01_x_x86c_arith_eh,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32ima_01_x_x86c_arith_eh_spec,
    [("rv32ima/01_atomic_operations", is_corpus=true,  arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64d_01_x_rv32c_arith_no_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64d_01_x_rv32c_arith_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64d_01_x_rv32c_arith_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64d_01_x_rv64c_arith_no_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64d_01_x_rv64c_arith_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64d_01_x_rv64c_arith_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64d_01_x_x86c_arith_no_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64d_01_x_x86c_arith_eh,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64d_01_x_x86c_arith_eh_spec,
    [("rv64d/01_rv64_double_precision_fp", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64i_01_x_rv32c_arith_no_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64i_01_x_rv32c_arith_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64i_01_x_rv32c_arith_eh_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64i_01_x_rv64c_arith_no_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64i_01_x_rv64c_arith_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64i_01_x_rv64c_arith_eh_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64i_01_x_x86c_arith_no_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64i_01_x_x86c_arith_eh,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64i_01_x_x86c_arith_eh_spec,
    [("rv64i/01_basic_64bit", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64im_01_x_rv32c_arith_no_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64im_01_x_rv32c_arith_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64im_01_x_rv32c_arith_eh_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64im_01_x_rv64c_arith_no_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64im_01_x_rv64c_arith_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64im_01_x_rv64c_arith_eh_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64im_01_x_x86c_arith_no_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64im_01_x_x86c_arith_eh,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64im_01_x_x86c_arith_eh_spec,
    [("rv64im/01_multiply_divide_64", is_corpus=true,  arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);

// ── C-C link tests ──────────────────────────────────────────────────────────────

link_c!(link_rv32c_arith_x_rv64c_arith_no_eh,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32c_arith_x_rv64c_arith_eh,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32c_arith_x_rv64c_arith_eh_spec,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv32c_arith_x_x86c_arith_no_eh,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv32c_arith_x_x86c_arith_eh,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv32c_arith_x_x86c_arith_eh_spec,
    [("E2E_RV32_ARITH", is_corpus=false, arch=Arch::Rv32, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);
link_c!(link_rv64c_arith_x_x86c_arith_no_eh,
    [("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::None, speculative=false);
link_c!(link_rv64c_arith_x_x86c_arith_eh,
    [("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=false);
link_c!(link_rv64c_arith_x_x86c_arith_eh_spec,
    [("E2E_RV64_ARITH", is_corpus=false, arch=Arch::Rv64, entry="entry_0"),
     ("E2E_X86_ARITH", is_corpus=false, arch=Arch::X86_64, entry="entry_1")],
    Eh::With, speculative=true);

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

// ── Native-backend corpus tests ─────────────────────────────────────────────────

native!(native_rv32d_01_no_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32d_01_eh, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32d_01_eh_spec, "rv32d/01_double_precision_fp", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32f_01_no_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32f_01_eh, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32f_01_eh_spec, "rv32f/01_single_precision_fp", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32fd_01_no_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32fd_01_eh, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32fd_01_eh_spec, "rv32fd/01_combined_fp", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32i_01_no_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32i_01_eh, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32i_01_eh_spec, "rv32i/01_integer_computational", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32i_02_no_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32i_02_eh, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32i_02_eh_spec, "rv32i/02_control_transfer", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32i_03_no_eh, "rv32i/03_load_store", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32i_03_eh, "rv32i/03_load_store", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32i_03_eh_spec, "rv32i/03_load_store", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32i_04_no_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32i_04_eh, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32i_04_eh_spec, "rv32i/04_edge_cases", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32i_05_no_eh, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32i_05_eh, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32i_05_eh_spec, "rv32i/05_simple_program", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32i_06_no_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32i_06_eh, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32i_06_eh_spec, "rv32i/06_nop_and_hints", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32i_07_no_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32i_07_eh, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32i_07_eh_spec, "rv32i/07_pseudo_instructions", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32i_zicsr_01_no_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32i_zicsr_01_eh, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32i_zicsr_01_eh_spec, "rv32i_zicsr/01_csr_instructions", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32im_01_no_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32im_01_eh, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32im_01_eh_spec, "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv32ima_01_no_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::None, speculative=false);
native!(native_rv32ima_01_eh, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::With, speculative=false);
native!(native_rv32ima_01_eh_spec, "rv32ima/01_atomic_operations", arch=Arch::Rv32, Eh::With, speculative=true);
native!(native_rv64d_01_no_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::None, speculative=false);
native!(native_rv64d_01_eh, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::With, speculative=false);
native!(native_rv64d_01_eh_spec, "rv64d/01_rv64_double_precision_fp", arch=Arch::Rv64, Eh::With, speculative=true);
native!(native_rv64i_01_no_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::None, speculative=false);
native!(native_rv64i_01_eh, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::With, speculative=false);
native!(native_rv64i_01_eh_spec, "rv64i/01_basic_64bit", arch=Arch::Rv64, Eh::With, speculative=true);
native!(native_rv64im_01_no_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::None, speculative=false);
native!(native_rv64im_01_eh, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::With, speculative=false);
native!(native_rv64im_01_eh_spec, "rv64im/01_multiply_divide_64", arch=Arch::Rv64, Eh::With, speculative=true);

// ── Native-backend C tests ──────────────────────────────────────────────────────

native_c!(native_rv32c_arith_no_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::None, speculative=false);
native_c!(native_rv32c_arith_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::With, speculative=false);
native_c!(native_rv32c_arith_eh_spec, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::With, speculative=true);
native_c!(native_rv64c_arith_no_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::None, speculative=false);
native_c!(native_rv64c_arith_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::With, speculative=false);
native_c!(native_rv64c_arith_eh_spec, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::With, speculative=true);
native_c!(native_x86c_arith_no_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::None, speculative=false);
native_c!(native_x86c_arith_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::With, speculative=false);
native_c!(native_x86c_arith_eh_spec, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::With, speculative=true);

// ── Native-backend WASM tests ───────────────────────────────────────────────────

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

// @generated-tests-end

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
