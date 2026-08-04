//! Runtime unreachable trap logging end-to-end.

use binary_io::{BinArch, BinOs};
use speet_host_api::integrated_host_api;
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::recompile_to_wasm_instrumented;
use speet_runtime::{link_guest_integrated, IntegratedNativeRuntime, LlvmToolchain, NativeRuntime};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

fn host_link_target() -> (BinArch, BinOs) {
    // Same-platform host output (aarch64 Mach-O on Apple Silicon; no Rosetta).
    let (os, arch) = speet_recompile::frontend::host_platform();
    (arch, os)
}

#[test]
fn unreach_trap_logs_guest_pc() {
    let tc = match LlvmToolchain::from_build_env() {
        Some(t) if t.is_available() => t,
        _ => {
            eprintln!("SKIP: LLVM not available");
            return;
        }
    };

    let (out_arch, out_os) = host_link_target();
    let guest_arch = if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
        BinArch::AArch64
    } else if cfg!(target_arch = "x86_64") {
        BinArch::X86_64
    } else {
        BinArch::AArch64
    };

    // Undefined AArch64 word on aarch64 guest; x86 uses HLT (0xF4) on x86 guest.
    let (text, start) = match guest_arch {
        BinArch::AArch64 => ([0xffu8, 0xff, 0xff, 0xff].as_slice(), 0x1000u64),
        BinArch::X86_64 => ([0xf4u8].as_slice(), 0x1000u64),
        _ => unreachable!(),
    };

    let (wasm, _) = recompile_to_wasm_instrumented(text, start, guest_arch);
    let obj = compile_wasm_to_object(&wasm, out_arch, out_os).expect("blitz");
    let dir = std::env::temp_dir().join(format!("speet_unreach_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("trap_guest");
    let host = integrated_host_api();
    let entry_param_count = speet_recompile::drive::entry_param_count(&wasm);
    let sp_idx = speet_recompile::drive::sp_param_index(guest_arch);
    let lr_idx = speet_recompile::drive::lr_param_index(guest_arch);
    let catalog = speet_recompile::guest_func_catalog::GuestFuncCatalog::from_wasm(
        &wasm,
        start,
        guest_arch,
        text.len(),
    );
    let entry_local = speet_recompile::drive::entry_local_func_idx(&wasm);
    let halt_local = catalog.halt_entry().local_func_idx;
    link_guest_integrated(
        &tc,
        &host,
        &obj,
        out_arch,
        out_os,
        entry_param_count,
        sp_idx,
        lr_idx,
        &catalog.to_stub_entries(),
        entry_local,
        halt_local,
        None,
        &[],
        &dir.join("work"),
        &exe,
    )
    .expect("link");

    let output = Command::new(&exe)
        .output()
        .expect("spawn");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("speet: unsupported instruction at guest pc=0x1000"),
        "expected unreachable log, got stderr={stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn integrated_analyze_exit42_and_wasm_instrumented() {
    let Some(path) = exit42_path() else {
        eprintln!("SKIP: no exit42 linked artifact");
        return;
    };
    let rt = IntegratedNativeRuntime::new(Arc::new(integrated_host_api()));
    let report = rt.analyze(&path).expect("analyze");
    assert!(report.suitable, "{report:?}");

    let bin = speet_runtime::load_binary(&path).expect("load");
    let text = bin
        .sections
        .iter()
        .find(|s| {
            s.name == ".text"
                || s.name == "__TEXT,__text"
                || s.name == "__text"
                || matches!(s.kind, binary_io::SectionKind::Text)
        })
        .expect("text");
    let (wasm, _) =
        speet_recompile::frontend::recompile_to_wasm_instrumented(&text.data, text.addr, bin.arch);
    speet_runtime::validate_wasm_public(&wasm).expect("valid wasm");
    let needle = b"__speet_log_unreachable";
    assert!(
        wasm.windows(needle.len()).any(|w| w == needle),
        "instrumented module missing log import"
    );
}

fn exit42_path() -> Option<std::path::PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus");
    if cfg!(target_arch = "aarch64") {
        let p = root.join("aarch64-macos/exit42.linked.macho");
        if p.is_file() {
            return Some(p);
        }
    }
    if cfg!(target_arch = "x86_64") {
        let p = root.join("x86_64-macos/exit42.linked.macho");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}
