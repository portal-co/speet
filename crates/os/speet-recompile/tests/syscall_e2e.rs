//! End-to-end syscall test: recompiled RV64 `exit(code)` terminates natively.
//!
//! Delegates link/run to `speet-runtime` (LLVM toolchain + generated shims).

#![cfg(target_os = "macos")]

use binary_io::{BinArch, BinOs};
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::recompile_rv64_to_wasm;
use speet_runtime::{default_host_api, Runtime};
use std::sync::Arc;

/// RV64: addi a0,x0,42 ; addi a7,x0,93 ; ecall
const EXIT_42: &[u8] = &[
    0x13, 0x05, 0xA0, 0x02, 0x93, 0x08, 0xD0, 0x05, 0x73, 0x00, 0x00, 0x00,
];

#[test]
#[ignore = "executes x86_64 via Rosetta (run on an x86 host or after Rosetta reset)"]
fn recompiled_rv64_exit_sets_code() {
    let mut rt = Runtime::new(Arc::new(default_host_api()));
    if !rt.llvm_available() {
        return;
    }
    let status = rt
        .recompile_rv64_and_run(EXIT_42, 0x1000, BinArch::X86_64, BinOs::MacOs)
        .expect("pipeline");
    assert_eq!(status.code(), Some(42));
}

#[cfg(target_arch = "aarch64")]
#[test]
fn recompiled_rv64_exit_sets_code_aarch64_native() {
    let mut rt = Runtime::new(Arc::new(default_host_api()));
    if !rt.llvm_available() {
        return;
    }
    let status = rt
        .recompile_rv64_and_run(EXIT_42, 0x1000, BinArch::AArch64, BinOs::MacOs)
        .expect("pipeline");
    assert_eq!(status.code(), Some(42));
}

#[test]
fn pipeline_reaches_linked_binary() {
    let rt = Runtime::new(Arc::new(default_host_api()));
    if !rt.llvm_available() {
        eprintln!("SKIP: LLVM not available");
        return;
    }
    let wasm = recompile_rv64_to_wasm(EXIT_42, 0x1000);
    let obj = compile_wasm_to_object(&wasm, BinArch::X86_64, BinOs::MacOs).expect("object");
    let dir = std::env::temp_dir().join(format!("speet_scl_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("exe");
    let entry_param_count = speet_recompile::drive::entry_param_count(&wasm);
    let halt_addr = speet_recompile::frontend::halt_addr(0x1000, EXIT_42.len());
    rt.link_guest_object(
        &obj,
        entry_param_count,
        speet_riscv::RV64_SP_PARAM_INDEX,
        halt_addr,
        Some(speet_riscv::RV64_RA_PARAM_INDEX),
        BinArch::X86_64,
        BinOs::MacOs,
        &exe,
    )
    .expect("link");
    assert!(exe.exists());
    let _ = std::fs::remove_dir_all(&dir);
}
