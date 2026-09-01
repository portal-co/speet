//! End-to-end thin runtime tests using the committed corpus.

use binary_io::{BinArch, BinOs};
use speet_runtime::{default_host_api, load_text_from_object, HostApi, Runtime};
use std::path::Path;

fn corpus_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/thin-runtime-corpus")
}

fn skip_no_llvm(rt: &Runtime) -> bool {
    if !rt.llvm_available() {
        eprintln!("SKIP: LLVM clang not available");
        return true;
    }
    false
}

fn skip_elf_link_on_macos(os: BinOs) -> bool {
    if cfg!(target_os = "macos") && matches!(os, BinOs::Linux) {
        eprintln!("SKIP: Linux ELF link requires a Linux sysroot on macOS hosts");
        return true;
    }
    false
}

/// RV64 exit(42) → x86_64 ELF output → link (execute only on Linux hosts).
#[test]
fn corpus_rv64_to_x86_64_elf() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    if skip_no_llvm(&rt) || skip_elf_link_on_macos(BinOs::Linux) {
        return;
    }
    let guest = corpus_root().join("rv64-linux/exit_42.elf");
    let (text, addr) = load_text_from_object(&guest).unwrap();
    let wasm = rt.recompile_rv64_text(&text, addr);
    let obj = rt
        .compile_to_object(&wasm, BinArch::X86_64, BinOs::Linux)
        .expect("blitz");
    let dir = std::env::temp_dir().join(format!("speet_rt_xelf_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("guest");
    let entry_param_count = speet_recompile::drive::entry_param_count(&wasm);
    let halt_addr = speet_recompile::frontend::halt_addr(addr, text.len());
    rt.link_guest_object(
        &obj,
        entry_param_count,
        speet_riscv::RV64_SP_PARAM_INDEX,
        halt_addr,
        Some(speet_riscv::RV64_RA_PARAM_INDEX),
        BinArch::X86_64,
        BinOs::Linux,
        &exe,
    )
    .expect("link");
    assert!(exe.exists());
    if cfg!(target_os = "linux") {
        let status = std::process::Command::new(&exe).status().expect("run");
        assert_eq!(status.code(), Some(42));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// RV64 exit(42) → aarch64 Mach-O output → link → spawn (dev host path).
#[test]
fn corpus_rv64_to_aarch64_macho() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    if skip_no_llvm(&rt) {
        return;
    }
    let guest = corpus_root().join("rv64-linux/exit_42.elf");
    let status = rt
        .run_corpus_object(&guest, BinArch::AArch64, BinOs::MacOs)
        .expect("pipeline");
    assert_eq!(status.code(), Some(42));
}

/// Corpus objects load and yield non-empty `.text`.
#[test]
fn corpus_guests_have_text() {
    let root = corpus_root();
    for rel in [
        "rv64-linux/exit_42.elf",
        "x86_64-linux/exit_42.elf",
        "aarch64-linux/exit_42.elf",
        "x86_64-macos/exit_42.macho",
        "aarch64-macos/exit_42.macho",
    ] {
        let path = root.join(rel);
        assert!(
            path.exists(),
            "missing committed corpus blob: {}",
            path.display()
        );
        let (text, _) = load_text_from_object(&path).expect("load text");
        assert!(!text.is_empty(), "empty .text in {}", path.display());
    }
}

/// Link-only smoke (no execute): proves LLVM link wiring without Rosetta/runtime quirks.
#[test]
fn corpus_link_only_x86_64_elf() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    if skip_no_llvm(&rt) || skip_elf_link_on_macos(BinOs::Linux) {
        return;
    }
    let guest = corpus_root().join("rv64-linux/exit_42.elf");
    let (text, addr) = load_text_from_object(&guest).unwrap();
    let wasm = rt.recompile_rv64_text(&text, addr);
    let obj = rt
        .compile_to_object(&wasm, BinArch::X86_64, BinOs::Linux)
        .expect("blitz");
    let dir = std::env::temp_dir().join(format!("speet_rt_link_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("linked");
    let entry_param_count = speet_recompile::drive::entry_param_count(&wasm);
    let halt_addr = speet_recompile::frontend::halt_addr(addr, text.len());
    rt.link_guest_object(
        &obj,
        entry_param_count,
        speet_riscv::RV64_SP_PARAM_INDEX,
        halt_addr,
        Some(speet_riscv::RV64_RA_PARAM_INDEX),
        BinArch::X86_64,
        BinOs::Linux,
        &exe,
    )
    .expect("link");
    assert!(exe.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Artifact cache returns the same WASM bytes for identical input.
#[test]
fn cache_hit_wasm() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    let guest = corpus_root().join("rv64-linux/exit_42.elf");
    let (text, addr) = load_text_from_object(&guest).unwrap();
    let w1 = rt.recompile_rv64_text(&text, addr);
    let w2 = rt.recompile_rv64_text(&text, addr);
    assert_eq!(w1, w2);
}

/// FilteredHostApi denies unknown ambient symbols.
#[test]
fn filtered_host_api_denies_printf() {
    use speet_runtime::FilteredHostApi;
    let api = FilteredHostApi::exit_write_only(default_host_api());
    assert!(api.resolve_ambient("printf").is_none());
    assert!(api.resolve_ambient("write").is_some());
}
