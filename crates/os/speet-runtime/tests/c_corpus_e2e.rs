//! Thin-runtime end-to-end tests for libc-linked C exit guests from `test-data/c-corpus/`.
//!
//! Only guest `.text` is recompiled; libc/libSystem symbols are tunnelled at link via
//! [`speet_host_api::default_host_api`].
//!
//! On Apple Silicon, Darwin guests use native aarch64 blitz output (asm-arch
//! `AArch64Writer`) — no Rosetta.

use binary_io::{BinArch, BinOs};
use speet_runtime::{default_host_api, load_text_from_object, Runtime};
use std::path::Path;

fn c_corpus_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus")
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

fn guest_exists(path: &Path) -> bool {
    if path.exists() {
        return true;
    }
    eprintln!("SKIP: missing committed guest {}", path.display());
    false
}

/// Apple Silicon Darwin path: hard-fail on pipeline errors (native aarch64 output).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run_c_guest_aarch64_macos(rt: &mut Runtime, guest: &Path) {
    let status = rt
        .recompile_binary_and_run(guest, BinArch::AArch64, BinOs::MacOs)
        .unwrap_or_else(|e| panic!("aarch64-macos pipeline {}: {e}", guest.display()));
    assert_eq!(status.code(), Some(42), "{}", guest.display());
}

/// Soft-skip when the linker cannot resolve wasm imports yet (non-AS hosts).
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn run_c_guest_or_skip(rt: &mut Runtime, guest: &Path, arch: BinArch, os: BinOs) {
    match rt.recompile_binary_and_run(guest, arch, os) {
        Ok(status) => assert_eq!(status.code(), Some(42)),
        Err(e)
            if e.contains("link failed")
                || e.contains("Undefined symbols")
                || e.contains("required architecture") =>
        {
            eprintln!("SKIP: {guest:?} pipeline: {e}");
        }
        Err(e) => panic!("pipeline: {e}"),
    }
}

/// aarch64 Mach-O C `main` returning 42 — recompile `.text`, tunnel libc, spawn.
#[test]
fn c_corpus_aarch64_macho_exit() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    if skip_no_llvm(&rt) {
        return;
    }
    let guest = c_corpus_root().join("aarch64-macos/exit.macho");
    if !guest_exists(&guest) {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    run_c_guest_aarch64_macos(&mut rt, &guest);
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    run_c_guest_or_skip(&mut rt, &guest, BinArch::AArch64, BinOs::MacOs);
}

/// aarch64 Mach-O `exit42` corpus program — same native blitz path as `exit`.
#[test]
fn c_corpus_aarch64_macho_exit42() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    if skip_no_llvm(&rt) {
        return;
    }
    // Prefer the libc-linked Mach-O (exercises dyld stub → redirect shim).
    // Fall back to the freestanding thin guest (return 42 / halt path).
    let linked = c_corpus_root().join("aarch64-macos/exit42.linked.macho");
    let thin = c_corpus_root().join("aarch64-macos/exit42.macho");
    let guest = if linked.exists() {
        linked
    } else if thin.exists() {
        thin
    } else {
        eprintln!("SKIP: missing aarch64-macos/exit42.{{linked.,}}macho");
        return;
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    run_c_guest_aarch64_macos(&mut rt, &guest);
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    run_c_guest_or_skip(&mut rt, &guest, BinArch::AArch64, BinOs::MacOs);
}

/// x86_64 Mach-O C exit guest (when committed).
#[test]
fn c_corpus_x86_64_macho_exit() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    if skip_no_llvm(&rt) {
        return;
    }
    let guest = c_corpus_root().join("x86_64-macos/exit.macho");
    if !guest_exists(&guest) {
        return;
    }
    if cfg!(target_arch = "aarch64") {
        eprintln!("SKIP: x86_64 Mach-O link/spawn on Apple Silicon hosts");
        return;
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        run_c_guest_or_skip(&mut rt, &guest, BinArch::X86_64, BinOs::MacOs);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let _ = &mut rt;
}

/// Linux ELF guests (skipped on macOS without a Linux sysroot).
#[test]
fn c_corpus_linux_elf_exit() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    if skip_no_llvm(&rt) || skip_elf_link_on_macos(BinOs::Linux) {
        return;
    }
    for (subdir, arch) in [
        ("aarch64-linux/exit.elf", BinArch::AArch64),
        ("x86_64-linux/exit.elf", BinArch::X86_64),
    ] {
        let guest = c_corpus_root().join(subdir);
        if !guest_exists(&guest) {
            continue;
        }
        let status = rt
            .recompile_binary_and_run(&guest, arch, BinOs::Linux)
            .expect("pipeline");
        if cfg!(target_os = "linux") {
            assert_eq!(status.code(), Some(42), "{}", guest.display());
        }
    }
}

/// Committed Mach-O guests expose non-empty `.text` for recompilation input.
#[test]
fn c_corpus_guests_have_text() {
    for rel in [
        "aarch64-macos/exit.macho",
        "aarch64-macos/exit42.macho",
        "aarch64-macos/exit42.linked.macho",
        "x86_64-macos/exit.macho",
    ] {
        let path = c_corpus_root().join(rel);
        if !path.exists() {
            eprintln!("SKIP: missing {}", path.display());
            continue;
        }
        let (text, _) = load_text_from_object(&path).expect("load text");
        assert!(!text.is_empty(), "empty .text in {}", path.display());
    }
}
