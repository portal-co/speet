//! Darwin/BSD → WASI Lane A integration tests.
//!
//! The basic write/exit cell is also in the generated matrix (`darwin_wasi!`).
//! GOT redirect remains here (needs `ExternalTargetTable` setup).

#[path = "harness/mod.rs"]
mod harness;

use harness::{run_preview1, FIXTURE_DARWIN_WRITE_EXIT};
use speet_plugin_api::external_target::{ExternalTargetTable, LibraryId};

#[test]
fn test_darwin_to_wasi_write_and_exit() {
    let wasm =
        speet_darwin_wasi::recompile_aarch64_darwin_wasi_to_wasm(FIXTURE_DARWIN_WRITE_EXIT, 0x1000);
    let state = run_preview1(&wasm, "_start", 520, b"hello\n").expect("preview1");
    assert_eq!(state.stdout, b"hello\n");
    assert_eq!(state.exit_code, Some(0));
}

/// Calls virtual PCs that a Mach-O loader would place into GOT/lazy-pointer
/// cells. Both `blr`s land on redirect shim slots, never on `svc`.
#[test]
fn test_darwin_got_redirect_write_and_exit() {
    const GOT_WRITE_EXIT: &[u8] = &[
        0x20, 0x00, 0x80, 0xD2,
        0x01, 0x41, 0x80, 0xD2,
        0xC2, 0x00, 0x80, 0xD2,
        0x83, 0x04, 0x82, 0xD2,
        0x60, 0x00, 0x3F, 0xD6,
        0x00, 0x00, 0x80, 0xD2,
        0x04, 0x05, 0x82, 0xD2,
        0x80, 0x00, 0x3F, 0xD6,
    ];
    let start_addr = 0x1000;
    let mut targets = ExternalTargetTable::new();
    targets.insert(LibraryId::MAIN_IMAGE, 0x1024, "_write");
    targets.insert(LibraryId::MAIN_IMAGE, 0x1028, "_exit");
    let wasm = speet_darwin_wasi::recompile_aarch64_darwin_wasi_to_wasm_with_targets(
        GOT_WRITE_EXIT,
        start_addr,
        &targets,
    );
    let state = run_preview1(&wasm, "_start", 520, b"hello\n").expect("preview1");
    assert_eq!(state.stdout, b"hello\n");
    assert_eq!(state.exit_code, Some(0));
}
