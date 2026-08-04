//! Linux → WASI Lane A integration tests.
//!
//! The write/exit cell is also in the generated matrix (`linux_wasi!`).

#[path = "harness/mod.rs"]
mod harness;

use harness::{run_preview1, FIXTURE_WRITE_EXIT};

#[test]
fn test_linux_to_wasi_write_and_exit() {
    let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm(FIXTURE_WRITE_EXIT, 0x1000);
    let state = run_preview1(&wasm, "_start", 520, b"hello\n").expect("preview1");
    assert_eq!(state.stdout, b"hello\n");
    assert_eq!(state.exit_code, Some(0));
}
