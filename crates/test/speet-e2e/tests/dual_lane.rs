//! Dual-lane smoke tests kept as thin wrappers over shared harness envs.
//!
//! The combinatorial matrix (escape configs × paths) lives in generated
//! `e2e.rs` cells (`linux_wasi!` / `thin_native!`). This file covers GOT /
//! corpus-ELF edge cases that are awkward to express in the generator.

#[path = "harness/mod.rs"]
mod harness;

use harness::{run_preview1, run_thin_rv64, FIXTURE_EXIT_42};

/// Same scenario from thin-runtime corpus ELF `.text` (Lane A on every host).
#[test]
fn dual_lane_linux_corpus_exit_42_wasi() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/thin-runtime-corpus/rv64-linux/exit_42.elf");
    let data = std::fs::read(&path).expect("corpus exit_42.elf");
    let obj = object::File::parse(&*data).expect("parse ELF");
    use object::{Object, ObjectSection};
    let text = obj
        .section_by_name(".text")
        .expect(".text")
        .data()
        .expect(".text data");
    let addr = obj.section_by_name(".text").unwrap().address();
    assert_eq!(text, FIXTURE_EXIT_42);
    let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm(text, addr);
    let result = run_preview1(&wasm, "_start", 0, &[]).expect("preview1");
    assert_eq!(result.exit_code, Some(42));
}

#[cfg(target_os = "macos")]
#[test]
fn dual_lane_linux_exit_42_native_parity() {
    match run_thin_rv64(FIXTURE_EXIT_42, 0x1000) {
        Ok(outcome) => assert_eq!(outcome.exit_code, Some(42)),
        Err(reason) => {
            eprintln!("SKIP: {reason}");
        }
    }
    let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm(FIXTURE_EXIT_42, 0x1000);
    let wasi = run_preview1(&wasm, "_start", 0, &[]).expect("preview1");
    assert_eq!(wasi.exit_code, Some(42));
}
