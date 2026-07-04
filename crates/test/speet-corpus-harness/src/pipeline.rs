//! End-to-end C corpus pipeline: translate → instrument → run.

use std::path::Path;

use wasm_encoder::{Function, ValType};

use crate::{
    assemble::{assemble_corpus_module, N_CORPUS_IMPORTS},
    guest_pc_to_func_idx, instrument_functions, load::entry_pc, run::run_corpus_module, CorpusArch,
};

pub fn run_c_corpus_file(
    arch: CorpusArch,
    text_elf: &Path,
    translate: impl FnOnce(&[u8], u64) -> (Vec<Function>, Vec<ValType>),
) {
    let (text, base, main_pc) = entry_pc(text_elf);
    let (mut fns, params) = translate(&text, base);
    assert!(!fns.is_empty(), "{}: no functions translated", text_elf.display());

    instrument_functions(&mut fns, N_CORPUS_IMPORTS);
    let entry_func_idx = guest_pc_to_func_idx(arch, base, main_pc, N_CORPUS_IMPORTS);
    let wasm = assemble_corpus_module(&fns, &params, entry_func_idx);

    wasmparser::validate(&wasm)
        .unwrap_or_else(|e| panic!("invalid wasm for {}: {e}", text_elf.display()));

    let state = run_corpus_module(&wasm, "_start")
        .unwrap_or_else(|e| panic!("run {} failed: {e}", text_elf.display()));

    assert!(
        state.unreachable_trap_hits.is_empty(),
        "{}: unreachable traps at func indices {:?}",
        text_elf.display(),
        state.unreachable_trap_hits
    );
    println!(
        "  ✓ {} — {} funcs, entry #{entry_func_idx}",
        text_elf.file_name().unwrap().to_string_lossy(),
        fns.len()
    );
}
