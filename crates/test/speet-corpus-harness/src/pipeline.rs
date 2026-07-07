//! End-to-end C corpus pipeline: translate → instrument → run → assert semantics.

use std::path::Path;

use wasm_encoder::{Function, ValType};

use crate::{
    assemble::{assemble_corpus_module, N_CORPUS_IMPORTS},
    digest::sha256_hex,
    expected::{expectation_for_triple, load_expected, TripleExpectation},
    guest_pc_to_func_idx, instrument_functions, load::entry_pc, run::run_corpus_module, CorpusArch,
};

pub fn run_c_corpus_file(
    arch: CorpusArch,
    text_elf: &Path,
    translate: impl FnOnce(&[u8], u64) -> (Vec<Function>, Vec<ValType>),
) {
    run_c_corpus_file_with_expected(arch, text_elf, None, translate);
}

pub fn run_c_corpus_file_with_expected(
    arch: CorpusArch,
    text_elf: &Path,
    expected: Option<&TripleExpectation>,
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

    if let Some(exp) = expected {
        assert_semantics(text_elf, &state, exp);
    }

    println!(
        "  ✓ {} — {} funcs, entry #{entry_func_idx}",
        text_elf.file_name().unwrap().to_string_lossy(),
        fns.len()
    );
}

/// Run a program directory's `expected.toml` against a `.text.elf` for a triple.
pub fn run_c_program_text(
    arch: CorpusArch,
    corpus_root: &Path,
    program: &str,
    triple: &str,
    text_elf: &Path,
    translate: impl FnOnce(&[u8], u64) -> (Vec<Function>, Vec<ValType>),
) {
    let expected_path = corpus_root.join("programs").join(program).join("expected.toml");
    let expectations = load_expected(&expected_path)
        .unwrap_or_else(|e| panic!("load {}: {e}", expected_path.display()));
    let triple_exp = expectation_for_triple(&expectations, triple)
        .unwrap_or_else(|| panic!("no [[triple]] name={triple} in {}", expected_path.display()));
    run_c_corpus_file_with_expected(arch, text_elf, Some(triple_exp), translate);
}

fn assert_semantics(path: &Path, state: &crate::run::RunState, exp: &TripleExpectation) {
    if let Some(code) = exp.exit_code {
        assert_eq!(
            state.exit_code,
            Some(code),
            "{}: exit_code",
            path.display()
        );
    }
    if let Some(ret) = exp.main_return {
        let actual = state.exit_code.or(state.main_return);
        assert_eq!(actual, Some(ret), "{}: main_return/exit", path.display());
    }
    if let Some(ref digest) = exp.stdout_sha256 {
        let got = sha256_hex(&state.stdout);
        assert_eq!(got, *digest, "{}: stdout digest", path.display());
    }
}
