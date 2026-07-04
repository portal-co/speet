//! Shared helpers for C corpus integration tests.

pub mod assemble;
pub mod instrument;
pub mod load;
pub mod run;

pub use assemble::{assemble_corpus_module, N_CORPUS_IMPORTS, TRAP_IMPORT_IDX};
pub use instrument::instrument_functions;
pub use load::{load_entry_offset, load_text_blob};
pub mod pipeline;

pub use pipeline::run_c_corpus_file;

/// WASM module index for a guest PC under the default slot formulas (no slot assigner).
pub fn guest_pc_to_func_idx(arch: CorpusArch, base_pc: u64, guest_pc: u64, n_imports: u32) -> u32 {
    let offset = guest_pc.wrapping_sub(base_pc);
    let slot = match arch {
        CorpusArch::X86_64 => offset,
        CorpusArch::AArch64 => offset / 4,
        CorpusArch::Riscv => offset / 2,
    };
    n_imports + slot as u32
}

#[derive(Clone, Copy, Debug)]
pub enum CorpusArch {
    X86_64,
    AArch64,
    Riscv,
}
