//! Corpus module assembly — delegates to `speet_recompile::frontend` with
//! [`ImportManifest::corpus_harness`] and [`EntityIndexSpace`] slot allocation.

use speet_host_api::ImportManifest;
use speet_recompile::frontend::{assemble_translated_module, host_capability_total};
use wasm_encoder::{Function, ValType};

/// Shared import manifest for corpus tests and wasmi execution.
pub fn corpus_manifest() -> ImportManifest {
    ImportManifest::corpus_harness()
}

/// Host-capability slot count for [`corpus_manifest`], read from
/// [`EntityIndexSpace`] — never hand-count this.
pub fn corpus_n_imports() -> u32 {
    host_capability_total(&corpus_manifest())
}

/// Build a runnable module exporting `_start` at `entry_func_idx`.
///
/// `entry_func_idx` is 0-based among translated functions only (before imports),
/// matching `speet_recompile::frontend::assemble_translated_module`.
pub fn assemble_corpus_module(
    fns: &[Function],
    params: &[ValType],
    entry_func_idx: u32,
) -> Vec<u8> {
    assemble_translated_module(
        &corpus_manifest(),
        fns,
        params.to_vec(),
        entry_func_idx,
        true,
    )
}
