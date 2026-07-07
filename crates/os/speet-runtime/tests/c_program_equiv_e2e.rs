//! Original vs recompiled equivalence for committed C corpus programs.

use speet_corpus_harness::load_manifest;
use speet_program_equivalence::{
    assert_original_matches_recompiled, corpus_roots, default_runtime, linked_exists,
};
use speet_guest_runner::PathPlanner;

#[test]
fn c_program_equiv_manifest_entries() {
    let planner = PathPlanner::from_env();
    let mut rt = default_runtime();
    if !rt.llvm_available() {
        eprintln!("SKIP: LLVM clang not available");
        return;
    }

    let mut tested = 0usize;
    for root in corpus_roots() {
        let manifest = root.join("manifest.toml");
        if !manifest.is_file() {
            continue;
        }
        let artifacts = load_manifest(&manifest).expect("manifest");
        for art in &artifacts {
            if art.program != "exit42" && art.program != "arith" {
                continue;
            }
            if !linked_exists(&root, art) {
                eprintln!(
                    "SKIP: no linked artifact for {} {}",
                    art.program, art.triple
                );
                continue;
            }
            assert_original_matches_recompiled(&root, art, &planner, &mut rt).unwrap_or_else(
                |e| panic!("equiv {} {}: {e}", art.program, art.triple),
            );
            tested += 1;
        }
    }
    if tested == 0 {
        eprintln!("SKIP: no linked guests available for equivalence");
    }
}
