//! Original vs recompiled equivalence for committed C corpus programs.

use binary_io::{BinArch, BinOs};
use speet_corpus_harness::load_manifest;
use speet_corpus_harness::manifest::Artifact;
use speet_program_equivalence::{
    assert_original_matches_recompiled, corpus_roots, default_runtime, linked_exists,
};
use speet_guest_runner::PathPlanner;
use speet_recompile::frontend::host_platform;

fn artifact_matches_host(art: &Artifact) -> bool {
    let (host_os, host_arch) = host_platform();
    let arch_ok = match (art.arch.as_str(), host_arch) {
        ("aarch64", BinArch::AArch64) => true,
        ("x86_64", BinArch::X86_64) => true,
        _ => false,
    };
    let os_ok = match (art.os.as_str(), host_os) {
        ("macos", BinOs::MacOs) => true,
        ("linux", BinOs::Linux) => true,
        _ => false,
    };
    arch_ok && os_ok
}

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
            if !artifact_matches_host(art) {
                eprintln!(
                    "SKIP: {} {} (guest platform ≠ host; v1 same-platform only)",
                    art.program, art.triple
                );
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
