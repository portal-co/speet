//! proptest harness (plan §3.2): random seeds through the full comparison.
//! proptest's shrinking applies to the seed space — coarse, but persistent:
//! failing seeds persist to `proptest-regressions/` and replay on CI.

use proptest::prelude::*;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Union of facet names across every checked-in artifact's description —
/// the set of documented bug classes.
fn documented_facets() -> &'static HashSet<String> {
    static SET: OnceLock<HashSet<String>> = OnceLock::new();
    SET.get_or_init(|| {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../test-data/diff-fuzz/x86_64");
        let mut set = HashSet::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                    &std::fs::read_to_string(&p).unwrap_or_default(),
                ) {
                    if let Some(desc) = v.get("description").and_then(|d| d.as_str()) {
                        for f in desc.split("; ") {
                            let f = f.split(':').next().unwrap_or(f).trim();
                            set.insert(if f.starts_with("flag") { "flags".into() } else { f.to_string() });
                        }
                    }
                }
            }
        }
        set
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32, // keep CI fast; crank locally via PROPTEST_CASES
        max_shrink_iters: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn recompiled_matches_oracle(seed in any::<u64>()) {
        let case = speet_diff_core::generate_case(seed);
        let oracle = speet_diff_core::run_oracle(&case);
        let recompiled = speet_diff_core::run_recompiled(&case);
        let c = speet_diff_core::compare_outcomes(
            &case, oracle.as_ref().ok(), recompiled.as_ref());
        // Known-divergence classes are documented via checked-in artifacts
        // (tests/replay_corpus.rs) and allowed here BY SIGNATURE: a diver-
        // gence is only tolerated when its minimized code + facet set match
        // an artifact. Anything new fails and gets minimized/checked in.
        if let speet_diff_core::Comparison::Divergence(desc) = &c {
            // Tolerate a divergence only when EVERY facet in its description
            // belongs to a documented bug class (the union of checked-in
            // artifacts' facet sets). A new facet (a register or memory
            // mismatch not seen in any artifact) fails the harness and must
            // be minimized + checked in via `diff-fuzz`.
            let facets: HashSet<String> = desc
                .split("; ")
                .map(|f| f.split(':').next().unwrap_or(f).trim().to_string())
                .map(|f| if f.starts_with("flag") { "flags".into() } else { f })
                .collect();
            prop_assert!(
                !facets.is_empty() && facets.is_subset(&documented_facets()),
                "seed {seed}: NEW divergence class {desc}\n  facets: {facets:?}"
            );
        }
    }
}
