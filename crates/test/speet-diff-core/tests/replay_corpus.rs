//! Regression corpus (plan §6.3): every checked-in divergence artifact must
//! still diverge. When a fix lands, the artifact's expectation flips to
//! `Comparison::Match` — update the artifact's `"expectation"` field to
//! `"fixed"` (or delete it) in the same commit.

use speet_diff_core::case::{FuzzCase, RegState};
use speet_diff_core::{compare_outcomes, run_oracle, run_recompiled, Comparison};
use std::path::PathBuf;

fn artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/diff-fuzz/x86_64")
}


#[test]
fn checked_in_divergences_still_diverge() {
    let dir = artifact_dir();
    let mut checked = 0;
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let case = speet_diff_core::case_from_json(&v);
        let oracle = run_oracle(&case);
        let recompiled = run_recompiled(&case);
        let c = compare_outcomes(&case, oracle.as_ref().ok(), recompiled.as_ref());
        let expectation = v.get("expectation").and_then(|e| e.as_str()).unwrap_or("divergence");
        match expectation {
            // Still-broken repro: must keep diverging (with the same facets).
            "divergence" => assert!(
                matches!(c, Comparison::Divergence(_)),
                "{}: expected a divergence, got {c:?}",
                path.display()
            ),
            // Fixed: the artifact documents a resolved bug — must match now.
            "fixed" => assert_eq!(c, Comparison::Match, "{}: regression!", path.display()),
            other => panic!("{}: unknown expectation {other:?}", path.display()),
        }
        checked += 1;
    }
    // The corpus must never silently empty out — divergence hunting should
    // keep it populated (add artifacts from `diff-fuzz` runs).
    assert!(checked > 0, "no artifacts in {}", dir.display());
}
