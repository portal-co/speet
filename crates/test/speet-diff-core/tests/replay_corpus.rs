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

fn case_from_json(v: &serde_json::Value) -> FuzzCase {
    let case = &v["case"];
    let hex = |s: &str| (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i+2], 16).unwrap()).collect::<Vec<u8>>();
    let g: Vec<u64> = case["regs"]["gprs"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    let arch = speet_diff_core::parse_arch_name(case["arch"].as_str());
    let mut gprs = [0u64; 32];
    for (i, v) in g.iter().enumerate() {
        gprs[i] = *v;
    }
    FuzzCase {
        arch,
        code: hex(case["code_hex"].as_str().unwrap()),
        entry_pc: case["entry_pc"].as_u64().unwrap(),
        regs: RegState {
            gprs,
            rip: case["regs"]["rip"].as_u64().unwrap(),
            zf: case["regs"]["zf"].as_bool().unwrap(),
            sf: case["regs"]["sf"].as_bool().unwrap(),
            cf: case["regs"]["cf"].as_bool().unwrap(),
            of: case["regs"]["of"].as_bool().unwrap(),
            pf: case["regs"]["pf"].as_bool().unwrap(),
            sp: case["regs"].get("sp").and_then(|x| x.as_u64()).unwrap_or(0),
        },
        data: hex(case["data_hex"].as_str().unwrap()),
        data_base: case["data_base"].as_u64().unwrap(),
        stack: hex(case["stack_hex"].as_str().unwrap()),
        stack_base: case["stack_base"].as_u64().unwrap(),
        read_only: case["read_only"].as_array().map(|a| a.iter().map(|p| (p[0].as_u64().unwrap(), p[1].as_u64().unwrap())).collect()).unwrap_or_default(),
        step_budget: case["step_budget"].as_u64().unwrap(),
        seed: case.get("seed").and_then(|s| s.as_u64()).unwrap_or(0),
    }
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
        let case = case_from_json(&v);
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
