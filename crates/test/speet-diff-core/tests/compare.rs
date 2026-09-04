//! End-to-end comparison tests: known-good cases must MATCH; a flag-repro
//! documents the constant-folding finding (see README findings).

use speet_diff_core::tests_common::make_case;
use speet_diff_core::{compare_outcomes, run_oracle, run_recompiled, Comparison};

fn verdict(code: Vec<u8>) -> (String, Comparison) {
    let case = make_case(code, 1);
    let o = run_oracle(&case);
    let r = run_recompiled(&case);
    (
        format!("oracle={o:?} recompiled={r:?}"),
        compare_outcomes(&case, o.as_ref().ok(), r.as_ref()),
    )
}

#[test]
fn mov_r64_matches() {
    // mov rax, 0x42; ret
    let (ctx, c) = verdict(vec![0x48, 0xB8, 0x42, 0, 0, 0, 0, 0, 0, 0, 0xC3]);
    assert_eq!(c, Comparison::Match, "{ctx}");
}

#[test]
fn xor_reg_flags_match() {
    // mov rax, 3; xor rax, r15 (0xAAAA seeded); ret — result 0xAAA9 has
    // even parity → oracle PF=1. The compared final state includes flags,
    // so this must match bit-exactly (non-const operand → not foldable).
    let (ctx, c) = verdict(vec![
        0x48, 0xB8, 0x03, 0, 0, 0, 0, 0, 0, 0, // mov rax, 3
        0x48, 0x31, 0xF8, // xor rax, r15
        0xC3,
    ]);
    assert_eq!(c, Comparison::Match, "{ctx}");
}

#[test]
#[ignore = "documents the constant-folding flag loss (see README findings)"]
fn folded_add_drops_pf() {
    // mov rax, 3 (const); add rax, 3 (const) → yecta folds to rax=6 and
    // skips the flag computation; oracle sets PF=1 for 6 (110b).
    let (ctx, c) = verdict(vec![
        0x48, 0xB8, 0x03, 0, 0, 0, 0, 0, 0, 0, // mov rax, 3 (imm64)
        0x48, 0x83, 0xC0, 0x03, // add rax, 3 (imm8 — const-foldable)
        0xC3,
    ]);
    assert!(
        matches!(&c, Comparison::Divergence(d) if d.contains("pf")),
        "expected a PF divergence, got {c:?} ({ctx})"
    );
}
