//! Divergence minimization (plan §6.2): instruction-level delta debugging.
//!
//! Decodes the case's instruction list, then repeatedly tries removing
//! single instructions, keeping each removal only when the divergence
//! persists. Removal changes byte offsets, so with the byte-granular slot
//! layout the shingle neighborhood shifts — a removal that loses the
//! divergence is simply rejected (the minimizer never *invents* cases).
//! The final `ret` and any instruction whose removal yields a no-op case
//! are preserved.

use crate::case::FuzzCase;
use crate::compare::{compare_outcomes, Comparison};
use crate::recompiled::{run_recompiled, RecompiledError};
use iced_x86::{Decoder, DecoderOptions};

/// Split `code` (excluding the trailing `ret`) into decoded instruction
/// byte-ranges via a sequential walk.
fn split_insns(code: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut off = 0;
    while off < code.len() {
        let mut dec = Decoder::with_ip(64, &code[off..], 0, DecoderOptions::NONE);
        let inst = dec.decode();
        if inst.is_invalid() {
            break;
        }
        let len = inst.len() as usize;
        out.push((off, off + len));
        off += len;
    }
    out
}

fn verdict_persists(case: &FuzzCase, first: &Comparison) -> bool {
    let oracle = crate::oracle::run_oracle(case);
    let recompiled = run_recompiled(case);
    matches!(
        compare_outcomes(case, oracle.as_ref().ok(), recompiled.as_ref()),
        Comparison::Divergence(_)
    )
}

/// Minimize `case` (which must currently diverge). Returns a case with as
/// many instructions removed as possible while the divergence persists.
pub fn minimize(case: &FuzzCase) -> FuzzCase {
    debug_assert!(
        verdict_persists(case, &Comparison::Divergence(String::new())),
        "minimize() called on a non-diverging case"
    );
    let body_len = case.code.len().saturating_sub(1); // strip trailing ret
    let insns = split_insns(&case.code[..body_len]);
    let mut current: Vec<(usize, usize)> = insns;

    loop {
        let mut removed = false;
        for i in 0..current.len() {
            let mut trial_ranges: Vec<(usize, usize)> = Vec::with_capacity(current.len() - 1);
            trial_ranges.extend_from_slice(&current[..i]);
            trial_ranges.extend_from_slice(&current[i + 1..]);
            let code = rebuild(&case.code, &trial_ranges);
            if code.len() <= 1 {
                continue; // keep at least the ret
            }
            let trial = with_code(case, code);
            if verdict_persists(&trial, &Comparison::Divergence(String::new())) {
                current = trial_ranges;
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }

    with_code(case, rebuild(&case.code, &current))
}

/// Re-encode: concatenate the surviving instruction ranges + trailing ret.
fn rebuild(code: &[u8], ranges: &[(usize, usize)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (s, e) in ranges {
        out.extend_from_slice(&code[*s..*e]);
    }
    out.push(0xC3);
    out
}

/// Clone the case with replaced code, re-deriving the register seeds that
/// depend on the code length (RBX data anchor unchanged; the sentinel is
/// embedded in the stack image at [SP] and must be rewritten).
fn with_code(case: &FuzzCase, code: Vec<u8>) -> FuzzCase {
    let mut c = case.clone();
    c.code = code;
    // Re-seed the halt sentinel at [SP] (the stack image is copied as-is,
    // but the sentinel value depends on the code length).
    let sp = c.regs.gprs[4];
    let off = (sp - c.stack_base) as usize;
    let sentinel = c.halt_sentinel().to_le_bytes();
    c.stack[off..off + 8].copy_from_slice(&sentinel);
    c
}
