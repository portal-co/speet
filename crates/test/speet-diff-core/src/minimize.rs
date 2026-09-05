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

/// Split `code` (excluding the trailing terminator) into instruction
/// byte-ranges: fixed-width words for aarch64/riscv64, a sequential iced
/// walk for x86-64.
fn split_insns(arch: crate::case::Arch, code: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    match arch {
        crate::case::Arch::AArch64 | crate::case::Arch::RiscV64 | crate::case::Arch::Arm
        | crate::case::Arch::RiscV32 | crate::case::Arch::Mips => {
            for off in (0..code.len()).step_by(4) {
                out.push((off, off + 4));
            }
        }
        crate::case::Arch::X86_64 | crate::case::Arch::X86_32 => {
            let mut off = 0;
            while off < code.len() {
                let bits = if arch == crate::case::Arch::X86_32 { 32 } else { 64 };
                let mut dec = Decoder::with_ip(bits, &code[off..], 0, DecoderOptions::NONE);
                let inst = dec.decode();
                if inst.is_invalid() {
                    break;
                }
                let len = inst.len() as usize;
                out.push((off, off + len));
                off += len;
            }
        }
    }
    out
}

fn verdict_persists(case: &FuzzCase, first: &Comparison) -> bool {
    let oracle = crate::oracle::run_oracle(case);
    let recompiled = run_recompiled(case);
    // NOTE: harness-internal errors (validate failures etc.) persist as
    // divergences on purpose — a broken module is a finding, not a
    // non-reproducing trial. To keep them meaningful the minimizer re-runs
    // the arch branch fixup on every rebuilt trial (`fixup_branches`), so
    // stale-offset validate failures can't masquerade as findings here.
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
    // Strip the arch terminator: x86 `ret` byte; aarch64 RET / rv64 JALR
    // are 4-byte words.
    let term = case.arch.code_align() as usize;
    let body_len = case.code.len().saturating_sub(term);
    let insns = split_insns(case.arch, &case.code[..body_len]);
    let mut current: Vec<(usize, usize)> = insns;

    loop {
        let mut removed = false;
        for i in 0..current.len() {
            let mut trial_ranges: Vec<(usize, usize)> = Vec::with_capacity(current.len() - 1);
            trial_ranges.extend_from_slice(&current[..i]);
            trial_ranges.extend_from_slice(&current[i + 1..]);
            let mut code = rebuild(&case.code, &trial_ranges, term);
            if code.len() <= 1 {
                continue; // keep at least the ret
            }
            crate::branchfix::fixup_branches(case.arch, &mut code);
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

    let mut code = rebuild(&case.code, &current, term);
    crate::branchfix::fixup_branches(case.arch, &mut code);
    with_code(case, code)
}

/// Re-encode: concatenate the surviving instruction ranges + trailing ret.
fn rebuild(code: &[u8], ranges: &[(usize, usize)], term: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for (s, e) in ranges {
        out.extend_from_slice(&code[*s..*e]);
    }
    // Re-append the arch terminator (the final term bytes of the original).
    out.extend_from_slice(&code[code.len() - term..]);
    out
}

/// Clone the case with replaced code, re-deriving the register seeds that
/// depend on the code length (RBX data anchor unchanged; the sentinel is
/// embedded in the stack image at [SP] and must be rewritten).
/// Public wrapper for tests / tooling: replace the code and re-derive the
/// length-dependent register seeds.
pub fn with_code_public(case: &FuzzCase, code: Vec<u8>) -> FuzzCase {
    with_code(case, code)
}

fn with_code(case: &FuzzCase, code: Vec<u8>) -> FuzzCase {
    let mut c = case.clone();
    c.code = code;
    // Re-seed the halt sentinel (the stack image is copied as-is, but the
    // sentinel value depends on the code length). Stack-based-ret archs
    // store it at [SP]; link-register archs carry it in the LR register.
    let sentinel = c.halt_sentinel().to_le_bytes();
    match c.arch {
        crate::case::Arch::X86_64 => {
            let sp = c.regs.gprs[4];
            let off = (sp - c.stack_base) as usize;
            c.stack[off..off + 8].copy_from_slice(&sentinel);
        }
        crate::case::Arch::AArch64 => {
            c.regs.gprs[30] = c.halt_sentinel();
        }
        crate::case::Arch::RiscV64 | crate::case::Arch::RiscV32 => {
            c.regs.gprs[1] = c.halt_sentinel();
        }
        crate::case::Arch::Arm => {
            c.regs.gprs[14] = c.halt_sentinel(); // LR
        }
        crate::case::Arch::Mips => {
            c.regs.gprs[31] = c.halt_sentinel(); // $ra
        }
        crate::case::Arch::X86_32 => {
            let sentinel = c.halt_sentinel().to_le_bytes();
            let sp = c.regs.gprs[4];
            let off = (sp - c.stack_base) as usize;
            c.stack[off..off + 4].copy_from_slice(&sentinel[..4]);
        }
    }
    c
}
