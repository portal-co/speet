//! Branch-target fixup for minimized fixed-width cases.
//!
//! Removing a word shifts every later instruction, so pre-patched branch
//! offsets go stale and their targets fall outside the code — the
//! recompiler then emits a call to an out-of-table function index and the
//! module fails validation. That failure is a *mechanical* artifact of
//! minimization, not a finding. [`fixup_branches`] rescans the branch
//! encodings (site lists are not available post-generation), recomputes
//! each target, and repoints only out-of-range branches at the final word
//! (the arch terminator = clean exit), leaving in-range branches untouched
//! so surviving semantics drive minimization.
//!
//! Any module that still fails validation after this fixup is a genuine
//! emission bug: the minimizer keeps such trials as divergences and the
//! driver writes an artifact for them.

use crate::case::Arch;

/// Fix out-of-range branch targets in `code` (word-aligned archs only;
/// x86 is byte-granular and position-independent — no-op there).
pub fn fixup_branches(arch: Arch, code: &mut [u8]) {
    if arch.code_align() != 4 {
        return;
    }
    let big_endian = arch == Arch::Mips;
    let code_len = code.len() as i64;
    let n_words = code.len() / 4;
    let last = code_len - 4; // terminator word = clean-exit target

    let read = |code: &[u8], i: usize| -> u32 {
        let b: [u8; 4] = code[i * 4..i * 4 + 4].try_into().unwrap();
        if big_endian {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        }
    };
    let write = |code: &mut [u8], i: usize, w: u32| {
        let b = if big_endian {
            w.to_be_bytes()
        } else {
            w.to_le_bytes()
        };
        code[i * 4..i * 4 + 4].copy_from_slice(&b);
    };

    for i in 0..n_words {
        let w = read(code, i);
        let pc = (i * 4) as i64;

        // (imm: signed displacement in words from the branch's reference
        // point; reencode: rebuild the word with a new displacement)
        let (target, reencode): (i64, Box<dyn Fn(i64) -> u32>) = match arch {
            Arch::AArch64 => {
                let op = w >> 24;
                if op == 0b0001_0100 {
                    // B: imm26 words from pc.
                    let imm = sext(w & 0x03FF_FFFF, 26);
                    (
                        pc + imm * 4,
                        Box::new(move |new_off: i64| {
                            (w & 0xFC00_0000) | ((new_off as u32) & 0x03FF_FFFF)
                        }),
                    )
                } else if op == 0b0101_0100
                    || matches!(op, 0b0011_0100 | 0b0011_0101 | 0b1011_0100 | 0b1011_0101)
                {
                    // B.cond (0x54) / CBZ / CBNZ (0x34/0x35/0xB4/0xB5):
                    // imm19 words from pc, at bits 23..5.
                    let imm = sext((w >> 5) & 0x7_FFFF, 19);
                    (
                        pc + imm * 4,
                        Box::new(move |new_off: i64| {
                            (w & 0xFFE0_001F) | (((new_off as u32) & 0x7_FFFF) << 5)
                        }),
                    )
                } else {
                    continue;
                }
            }
            Arch::Arm => {
                // B/BL: cond 0xE, op field bits 27..25 == 0b101; imm24
                // words from pc+8.
                if (w >> 28) != 0xE || ((w >> 25) & 0x7) != 0b101 {
                    continue;
                }
                let imm = sext(w & 0x00FF_FFFF, 24);
                (
                    pc + 8 + imm * 4,
                    Box::new(move |new_off: i64| {
                        (w & 0xFF00_0000) | ((new_off as u32) & 0x00FF_FFFF)
                    }),
                )
            }
            Arch::RiscV64 | Arch::RiscV32 => {
                // B-type (BEQ/BNE/...): opcode 0b1100011; 13-bit signed
                // byte offset from pc, split across imm[12|10:5] and
                // imm[4:1|11].
                if (w & 0x7F) != 0b110_0011 {
                    continue;
                }
                let imm = b_type_imm(w);
                (
                    pc + imm,
                    Box::new(move |new_target: i64| encode_b_type(w, (new_target - pc) as i32)),
                )
            }
            Arch::Mips => {
                // BEQ(4)/BNE(5): imm16 words from pc+4 (delay slot).
                let op = w >> 26;
                if op != 4 && op != 5 {
                    continue;
                }
                let imm = sext(w & 0xFFFF, 16);
                (
                    pc + 4 + imm * 4,
                    Box::new(move |new_off: i64| (w & 0xFFFF_0000) | ((new_off as u32) & 0xFFFF)),
                )
            }
            Arch::X86_64 | Arch::X86_32 => continue,
        };

        if target < 0 || target >= code_len {
            // Repoint at the terminator word (clean exit).
            let new_off = (last - reference(arch, pc)) / 4;
            let patched = match arch {
                Arch::RiscV64 | Arch::RiscV32 => reencode(last),
                _ => reencode(new_off),
            };
            write(code, i, patched);
        }
    }
}

/// Branch displacement reference point: RV offsets are byte displacements
/// from pc itself; the others are word displacements from their reference.
fn reference(arch: Arch, pc: i64) -> i64 {
    match arch {
        Arch::AArch64 => pc,
        Arch::Arm => pc + 8,
        Arch::Mips => pc + 4,
        Arch::RiscV64 | Arch::RiscV32 => pc,
        Arch::X86_64 | Arch::X86_32 => pc,
    }
}

fn sext(v: u32, bits: u32) -> i64 {
    let shift = 32 - bits;
    ((v << shift) as i32 >> shift) as i64
}

/// Decode a B-type immediate (13-bit signed byte offset).
fn b_type_imm(w: u32) -> i64 {
    let imm12 = ((w >> 31) & 0x1) << 12
        | ((w >> 25) & 0x3F) << 5
        | ((w >> 8) & 0xF) << 1
        | ((w >> 7) & 0x1) << 11;
    sext(imm12, 13)
}

/// Re-encode a B-type word with a new byte displacement.
fn encode_b_type(w: u32, disp: i32) -> u32 {
    let d = disp as u32;
    let imm = ((d >> 12) & 0x1) << 31
        | ((d >> 5) & 0x3F) << 25
        | ((d >> 1) & 0xF) << 8
        | ((d >> 11) & 0x1) << 7;
    (w & 0x01F0_007F) | imm
}
