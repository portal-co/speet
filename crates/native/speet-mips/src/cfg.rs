//! [`MipsCfgDecoder`] — MIPS32 implementation of [`CfgDecoder`].
//!
//! Decodes individual MIPS32 instructions using `rabbitizer` and returns
//! their static control-flow successors for use with
//! [`speet_reach::compute_reachable`].
//!
//! # Delay slots
//!
//! MIPS has a branch delay slot: the instruction immediately after a branch
//! is always executed before the branch takes effect.  For reachability
//! purposes we conservatively treat the delay-slot instruction as reachable
//! (it is always executed) by marking `fallthrough = true` for all branch
//! and jump instructions.  The instruction *two* words after (the first
//! instruction at the branch-not-taken path) is handled by ordinary
//! sequential reachability once the delay-slot instruction is reached.

use rabbitizer::{InstrCategory, InstrId, Instruction};
use speet_reach::{CfgDecoder, CfgEdges};

extern crate alloc;
use alloc::vec;

/// MIPS32 implementation of [`CfgDecoder`].
///
/// Zero-cost unit struct; create with `MipsCfgDecoder`.
pub struct MipsCfgDecoder;

/// MIPS instructions are always 4 bytes.
const MIPS_INSN_LEN: u32 = 4;

impl CfgDecoder for MipsCfgDecoder {
    fn decode_edges(&self, pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
        if bytes.len() < 4 {
            return None;
        }

        let word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let inst = Instruction::new(word, pc as u32, InstrCategory::CPU);

        let edges = match inst.unique_id {
            // ── Unconditional direct jump ──────────────────────────────────
            InstrId::cpu_j => {
                let target = mips_j_target(pc as u32, inst.get_instr_index());
                // Delay slot: the instruction at pc+4 is always executed first.
                // After the delay slot, execution moves to `target`.
                // We mark fallthrough = true to reach the delay slot, and add
                // `target` as a static successor.
                CfgEdges {
                    static_successors: vec![target as u64],
                    fallthrough: true, // delay slot at pc+4 is always executed
                    insn_len: MIPS_INSN_LEN,
                    has_indirect: false,
                }
            }

            // ── Direct call ────────────────────────────────────────────────
            InstrId::cpu_jal => {
                let target = mips_j_target(pc as u32, inst.get_instr_index());
                CfgEdges {
                    static_successors: vec![target as u64],
                    fallthrough: true, // delay slot + return site both reachable
                    insn_len: MIPS_INSN_LEN,
                    has_indirect: false,
                }
            }

            // ── Indirect jump (JR) ─────────────────────────────────────────
            InstrId::cpu_jr => CfgEdges {
                static_successors: vec![],
                fallthrough: true, // delay slot is always executed
                insn_len: MIPS_INSN_LEN,
                has_indirect: true,
            },

            // ── Indirect call (JALR) ───────────────────────────────────────
            InstrId::cpu_jalr => CfgEdges {
                static_successors: vec![],
                fallthrough: true, // delay slot + return site both reachable
                insn_len: MIPS_INSN_LEN,
                has_indirect: true,
            },

            // ── Conditional branches ───────────────────────────────────────
            InstrId::cpu_beq
            | InstrId::cpu_bne
            | InstrId::cpu_blez
            | InstrId::cpu_bgtz
            | InstrId::cpu_bltz
            | InstrId::cpu_bgez
            | InstrId::cpu_beql
            | InstrId::cpu_bnel
            | InstrId::cpu_bltzl
            | InstrId::cpu_bgezl
            | InstrId::cpu_blezl
            | InstrId::cpu_bgtzl => {
                let target = mips_branch_target(pc as u32, inst.get_immediate() as i16);
                CfgEdges {
                    static_successors: vec![target as u64],
                    fallthrough: true, // delay slot + not-taken path both reachable
                    insn_len: MIPS_INSN_LEN,
                    has_indirect: false,
                }
            }

            // ── All other instructions fall through ────────────────────────
            _ => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len: MIPS_INSN_LEN,
                has_indirect: false,
            },
        };

        Some(edges)
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Compute a MIPS J/JAL target from the instruction index field.
///
/// MIPS J-format: target = `(pc & 0xF000_0000) | (instr_index << 2)`.
fn mips_j_target(pc: u32, instr_index: u32) -> u32 {
    (pc & 0xF000_0000) | (instr_index << 2)
}

/// Compute a MIPS branch target from a signed 16-bit immediate.
///
/// MIPS branch offset = sign-extended 16-bit immediate × 4, added to `pc + 4`.
fn mips_branch_target(pc: u32, imm16: i16) -> u32 {
    let offset = (imm16 as i32) << 2;
    (pc as i32).wrapping_add(4).wrapping_add(offset) as u32
}
