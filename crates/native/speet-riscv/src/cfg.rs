//! [`RiscVCfgDecoder`] — RISC-V implementation of [`CfgDecoder`].
//!
//! Decodes individual RISC-V instructions (both compressed C-extension and
//! standard 32-bit) and returns their static control-flow successors for use
//! with [`speet_reach::compute_reachable`].

use rv_asm::{Inst, IsCompressed, Xlen};
use speet_reach::{CfgDecoder, CfgEdges};

extern crate alloc;
use alloc::vec;

/// RISC-V implementation of [`CfgDecoder`].
///
/// Create one per translation session with the same [`Xlen`] you pass to the
/// [`RiscVRecompiler`](crate::RiscVRecompiler).
pub struct RiscVCfgDecoder {
    /// Whether to use RV64 (64-bit) or RV32 (32-bit) address arithmetic.
    pub xlen: Xlen,
}

impl CfgDecoder for RiscVCfgDecoder {
    fn decode_edges(&self, pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
        if bytes.len() < 2 {
            return None;
        }

        // Read a 32-bit word; for compressed instructions only 16 bits matter.
        let word = if bytes.len() >= 4 {
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            u16::from_le_bytes([bytes[0], bytes[1]]) as u32
        };

        let (inst, is_compressed) = Inst::decode(word, self.xlen).ok()?;
        let insn_len: u32 = match is_compressed {
            IsCompressed::Yes => 2,
            IsCompressed::No => 4,
        };

        let edges = match inst {
            // ── Unconditional direct jump and link ─────────────────────────
            Inst::Jal { offset, dest } => {
                let target = self.branch_target(pc, offset.as_i32());
                // dest == x0: pure jump (no link register written) → no fallthrough.
                // dest != x0: call (return address in dest) → fallthrough = return site.
                CfgEdges {
                    static_successors: vec![target],
                    fallthrough: dest.0 != 0,
                    insn_len,
                    has_indirect: false,
                }
            }

            // ── Indirect jump and link ─────────────────────────────────────
            Inst::Jalr { dest, .. } => {
                // Target is runtime-computed (base + offset).
                // dest != x0: stores return address → call, fallthrough = return site.
                // dest == x0: pure indirect jump or return → no fallthrough.
                CfgEdges {
                    static_successors: vec![],
                    fallthrough: dest.0 != 0,
                    insn_len,
                    has_indirect: true,
                }
            }

            // ── Conditional branches ───────────────────────────────────────
            Inst::Beq { offset, .. }
            | Inst::Bne { offset, .. }
            | Inst::Blt { offset, .. }
            | Inst::Bge { offset, .. }
            | Inst::Bltu { offset, .. }
            | Inst::Bgeu { offset, .. } => {
                let target = self.branch_target(pc, offset.as_i32());
                // Both the taken target and the fallthrough (not-taken) are reachable.
                CfgEdges {
                    static_successors: vec![target],
                    fallthrough: true,
                    insn_len,
                    has_indirect: false,
                }
            }

            // ── All other instructions fall through ────────────────────────
            _ => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },
        };

        Some(edges)
    }
}

impl RiscVCfgDecoder {
    /// Compute a PC-relative branch target, respecting the configured XLEN.
    fn branch_target(&self, pc: u64, offset: i32) -> u64 {
        match self.xlen {
            Xlen::Rv32 => (pc as i32).wrapping_add(offset) as u32 as u64,
            Xlen::Rv64 => (pc as i64).wrapping_add(offset as i64) as u64,
        }
    }
}
