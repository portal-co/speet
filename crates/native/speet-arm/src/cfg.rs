//! [`ArmCfgDecoder`] — AArch32 (A32) implementation of [`CfgDecoder`].
//!
//! Fixed 4-byte instruction width for A32. Thumb-2 is not modeled here;
//! see the crate root docs.

use speet_reach::{CfgDecoder, CfgEdges};

extern crate alloc;
use alloc::vec;

/// AArch32 (ARM state) CFG decoder — 4-byte slots.
pub struct ArmCfgDecoder;

#[inline]
fn sign_ext24(imm24: u32) -> i64 {
    let shift = 32 - 24;
    ((imm24 << shift) as i32 >> shift) as i64
}

impl CfgDecoder for ArmCfgDecoder {
    fn decode_edges(&self, pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
        if bytes.len() < 4 {
            return None;
        }
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let insn_len = 4u32;

        // B / BL: cond | 101 | L | imm24
        if (word >> 25) & 0x7 == 0b101 {
            let is_bl = (word >> 24) & 1 != 0;
            let imm24 = word & 0x00FF_FFFF;
            let target = pc.wrapping_add_signed(sign_ext24(imm24) * 4 + 8);
            return Some(CfgEdges {
                static_successors: vec![target],
                fallthrough: is_bl,
                insn_len,
                has_indirect: false,
            });
        }

        // BX / BLX (register): … 0001 0010 1111 1111 1111 00L1 Rm
        if (word & 0x0FFF_FFF0) == 0x012F_FF10 || (word & 0x0FFF_FFF0) == 0x012F_FF30 {
            let is_blx = (word & 0x0FFF_FFF0) == 0x012F_FF30;
            return Some(CfgEdges {
                static_successors: vec![],
                fallthrough: is_blx,
                insn_len,
                has_indirect: true,
            });
        }

        Some(CfgEdges {
            static_successors: vec![],
            fallthrough: true,
            insn_len,
            has_indirect: false,
        })
    }
}
