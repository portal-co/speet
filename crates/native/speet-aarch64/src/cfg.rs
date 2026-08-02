//! [`AArch64CfgDecoder`] — AArch64 implementation of [`CfgDecoder`].
//!
//! Fixed 4-byte instruction width. Decodes static branch/call targets for use
//! with [`speet_reach::compute_reachable`] / [`speet_reach::PcSlotMap`].

use disarm64::decoder_full::{
    BRANCH_IMM, BRANCH_REG, COMPBRANCH, CONDBRANCH, Operation,
};
use speet_reach::{CfgDecoder, CfgEdges};

extern crate alloc;
use alloc::vec;

/// AArch64 implementation of [`CfgDecoder`].
pub struct AArch64CfgDecoder;

#[inline(always)]
fn imm26(w: u32) -> u32 {
    w & 0x3FF_FFFF
}

#[inline(always)]
fn imm19(w: u32) -> u32 {
    (w >> 5) & 0x7_FFFF
}

#[inline(always)]
fn sign_ext_n(v: u32, n: u8) -> i64 {
    let shift = 64 - n as i64;
    ((v as i64) << shift) >> shift
}

#[inline(always)]
fn sign_ext26(v: u32) -> i64 {
    sign_ext_n(v & 0x3FF_FFFF, 26)
}

#[inline(always)]
fn sign_ext19(v: u32) -> i64 {
    sign_ext_n(v & 0x7_FFFF, 19)
}

impl CfgDecoder for AArch64CfgDecoder {
    fn decode_edges(&self, pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
        if bytes.len() < 4 {
            return None;
        }
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let insn_len = 4u32;

        let Some(opcode) = disarm64::decoder::decode(word) else {
            return Some(CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            });
        };

        let edges = match opcode.operation {
            Operation::BRANCH_IMM(ref inner) => {
                let (w, is_bl) = match inner {
                    BRANCH_IMM::B_ADDR_PCREL26(x) => (x.0, false),
                    BRANCH_IMM::BL_ADDR_PCREL26(x) => (x.0, true),
                    _ => {
                        return Some(CfgEdges {
                            static_successors: vec![],
                            fallthrough: true,
                            insn_len,
                            has_indirect: false,
                        });
                    }
                };
                let target = pc.wrapping_add_signed(sign_ext26(imm26(w)) * 4);
                CfgEdges {
                    static_successors: vec![target],
                    fallthrough: is_bl,
                    insn_len,
                    has_indirect: false,
                }
            }
            Operation::CONDBRANCH(ref inner) => {
                let w = match inner {
                    CONDBRANCH::B__ADDR_PCREL19(x) => x.0,
                    CONDBRANCH::BC__ADDR_PCREL19(x) => x.0,
                    _ => {
                        return Some(CfgEdges {
                            static_successors: vec![],
                            fallthrough: true,
                            insn_len,
                            has_indirect: false,
                        });
                    }
                };
                let target = pc.wrapping_add_signed(sign_ext19(imm19(w)) * 4);
                CfgEdges {
                    static_successors: vec![target],
                    fallthrough: true,
                    insn_len,
                    has_indirect: false,
                }
            }
            Operation::COMPBRANCH(ref inner) => {
                let w = match inner {
                    COMPBRANCH::CBZ_Rt_ADDR_PCREL19(x) => x.0,
                    COMPBRANCH::CBNZ_Rt_ADDR_PCREL19(x) => x.0,
                    _ => {
                        return Some(CfgEdges {
                            static_successors: vec![],
                            fallthrough: true,
                            insn_len,
                            has_indirect: false,
                        });
                    }
                };
                let target = pc.wrapping_add_signed(sign_ext19(imm19(w)) * 4);
                CfgEdges {
                    static_successors: vec![target],
                    fallthrough: true,
                    insn_len,
                    has_indirect: false,
                }
            }
            Operation::BRANCH_REG(ref inner) => {
                let is_call = matches!(inner, BRANCH_REG::BLR_Rn(_));
                CfgEdges {
                    static_successors: vec![],
                    fallthrough: is_call,
                    insn_len,
                    has_indirect: true,
                }
            }
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
