//! [`X86CfgDecoder`] — x86-64 implementation of [`CfgDecoder`].
//!
//! Decodes individual x86-64 instructions using `iced-x86` and returns their
//! static control-flow successors for use with
//! [`speet_reach::compute_reachable`].

use iced_x86::{Decoder, DecoderOptions, FlowControl};
use speet_reach::{CfgDecoder, CfgEdges};

extern crate alloc;
use alloc::vec;

/// x86-64 implementation of [`CfgDecoder`].
///
/// Zero-cost unit struct; create with `X86CfgDecoder`.
pub struct X86CfgDecoder;

impl CfgDecoder for X86CfgDecoder {
    fn decode_edges(&self, pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
        if bytes.is_empty() {
            return None;
        }

        let mut dec = Decoder::with_ip(64, bytes, pc, DecoderOptions::NONE);
        if !dec.can_decode() {
            return None;
        }
        let inst = dec.decode();
        if inst.is_invalid() {
            return None;
        }

        let insn_len = inst.len() as u32;

        let edges = match inst.flow_control() {
            // Normal sequential instruction — fall through only.
            FlowControl::Next => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },

            // Unconditional direct jump — one static target, no fallthrough.
            FlowControl::UnconditionalBranch => {
                let target = inst.near_branch64();
                CfgEdges {
                    static_successors: vec![target],
                    fallthrough: false,
                    insn_len,
                    has_indirect: false,
                }
            }

            // Indirect jump (JMP r/m) — target unknown at analysis time.
            FlowControl::IndirectBranch => CfgEdges {
                static_successors: vec![],
                fallthrough: false,
                insn_len,
                has_indirect: true,
            },

            // Conditional branch — taken target + fallthrough.
            FlowControl::ConditionalBranch => {
                let target = inst.near_branch64();
                CfgEdges {
                    static_successors: vec![target],
                    fallthrough: true,
                    insn_len,
                    has_indirect: false,
                }
            }

            // Return — no static successors (return address on stack).
            FlowControl::Return => CfgEdges {
                static_successors: vec![],
                fallthrough: false,
                insn_len,
                has_indirect: true,
            },

            // Direct call — callee target + fallthrough (return site).
            FlowControl::Call => {
                let target = inst.near_branch64();
                CfgEdges {
                    static_successors: vec![target],
                    fallthrough: true,
                    insn_len,
                    has_indirect: false,
                }
            }

            // Indirect call (CALL r/m) — target unknown, but return site reachable.
            FlowControl::IndirectCall => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: true,
            },

            // Interrupt / syscall (INT, SYSCALL, SYSENTER) — treat as a call:
            // execution returns to the next instruction after the syscall handler.
            FlowControl::Interrupt => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },

            // XBEGIN/XABORT/XEND — transactional memory; conservatively fall through.
            FlowControl::XbeginXabortXend => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },

            // Exception-generating instructions (UD0, UD1, UD2, HLT, …) —
            // treat as sinks: no fallthrough, no successors.
            FlowControl::Exception => CfgEdges {
                static_successors: vec![],
                fallthrough: false,
                insn_len,
                has_indirect: false,
            },
        };

        Some(edges)
    }
}
