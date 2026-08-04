//! [`X86_32CfgDecoder`] — i686 implementation of [`CfgDecoder`].

use iced_x86::{Decoder, DecoderOptions, FlowControl};
use speet_reach::{CfgDecoder, CfgEdges};

extern crate alloc;
use alloc::vec;

/// i686 CFG decoder (iced-x86 bitness 32).
pub struct X86_32CfgDecoder;

impl CfgDecoder for X86_32CfgDecoder {
    fn decode_edges(&self, pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
        if bytes.is_empty() {
            return None;
        }
        let mut dec = Decoder::with_ip(32, bytes, pc, DecoderOptions::NONE);
        if !dec.can_decode() {
            return None;
        }
        let inst = dec.decode();
        if inst.is_invalid() {
            return None;
        }
        let insn_len = inst.len() as u32;
        let edges = match inst.flow_control() {
            FlowControl::Next => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },
            FlowControl::UnconditionalBranch => CfgEdges {
                static_successors: vec![inst.near_branch32() as u64],
                fallthrough: false,
                insn_len,
                has_indirect: false,
            },
            FlowControl::IndirectBranch => CfgEdges {
                static_successors: vec![],
                fallthrough: false,
                insn_len,
                has_indirect: true,
            },
            FlowControl::ConditionalBranch => CfgEdges {
                static_successors: vec![inst.near_branch32() as u64],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },
            FlowControl::Return => CfgEdges {
                static_successors: vec![],
                fallthrough: false,
                insn_len,
                has_indirect: true,
            },
            FlowControl::Call => CfgEdges {
                static_successors: vec![inst.near_branch32() as u64],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },
            FlowControl::IndirectCall => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: true,
            },
            FlowControl::Interrupt => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },
            FlowControl::XbeginXabortXend => CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len,
                has_indirect: false,
            },
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
