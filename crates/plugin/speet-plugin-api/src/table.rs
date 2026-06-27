//! Table plugin — mirrors `yecta::IndirectJumpHandler`.

use alloc::vec::Vec;

use crate::error::PResult;
use crate::imports::HostImports;
use crate::snippet::CodeSnippet;
use crate::wire::{WireDecode, WireEncode, WireError};

/// Mirrors `yecta::IndirectJumpKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginIndirectJumpKind {
    /// Dispatch via a WASM table; the `u32` is the `TableIdx`.
    Table(u32),
    /// Dispatch via a `funcref` already on the stack.
    Ref,
}

impl WireEncode for PluginIndirectJumpKind {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            PluginIndirectJumpKind::Table(idx) => {
                0u8.encode(out);
                idx.encode(out);
            }
            PluginIndirectJumpKind::Ref => 1u8.encode(out),
        }
    }
}
impl WireDecode for PluginIndirectJumpKind {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        match tag {
            0 => {
                let (idx, rest) = u32::decode(rest)?;
                Ok((PluginIndirectJumpKind::Table(idx), rest))
            }
            1 => Ok((PluginIndirectJumpKind::Ref, rest)),
            _ => Err(WireError),
        }
    }
}

/// Method tags for the `TablePlugin` wire protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableMethod {
    IndirectJump = 0,
}

/// Indirect-call dispatch. Mirrors `yecta::IndirectJumpHandler`.
///
/// `&self`, not `&mut self` — see `memory` module docs for why (shared
/// `Arc<dyn Trait>` storage for host-entity imports).
pub trait TablePlugin: Send + Sync {
    /// `target_local` holds the runtime indirect-jump target value. Returns
    /// the dispatch kind plus a snippet computing whatever table/ref value
    /// that kind needs.
    fn indirect_jump(&self, target_local: u32) -> PResult<(PluginIndirectJumpKind, CodeSnippet)>;

    fn bind_imports(&self, _imports: &dyn HostImports) {}
}
