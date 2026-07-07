//! Arch plugin — mirrors `speet_link_core::Recompile` via a command-stream
//! protocol, since translation is a *stateful decode loop* against
//! `ReactorContext`, not a single request/response.
//!
//! Scoped down for v1: covers the common open/feed/jump/indirect-jump/
//! seal/request-more/done shape. Does not expose the full `ji`/`EscapeTag`/
//! speculative-call machinery, lazy-store lookahead, or OOB lookup-stub
//! dispatch — those stay host-side adapter defaults a plugin can't directly
//! invoke in v1. See `docs/guides/plugin-api.md`.

use alloc::vec::Vec;

use crate::error::PResult;
use crate::imports::HostImports;
use crate::snippet::{CodeSnippet, PluginValType};
use crate::wire::{WireDecode, WireEncode, WireError};

/// One step of the decode loop. Mirrors the small set of `ReactorContext`
/// methods a native arch recompiler calls during translation.
#[derive(Debug, Clone, PartialEq)]
pub enum ArchOp {
    /// Open a new function; `len` mirrors `ReactorContext::next_with`'s
    /// control-flow-depth parameter.
    OpenFn { len: u32 },
    /// Append a snippet into the function opened by the most recent `OpenFn`.
    Feed { snippet: CodeSnippet },
    /// Unconditional tail-call jump. Mirrors `ReactorContext::jmp`.
    Jump { target_pc: u64, params: u32 },
    /// Indirect jump/call (common case only — see module docs). `snippet`
    /// computes any pre-dispatch address; `params` forwards to the target.
    IndirectJump { snippet: CodeSnippet, params: u32 },
    /// Close the current function with a terminal snippet. Mirrors
    /// `ReactorContext::seal_fn`.
    Seal { snippet: CodeSnippet },
    /// Ask the host for more guest bytes at `guest_addr` (decode-window
    /// refill, or a newly discovered reachable address).
    RequestBytes { guest_addr: u64, max_len: u32 },
    /// Decode loop finished for this translation unit.
    Done,
}

impl WireEncode for ArchOp {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            ArchOp::OpenFn { len } => {
                0u8.encode(out);
                len.encode(out);
            }
            ArchOp::Feed { snippet } => {
                1u8.encode(out);
                snippet.encode(out);
            }
            ArchOp::Jump { target_pc, params } => {
                2u8.encode(out);
                target_pc.encode(out);
                params.encode(out);
            }
            ArchOp::IndirectJump { snippet, params } => {
                3u8.encode(out);
                snippet.encode(out);
                params.encode(out);
            }
            ArchOp::Seal { snippet } => {
                4u8.encode(out);
                snippet.encode(out);
            }
            ArchOp::RequestBytes {
                guest_addr,
                max_len,
            } => {
                5u8.encode(out);
                guest_addr.encode(out);
                max_len.encode(out);
            }
            ArchOp::Done => 6u8.encode(out),
        }
    }
}
impl WireDecode for ArchOp {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (len, rest) = u32::decode(rest)?;
                (ArchOp::OpenFn { len }, rest)
            }
            1 => {
                let (snippet, rest) = CodeSnippet::decode(rest)?;
                (ArchOp::Feed { snippet }, rest)
            }
            2 => {
                let (target_pc, rest) = u64::decode(rest)?;
                let (params, rest) = u32::decode(rest)?;
                (ArchOp::Jump { target_pc, params }, rest)
            }
            3 => {
                let (snippet, rest) = CodeSnippet::decode(rest)?;
                let (params, rest) = u32::decode(rest)?;
                (ArchOp::IndirectJump { snippet, params }, rest)
            }
            4 => {
                let (snippet, rest) = CodeSnippet::decode(rest)?;
                (ArchOp::Seal { snippet }, rest)
            }
            5 => {
                let (guest_addr, rest) = u64::decode(rest)?;
                let (max_len, rest) = u32::decode(rest)?;
                (
                    ArchOp::RequestBytes {
                        guest_addr,
                        max_len,
                    },
                    rest,
                )
            }
            6 => (ArchOp::Done, rest),
            _ => return Err(WireError),
        })
    }
}

/// Method tags for the `ArchPlugin` wire protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchMethod {
    ResetForNextBinary = 0,
    CountFns = 1,
    Step = 2,
    DeclareParams = 3,
}

/// A whole CPU/bytecode architecture's recompile environment. Mirrors
/// `speet_link_core::Recompile<Context, E, F>`.
///
/// `&self` throughout, even though the decode loop is inherently stateful
/// (current PC, open-function bookkeeping) — see `memory` module docs for
/// why every plugin trait here is `&self`. Implementations hold their decode
/// state behind a `Mutex`/`RwLock` (never bare `Cell`/`RefCell`, since
/// `ArchPlugin: Send + Sync`).
pub trait ArchPlugin: Send + Sync {
    /// Per-binary reset. `args` is plugin-defined bytes — the plugin author
    /// documents their own encoding (mirrors `Recompile::BinaryArgs`, which
    /// is per-implementation in the internal trait too).
    fn reset_for_next_binary(&self, args: &[u8]);

    /// Lightweight pre-pass: count the WASM functions this binary will
    /// produce. Mirrors `Recompile::count_fns`.
    fn count_fns(&self, bytes: &[u8]) -> u32;

    /// Drive one step of the decode loop. `feedback` is `None` on the first
    /// call and `Some(bytes)` for the guest bytes satisfying the most recent
    /// `RequestBytes`.
    fn step(&self, feedback: Option<&[u8]>) -> PResult<ArchOp>;

    /// Register-file layout declaration (the param half only — trap
    /// installation stays host-side).
    fn declare_params(&self) -> Vec<PluginValType>;

    fn bind_imports(&self, _imports: &dyn HostImports) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_op_roundtrip() {
        let ops = [
            ArchOp::OpenFn { len: 2 },
            ArchOp::Feed {
                snippet: CodeSnippet::from_instructions(&[wasm_encoder::Instruction::Nop]),
            },
            ArchOp::Jump {
                target_pc: 0x1008,
                params: 17,
            },
            ArchOp::IndirectJump {
                snippet: CodeSnippet::empty(),
                params: 3,
            },
            ArchOp::Seal {
                snippet: CodeSnippet::empty(),
            },
            ArchOp::RequestBytes {
                guest_addr: 0x2000,
                max_len: 16,
            },
            ArchOp::Done,
        ];
        for op in ops {
            let mut out = Vec::new();
            op.encode(&mut out);
            let (decoded, rest) = ArchOp::decode(&out).unwrap();
            assert_eq!(decoded, op);
            assert!(rest.is_empty());
        }
    }
}
