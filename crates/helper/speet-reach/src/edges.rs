//! [`CfgDecoder`] — architecture-agnostic CFG edge extraction.
//!
//! Each architecture frontend implements [`CfgDecoder`] to expose the static
//! control-flow successors of a single instruction without performing full
//! WASM translation.  [`compute_reachable`](crate::compute_reachable) uses
//! these edges to expand a seed set of PCs transitively.

use alloc::vec::Vec;

// ── CfgEdges ─────────────────────────────────────────────────────────────────

/// Static control-flow successors of one decoded instruction.
#[derive(Debug, Clone)]
pub struct CfgEdges {
    /// Statically-known jump/branch/call targets.
    ///
    /// For conditional branches: the *taken* target only; the not-taken path
    /// is the fallthrough (see [`fallthrough`](Self::fallthrough)).
    ///
    /// For calls: the callee entry point; the return site is the fallthrough.
    pub static_successors: Vec<u64>,

    /// Whether execution may continue to `pc + insn_len` without jumping.
    ///
    /// `true` for: normal (non-branching) instructions, conditional branches
    /// (not-taken path), and call instructions (return site).
    ///
    /// `false` for: unconditional jumps and return instructions.
    pub fallthrough: bool,

    /// Byte length of this instruction.
    ///
    /// Used by [`compute_reachable`](crate::compute_reachable) to compute the
    /// fallthrough PC as `pc + insn_len`.
    pub insn_len: u32,

    /// Whether this instruction has a runtime-computed (indirect) successor.
    ///
    /// When `true`, the full set of successors is not statically knowable.
    /// External tools must add indirect targets to
    /// [`ReachabilitySpec::seeds`](crate::ReachabilitySpec::seeds) manually.
    pub has_indirect: bool,
}

// ── CfgDecoder trait ─────────────────────────────────────────────────────────

/// Extracts static control-flow edges from a single instruction.
///
/// Implement this trait for each architecture frontend so that
/// [`compute_reachable`](crate::compute_reachable) can perform a BFS over the
/// binary's CFG without invoking the full WASM-emitting translation pipeline.
///
/// # Contract
///
/// - `bytes` contains the raw machine code starting at `pc`.
/// - If the bytes are too short or cannot be decoded, return `None`.
/// - The implementation must not modify any state — this is a pure decode.
/// - `insn_len` in the returned [`CfgEdges`] must be ≥ 1.
pub trait CfgDecoder {
    /// Decode the instruction at `pc` from `bytes` and return its edges.
    ///
    /// Returns `None` if decoding fails (invalid encoding, truncated bytes,
    /// or PC outside the binary).
    fn decode_edges(&self, pc: u64, bytes: &[u8]) -> Option<CfgEdges>;
}
