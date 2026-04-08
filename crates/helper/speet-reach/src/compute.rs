//! [`ReachableSet`] and [`compute_reachable`] — transitive reachability over
//! a binary's control-flow graph.

use alloc::collections::{BTreeSet, VecDeque};

use crate::{CfgDecoder, ReachabilitySpec};

// ── ReachableSet ─────────────────────────────────────────────────────────────

/// All transitively reachable instruction PCs, including the original seeds.
///
/// Produced by [`compute_reachable`].  Pass to
/// [`ReachabilityFilter::new`](crate::ReachabilityFilter::new) to install the
/// filter as an [`InstructionTrap`](speet_traps::InstructionTrap).
#[derive(Debug, Clone, Default)]
pub struct ReachableSet {
    /// The set of reachable PCs.
    pub pcs: BTreeSet<u64>,
}

impl ReachableSet {
    /// Returns `true` if `pc` is in the reachable set.
    #[inline]
    pub fn contains(&self, pc: u64) -> bool {
        self.pcs.contains(&pc)
    }

    /// Number of reachable instructions.
    #[inline]
    pub fn len(&self) -> usize {
        self.pcs.len()
    }

    /// Returns `true` if no PCs are reachable.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pcs.is_empty()
    }
}

// ── compute_reachable ─────────────────────────────────────────────────────────

/// Compute the set of transitively reachable instruction PCs by BFS over the
/// binary's control-flow graph.
///
/// Starting from the seeds in `spec`, the algorithm follows:
/// - **Static successors** — statically-known jump/branch/call targets
///   reported in [`CfgEdges::static_successors`](crate::CfgEdges::static_successors).
/// - **Fallthroughs** — the next sequential instruction (`pc + insn_len`)
///   when [`CfgEdges::fallthrough`](crate::CfgEdges::fallthrough) is `true`.
///
/// Edges that point outside `[base_addr, base_addr + bytes.len())` are
/// silently skipped.  Indirect targets (runtime-computed) are not followed;
/// the external tool must include them in `spec.seeds` if known.
///
/// # Arguments
///
/// * `spec`      — seed PCs from the external tool.
/// * `bytes`     — raw binary bytes to analyze.
/// * `base_addr` — virtual address of `bytes[0]`.
/// * `decoder`   — architecture-specific [`CfgDecoder`].
pub fn compute_reachable(
    spec: &ReachabilitySpec,
    bytes: &[u8],
    base_addr: u64,
    decoder: &dyn CfgDecoder,
) -> ReachableSet {
    let end_addr = base_addr.saturating_add(bytes.len() as u64);
    let in_range = |pc: u64| pc >= base_addr && pc < end_addr;

    let mut reachable = BTreeSet::new();
    let mut worklist: VecDeque<u64> = spec.seeds.iter().copied().filter(|&pc| in_range(pc)).collect();

    while let Some(pc) = worklist.pop_front() {
        if !reachable.insert(pc) {
            continue; // already visited
        }

        let offset = (pc - base_addr) as usize;
        let Some(edges) = decoder.decode_edges(pc, &bytes[offset..]) else {
            continue; // decode failure — treat as sink
        };

        // Follow static successors within the binary.
        for &target in &edges.static_successors {
            if in_range(target) && !reachable.contains(&target) {
                worklist.push_back(target);
            }
        }

        // Follow fallthrough.
        if edges.fallthrough {
            let next_pc = pc.wrapping_add(edges.insn_len as u64);
            if in_range(next_pc) && !reachable.contains(&next_pc) {
                worklist.push_back(next_pc);
            }
        }
    }

    ReachableSet { pcs: reachable }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CfgEdges, ReachabilitySpec};
    use alloc::vec;

    /// A minimal synthetic decoder: treat every 4-byte word as one instruction
    /// that falls through, except the word `0xFFFFFFFF` which is a sink (no
    /// fallthrough, no successors).
    struct SyntheticDecoder;

    impl CfgDecoder for SyntheticDecoder {
        fn decode_edges(&self, _pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
            if bytes.len() < 4 {
                return None;
            }
            let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if word == 0xFFFFFFFF {
                // Sink instruction (e.g. unreachable / infinite loop body).
                Some(CfgEdges {
                    static_successors: vec![],
                    fallthrough: false,
                    insn_len: 4,
                    has_indirect: false,
                })
            } else {
                // Normal instruction: fall through.
                Some(CfgEdges {
                    static_successors: vec![],
                    fallthrough: true,
                    insn_len: 4,
                    has_indirect: false,
                })
            }
        }
    }

    /// A decoder that emits an unconditional jump from address 0x1000 to 0x2000.
    struct JumpDecoder;

    impl CfgDecoder for JumpDecoder {
        fn decode_edges(&self, pc: u64, _bytes: &[u8]) -> Option<CfgEdges> {
            if pc == 0x1000 {
                // Unconditional jump to 0x2000.
                Some(CfgEdges {
                    static_successors: vec![0x2000],
                    fallthrough: false,
                    insn_len: 4,
                    has_indirect: false,
                })
            } else {
                // Normal instruction.
                Some(CfgEdges {
                    static_successors: vec![],
                    fallthrough: true,
                    insn_len: 4,
                    has_indirect: false,
                })
            }
        }
    }

    #[test]
    fn seed_alone_is_reachable() {
        // A single-instruction binary; seed at its address should be reachable.
        let bytes = [0u8; 4];
        let spec = ReachabilitySpec::single(0x1000);
        let set = compute_reachable(&spec, &bytes, 0x1000, &SyntheticDecoder);
        assert!(set.contains(0x1000));
    }

    #[test]
    fn fallthrough_extends_set() {
        // 3 NOP-like instructions at 0x1000, 0x1004, 0x1008.
        // Seed at 0x1000 — expect all three to be reachable.
        let bytes = [0u8; 12];
        let spec = ReachabilitySpec::single(0x1000);
        let set = compute_reachable(&spec, &bytes, 0x1000, &SyntheticDecoder);
        assert!(set.contains(0x1000));
        assert!(set.contains(0x1004));
        assert!(set.contains(0x1008));
    }

    #[test]
    fn sink_stops_fallthrough() {
        // Instruction at 0x1000 is a sink; 0x1004 is not reachable from 0x1000.
        let bytes: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
        let spec = ReachabilitySpec::single(0x1000);
        let set = compute_reachable(&spec, &bytes, 0x1000, &SyntheticDecoder);
        assert!(set.contains(0x1000));
        assert!(!set.contains(0x1004));
    }

    #[test]
    fn static_jump_target_reached() {
        // Binary spans 0x1000..0x2010 (large enough for both endpoints).
        let bytes = vec![0u8; 0x1010];
        let spec = ReachabilitySpec::single(0x1000);
        let set = compute_reachable(&spec, &bytes, 0x1000, &JumpDecoder);
        // 0x1000 is the jump; it jumps to 0x2000 (no fallthrough).
        assert!(set.contains(0x1000));
        assert!(set.contains(0x2000));
        // 0x1004 is NOT reachable (jump is unconditional, no fallthrough).
        assert!(!set.contains(0x1004));
    }

    #[test]
    fn out_of_range_seeds_ignored() {
        let bytes = [0u8; 4];
        let spec = ReachabilitySpec::new([0x9999]); // outside [0x1000, 0x1004)
        let set = compute_reachable(&spec, &bytes, 0x1000, &SyntheticDecoder);
        assert!(set.is_empty());
    }

    #[test]
    fn out_of_range_edge_not_followed() {
        // Jump decoder at 0x1000 jumps to 0x2000, but binary only covers 0x1000..0x1004.
        let bytes = [0u8; 4];
        let spec = ReachabilitySpec::single(0x1000);
        let set = compute_reachable(&spec, &bytes, 0x1000, &JumpDecoder);
        assert!(set.contains(0x1000));
        assert!(!set.contains(0x2000)); // out of range, not followed
    }

    #[test]
    fn backward_edge_revisit_ok() {
        // Two instructions: 0x1000 falls through to 0x1004; 0x1004 jumps back
        // to 0x1000.  Both should be reachable; no infinite loop in BFS.
        struct LoopDecoder;
        impl CfgDecoder for LoopDecoder {
            fn decode_edges(&self, pc: u64, _bytes: &[u8]) -> Option<CfgEdges> {
                if pc == 0x1004 {
                    Some(CfgEdges {
                        static_successors: vec![0x1000],
                        fallthrough: false,
                        insn_len: 4,
                        has_indirect: false,
                    })
                } else {
                    Some(CfgEdges {
                        static_successors: vec![],
                        fallthrough: true,
                        insn_len: 4,
                        has_indirect: false,
                    })
                }
            }
        }
        let bytes = [0u8; 8];
        let spec = ReachabilitySpec::single(0x1000);
        let set = compute_reachable(&spec, &bytes, 0x1000, &LoopDecoder);
        assert!(set.contains(0x1000));
        assert!(set.contains(0x1004));
        assert_eq!(set.len(), 2); // no duplicates, no infinite loop
    }
}
