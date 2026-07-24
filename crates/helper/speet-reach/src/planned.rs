//! `portal-lazy-transform`-based alternative to [`crate::compute_reachable`]'s
//! internal worklist, gated behind the `lazy-reach-plan` feature.
//!
//! `describe` performs exactly the same purely structural edge lookup
//! (`CfgDecoder::decode_edges`, filtered to in-range static successors and
//! fallthrough) as the default `VecDeque`-based implementation -- this module
//! only swaps *how* the BFS closure is computed, not what edges it follows.
//! [`ReachableSet::pcs`] is an unordered `BTreeSet`, so the discovered *set*
//! is identical between the two implementations; only internal traversal
//! order (irrelevant to any caller) can differ.

use alloc::vec::Vec;
use core::convert::Infallible;

use portal_lazy_transform::{
    AssemblyError, AssemblyLimits, Demand, Fragment, NoopObserver, PlanSource, ResolvedInputs,
};

use crate::{CfgDecoder, ReachabilitySpec, compute::ReachableSet};

/// The single facet this plan resolves: "is this PC reachable."
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reach;

struct ReachCtx<'a> {
    bytes: &'a [u8],
    base_addr: u64,
    decoder: &'a dyn CfgDecoder,
}

impl<'a> ReachCtx<'a> {
    fn in_range(&self, pc: u64) -> bool {
        let end_addr = self.base_addr.saturating_add(self.bytes.len() as u64);
        pc >= self.base_addr && pc < end_addr
    }
}

/// Stateless `PlanSource` functor -- purely structural discovery, no
/// resolution step (nothing to resolve for a reachability-only query).
#[derive(Default)]
struct ReachSource;

impl<'a> PlanSource<ReachCtx<'a>> for ReachSource {
    type Node = u64;
    type Artifact = u64;
    type Facet = Reach;
    type Metadata = ();
    type Error = Infallible;

    fn root_for(
        &self,
        _context: &ReachCtx<'a>,
        demand: &Demand<u64, Reach>,
    ) -> Result<u64, Infallible> {
        Ok(demand.artifact)
    }

    fn describe(
        &mut self,
        context: &ReachCtx<'a>,
        node: &u64,
        _demand: &Demand<u64, Reach>,
    ) -> Result<Fragment<u64, u64, Reach, ()>, Infallible> {
        let pc = *node;
        if !context.in_range(pc) {
            return Ok(Fragment::new(pc, (), Vec::new()));
        }
        let offset = (pc - context.base_addr) as usize;
        let Some(edges) = context.decoder.decode_edges(pc, &context.bytes[offset..]) else {
            return Ok(Fragment::new(pc, (), Vec::new())); // decode failure — treat as sink
        };

        let mut deps = Vec::new();
        for &target in &edges.static_successors {
            if context.in_range(target) {
                deps.push(Demand::new(target, Reach));
            }
        }
        if edges.fallthrough {
            let next_pc = pc.wrapping_add(edges.insn_len as u64);
            if context.in_range(next_pc) {
                deps.push(Demand::new(next_pc, Reach));
            }
        }
        Ok(Fragment::new(pc, (), deps))
    }

    fn resolve(
        &mut self,
        _context: &mut ReachCtx<'a>,
        _node: &u64,
        _facet: &Reach,
        _inputs: &mut dyn ResolvedInputs<u64, Reach>,
    ) -> Result<(), Infallible> {
        Ok(())
    }
}

/// `portal_lazy_transform::assemble_bfs`-based equivalent of
/// [`crate::compute_reachable`], unbounded (matches its lack of a limits
/// parameter -- this can never return an assembly error).
pub fn compute_reachable_via_plan(
    spec: &ReachabilitySpec,
    bytes: &[u8],
    base_addr: u64,
    decoder: &dyn CfgDecoder,
) -> ReachableSet {
    compute_reachable_via_plan_with_limits(spec, bytes, base_addr, decoder, AssemblyLimits::default())
        .expect("unbounded AssemblyLimits with an Infallible source cannot error")
}

/// Like [`compute_reachable_via_plan`] but with explicit `AssemblyLimits`,
/// returning an error if a bound is exceeded rather than growing unbounded.
pub fn compute_reachable_via_plan_with_limits(
    spec: &ReachabilitySpec,
    bytes: &[u8],
    base_addr: u64,
    decoder: &dyn CfgDecoder,
    limits: AssemblyLimits,
) -> Result<ReachableSet, AssemblyError<Infallible>> {
    let ctx = ReachCtx {
        bytes,
        base_addr,
        decoder,
    };
    let roots: Vec<_> = spec
        .seeds
        .iter()
        .copied()
        .filter(|&pc| ctx.in_range(pc))
        .map(|pc| Demand::new(pc, Reach))
        .collect();

    let mut source = ReachSource;
    let plan = portal_lazy_transform::assemble_bfs(&mut source, &ctx, roots, limits, &mut NoopObserver)?;
    Ok(ReachableSet {
        pcs: plan.nodes.into_iter().map(|n| n.node).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::compute_reachable_eager_worklist;
    use alloc::vec;

    struct JumpDecoder;
    impl CfgDecoder for JumpDecoder {
        fn decode_edges(&self, pc: u64, _bytes: &[u8]) -> Option<crate::CfgEdges> {
            if pc == 0x1000 {
                Some(crate::CfgEdges {
                    static_successors: vec![0x2000],
                    fallthrough: false,
                    insn_len: 4,
                    has_indirect: false,
                })
            } else {
                Some(crate::CfgEdges {
                    static_successors: vec![],
                    fallthrough: true,
                    insn_len: 4,
                    has_indirect: false,
                })
            }
        }
    }

    struct LoopDecoder;
    impl CfgDecoder for LoopDecoder {
        fn decode_edges(&self, pc: u64, _bytes: &[u8]) -> Option<crate::CfgEdges> {
            if pc == 0x1004 {
                Some(crate::CfgEdges {
                    static_successors: vec![0x1000],
                    fallthrough: false,
                    insn_len: 4,
                    has_indirect: false,
                })
            } else {
                Some(crate::CfgEdges {
                    static_successors: vec![],
                    fallthrough: true,
                    insn_len: 4,
                    has_indirect: false,
                })
            }
        }
    }

    /// Both implementations must discover the identical reachable set for a
    /// static-jump CFG.
    #[test]
    fn matches_eager_worklist_for_static_jump() {
        let bytes = vec![0u8; 0x1010];
        let spec = ReachabilitySpec::single(0x1000);
        let eager = compute_reachable_eager_worklist(&spec, &bytes, 0x1000, &JumpDecoder);
        let planned = compute_reachable_via_plan(&spec, &bytes, 0x1000, &JumpDecoder);
        assert_eq!(eager.pcs, planned.pcs);
        assert!(planned.contains(0x1000));
        assert!(planned.contains(0x2000));
        assert!(!planned.contains(0x1004));
    }

    /// Both implementations must handle a back edge (loop) identically and
    /// terminate (no infinite loop / duplicate visits).
    #[test]
    fn matches_eager_worklist_for_loop() {
        let bytes = [0u8; 8];
        let spec = ReachabilitySpec::single(0x1000);
        let eager = compute_reachable_eager_worklist(&spec, &bytes, 0x1000, &LoopDecoder);
        let planned = compute_reachable_via_plan(&spec, &bytes, 0x1000, &LoopDecoder);
        assert_eq!(eager.pcs, planned.pcs);
        assert_eq!(planned.len(), 2);
    }

    /// Out-of-range seeds and edge targets are dropped identically by both.
    #[test]
    fn matches_eager_worklist_for_out_of_range() {
        let bytes = [0u8; 4];
        let spec = ReachabilitySpec::single(0x1000);
        let eager = compute_reachable_eager_worklist(&spec, &bytes, 0x1000, &JumpDecoder);
        let planned = compute_reachable_via_plan(&spec, &bytes, 0x1000, &JumpDecoder);
        assert_eq!(eager.pcs, planned.pcs);
        assert!(!planned.contains(0x2000));
    }

    #[test]
    fn with_limits_reports_node_limit_error_instead_of_panicking() {
        let bytes = vec![0u8; 0x2000];
        let spec = ReachabilitySpec::single(0x1000);
        let limits = AssemblyLimits {
            max_nodes: 1,
            max_edges: usize::MAX,
            max_depth: usize::MAX,
        };
        let err = compute_reachable_via_plan_with_limits(&spec, &bytes, 0x1000, &JumpDecoder, limits)
            .unwrap_err();
        assert!(matches!(err, AssemblyError::NodeLimit { limit: 1 }));
    }
}
