//! [`ReachabilitySpec`] — seed PCs provided by an external tool.

use alloc::collections::BTreeSet;

/// Entry-point PCs provided by an external tool.
///
/// These addresses are the *seeds* for [`compute_reachable`](crate::compute_reachable).
/// The function performs a BFS from these seeds, following jumps, calls,
/// fallthroughs, and branches to build the full transitive [`ReachableSet`](crate::ReachableSet).
///
/// # Indirect jump targets
///
/// Static analysis cannot determine runtime-computed targets (indirect jumps
/// and calls).  If an external tool knows these targets (e.g. from profiling
/// or a vtable map), it should include them here so the BFS can follow them.
///
/// # Cross-binary edges
///
/// [`compute_reachable`](crate::compute_reachable) operates on a single
/// contiguous byte slice.  Edges that point outside the slice are silently
/// ignored.  Add entry points of separately-compiled units to `seeds` to
/// include them.
#[derive(Debug, Clone, Default)]
pub struct ReachabilitySpec {
    /// Known-reachable instruction addresses.
    pub seeds: BTreeSet<u64>,
}

impl ReachabilitySpec {
    /// Construct a spec from an iterator of seed PCs.
    pub fn new(seeds: impl IntoIterator<Item = u64>) -> Self {
        Self { seeds: seeds.into_iter().collect() }
    }

    /// Convenience constructor for a spec with a single entry point.
    pub fn single(pc: u64) -> Self {
        let mut seeds = BTreeSet::new();
        seeds.insert(pc);
        Self { seeds }
    }

    /// Add a seed PC to an existing spec.
    pub fn add(&mut self, pc: u64) {
        self.seeds.insert(pc);
    }
}
