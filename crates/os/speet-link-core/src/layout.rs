//! [`IndexSpace`] and [`EntityIndexSpace`] — pre-declared index ranges for all
//! five WASM entity kinds in multi-binary modules.
//!
//! See `docs/entity-index-space.md` for the full design rationale.

#![allow(unused)]
extern crate alloc;
use alloc::vec::Vec;

/// Handle to a declared range within a single [`IndexSpace`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IndexSlot(pub u32);

/// Pre-declared index space for one WASM entity kind.
///
/// Built during the **registration phase** of
/// [`FuncSchedule`](crate::schedule::FuncSchedule): each binary calls
/// [`append`](Self::append) to declare how many entities it will produce,
/// receiving an [`IndexSlot`] handle whose base index is immediately known.
///
/// Once all binaries are registered the space is frozen and
/// [`base`](Self::base) returns absolute WASM indices before any emission
/// begins.
#[derive(Default)]
pub struct IndexSpace {
    counts: Vec<u32>,
    bases:  Vec<u32>,
    total:  u32,
}

impl IndexSpace {
    /// Create an empty space (no slots declared).
    pub fn empty() -> Self {
        Self { counts: Vec::new(), bases: Vec::new(), total: 0 }
    }

    /// Declare a range of `count` contiguous indices.
    ///
    /// Returns a slot whose base immediately follows all previously declared
    /// slots.
    pub fn append(&mut self, count: u32) -> IndexSlot {
        let slot = IndexSlot(self.counts.len() as u32);
        self.bases.push(self.total);
        self.counts.push(count);
        self.total += count;
        slot
    }

    /// Absolute WASM index of the first element in `slot`.
    pub fn base(&self, slot: IndexSlot) -> u32 {
        self.bases[slot.0 as usize]
    }

    /// Declared count for `slot`.
    pub fn count(&self, slot: IndexSlot) -> u32 {
        self.counts[slot.0 as usize]
    }

    /// Total declared count across all slots.
    pub fn total(&self) -> u32 {
        self.total
    }
}

/// Unified pre-declaration covering all six WASM entity kinds.
///
/// See `docs/entity-index-space.md`.
#[derive(Default)]
pub struct EntityIndexSpace {
    pub types:     IndexSpace,
    pub functions: IndexSpace,
    pub memories:  IndexSpace,
    pub tables:    IndexSpace,
    pub tags:      IndexSpace,
    pub globals:   IndexSpace,
}

impl EntityIndexSpace {
    pub fn empty() -> Self {
        Self {
            types:     IndexSpace::empty(),
            functions: IndexSpace::empty(),
            memories:  IndexSpace::empty(),
            tables:    IndexSpace::empty(),
            tags:      IndexSpace::empty(),
            globals:   IndexSpace::empty(),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy aliases — kept during the migration; will be removed once all call
// sites are updated to EntityIndexSpace.
// ---------------------------------------------------------------------------

/// Alias for [`IndexSlot`] — use `IndexSlot` in new code.
pub type FuncSlot = IndexSlot;

/// Alias for [`IndexSpace`] — use `IndexSpace` in new code.
pub type FuncLayout = IndexSpace;
