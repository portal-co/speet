//! [`FuncLayout`] — pre-declared function index ranges for multi-binary modules.

#![allow(unused)]
extern crate alloc;
use alloc::vec::Vec;

/// Handle to a declared function range inside a [`FuncLayout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncSlot(pub u32);

/// Pre-declared function index layout for a multi-binary WASM module.
///
/// Built during the **registration phase** of [`FuncSchedule`](crate::schedule::FuncSchedule):
/// each binary calls [`append`](Self::append) with its function count, receiving a
/// [`FuncSlot`] handle.  Once all binaries are registered the layout is final and
/// [`base`](Self::base) returns absolute WASM function indices suitable for
/// constructing [`IndexOffsets`] before any translation begins.
pub struct FuncLayout {
    counts: Vec<u32>,
    bases:  Vec<u32>,
    total:  u32,
}

impl FuncLayout {
    /// Create an empty layout (no slots declared).
    pub fn empty() -> Self {
        Self { counts: Vec::new(), bases: Vec::new(), total: 0 }
    }

    /// Declare a binary that will produce `count` functions.
    ///
    /// Returns a slot whose base immediately follows all previously declared
    /// slots.  Calling this repeatedly builds a contiguous, non-overlapping
    /// index range for each binary.
    pub fn append(&mut self, count: u32) -> FuncSlot {
        let slot = FuncSlot(self.counts.len() as u32);
        self.bases.push(self.total);
        self.counts.push(count);
        self.total += count;
        slot
    }

    /// Absolute WASM function index of the first function in `slot`.
    pub fn base(&self, slot: FuncSlot) -> u32 {
        self.bases[slot.0 as usize]
    }

    /// Declared function count for `slot`.
    pub fn count(&self, slot: FuncSlot) -> u32 {
        self.counts[slot.0 as usize]
    }

    /// Total declared function count across all slots.
    pub fn total(&self) -> u32 {
        self.total
    }
}
