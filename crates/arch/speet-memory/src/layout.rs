//! [`LocalLayout`] — compile-a-local-variable-map from `(count, ValType)` groups.

use alloc::vec::Vec;
use wasm_encoder::ValType;

/// A single named region within a [`LocalLayout`].
///
/// Create instances with [`LocalLayout::build`] and look up base indices with
/// [`LocalLayout::base`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalSlot(pub usize);

/// A frozen map from `(count, ValType)` groups to contiguous wasm local indices.
///
/// # Building a layout
///
/// ```ignore
/// use speet_memory::layout::{LocalLayout, LocalSlot};
/// use wasm_encoder::ValType;
///
/// let (layout, [regs, pc, flags, temps]) = LocalLayout::build([
///     (16, ValType::I64),  // 16 GP registers  → locals 0-15
///     (1,  ValType::I32),  // PC               → local 16
///     (5,  ValType::I32),  // flags ZF..PF     → locals 17-21
///     (4,  ValType::I64),  // 4 temps          → locals 22-25
/// ]);
///
/// assert_eq!(layout.base(pc), 16);
/// assert_eq!(layout.base(temps), 22);
/// ```
///
/// # Iterating
///
/// `LocalLayout` implements `Iterator<Item = (u32, ValType)>` (the format
/// expected by `wasm_encoder::Function::new`).
pub struct LocalLayout {
    /// (count, type, base_local_index) for each slot in insertion order.
    slots: Vec<(u32, ValType, u32)>,
}

impl LocalLayout {
    /// Build a `LocalLayout` from a fixed-size array of `(count, ValType)` groups.
    ///
    /// Returns the frozen layout together with a same-length array of
    /// [`LocalSlot`] handles, one per group, in the same order.
    ///
    /// Generic over `N` so the slot-handle array is stack-allocated and the
    /// caller gets named handles without needing a `Vec`.
    pub fn build<const N: usize>(groups: [(u32, ValType); N]) -> (Self, [LocalSlot; N]) {
        let mut slots = Vec::with_capacity(N);
        let mut handles = [LocalSlot(0); N];
        let mut cursor: u32 = 0;
        for (i, (count, ty)) in groups.iter().enumerate() {
            handles[i] = LocalSlot(i);
            slots.push((*count, *ty, cursor));
            cursor += count;
        }
        (Self { slots }, handles)
    }

    /// Build a layout from a slice, returning a `Vec` of slot handles.
    pub fn build_dynamic(groups: &[(u32, ValType)]) -> (Self, Vec<LocalSlot>) {
        let mut slots = Vec::with_capacity(groups.len());
        let mut handles = Vec::with_capacity(groups.len());
        let mut cursor: u32 = 0;
        for (i, &(count, ty)) in groups.iter().enumerate() {
            handles.push(LocalSlot(i));
            slots.push((count, ty, cursor));
            cursor += count;
        }
        (Self { slots }, handles)
    }

    /// Return the first wasm local index for `slot`.
    ///
    /// # Panics
    /// Panics if `slot` was not produced by this layout.
    #[inline]
    pub fn base(&self, slot: LocalSlot) -> u32 {
        self.slots[slot.0].2
    }

    /// Return the wasm local index for the *n*-th element inside `slot`
    /// (0-based).
    ///
    /// # Panics
    /// Panics if `n >= count` for the slot.
    #[inline]
    pub fn local(&self, slot: LocalSlot, n: u32) -> u32 {
        let (count, _, base) = self.slots[slot.0];
        assert!(n < count, "local index {n} out of range for slot (count={count})");
        base + n
    }

    /// Return the count of locals in `slot`.
    #[inline]
    pub fn count(&self, slot: LocalSlot) -> u32 {
        self.slots[slot.0].0
    }

    /// Return the `ValType` of `slot`.
    #[inline]
    pub fn val_type(&self, slot: LocalSlot) -> ValType {
        self.slots[slot.0].1
    }

    /// Total number of wasm locals declared by this layout.
    #[inline]
    pub fn total_locals(&self) -> u32 {
        self.slots.last().map(|&(count, _, base)| base + count).unwrap_or(0)
    }

    /// Iterate over `(count, ValType)` pairs in insertion order.
    ///
    /// The yielded pairs are directly usable as arguments to
    /// `wasm_encoder::Function::new`.
    pub fn iter(&self) -> impl Iterator<Item = (u32, ValType)> + '_ {
        self.slots.iter().map(|&(count, ty, _)| (count, ty))
    }
}

impl core::fmt::Debug for LocalLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list()
            .entries(self.slots.iter().map(|&(count, ty, base)| {
                (base..base + count, ty)
            }))
            .finish()
    }
}
