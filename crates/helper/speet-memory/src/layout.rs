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
/// When the absolute starting local index is known at build time, pass it to
/// [`LocalLayout::build`] directly (the default base offset is 0, so indices
/// start at 0 unless changed):
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
/// # Deferred base assignment
///
/// Sometimes the layout is built before the absolute starting index is known
/// (e.g. a trap that appends its locals after architecture-defined ones).
/// Use [`LocalLayout::set_base`] once the starting index is determined:
///
/// ```ignore
/// let (mut layout, [scratch, counter]) = LocalLayout::build([
///     (2, ValType::I32),  // two scratch i32s → relative 0, 1
///     (1, ValType::I64),  // one i64 counter  → relative 2
/// ]);
/// // Arch uses 66 locals (total_params=2, arch=64) before these.
/// layout.set_base(66);
/// assert_eq!(layout.base(scratch), 66);
/// assert_eq!(layout.base(counter), 68);
/// ```
///
/// The builder method [`LocalLayout::with_base`] is also available for
/// fluent construction.
///
/// # Iterating
///
/// [`LocalLayout::iter`] yields `(count, ValType)` pairs in insertion order,
/// suitable as arguments to `wasm_encoder::Function::new`.
pub struct LocalLayout {
    /// (count, type, relative_offset) for each slot in insertion order.
    /// Relative offset is from 0 (independent of `base_offset`).
    slots: Vec<(u32, ValType, u32)>,
    /// Added to every relative offset when resolving absolute local indices.
    /// Set via [`set_base`](Self::set_base); defaults to `0`.
    base_offset: u32,
}

impl LocalLayout {
    /// Build a `LocalLayout` from a fixed-size array of `(count, ValType)` groups.
    ///
    /// Returns the frozen layout together with a same-length array of
    /// [`LocalSlot`] handles, one per group, in the same order.
    ///
    /// Generic over `N` so the slot-handle array is stack-allocated and the
    /// caller gets named handles without needing a `Vec`.
    ///
    /// The initial base offset is `0`.  Call [`set_base`](Self::set_base) or
    /// use [`with_base`](Self::with_base) if absolute indices start elsewhere.
    pub fn build<const N: usize>(groups: [(u32, ValType); N]) -> (Self, [LocalSlot; N]) {
        let mut slots = Vec::with_capacity(N);
        let mut handles = [LocalSlot(0); N];
        let mut cursor: u32 = 0;
        for (i, (count, ty)) in groups.iter().enumerate() {
            handles[i] = LocalSlot(i);
            slots.push((*count, *ty, cursor));
            cursor += count;
        }
        (Self { slots, base_offset: 0 }, handles)
    }

    /// Build a layout from a slice, returning a `Vec` of slot handles.
    ///
    /// The initial base offset is `0`.  Call [`set_base`](Self::set_base) or
    /// use [`with_base`](Self::with_base) if absolute indices start elsewhere.
    pub fn build_dynamic(groups: &[(u32, ValType)]) -> (Self, Vec<LocalSlot>) {
        let mut slots = Vec::with_capacity(groups.len());
        let mut handles = Vec::with_capacity(groups.len());
        let mut cursor: u32 = 0;
        for (i, &(count, ty)) in groups.iter().enumerate() {
            handles.push(LocalSlot(i));
            slots.push((count, ty, cursor));
            cursor += count;
        }
        (Self { slots, base_offset: 0 }, handles)
    }

    /// Assign the base offset for all absolute local index calculations.
    ///
    /// After this call, [`base`](Self::base) and [`local`](Self::local) return
    /// indices offset by `offset`.  For example, if `offset = 66` then what
    /// was slot 0 at relative index 0 now resolves to absolute index 66.
    ///
    /// This method is idempotent: calling it again with a different value
    /// simply replaces the previous base.
    #[inline]
    pub fn set_base(&mut self, offset: u32) {
        self.base_offset = offset;
    }

    /// Return a copy of this layout with the given base offset applied.
    ///
    /// Equivalent to `layout.set_base(offset); layout` but usable in a
    /// builder chain:
    ///
    /// ```ignore
    /// let (layout, [scratch]) = LocalLayout::build([(3, ValType::I32)]);
    /// let layout = layout.with_base(total_params + arch_locals);
    /// ```
    #[inline]
    pub fn with_base(mut self, offset: u32) -> Self {
        self.base_offset = offset;
        self
    }

    /// Return the current base offset.
    #[inline]
    pub fn base_offset(&self) -> u32 {
        self.base_offset
    }

    /// Return the first absolute wasm local index for `slot`.
    ///
    /// # Panics
    /// Panics if `slot` was not produced by this layout.
    #[inline]
    pub fn base(&self, slot: LocalSlot) -> u32 {
        self.base_offset + self.slots[slot.0].2
    }

    /// Return the absolute wasm local index for the *n*-th element inside
    /// `slot` (0-based).
    ///
    /// # Panics
    /// Panics if `n >= count` for the slot.
    #[inline]
    pub fn local(&self, slot: LocalSlot, n: u32) -> u32 {
        let (count, _, rel) = self.slots[slot.0];
        assert!(n < count, "local index {n} out of range for slot (count={count})");
        self.base_offset + rel + n
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

    /// Total number of wasm locals declared by this layout (the span, not
    /// including the base offset).
    #[inline]
    pub fn total_locals(&self) -> u32 {
        self.slots.last().map(|&(count, _, rel)| rel + count).unwrap_or(0)
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
            .entries(self.slots.iter().map(|&(count, ty, rel)| {
                let abs = self.base_offset + rel;
                (abs..abs + count, ty)
            }))
            .finish()
    }
}
