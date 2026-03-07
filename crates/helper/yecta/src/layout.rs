//! [`LocalLayout`] — a growable map from named `(count, ValType)` groups to
//! contiguous wasm local indices.
//!
//! `LocalLayout` serves two distinct roles in the speet recompiler pipeline,
//! both of which share the same resolution mechanism:
//!
//! ## Role 1 — architectural locals (constructed up-front)
//!
//! A recompiler calls [`LocalLayout::build`] once with all its local groups in
//! order.  The resulting [`LocalSlot`] handles are used throughout translation
//! to look up absolute wasm indices.  An optional [`LocalLayout::set_base`]
//! call shifts all indices if the locals do not start at 0 (e.g. they follow
//! wasm function parameters).
//!
//! ```ignore
//! let (layout, [regs, pc, flags, temps]) = LocalLayout::build([
//!     (32, ValType::I64),  // 32 GP registers → locals 0-31
//!     (1,  ValType::I64),  // PC              → local 32
//!     (5,  ValType::I32),  // flags ZF..PF    → locals 33-37
//!     (4,  ValType::I64),  // 4 temps         → locals 38-41
//! ]);
//! // locals start right after total_params wasm params:
//! layout.set_base(total_params);
//! assert_eq!(layout.base(pc), total_params + 32);
//! ```
//!
//! ## Role 2 — trap-contributed parameters and locals (grown incrementally)
//!
//! [`TrapConfig`](crate) uses an empty `LocalLayout` to collect parameter and
//! local slots from installed traps:
//!
//! ```ignore
//! let mut params = LocalLayout::empty();
//! // Each installed trap appends its own parameter groups and stores the
//! // returned LocalSlot handles in its own fields for later use:
//! my_trap.declare_params(&mut params);
//! // After all traps have declared, set the absolute base:
//! params.set_base(base_params);
//! let total_params = base_params + params.total_locals();
//! ```
//!
//! The same pattern applies to per-function locals:
//!
//! ```ignore
//! let mut locals = LocalLayout::empty();
//! my_trap.declare_locals(&mut locals);
//! // Per function, update the base when the first trap-local index is known:
//! locals.set_base(total_params + arch_local_count);
//! ```
//!
//! ## Iterating
//!
//! [`LocalLayout::iter`] yields `(count, ValType)` pairs in insertion order,
//! directly usable as arguments to `wasm_encoder::Function::new`.

use alloc::vec::Vec;
use wasm_encoder::ValType;

/// A handle to a named group of locals within a [`LocalLayout`].
///
/// Obtained from [`LocalLayout::build`], [`LocalLayout::build_dynamic`], or
/// [`LocalLayout::append`].  Pass to [`LocalLayout::base`] or
/// [`LocalLayout::local`] to obtain absolute wasm local indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalSlot(pub usize);

/// A map from `(count, ValType)` groups to contiguous wasm local indices.
///
/// See the [module documentation](self) for usage patterns.
pub struct LocalLayout {
    /// `(count, type, relative_offset)` for each slot, in insertion order.
    /// `relative_offset` is the offset from `base_offset` (always starts at 0
    /// for the first slot regardless of base).
    slots: Vec<(u32, ValType, u32)>,
    /// Added to every relative offset when resolving absolute local indices.
    /// Defaults to `0`; set via [`set_base`](Self::set_base).
    base_offset: u32,
}

impl LocalLayout {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create an empty layout with no slots and base offset `0`.
    ///
    /// Slots are added incrementally via [`append`](Self::append).
    #[inline]
    pub fn empty() -> Self {
        Self { slots: Vec::new(), base_offset: 0 }
    }

    /// Create an empty layout with the given base offset already set.
    ///
    /// Equivalent to `LocalLayout::empty()` followed by `set_base(offset)`.
    #[inline]
    pub fn empty_with_base(offset: u32) -> Self {
        Self { slots: Vec::new(), base_offset: offset }
    }

    /// Build a `LocalLayout` from a fixed-size array of `(count, ValType)` groups.
    ///
    /// Returns the frozen layout together with a same-length array of
    /// [`LocalSlot`] handles, one per group, in the same order.
    ///
    /// Generic over `N` so the slot-handle array is stack-allocated.
    ///
    /// The initial base offset is `0`.  Call [`set_base`](Self::set_base) or
    /// chain [`with_base`](Self::with_base) when absolute indices start
    /// elsewhere.
    pub fn build<const N: usize>(groups: [(u32, ValType); N]) -> (Self, [LocalSlot; N]) {
        let mut layout = Self::empty();
        let mut handles = [LocalSlot(0); N];
        for (i, (count, ty)) in groups.iter().enumerate() {
            handles[i] = layout.append(*count, *ty);
        }
        (layout, handles)
    }

    /// Build a layout from a slice, returning a `Vec` of slot handles.
    ///
    /// The initial base offset is `0`.
    pub fn build_dynamic(groups: &[(u32, ValType)]) -> (Self, Vec<LocalSlot>) {
        let mut layout = Self::empty();
        let handles = groups.iter().map(|&(count, ty)| layout.append(count, ty)).collect();
        (layout, handles)
    }

    // ── Growing ───────────────────────────────────────────────────────────────

    /// Append a group of `count` locals of type `ty`, returning a slot handle.
    ///
    /// The new group immediately follows any previously appended groups.  The
    /// returned [`LocalSlot`] can be stored and later passed to
    /// [`base`](Self::base) or [`local`](Self::local) to obtain absolute wasm
    /// local indices (after [`set_base`](Self::set_base) has been called if
    /// needed).
    ///
    /// This is the primary entry point for traps that participate in the
    /// incremental declaration protocol:
    ///
    /// ```ignore
    /// fn declare_params(&mut self, params: &mut LocalLayout) {
    ///     self.depth_slot = params.append(1, ValType::I32);
    /// }
    /// ```
    pub fn append(&mut self, count: u32, ty: ValType) -> LocalSlot {
        let rel = self.total_locals(); // start of new group (relative)
        let slot = LocalSlot(self.slots.len());
        self.slots.push((count, ty, rel));
        slot
    }

    // ── Base management ───────────────────────────────────────────────────────

    /// Assign (or replace) the base offset for all absolute-index calculations.
    ///
    /// After this call, [`base`](Self::base) and [`local`](Self::local) return
    /// indices offset by `offset`.  For example, after `set_base(32)`, a slot
    /// at relative offset 0 resolves to absolute index 32.
    #[inline]
    pub fn set_base(&mut self, offset: u32) {
        self.base_offset = offset;
    }

    /// Return a copy of this layout with the given base offset applied.
    ///
    /// Useful in builder chains:
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

    // ── Index resolution ──────────────────────────────────────────────────────

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
    /// Panics if `n >= count` for the slot or if `slot` is not from this layout.
    #[inline]
    pub fn local(&self, slot: LocalSlot, n: u32) -> u32 {
        let (count, _, rel) = self.slots[slot.0];
        assert!(n < count, "local index {n} out of range for slot (count={count})");
        self.base_offset + rel + n
    }

    // ── Inspection ───────────────────────────────────────────────────────────

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

    /// Total number of wasm locals declared by this layout (span, not
    /// including the base offset).
    ///
    /// After `set_base(b)`, the locals occupy `b .. b + total_locals()`.
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

    /// Returns `true` if the layout has no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
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
