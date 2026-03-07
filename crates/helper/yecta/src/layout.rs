//! [`LocalLayout`] — a growable map from named `(count, ValType)` groups to
//! contiguous wasm local indices.
//!
//! `LocalLayout` serves two distinct roles in the speet recompiler pipeline,
//! both of which share the same resolution mechanism:
//!
//! ## Role 1 — architectural params (declared once)
//!
//! A recompiler appends its parameter groups in order to a single layout,
//! starting from index 0.  Traps then append their own parameter groups.
//! A [`Mark`] is placed after all params.  Because every group is appended in
//! insertion order and the layout starts at 0, [`LocalLayout::local`] and
//! [`LocalLayout::base`] return correct **absolute** wasm local indices
//! immediately — no `set_base` call is needed.
//!
//! ```ignore
//! let mut layout = LocalLayout::empty();
//! // Architecture params (appended first → indices 0, 1, …):
//! let regs = layout.append(32, ValType::I64);  // params 0-31
//! let pc   = layout.append(1,  ValType::I32);  // param  32
//! // Trap params (appended next → indices 33, …):
//! trap.declare_params(&mut layout);
//! let params_mark = layout.mark();             // total_locals() == total_params
//! ```
//!
//! ## Role 2 — per-function locals (mark + rewind)
//!
//! Per-function non-param locals are appended after the params mark and
//! yielded to the function constructor via [`LocalLayout::iter_since`].  At
//! the start of each new function the layout is rewound to the params mark
//! so the same slots can be re-declared with fresh indices:
//!
//! ```ignore
//! // In init_function():
//! layout.rewind(&params_mark);           // discard previous function's locals
//! let temps = layout.append(num_temps, ValType::I64);
//! let pool  = layout.append(N_POOL, ValType::I32);
//! trap.declare_locals(&mut layout);      // trap appends its own locals
//! reactor.next_with(ctx, f(&mut layout.iter_since(&params_mark)), depth)?;
//! // layout.base(pool) now gives the correct absolute wasm local index
//! ```
//!
//! ## Iterating
//!
//! [`LocalLayout::iter`] yields all `(count, ValType)` pairs in insertion
//! order.  [`LocalLayout::iter_since`] yields only the pairs added after a
//! given [`Mark`], suitable for passing to `wasm_encoder::Function::new`.

use alloc::vec::Vec;
use wasm_encoder::ValType;

// ── Mark ─────────────────────────────────────────────────────────────────────

/// An opaque snapshot of a [`LocalLayout`]'s size at a given moment.
///
/// Obtained from [`LocalLayout::mark`].  Pass to [`LocalLayout::rewind`] to
/// truncate the layout back to that snapshot, or to
/// [`LocalLayout::iter_since`] to iterate only the groups added after the
/// mark.
///
/// The `total_locals` field is the total span of locals at the time the mark
/// was taken — conveniently equal to the *total parameter count* when the
/// mark is placed right after all parameter groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mark {
    /// Number of slots in the layout at mark time.
    pub slot_count: usize,
    /// Cumulative local count at mark time (`total_locals()` value).
    pub total_locals: u32,
}

// ── LocalSlot ────────────────────────────────────────────────────────────────

/// A handle to a named group of locals within a [`LocalLayout`].
///
/// Obtained from [`LocalLayout::build`], [`LocalLayout::build_dynamic`], or
/// [`LocalLayout::append`].  Pass to [`LocalLayout::base`] or
/// [`LocalLayout::local`] to obtain absolute wasm local indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalSlot(pub usize);

// ── LocalLayout ──────────────────────────────────────────────────────────────

/// A map from `(count, ValType)` groups to contiguous wasm local indices.
///
/// See the [module documentation](self) for usage patterns.
pub struct LocalLayout {
    /// `(count, type, offset)` for each slot in insertion order.
    /// `offset` is the cumulative count of locals in all *preceding* slots —
    /// i.e. the absolute wasm local index of the first local in this slot.
    slots: Vec<(u32, ValType, u32)>,
}

impl LocalLayout {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create an empty layout with no slots.
    ///
    /// Slots are added incrementally via [`append`](Self::append).
    #[inline]
    pub fn empty() -> Self {
        Self { slots: Vec::new() }
    }

    /// Build a `LocalLayout` from a fixed-size array of `(count, ValType)` groups.
    ///
    /// Returns the frozen layout together with a same-length array of
    /// [`LocalSlot`] handles, one per group, in the same order.  The first
    /// group starts at absolute index 0 unless [`rewind`](Self::rewind) has
    /// previously placed the layout at a non-zero start (which cannot happen
    /// with this constructor).
    pub fn build<const N: usize>(groups: [(u32, ValType); N]) -> (Self, [LocalSlot; N]) {
        let mut layout = Self::empty();
        let mut handles = [LocalSlot(0); N];
        for (i, (count, ty)) in groups.iter().enumerate() {
            handles[i] = layout.append(*count, *ty);
        }
        (layout, handles)
    }

    /// Build a layout from a slice, returning a `Vec` of slot handles.
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
    /// local indices.
    ///
    /// This is the primary entry point for both architecture-defined param
    /// groups and trap-defined groups:
    ///
    /// ```ignore
    /// fn declare_params(&mut self, params: &mut LocalLayout) {
    ///     self.depth_slot = params.append(1, ValType::I32);
    /// }
    /// ```
    pub fn append(&mut self, count: u32, ty: ValType) -> LocalSlot {
        let offset = self.total_locals(); // absolute start of new group
        let slot = LocalSlot(self.slots.len());
        self.slots.push((count, ty, offset));
        slot
    }

    // ── Mark / rewind ─────────────────────────────────────────────────────────

    /// Capture the current layout size as a [`Mark`].
    ///
    /// The mark's `total_locals` field equals the current
    /// [`total_locals()`](Self::total_locals) value.  When the mark is placed
    /// after all parameter groups, this equals the total wasm parameter count.
    #[inline]
    pub fn mark(&self) -> Mark {
        Mark {
            slot_count:   self.slots.len(),
            total_locals: self.total_locals(),
        }
    }

    /// Truncate the layout back to `mark`, removing all slots appended since.
    ///
    /// Slot handles obtained *after* the mark (in e.g. `declare_locals`) are
    /// invalidated by this call.  Slot handles obtained *before* the mark
    /// remain valid and return the same indices as before.
    ///
    /// # Panics
    /// Panics in debug builds if `mark.total_locals` doesn't match the
    /// cumulative count after truncation.
    #[inline]
    pub fn rewind(&mut self, mark: &Mark) {
        self.slots.truncate(mark.slot_count);
        debug_assert_eq!(self.total_locals(), mark.total_locals);
    }

    // ── Index resolution ──────────────────────────────────────────────────────

    /// Return the first absolute wasm local index for `slot`.
    ///
    /// # Panics
    /// Panics if `slot` was not produced by this layout.
    #[inline]
    pub fn base(&self, slot: LocalSlot) -> u32 {
        self.slots[slot.0].2
    }

    /// Return the absolute wasm local index for the *n*-th element inside
    /// `slot` (0-based).
    ///
    /// # Panics
    /// Panics if `n >= count` for the slot or if `slot` is not from this layout.
    #[inline]
    pub fn local(&self, slot: LocalSlot, n: u32) -> u32 {
        let (count, _, offset) = self.slots[slot.0];
        assert!(n < count, "local index {n} out of range for slot (count={count})");
        offset + n
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

    /// Total number of wasm locals declared by this layout (span from 0).
    ///
    /// After appending param groups, this equals the total wasm parameter
    /// count and is also stored in [`Mark::total_locals`] when
    /// [`mark`](Self::mark) is called.
    #[inline]
    pub fn total_locals(&self) -> u32 {
        self.slots.last().map(|&(count, _, offset)| offset + count).unwrap_or(0)
    }

    /// Iterate over all `(count, ValType)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, ValType)> + '_ {
        self.slots.iter().map(|&(count, ty, _)| (count, ty))
    }

    /// Iterate over `(count, ValType)` pairs for slots added *after* `mark`.
    ///
    /// The yielded pairs are directly usable as arguments to
    /// `wasm_encoder::Function::new` for declaring the non-param locals of a
    /// wasm function.
    pub fn iter_since<'a>(&'a self, mark: &Mark) -> impl Iterator<Item = (u32, ValType)> + 'a {
        self.slots[mark.slot_count..].iter().map(|&(count, ty, _)| (count, ty))
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
            .entries(self.slots.iter().map(|&(count, ty, offset)| {
                (offset..offset + count, ty)
            }))
            .finish()
    }
}
