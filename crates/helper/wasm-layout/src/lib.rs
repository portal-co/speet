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
//!
//! ## Cell system
//!
//! A **cell** identifies a unique combination of WASM function-type parameters
//! and non-parameter locals.  Cells enable trap implementations to allocate
//! per-function scratch locals that are deduplicated across functions with the
//! same signature, and to look up those locals by a stable [`CellIdx`] handle
//! rather than recomputing indices.
//!
//! ### Registration
//!
//! [`CellRegistry`] maps signatures to [`CellIdx`] handles.  Two registration
//! paths exist:
//!
//! - **Host-keyed** ([`CellRegistry::register`]): called *after* all params and
//!   locals have been appended to the layout.  Extracts params via
//!   `layout.iter_before(&mark)` and locals via `layout.iter_since(&mark)`.
//!   Used by native arch recompilers (RISC-V, x86-64).
//!
//! - **Guest-keyed** ([`CellRegistry::register_for_guest`]): called *before*
//!   `declare_locals` — as soon as the parsed guest function type and declared
//!   locals are known.  Used by the WASM frontend where functions are not
//!   addressed by guest PC.
//!
//! ### Protocol (host-keyed, three phases)
//!
//! ```ignore
//! // Phase 1 — once per recompiler instance:
//! recompiler.setup_traps(&mut rctx);      // arch params → trap.declare_params → mark
//!
//! // Phase 2 — once per function (inside init_function):
//! rctx.layout_mut().rewind(&mark);        // discard previous function's locals
//! rctx.layout_mut().append(…);           // arch non-param locals
//! rctx.declare_trap_locals();             // trap appends its per-function locals
//! let cell = rctx.alloc_cell();           // register → CellIdx
//!
//! // Phase 3 — firing:
//! rctx.on_instruction(&info, ctx)?;       // trap fires, resolves locals via cell
//! rctx.on_jump(&info, ctx)?;
//! ```
//!
//! ### Resolving indices
//!
//! Traps store [`LocalSlot`] handles from their `declare_params` / `declare_locals`
//! calls.  They resolve absolute WASM local indices at fire-time via:
//!
//! ```ignore
//! let idx = trap_ctx.layout().local(self.my_slot, 0);
//! trap_ctx.emit(ctx, &Instruction::LocalGet(idx))?;
//! ```
//!
//! Both param and local slots live in the same layout (params first, then locals),
//! so `local(slot, n)` returns the correct absolute index without any extra offset.

#![no_std]

extern crate alloc;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct LocalSlot(pub(crate) usize);

// ── LocalLayout ──────────────────────────────────────────────────────────────

/// A map from `(count, ValType)` groups to contiguous wasm local indices.
///
/// See the [module documentation](self) for usage patterns.
#[derive(Clone)]
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
        let handles = groups
            .iter()
            .map(|&(count, ty)| layout.append(count, ty))
            .collect();
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
            slot_count: self.slots.len(),
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
        assert!(
            n < count,
            "local index {n} out of range for slot (count={count})"
        );
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
        self.slots
            .last()
            .map(|&(count, _, offset)| offset + count)
            .unwrap_or(0)
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
        self.slots[mark.slot_count..]
            .iter()
            .map(|&(count, ty, _)| (count, ty))
    }

    /// Iterate over `(count, ValType)` pairs for slots *before* `mark`.
    ///
    /// The yielded pairs span the parameter portion of the layout — i.e. the
    /// slots appended before the params [`Mark`] was placed.  Suitable as the
    /// `params` argument to [`CellRegistry::register`]:
    ///
    /// ```ignore
    /// let cell = registry.register(
    ///     layout.iter_before(&locals_mark),  // param groups
    ///     layout.iter_since(&locals_mark),   // local groups
    /// );
    /// ```
    pub fn iter_before<'a>(&'a self, mark: &Mark) -> impl Iterator<Item = (u32, ValType)> + 'a {
        self.slots[..mark.slot_count]
            .iter()
            .map(|&(count, ty, _)| (count, ty))
    }

    /// Returns `true` if the layout has no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellIdx(pub u32);

// ── CellSignature ─────────────────────────────────────────────────────────────

/// The canonical key identifying a unique cell.
///
/// Produced and stored by [`CellRegistry`].  Two registration paths exist,
/// and they occupy disjoint namespaces because the field groups they populate
/// are mutually exclusive:
///
/// ## Host-keyed cells (native recompilers)
///
/// Registered via [`CellRegistry::register`] *after* both `declare_params`
/// and `declare_locals` have finished appending to the [`LocalLayout`].  The
/// `params` and `locals` fields are populated from the host layout; the
/// `guest_*` fields are left empty.
///
/// ## Guest-keyed cells (WASM frontend)
///
/// Registered via [`CellRegistry::register_for_guest`] *before*
/// `declare_locals` is called — as soon as the guest function type and
/// declared locals are known from the parsed WASM binary.  The `guest_params`,
/// `guest_results`, and `guest_locals` fields are populated; `params` and
/// `locals` remain empty until a future pass fills them in.
///
/// Because host-keyed entries always have empty `guest_*` fields and
/// guest-keyed entries always have empty `params`/`locals`, the two namespaces
/// are disjoint and equality comparison never produces false collisions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellSignature {
    /// Flat wasm parameter types in declaration order (one per wasm param).
    /// Populated by host-keyed registration; empty for guest-keyed cells.
    pub params: Vec<ValType>,
    /// Non-param locals as `(count, type)` groups, in declaration order.
    /// Populated by host-keyed registration; empty for guest-keyed cells.
    pub locals: Vec<(u32, ValType)>,
    /// Guest function-type parameter types, in declaration order.
    /// Populated by guest-keyed registration; empty for host-keyed cells.
    pub guest_params: Vec<ValType>,
    /// Guest function-type result types, in declaration order.
    /// Populated by guest-keyed registration; empty for host-keyed cells.
    pub guest_results: Vec<ValType>,
    /// Guest declared non-param locals as `(count, type)` groups, in
    /// declaration order (from the function body's locals section).
    /// Populated by guest-keyed registration; empty for host-keyed cells.
    pub guest_locals: Vec<(u32, ValType)>,
}

// ── CellRegistry ──────────────────────────────────────────────────────────────

/// A registry that allocates [`CellIdx`] handles for unique
/// (function-type params, non-param locals) combinations.
///
/// Each distinct signature is assigned exactly one [`CellIdx`].  Calling
/// either registration method with an already-seen signature returns the
/// existing handle; a new `CellIdx` is minted only on first encounter.
///
/// Two registration paths exist — see [`CellSignature`] for the contract each
/// one fills.
///
/// ## Host-keyed path (native recompilers)
///
/// ```ignore
/// // (inside init_function, after traps.declare_locals has returned):
/// let cell = self.cell_registry.register(
///     self.layout.iter_before(&self.locals_mark),  // param groups
///     self.layout.iter_since(&self.locals_mark),   // local groups
/// );
/// self.current_cell = cell;
/// ```
///
/// ## Guest-keyed path (WASM frontend)
///
/// ```ignore
/// // (before declare_locals — as soon as guest info is available):
/// let cell = registry.register_for_guest(
///     func_type.params_val_types(),
///     func_type.results_val_types(),
///     guest_locals.iter().copied(),
/// );
/// mapper.declare_locals(cell, layout);
/// traps.declare_locals(cell, layout);
/// ```
pub struct CellRegistry {
    /// `(signature, assigned_index)` pairs in insertion order.
    /// Linear scan is used for lookup because a given recompiler instance
    /// produces at most a handful of distinct signatures.
    entries: Vec<(CellSignature, CellIdx)>,
}

impl CellRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Look up or insert a host-keyed cell for the given `(params, locals)` signature.
    ///
    /// `params` should iterate the `(count, ValType)` groups for the
    /// function's parameter slots — i.e. `layout.iter_before(&locals_mark)`.
    ///
    /// `locals` should iterate the `(count, ValType)` groups for the
    /// non-parameter locals — i.e. `layout.iter_since(&locals_mark)`.
    ///
    /// If an identical `(params, locals)` combination was registered before,
    /// the existing [`CellIdx`] is returned unchanged.  Otherwise a new
    /// [`CellIdx`] is allocated sequentially from 0.
    ///
    /// The `guest_*` fields of the stored [`CellSignature`] are empty;
    /// use [`register_for_guest`](Self::register_for_guest) for the WASM
    /// frontend's guest-keyed path.
    pub fn register(
        &mut self,
        params: impl IntoIterator<Item = (u32, ValType)>,
        locals: impl IntoIterator<Item = (u32, ValType)>,
    ) -> CellIdx {
        let sig = CellSignature {
            params: params
                .into_iter()
                .flat_map(|(c, t)| core::iter::repeat(t).take(c as usize))
                .collect(),
            locals: locals.into_iter().collect(),
            guest_params: Vec::new(),
            guest_results: Vec::new(),
            guest_locals: Vec::new(),
        };
        for (s, idx) in &self.entries {
            if s == &sig {
                return *idx;
            }
        }
        let idx = CellIdx(self.entries.len() as u32);
        self.entries.push((sig, idx));
        idx
    }

    /// Look up or insert a guest-keyed cell for the given
    /// `(guest_params, guest_results, guest_locals)` triple.
    ///
    /// Call this *before* `declare_locals` — as soon as the guest function
    /// type and declared locals are available from the parsed WASM binary.
    /// The returned [`CellIdx`] can then be passed directly to every
    /// `declare_locals` call (mapper and trap) for that function.
    ///
    /// The `params` and `locals` fields of the stored [`CellSignature`] are
    /// left empty (they belong to the host-layout namespace and will be filled
    /// in by a future pass).  Use [`register`](Self::register) for native
    /// recompilers that key on the host layout instead.
    pub fn register_for_guest(
        &mut self,
        guest_params: impl IntoIterator<Item = ValType>,
        guest_results: impl IntoIterator<Item = ValType>,
        guest_locals: impl IntoIterator<Item = (u32, ValType)>,
    ) -> CellIdx {
        let sig = CellSignature {
            params: Vec::new(),
            locals: Vec::new(),
            guest_params: guest_params.into_iter().collect(),
            guest_results: guest_results.into_iter().collect(),
            guest_locals: guest_locals.into_iter().collect(),
        };
        for (s, idx) in &self.entries {
            if s == &sig {
                return *idx;
            }
        }
        let idx = CellIdx(self.entries.len() as u32);
        self.entries.push((sig, idx));
        idx
    }

    /// Return the signature for a previously allocated cell.
    ///
    /// # Panics
    /// Panics if `cell` was not produced by this registry.
    pub fn signature(&self, cell: CellIdx) -> &CellSignature {
        &self.entries[cell.0 as usize].0
    }

    /// Total number of unique cells registered so far.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no cells have been registered yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for CellRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── LocalAllocator ────────────────────────────────────────────────────────────

/// Trait for types that allocate and resolve wasm local variable slots.
///
/// `LocalLayout` is the canonical implementation.  The trait exists so that
/// code receiving a layout through a trait object (e.g. [`TrapContext`]) does
/// not need to name the concrete type.
///
/// All methods mirror the corresponding [`LocalLayout`] inherent methods.
///
/// [`TrapContext`]: https://docs.rs/speet-traps/latest/speet_traps/context/struct.TrapContext.html
pub trait LocalAllocator {
    /// Append a group of `count` locals of type `ty` and return a slot handle.
    fn append(&mut self, count: u32, ty: ValType) -> LocalSlot;

    /// Return the first absolute wasm local index for `slot`.
    fn base(&self, slot: LocalSlot) -> u32;

    /// Return the absolute wasm local index for the *n*-th element in `slot`.
    fn local(&self, slot: LocalSlot, n: u32) -> u32;

    /// Capture the current layout size as a [`Mark`].
    fn mark(&self) -> Mark;

    /// Truncate the layout back to `mark`.
    fn rewind(&mut self, mark: &Mark);

    /// Total number of wasm locals declared by this layout.
    fn total_locals(&self) -> u32;
}

impl LocalAllocator for LocalLayout {
    #[inline]
    fn append(&mut self, count: u32, ty: ValType) -> LocalSlot {
        self.append(count, ty)
    }
    #[inline]
    fn base(&self, slot: LocalSlot) -> u32 {
        self.base(slot)
    }
    #[inline]
    fn local(&self, slot: LocalSlot, n: u32) -> u32 {
        self.local(slot, n)
    }
    #[inline]
    fn mark(&self) -> Mark {
        self.mark()
    }
    #[inline]
    fn rewind(&mut self, mark: &Mark) {
        self.rewind(mark)
    }
    #[inline]
    fn total_locals(&self) -> u32 {
        self.total_locals()
    }
}

// ── LocalDeclarator ───────────────────────────────────────────────────────────

/// Declares wasm parameter and local slots into a [`LocalLayout`].
///
/// Implement this trait to reserve [`LocalSlot`] handles during recompiler
/// setup.  Both methods default to no-ops; only types that need slots must
/// override them.
///
/// # Protocol
///
/// * [`declare_params`](Self::declare_params) — called **once** before the
///   params [`Mark`] is placed.  Slots appended here survive `return_call`
///   chains (they are wasm function *parameters*).
/// * [`declare_locals`](Self::declare_locals) — called **once per function**
///   after the params mark.  Slots appended here are reset to zero at the
///   start of each new wasm function (scratch locals).
pub trait LocalDeclarator {
    /// Append parameter-level slots (persist across `return_call` chains).
    #[allow(unused_variables)]
    fn declare_params(&mut self, cell: CellIdx, params: &mut LocalLayout) {}

    /// Append per-function scratch slots (reset each new wasm function).
    #[allow(unused_variables)]
    fn declare_locals(&mut self, cell: CellIdx, locals: &mut LocalLayout) {}
}

impl core::fmt::Debug for LocalLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list()
            .entries(
                self.slots
                    .iter()
                    .map(|&(count, ty, offset)| (offset..offset + count, ty)),
            )
            .finish()
    }
}
