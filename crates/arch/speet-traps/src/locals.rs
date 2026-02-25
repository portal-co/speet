//! [`ExtraLocals`] — per-trap wasm local declarations and index resolution.
//!
//! Every trap implementation may declare a fixed set of additional wasm locals
//! that it needs per translated function (e.g. a shadow-stack pointer, a
//! CFI tag, a per-function counter).  These locals are declared alongside the
//! architecture's own locals when a new wasm function is opened, and the trap
//! accesses them by absolute index via [`ExtraLocals::local`].
//!
//! ## Lifecycle
//!
//! 1. **Construction** — the trap calls [`ExtraLocals::new`] at startup,
//!    listing `(count, ValType)` groups in stable order.
//!
//! 2. **Chaining into the function declaration** — the recompiler's
//!    `init_function` iterates [`ExtraLocals::iter`] and appends the groups to
//!    its own local list before calling `reactor.next_with`.
//!
//! 3. **Base assignment** — after the architecture has counted its own locals,
//!    it calls [`ExtraLocals::set_base`] with that count so that
//!    [`ExtraLocals::local`] returns correct absolute indices.
//!
//! 4. **Use inside a trap** — the trap calls `trap_ctx.locals().local(n)` to
//!    obtain the wasm local index for slot `n` and uses it in emitted
//!    `local.get` / `local.set` / `local.tee` instructions.

use alloc::vec::Vec;
use wasm_encoder::ValType;

/// A block of wasm locals reserved for a single trap implementation.
///
/// The block is described as a sequence of `(count, ValType)` groups (the same
/// format accepted by `wasm_encoder::Function::new` and by
/// `reactor.next_with`).  The absolute wasm local index of any element in the
/// block is not known at construction time; it is assigned by the recompiler's
/// `init_function` via [`ExtraLocals::set_base`].
///
/// An `ExtraLocals` with zero groups is a valid no-op value (returned by
/// [`ExtraLocals::none`]).
#[derive(Debug, Clone)]
pub struct ExtraLocals {
    /// Declared groups, in order: `(count, type)`.
    groups: Vec<(u32, ValType)>,
    /// Absolute wasm local index of the first local in this block.
    /// `u32::MAX` until [`set_base`](Self::set_base) is called.
    base: u32,
}

impl ExtraLocals {
    /// Declare a block of locals described by `groups`.
    ///
    /// `groups` follows the same `(count, ValType)` convention used by
    /// `wasm_encoder::Function::new`: each pair declares `count` consecutive
    /// locals of the given type.  The groups are stored in order; indices
    /// assigned by [`local`](Self::local) reflect that order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Two i32 locals followed by one i64 local.
    /// let extra = ExtraLocals::new([(2, ValType::I32), (1, ValType::I64)]);
    /// // After set_base(100): local(0) = 100, local(1) = 101, local(2) = 102.
    /// ```
    pub fn new(groups: impl IntoIterator<Item = (u32, ValType)>) -> Self {
        Self {
            groups: groups.into_iter().collect(),
            base: u32::MAX,
        }
    }

    /// An empty `ExtraLocals` that declares no locals.
    ///
    /// Use this as the default when a trap needs no per-function state.
    pub fn none() -> Self {
        Self { groups: Vec::new(), base: 0 }
    }

    /// Iterate the `(count, ValType)` groups suitable for chaining into a
    /// `reactor.next_with` local declaration.
    pub fn iter(&self) -> impl Iterator<Item = (u32, ValType)> + '_ {
        self.groups.iter().copied()
    }

    /// Total number of individual locals declared (sum of all counts).
    pub fn total_count(&self) -> u32 {
        self.groups.iter().map(|(n, _)| n).sum()
    }

    /// The absolute wasm local index of the first local in this block.
    ///
    /// Only valid after [`set_base`](Self::set_base) has been called by the
    /// recompiler's `init_function`.  Panics in debug mode if called before
    /// the base is set.
    pub fn base(&self) -> u32 {
        debug_assert!(
            self.base != u32::MAX,
            "ExtraLocals::base() called before set_base()"
        );
        self.base
    }

    /// Return the absolute wasm local index for the `n`th individual local
    /// across all groups (zero-based, in declaration order).
    ///
    /// For example, if the groups are `[(2, I32), (1, I64)]` and the base is
    /// 100, then `local(0)` = 100, `local(1)` = 101, `local(2)` = 102.
    ///
    /// Panics if `n >= total_count()` or if the base has not been set.
    pub fn local(&self, n: u32) -> u32 {
        debug_assert!(
            self.base != u32::MAX,
            "ExtraLocals::local() called before set_base()"
        );
        debug_assert!(
            n < self.total_count(),
            "ExtraLocals::local({n}) out of range (total {})",
            self.total_count()
        );
        self.base + n
    }

    /// Assign the absolute base index.
    ///
    /// Called exactly once per translated function by the recompiler's
    /// `init_function`, after it has determined the total count of its own
    /// architecture-defined locals.  The trap's locals immediately follow in
    /// the wasm function's local vector.
    ///
    /// This method is `pub(crate)` — only `TrapConfig` (in the same crate) is
    /// permitted to call it.  Architecture recompilers go through
    /// [`TrapConfig::set_extra_locals_base`] instead.
    pub(crate) fn set_base(&mut self, base: u32) {
        self.base = base;
    }
}
