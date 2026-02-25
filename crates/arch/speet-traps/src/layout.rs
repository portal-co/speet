//! [`FunctionLayout`] — the two-phase protocol between a recompiler and the
//! trap system for declaring wasm parameters and locals.
//!
//! ## Why this exists
//!
//! Yecta models guest instructions as individual wasm functions that chain to
//! each other via `return_call`.  State that must survive across function
//! boundaries must live in **parameters** (wasm locals 0..params-1), because
//! `return_call N` passes `local.get 0 .. local.get (params-1)` to the next
//! function.  State that is only needed within a single function can live in
//! ordinary locals (declared in `next_with`'s iterator argument) beyond the
//! parameter range.
//!
//! Traps may need either kind:
//! - [`RopDetectTrap`](crate::impls::RopDetectTrap) maintains a **call-depth
//!   counter** that must survive across function boundaries → it is a
//!   **parameter**.
//! - [`CfiReturnTrap`](crate::impls::CfiReturnTrap) uses a scratch index
//!   local that is only needed within the function body → it is a **local**.
//!
//! ## The two-phase protocol
//!
//! ### Phase 1 — setup (once per recompiler instance)
//!
//! Before translation begins, the recompiler calls
//! [`FunctionLayout::new`] with its base parameter count, then passes the
//! layout to [`TrapConfig::setup`].  `setup` appends any trap-owned
//! parameter declarations and returns the total parameter count.  The
//! recompiler stores this and uses it as the `params` argument to every
//! `jmp` / `ji` / `ji_with_params` call for the lifetime of the recompiler.
//!
//! ```text
//! let mut layout = FunctionLayout::new(base_params);
//! let total_params = config.setup(&mut layout);
//! self.total_params = total_params;
//! ```
//!
//! ### Phase 2 — per function (inside `init_function`)
//!
//! Each time `init_function` is called the recompiler:
//!
//! 1. Builds an iterator of its own `(count, ValType)` local groups (not
//!    counting parameters — those are implicit).
//! 2. Calls [`TrapConfig::extend_locals`] with a mutable reference to that
//!    iterator so the trap system can append its own non-param local groups.
//! 3. Passes the extended iterator to `reactor.next_with`.
//! 4. Calls [`TrapConfig::set_local_base`] with the number of arch-defined
//!    non-param locals so the trap system can compute absolute local indices
//!    for its own non-param locals.
//!
//! ```text
//! // Inside init_function:
//! let mut locals_iter = arch_locals.into_iter();
//! let arch_local_count: u32 = arch_locals.iter().map(|(n,_)| n).sum();
//! let extended = config.extend_locals(&mut locals_iter);
//! reactor.next_with(ctx, f(&mut extended), 2)?;
//! config.set_local_base(total_params + arch_local_count);
//! ```
//!
//! Note: `set_local_base` receives `total_params + arch_local_count` because
//! wasm local indices run `[params | arch non-param locals | trap non-param
//! locals]` — the trap's non-param locals start after both the params and the
//! arch non-param locals.
//!
//! ## Parameter index layout
//!
//! ```text
//! local 0                     … base_params-1      : recompiler params (regs, PC, …)
//! local base_params           … total_params-1     : trap params (depth counter, …)
//! local total_params          … total_params+arch_locals-1  : arch non-param locals
//! local total_params+arch_locals … (end)           : trap non-param locals
//! ```
//!
//! Both [`ExtraParams`] and [`ExtraLocals`] record their `base` so that a
//! trap implementation can convert a slot index `n` into an absolute wasm
//! local index via [`ExtraParams::param`] and [`ExtraLocals::local`].

use alloc::vec::Vec;
use wasm_encoder::ValType;

// ── ExtraParams ───────────────────────────────────────────────────────────────

/// A block of wasm **parameters** (locals 0..params-1) reserved for a single
/// trap implementation.
///
/// Parameters survive `return_call` chains and are therefore suitable for any
/// state that must carry over from one translated instruction function to the
/// next.
///
/// The absolute base index within the parameter list is not known at
/// construction time; it is assigned by [`TrapConfig::setup`] via
/// [`ExtraParams::set_base`].
#[derive(Debug, Clone)]
pub struct ExtraParams {
    /// `(count, type)` groups, in declaration order.
    groups: Vec<(u32, ValType)>,
    /// Absolute wasm local index of the first parameter in this block.
    /// `u32::MAX` until [`set_base`](Self::set_base) is called.
    base: u32,
}

impl ExtraParams {
    /// Declare a block of parameters described by `groups`.
    ///
    /// `groups` follows the same `(count, ValType)` convention as
    /// `wasm_encoder::Function::new`: each pair declares `count` consecutive
    /// parameters of the given type.
    pub fn new(groups: impl IntoIterator<Item = (u32, ValType)>) -> Self {
        Self {
            groups: groups.into_iter().collect(),
            base: u32::MAX,
        }
    }

    /// No extra parameters (default for traps that need no cross-function
    /// state).
    pub fn none() -> Self {
        Self { groups: Vec::new(), base: 0 }
    }

    /// Iterate the `(count, ValType)` groups.
    pub fn iter(&self) -> impl Iterator<Item = (u32, ValType)> + '_ {
        self.groups.iter().copied()
    }

    /// Total number of individual parameters declared (sum of all counts).
    pub fn total_count(&self) -> u32 {
        self.groups.iter().map(|(n, _)| n).sum()
    }

    /// The absolute wasm local index of the first parameter in this block.
    ///
    /// Only valid after [`set_base`](Self::set_base) has been called by
    /// [`TrapConfig::setup`].  Panics in debug mode otherwise.
    pub fn base(&self) -> u32 {
        debug_assert!(
            self.base != u32::MAX,
            "ExtraParams::base() called before set_base()"
        );
        self.base
    }

    /// Return the absolute wasm local index of the `n`th individual parameter
    /// in this block (zero-based, in declaration order).
    ///
    /// Panics if `n >= total_count()` or if the base has not been set.
    pub fn param(&self, n: u32) -> u32 {
        debug_assert!(
            self.base != u32::MAX,
            "ExtraParams::param() called before set_base()"
        );
        debug_assert!(
            n < self.total_count(),
            "ExtraParams::param({n}) out of range (total {})",
            self.total_count()
        );
        self.base + n
    }

    /// Assign the absolute base index.
    ///
    /// Called by [`TrapConfig::setup`] after summing the recompiler's own
    /// parameter count.
    pub(crate) fn set_base(&mut self, base: u32) {
        self.base = base;
    }
}

// ── FunctionLayout ────────────────────────────────────────────────────────────

/// Carries the parameter and local layout information through the two-phase
/// protocol between a recompiler and [`TrapConfig`].
///
/// The recompiler creates one `FunctionLayout` per setup call (i.e. once when
/// the recompiler is constructed or reconfigured) and one per `init_function`
/// call.  See the [module documentation](self) for the full protocol.
///
/// This struct is intentionally simple — it is a data carrier, not a
/// controller.  The logic lives in [`TrapConfig::setup`] and
/// [`TrapConfig::extend_locals`].
pub struct FunctionLayout {
    /// The recompiler's own parameter count.
    pub base_params: u32,
    /// Total parameter count after trap contributions are appended.
    /// Set by [`TrapConfig::setup`]; read by the recompiler after setup.
    pub total_params: u32,
    /// Trap-contributed parameter groups, in declaration order.
    /// Populated by [`TrapConfig::setup`].
    pub(crate) extra_param_groups: Vec<(u32, ValType)>,
}

impl FunctionLayout {
    /// Create a new layout with the recompiler's base parameter count.
    ///
    /// `base_params` is the number of wasm parameters the recompiler's
    /// generated function type declares — e.g. 66 for RISC-V (32 int + 32
    /// float + PC + expected_RA).
    pub fn new(base_params: u32) -> Self {
        Self {
            base_params,
            total_params: base_params,
            extra_param_groups: Vec::new(),
        }
    }

    /// Iterate the extra `(count, ValType)` parameter groups contributed by
    /// traps, in declaration order.
    ///
    /// The recompiler's own parameters (locals 0..base_params-1) are not
    /// included — those are part of the function type the recompiler already
    /// knows about.  Only the *additional* groups appended by traps are here.
    ///
    /// Use this to extend the function type when registering translated
    /// functions with the wasm module.
    pub fn extra_param_iter(&self) -> impl Iterator<Item = (u32, ValType)> + '_ {
        self.extra_param_groups.iter().copied()
    }
}
