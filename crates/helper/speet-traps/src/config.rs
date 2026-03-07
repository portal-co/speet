//! [`TrapConfig`] — the single object a recompiler embeds and delegates to.
//!
//! ## Overview
//!
//! A recompiler (e.g. `RiscVRecompiler`) embeds one `TrapConfig` field and
//! calls methods at two phases:
//!
//! ### Phase 1 — setup (once per recompiler instance)
//!
//! ```text
//! self.total_params = self.traps.setup(Self::BASE_PARAMS);
//! ```
//!
//! `setup` calls `declare_params` on each installed trap, appending their
//! parameter groups to an internal [`LocalLayout`].  It then sets the layout
//! base to `base_params` and returns `base_params + trap_param_count` — the
//! total wasm function parameter count.  The recompiler stores this and uses
//! it as the `params` argument to every `jmp`, `ji`, and `ji_with_params` call.
//!
//! ### Phase 2 — per function (in `init_function`)
//!
//! ```text
//! let arch_locals: &[(u32, ValType)] = &[…];
//! let arch_local_count: u32 = arch_locals.iter().map(|(n,_)| n).sum();
//! let all_locals: Vec<_> = arch_locals.iter().copied()
//!     .chain(self.traps.locals_iter())
//!     .collect();
//! reactor.next_with(ctx, f(&mut all_locals.into_iter()), 2)?;
//! self.traps.set_local_base(self.total_params + arch_local_count);
//! ```
//!
//! `locals_iter()` yields the trap-contributed `(count, ValType)` groups to
//! chain after the arch locals.  `set_local_base` updates the base offset of
//! the internal locals layout so that `trap_ctx.locals().local(slot, n)`
//! returns correct absolute indices for the current function.
//!
//! ### Firing
//!
//! ```text
//! // at translate_instruction start:
//! if self.traps.on_instruction(&info, ctx, &mut reactor)? == TrapAction::Skip {
//!     return Ok(());
//! }
//! // at each jump site:
//! if self.traps.on_jump(&info, ctx, &mut reactor)? == TrapAction::Skip {
//!     return Ok(());
//! }
//! ```
//!
//! ## Local / parameter layout
//!
//! ```text
//! local 0           … base_params-1              recompiler params (regs, PC, …)
//! local base_params … total_params-1             trap params (depth counter, …)
//! local total_params … total_params+arch_locals-1  arch non-param locals
//! local total_params+arch_locals … (end)         trap non-param locals
//! ```
//!
//! Both the params layout and the locals layout are owned by `TrapConfig`.
//! Traps receive read-only references via [`TrapContext::params`] and
//! [`TrapContext::locals`].

use wasm_encoder::ValType;
use wax_core::build::InstructionSink;
use yecta::LocalLayout;

use crate::context::TrapContext;
use crate::insn::{InstructionInfo, InstructionTrap, TrapAction};
use crate::jump::{JumpInfo, JumpTrap};

// ── TrapConfig ────────────────────────────────────────────────────────────────

/// The configuration object a recompiler embeds.
///
/// Holds an optional [`InstructionTrap`] and an optional [`JumpTrap`], each
/// contributing parameter and local slots to shared [`LocalLayout`]s.  When
/// no traps are installed, all methods are no-ops with zero overhead.
///
/// ## Lifetimes
///
/// * `'cb` — lifetime of the borrowed trap implementations.
/// * `'ctx` — lifetime of any data the callbacks capture.
pub struct TrapConfig<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>> {
    // ── Instruction trap ──────────────────────────────────────────────────
    insn_trap: Option<&'cb mut (dyn InstructionTrap<Context, E, F> + 'ctx)>,

    // ── Jump trap ─────────────────────────────────────────────────────────
    jump_trap: Option<&'cb mut (dyn JumpTrap<Context, E, F> + 'ctx)>,

    // ── Shared layouts ────────────────────────────────────────────────────
    /// Parameter slots contributed by all installed traps.
    /// Base is set in [`setup`](Self::setup).
    params_layout: LocalLayout,
    /// Non-param local slots contributed by all installed traps.
    /// Base is updated per-function in [`set_local_base`](Self::set_local_base).
    locals_layout: LocalLayout,
}

impl<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>> Default
    for TrapConfig<'cb, 'ctx, Context, E, F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>>
    TrapConfig<'cb, 'ctx, Context, E, F>
{
    /// Create an empty `TrapConfig` with no traps installed.
    pub fn new() -> Self {
        Self {
            insn_trap: None,
            jump_trap: None,
            params_layout: LocalLayout::empty(),
            locals_layout: LocalLayout::empty(),
        }
    }

    // ── Installation ─────────────────────────────────────────────────────

    /// Install an instruction trap and collect its slot declarations.
    ///
    /// Calls `trap.declare_params(&mut params_layout)` and
    /// `trap.declare_locals(&mut locals_layout)` so the trap can append its
    /// groups to the shared layouts and store the returned [`LocalSlot`]
    /// handles in its own fields.
    ///
    /// Must be called **before** [`setup`](Self::setup).  Installing a trap
    /// after `setup` is not supported and will produce incorrect indices.
    ///
    /// [`LocalSlot`]: yecta::LocalSlot
    pub fn set_instruction_trap(
        &mut self,
        trap: &'cb mut (dyn InstructionTrap<Context, E, F> + 'ctx),
    ) {
        trap.declare_params(&mut self.params_layout);
        trap.declare_locals(&mut self.locals_layout);
        self.insn_trap = Some(trap);
    }

    /// Install a jump trap and collect its slot declarations.
    ///
    /// See [`set_instruction_trap`](Self::set_instruction_trap) for the
    /// protocol.
    pub fn set_jump_trap(
        &mut self,
        trap: &'cb mut (dyn JumpTrap<Context, E, F> + 'ctx),
    ) {
        trap.declare_params(&mut self.params_layout);
        trap.declare_locals(&mut self.locals_layout);
        self.jump_trap = Some(trap);
    }

    /// Remove the instruction trap and clear all declared slots.
    ///
    /// This resets both the params and locals layouts.  Any previously stored
    /// [`LocalSlot`] handles from the removed trap become invalid.  If a jump
    /// trap is still installed, it must be reinstalled to re-declare its slots.
    ///
    /// [`LocalSlot`]: yecta::LocalSlot
    pub fn clear_instruction_trap(&mut self) {
        self.insn_trap = None;
        self.params_layout = LocalLayout::empty();
        self.locals_layout = LocalLayout::empty();
        // Re-declare jump trap slots since we wiped the layouts.
        if let Some(trap) = self.jump_trap.as_mut().map(|t| &mut **t as *mut _) {
            // SAFETY: we hold an exclusive borrow on self; the trap pointer is
            // valid as long as 'cb is alive.
            let trap: &mut (dyn JumpTrap<Context, E, F> + 'ctx) = unsafe { &mut *trap };
            trap.declare_params(&mut self.params_layout);
            trap.declare_locals(&mut self.locals_layout);
        }
    }

    /// Remove the jump trap and clear its declared slots.
    ///
    /// See [`clear_instruction_trap`](Self::clear_instruction_trap) for
    /// caveats.
    pub fn clear_jump_trap(&mut self) {
        self.jump_trap = None;
        self.params_layout = LocalLayout::empty();
        self.locals_layout = LocalLayout::empty();
        if let Some(trap) = self.insn_trap.as_mut().map(|t| &mut **t as *mut _) {
            let trap: &mut (dyn InstructionTrap<Context, E, F> + 'ctx) = unsafe { &mut *trap };
            trap.declare_params(&mut self.params_layout);
            trap.declare_locals(&mut self.locals_layout);
        }
    }

    // ── Phase 1: setup ────────────────────────────────────────────────────

    /// **Phase 1** — fix the parameter base and return the total parameter count.
    ///
    /// Call this once after installing all traps, before translation begins:
    ///
    /// ```ignore
    /// self.total_params = self.traps.setup(Self::BASE_PARAMS);
    /// ```
    ///
    /// Sets the base offset of the internal params layout to `base_params`,
    /// so that every [`LocalSlot`] stored by traps during
    /// `declare_params` resolves to the correct absolute wasm local index.
    ///
    /// Returns `base_params + total_trap_param_count`.  The recompiler must
    /// store this and pass it as `params` to every `jmp` / `ji` /
    /// `ji_with_params` call.
    ///
    /// [`LocalSlot`]: yecta::LocalSlot
    pub fn setup(&mut self, base_params: u32) -> u32 {
        self.params_layout.set_base(base_params);
        base_params + self.params_layout.total_locals()
    }

    // ── Phase 2: per-function locals ──────────────────────────────────────

    /// **Phase 2a** — yield the trap-contributed `(count, ValType)` local groups.
    ///
    /// Chain the result after the arch-defined local groups when calling
    /// `reactor.next_with`:
    ///
    /// ```ignore
    /// let all_locals: Vec<_> = arch_locals.iter().copied()
    ///     .chain(self.traps.locals_iter())
    ///     .collect();
    /// reactor.next_with(ctx, f(&mut all_locals.into_iter()), depth)?;
    /// self.traps.set_local_base(total_params + arch_local_count);
    /// ```
    pub fn locals_iter(&self) -> impl Iterator<Item = (u32, ValType)> + '_ {
        self.locals_layout.iter()
    }

    /// **Phase 2b** — update the base offset of the locals layout for the
    /// current function.
    ///
    /// `first_trap_local` is the absolute wasm local index of the first
    /// trap-owned non-param local.  This equals
    /// `total_params + arch_non_param_local_count`.
    ///
    /// After this call, `trap_ctx.locals().local(slot, n)` returns the
    /// correct absolute index for all slots declared via `declare_locals`.
    pub fn set_local_base(&mut self, first_trap_local: u32) {
        self.locals_layout.set_base(first_trap_local);
    }

    // ── Param layout accessor ─────────────────────────────────────────────

    /// Read-only access to the shared parameter layout.
    ///
    /// The iterator over this layout's `(count, ValType)` groups can be used
    /// to extend the wasm function type when registering translated functions
    /// with the module (only the trap-contributed groups, not the recompiler's
    /// own base params).
    pub fn params_layout(&self) -> &LocalLayout {
        &self.params_layout
    }

    // ── Firing ────────────────────────────────────────────────────────────

    /// Fire the instruction trap (if installed).
    ///
    /// Returns [`TrapAction::Continue`] when no trap is installed.  When the
    /// trap returns [`TrapAction::Skip`], emits `skip_snippet` before
    /// returning `Skip`.
    pub fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        sink: &mut F,
    ) -> Result<TrapAction, E> {
        let trap = match self.insn_trap.as_mut() {
            Some(t) => t,
            None    => return Ok(TrapAction::Continue),
        };
        let mut trap_ctx = TrapContext::new(sink, &self.params_layout, &self.locals_layout);
        let action = trap.on_instruction(info, ctx, &mut trap_ctx)?;
        if action == TrapAction::Skip {
            let mut trap_ctx2 = TrapContext::new(sink, &self.params_layout, &self.locals_layout);
            self.insn_trap
                .as_ref()
                .unwrap()
                .skip_snippet(info, ctx, &mut trap_ctx2)?;
        }
        Ok(action)
    }

    /// Fire the jump trap (if installed).
    ///
    /// Returns [`TrapAction::Continue`] when no trap is installed.  When the
    /// trap returns [`TrapAction::Skip`], emits `skip_snippet` before
    /// returning `Skip`.
    pub fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        sink: &mut F,
    ) -> Result<TrapAction, E> {
        let trap = match self.jump_trap.as_mut() {
            Some(t) => t,
            None    => return Ok(TrapAction::Continue),
        };
        let mut trap_ctx = TrapContext::new(sink, &self.params_layout, &self.locals_layout);
        let action = trap.on_jump(info, ctx, &mut trap_ctx)?;
        if action == TrapAction::Skip {
            let mut trap_ctx2 = TrapContext::new(sink, &self.params_layout, &self.locals_layout);
            self.jump_trap
                .as_ref()
                .unwrap()
                .skip_snippet(info, ctx, &mut trap_ctx2)?;
        }
        Ok(action)
    }
}
