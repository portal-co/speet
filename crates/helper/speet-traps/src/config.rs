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
//! let mut layout = FunctionLayout::new(base_params);
//! let total_params = config.setup(&mut layout);
//! self.total_params = total_params;  // store; use as params in every jmp/ji
//! ```
//!
//! `setup` iterates all installed traps, appends their `extra_params()` groups
//! to the layout, sets the `ExtraParams` base on each trap, and returns the
//! final total parameter count.  The recompiler must use this count as the
//! `params` argument to every `jmp`, `ji`, and `ji_with_params` call.
//!
//! ### Phase 2 — per function (in `init_function`)
//!
//! ```text
//! let arch_locals: &[(u32, ValType)] = &[…];
//! let arch_local_count: u32 = arch_locals.iter().map(|(n,_)| n).sum();
//! let mut base_iter = arch_locals.iter().copied();
//! let extended = config.extend_locals(&mut base_iter);
//! reactor.next_with(ctx, f(&mut extended), 2)?;
//! config.set_local_base(total_params + arch_local_count);
//! ```
//!
//! `extend_locals` chains the trap's `extra_locals()` groups after the arch
//! iterator.  `set_local_base` tells each trap where in the wasm local index
//! space its non-param locals live.
//!
//! ### Firing
//!
//! ```text
//! // at translate_instruction start:
//! if config.on_instruction(&info, ctx, &mut reactor)? == TrapAction::Skip {
//!     return Ok(());
//! }
//! // at each jump site:
//! if config.on_jump(&info, ctx, &mut reactor)? == TrapAction::Skip {
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
//! Parameters (the first `total_params` locals) survive `return_call` chains;
//! non-param locals are reset to zero on each new function.

use alloc::{boxed::Box, vec::Vec};
use wasm_encoder::ValType;
use wax_core::build::InstructionSink;

use crate::context::TrapContext;
use crate::insn::{InstructionInfo, InstructionTrap, TrapAction};
use crate::jump::{JumpInfo, JumpTrap};
use crate::layout::{ExtraParams, FunctionLayout};
use crate::locals::ExtraLocals;

// ── TrapConfig ────────────────────────────────────────────────────────────────

/// The configuration object a recompiler embeds.
///
/// Holds an optional [`InstructionTrap`] and an optional [`JumpTrap`], each
/// with their associated [`ExtraParams`] (cross-function) and [`ExtraLocals`]
/// (per-function).  When no traps are installed, all methods are no-ops with
/// zero overhead.
///
/// ## Lifetimes
///
/// * `'cb` — lifetime of the borrowed trap implementations.
/// * `'ctx` — lifetime of any data the callbacks capture.
pub struct TrapConfig<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>> {
    // ── Instruction trap ──────────────────────────────────────────────────
    insn_trap:   Option<&'cb mut (dyn InstructionTrap<Context, E, F> + 'ctx)>,
    insn_params: ExtraParams,
    insn_locals: ExtraLocals,

    // ── Jump trap ─────────────────────────────────────────────────────────
    jump_trap:   Option<&'cb mut (dyn JumpTrap<Context, E, F> + 'ctx)>,
    jump_params: ExtraParams,
    jump_locals: ExtraLocals,
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
            insn_trap:   None,
            insn_params: ExtraParams::none(),
            insn_locals: ExtraLocals::none(),
            jump_trap:   None,
            jump_params: ExtraParams::none(),
            jump_locals: ExtraLocals::none(),
        }
    }

    // ── Installation ─────────────────────────────────────────────────────

    /// Install an instruction trap.
    ///
    /// Snapshots `trap.extra_params()` and `trap.extra_locals()`.  Parameter
    /// bases are assigned by the next call to [`setup`](Self::setup); local
    /// bases are assigned per-function by
    /// [`set_local_base`](Self::set_local_base).
    pub fn set_instruction_trap(
        &mut self,
        trap: &'cb mut (dyn InstructionTrap<Context, E, F> + 'ctx),
    ) {
        self.insn_params = trap.extra_params();
        self.insn_locals = trap.extra_locals();
        self.insn_trap   = Some(trap);
    }

    /// Install a jump trap.
    pub fn set_jump_trap(
        &mut self,
        trap: &'cb mut (dyn JumpTrap<Context, E, F> + 'ctx),
    ) {
        self.jump_params = trap.extra_params();
        self.jump_locals = trap.extra_locals();
        self.jump_trap   = Some(trap);
    }

    /// Remove the instruction trap.
    pub fn clear_instruction_trap(&mut self) {
        self.insn_trap   = None;
        self.insn_params = ExtraParams::none();
        self.insn_locals = ExtraLocals::none();
    }

    /// Remove the jump trap.
    pub fn clear_jump_trap(&mut self) {
        self.jump_trap   = None;
        self.jump_params = ExtraParams::none();
        self.jump_locals = ExtraLocals::none();
    }

    // ── Phase 1: setup ────────────────────────────────────────────────────

    /// **Phase 1** — append trap parameters to `layout` and return the total
    /// parameter count.
    ///
    /// Call this once after installing traps, before translation begins:
    ///
    /// ```ignore
    /// let mut layout = FunctionLayout::new(self.base_params());
    /// self.total_params = self.traps.setup(&mut layout);
    /// ```
    ///
    /// `setup` does three things for each installed trap:
    ///
    /// 1. Appends the trap's `extra_params()` groups to `layout`.
    /// 2. Sets the `ExtraParams` base on the stored snapshot so that
    ///    `ExtraParams::param(n)` returns the correct absolute index.
    /// 3. Advances an internal cursor so insn-trap params precede jump-trap
    ///    params in the layout.
    ///
    /// Returns `layout.total_params` (the recompiler's own params plus all
    /// trap params).  The recompiler must pass this value as `params` to
    /// every `jmp` / `ji` / `ji_with_params` call.
    pub fn setup(&mut self, layout: &mut FunctionLayout) -> u32 {
        // Insn-trap params come first (right after recompiler's own params).
        if self.insn_params.total_count() > 0 {
            let base = layout.total_params;
            self.insn_params.set_base(base);
            for g in self.insn_params.iter() {
                layout.extra_param_groups.push(g);
                layout.total_params += g.0;
            }
        }
        // Jump-trap params come after insn-trap params.
        if self.jump_params.total_count() > 0 {
            let base = layout.total_params;
            self.jump_params.set_base(base);
            for g in self.jump_params.iter() {
                layout.extra_param_groups.push(g);
                layout.total_params += g.0;
            }
        }
        layout.total_params
    }

    // ── Phase 2: per-function locals ──────────────────────────────────────

    /// **Phase 2a** — produce an iterator of all local `(count, ValType)`
    /// groups: the arch-defined groups first, then the trap-contributed groups.
    ///
    /// The result is passed directly to `reactor.next_with(ctx, f(&mut
    /// extended), depth)` inside `init_function`.  The iterator is lazy and
    /// chains without allocating.
    ///
    /// ```ignore
    /// let arch_iter = arch_locals.iter().copied();
    /// let extended  = self.traps.extend_locals(arch_iter);
    /// reactor.next_with(ctx, f(&mut extended), 2)?;
    /// ```
    pub fn extend_locals<I>(&self, arch: I)
        -> impl Iterator<Item = (u32, ValType)> + '_
    where
        I: Iterator<Item = (u32, ValType)> + 'static,
    {
        arch.chain(self.insn_locals.iter()).chain(self.jump_locals.iter())
    }

    /// **Phase 2b** — assign absolute local indices to trap non-param locals.
    ///
    /// `first_trap_local` is the index of the first trap-owned non-param local
    /// in the wasm function's local vector.  This equals
    /// `total_params + arch_non_param_local_count`.
    ///
    /// The insn-trap locals block starts at `first_trap_local`; the jump-trap
    /// locals block immediately follows.
    pub fn set_local_base(&mut self, first_trap_local: u32) {
        let insn_base = first_trap_local;
        let jump_base = insn_base + self.insn_locals.total_count();
        self.insn_locals.set_base(insn_base);
        self.jump_locals.set_base(jump_base);
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
        let mut trap_ctx = TrapContext::new(sink, &self.insn_locals, &self.insn_params);
        let action = trap.on_instruction(info, ctx, &mut trap_ctx)?;
        if action == TrapAction::Skip {
            let mut trap_ctx2 = TrapContext::new(sink, &self.insn_locals, &self.insn_params);
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
        let mut trap_ctx = TrapContext::new(sink, &self.jump_locals, &self.jump_params);
        let action = trap.on_jump(info, ctx, &mut trap_ctx)?;
        if action == TrapAction::Skip {
            let mut trap_ctx2 = TrapContext::new(sink, &self.jump_locals, &self.jump_params);
            self.jump_trap
                .as_ref()
                .unwrap()
                .skip_snippet(info, ctx, &mut trap_ctx2)?;
        }
        Ok(action)
    }

    // ── Accessors for trap params (used when emitting jump fixups) ─────────

    /// The `ExtraParams` block for the instruction trap.
    ///
    /// Recompilers that need to emit `local.get` / `local.set` for trap
    /// params at jump sites (e.g. to pass the ROP depth counter to the next
    /// function) can use these to retrieve absolute wasm local indices.
    pub fn insn_params(&self) -> &ExtraParams {
        &self.insn_params
    }

    /// The `ExtraParams` block for the jump trap.
    pub fn jump_params(&self) -> &ExtraParams {
        &self.jump_params
    }

    /// Install an instruction trap with an explicit `ExtraLocals` descriptor.
    ///
    /// Use this when installing a `Vec<Box<dyn InstructionTrap<…>>>` whose
    /// element-level locals have been pre-merged by [`merge_insn_trap_locals`].
    pub fn set_instruction_trap_with_locals(
        &mut self,
        trap: &'cb mut (dyn InstructionTrap<Context, E, F> + 'ctx),
        locals: ExtraLocals,
    ) {
        self.insn_params = trap.extra_params();
        self.insn_locals = locals;
        self.insn_trap   = Some(trap);
    }

    /// Install a jump trap with an explicit `ExtraLocals` descriptor.
    pub fn set_jump_trap_with_locals(
        &mut self,
        trap: &'cb mut (dyn JumpTrap<Context, E, F> + 'ctx),
        locals: ExtraLocals,
    ) {
        self.jump_params = trap.extra_params();
        self.jump_locals = locals;
        self.jump_trap   = Some(trap);
    }
}

// ── Vec helpers ───────────────────────────────────────────────────────────────

/// Merge the `extra_locals()` of every element in a slice of boxed
/// instruction traps into a single flat `ExtraLocals`.
pub fn merge_insn_trap_locals<Context, E, F: InstructionSink<Context, E>>(
    traps: &[Box<dyn InstructionTrap<Context, E, F> + '_>],
) -> ExtraLocals {
    let groups: Vec<(u32, ValType)> = traps
        .iter()
        .flat_map(|t: &Box<dyn InstructionTrap<Context, E, F> + '_>| {
            t.extra_locals().iter().collect::<Vec<_>>()
        })
        .collect();
    ExtraLocals::new(groups)
}

/// Merge the `extra_locals()` of every element in a slice of boxed jump
/// traps into a single flat `ExtraLocals`.
pub fn merge_jump_trap_locals<Context, E, F: InstructionSink<Context, E>>(
    traps: &[Box<dyn JumpTrap<Context, E, F> + '_>],
) -> ExtraLocals {
    let groups: Vec<(u32, ValType)> = traps
        .iter()
        .flat_map(|t: &Box<dyn JumpTrap<Context, E, F> + '_>| {
            t.extra_locals().iter().collect::<Vec<_>>()
        })
        .collect();
    ExtraLocals::new(groups)
}
