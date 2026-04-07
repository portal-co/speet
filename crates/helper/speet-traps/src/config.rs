//! [`TrapConfig`] — the single object a recompiler embeds and delegates to.
//!
//! ## Overview
//!
//! A recompiler (e.g. `RiscVRecompiler`) embeds one `TrapConfig` field and
//! calls methods at three phases.  The arch recompiler **owns** the
//! [`LocalLayout`] and passes it to `TrapConfig`; `TrapConfig` never owns a
//! layout itself.
//!
//! ### Phase 1 — parameter setup (once per recompiler instance)
//!
//! After appending all arch parameter groups to the layout, the recompiler
//! calls `traps.declare_params(&mut layout)` so each installed trap can append
//! its own parameter groups.  The recompiler then places a [`Mark`] on the
//! layout to capture `total_params`:
//!
//! ```ignore
//! let mut layout = LocalLayout::empty();
//! // Arch params (32 int regs, PC, …):
//! let regs = layout.append(32, ValType::I64);
//! let pc   = layout.append(1,  ValType::I32);
//! // Trap params:
//! self.traps.declare_params(&mut layout);
//! self.locals_mark = layout.mark();            // total_locals == total_params
//! self.total_params = self.locals_mark.total_locals;
//! ```
//!
//! ### Phase 2 — per-function local setup (in `init_function`)
//!
//! At the start of each new function, after rewinding and appending arch
//! non-param locals, the recompiler calls `traps.declare_locals(&mut layout)`:
//!
//! ```ignore
//! layout.rewind(&self.locals_mark);
//! let temps = layout.append(num_temps, ValType::I64);
//! let pool  = layout.append(N_POOL, ValType::I32);
//! self.traps.declare_locals(&mut layout);
//! reactor.next_with(ctx, f(&mut layout.iter_since(&self.locals_mark)), depth)?;
//! ```
//!
//! ### Phase 3 — firing
//!
//! At each instruction and jump site the recompiler calls `on_instruction` /
//! `on_jump`, passing a shared `&LocalLayout` so traps can resolve their slots:
//!
//! ```ignore
//! // at translate_instruction start:
//! if self.traps.on_instruction(&info, ctx, &mut reactor, &self.layout)? == TrapAction::Skip {
//!     return Ok(());
//! }
//! // at each jump site:
//! if self.traps.on_jump(&info, ctx, &mut reactor, &self.layout)? == TrapAction::Skip {
//!     return Ok(());
//! }
//! ```
//!
//! ## Local / parameter layout
//!
//! ```text
//! local 0           … arch_params-1              arch params (regs, PC, …)
//! local arch_params … total_params-1             trap params (depth counter, …)
//! local total_params … total_params+arch_locals-1  arch non-param locals
//! local total_params+arch_locals … (end)         trap non-param locals
//! ```
//!
//! The layout is owned by the arch recompiler and shared (read-only) with
//! traps through [`TrapContext::layout`].

use yecta::layout::CellIdx;
use yecta::{EmitSink, LocalAllocator, LocalLayout};

use crate::context::TrapContext;
use crate::insn::{InstructionInfo, InstructionTrap, TrapAction};
use crate::jump::{JumpInfo, JumpTrap};

// ── TrapConfig ────────────────────────────────────────────────────────────────

/// The configuration object a recompiler embeds.
///
/// Holds an optional [`InstructionTrap`] and an optional [`JumpTrap`].  All
/// methods are no-ops when no trap is installed.  The arch recompiler owns
/// the [`LocalLayout`] and passes it when calling the declare and firing
/// methods.
///
/// ## Lifetimes
///
/// * `'cb` — lifetime of the borrowed trap implementations.
/// * `'ctx` — lifetime of any data the callbacks capture.
pub struct TrapConfig<'cb, 'ctx, Context, E> {
    insn_trap: Option<&'cb mut (dyn InstructionTrap<Context, E> + 'ctx)>,
    jump_trap: Option<&'cb mut (dyn JumpTrap<Context, E> + 'ctx)>,
}

impl<'cb, 'ctx, Context, E> Default for TrapConfig<'cb, 'ctx, Context, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'cb, 'ctx, Context, E> TrapConfig<'cb, 'ctx, Context, E> {
    /// Create an empty `TrapConfig` with no traps installed.
    pub fn new() -> Self {
        Self {
            insn_trap: None,
            jump_trap: None,
        }
    }

    // ── Installation ──────────────────────────────────────────────────────

    /// Install an instruction trap.
    ///
    /// After installing all traps the recompiler must call
    /// [`declare_params`](Self::declare_params) and later (per function)
    /// [`declare_locals`](Self::declare_locals), passing its owned
    /// [`LocalLayout`] both times.
    pub fn set_instruction_trap(
        &mut self,
        trap: &'cb mut (dyn InstructionTrap<Context, E> + 'ctx),
    ) {
        self.insn_trap = Some(trap);
    }

    /// Install a jump trap.
    ///
    /// See [`set_instruction_trap`](Self::set_instruction_trap) for the
    /// post-install protocol.
    pub fn set_jump_trap(&mut self, trap: &'cb mut (dyn JumpTrap<Context, E> + 'ctx)) {
        self.jump_trap = Some(trap);
    }

    /// Remove the instruction trap.
    ///
    /// After clearing a trap, the recompiler must rebuild the layout from
    /// scratch (re-append arch params, re-call `declare_params`, re-mark) to
    /// ensure correct indices for any remaining traps.
    pub fn clear_instruction_trap(&mut self) {
        self.insn_trap = None;
    }

    /// Remove the jump trap.
    ///
    /// See [`clear_instruction_trap`](Self::clear_instruction_trap) for the
    /// re-setup requirement.
    pub fn clear_jump_trap(&mut self) {
        self.jump_trap = None;
    }

    // ── Phase 1: declare params ───────────────────────────────────────────

    /// **Phase 1** — let each installed trap append its parameter groups to
    /// `layout`.
    ///
    /// Call this after the arch recompiler has appended its own parameter
    /// groups, before placing the [`Mark`](yecta::Mark).  Each trap will call
    /// [`LocalLayout::append`] for its parameter groups and store the returned
    /// [`LocalSlot`](yecta::LocalSlot) handles in its own fields.
    pub fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        if let Some(t) = self.insn_trap.as_mut() {
            t.declare_params(cell, layout);
        }
        if let Some(t) = self.jump_trap.as_mut() {
            t.declare_params(cell, layout);
        }
    }

    // ── Phase 2: declare locals ───────────────────────────────────────────

    /// **Phase 2** — let each installed trap append its non-param local groups
    /// to `layout`.
    ///
    /// Call this after the arch recompiler has appended its own per-function
    /// local groups (temps, pool slots, etc.) and before calling
    /// `reactor.next_with`.  Each trap calls [`LocalLayout::append`] for its
    /// local groups and stores the returned handles.
    pub fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        if let Some(t) = self.insn_trap.as_mut() {
            t.declare_locals(cell, layout);
        }
        if let Some(t) = self.jump_trap.as_mut() {
            t.declare_locals(cell, layout);
        }
    }

    // ── Phase 3: firing ───────────────────────────────────────────────────

    /// Fire the instruction trap (if installed).
    ///
    /// `sink` is the emission sink (usually `&mut Reactor<…>` coerced to
    /// `&mut dyn EmitSink<Context, E>`).  `layout` is the arch recompiler's
    /// unified layout; both are passed to [`TrapContext`] so traps can emit
    /// instructions and resolve their [`LocalSlot`](yecta::LocalSlot) handles.
    ///
    /// Returns [`TrapAction::Continue`] when no trap is installed.
    pub fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        sink: &mut dyn EmitSink<Context, E>,
        layout: &dyn LocalAllocator,
    ) -> Result<TrapAction, E> {
        let trap = match self.insn_trap.as_mut() {
            Some(t) => t,
            None => return Ok(TrapAction::Continue),
        };
        let mut trap_ctx = TrapContext::new(sink, layout);
        let action = trap.on_instruction(info, ctx, &mut trap_ctx)?;
        if action == TrapAction::Skip {
            let mut trap_ctx2 = TrapContext::new(sink, layout);
            self.insn_trap
                .as_ref()
                .unwrap()
                .skip_snippet(info, ctx, &mut trap_ctx2)?;
        }
        Ok(action)
    }

    /// Fire the jump trap (if installed).
    ///
    /// `sink` and `layout` are the same as for [`on_instruction`](Self::on_instruction).
    ///
    /// Returns [`TrapAction::Continue`] when no trap is installed.
    pub fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        sink: &mut dyn EmitSink<Context, E>,
        layout: &dyn LocalAllocator,
    ) -> Result<TrapAction, E> {
        let trap = match self.jump_trap.as_mut() {
            Some(t) => t,
            None => return Ok(TrapAction::Continue),
        };
        let mut trap_ctx = TrapContext::new(sink, layout);
        let action = trap.on_jump(info, ctx, &mut trap_ctx)?;
        if action == TrapAction::Skip {
            let mut trap_ctx2 = TrapContext::new(sink, layout);
            self.jump_trap
                .as_ref()
                .unwrap()
                .skip_snippet(info, ctx, &mut trap_ctx2)?;
        }
        Ok(action)
    }
}
