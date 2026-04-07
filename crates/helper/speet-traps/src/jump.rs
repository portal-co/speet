//! [`JumpTrap`] — per-control-flow-transfer trap hook.
//!
//! A `JumpTrap` fires immediately before every control-flow transfer the
//! recompiler is about to emit: unconditional branches, conditional branches,
//! calls, returns, indirect jumps, and system calls.
//!
//! ## Firing order at a jump site
//!
//! ```text
//! <instruction body up to the jump>
//! TrapConfig::on_jump(…)    ← JumpTrap fires HERE
//! if TrapAction::Continue:
//!     reactor.jmp / ji / ji_with_params(…)
//! else (TrapAction::Skip):
//!     JumpTrap::skip_snippet(…)   ← replaces the jump entirely
//! ```
//!
//! ## Target local
//!
//! For indirect jumps the computed target is not statically known.  Before
//! calling `on_jump`, the recompiler places the runtime target value into a
//! dedicated wasm local and reports its index in
//! [`JumpInfo::target_local`].  The trap can read this local to inspect or
//! validate the target (e.g. for ROP detection or CFI).
//!
//! For static jumps `target_local` is `None`; the static target is available
//! as [`JumpInfo::target_pc`].
//!
//! ## Local / parameter declaration
//!
//! Same protocol as [`InstructionTrap`](crate::insn::InstructionTrap):
//! override [`declare_locals`](JumpTrap::declare_locals) and/or
//! [`declare_params`](JumpTrap::declare_params), append groups via
//! [`LocalLayout::append`], store the returned [`LocalSlot`] handles in the
//! trap struct, and later use `trap_ctx.locals().local(slot, n)` and
//! `trap_ctx.params().local(slot, n)` inside `on_jump` / `skip_snippet`.
//!
//! ## Blanket impl for closures
//!
//! Any `FnMut(&JumpInfo, &mut Context, &mut TrapContext<…>) -> Result<TrapAction, E>`
//! closure automatically implements `JumpTrap`.

use alloc::boxed::Box;
use wasm_encoder::Instruction;
use yecta::layout::CellIdx;
use yecta::{LocalDeclarator, LocalLayout};

use crate::context::TrapContext;
use crate::insn::TrapAction;

// ── JumpKind ──────────────────────────────────────────────────────────────────

/// The kind of control-flow transfer at a jump site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JumpKind {
    /// Unconditional direct branch / jump.
    DirectJump,
    /// Conditional branch (one of two paths; trap fires for the taken path).
    ConditionalBranch,
    /// Subroutine call saving a return address into a link register.
    Call,
    /// Return from subroutine (consumes the link register / stack return addr).
    Return,
    /// Indirect branch: target computed from a register at runtime.
    IndirectJump,
    /// Indirect call: target computed at runtime, link register written.
    IndirectCall,
    /// System call / environment call (ECALL, SYSCALL, INT, …).
    Syscall,
}

// ── JumpInfo ──────────────────────────────────────────────────────────────────

/// Metadata about the control-flow transfer that fired the trap.
pub struct JumpInfo {
    /// Guest PC of the instruction that produced this transfer.
    pub source_pc: u64,
    /// Static target address, if known.  `None` for indirect jumps.
    pub target_pc: Option<u64>,
    /// Wasm local holding the runtime target address (for indirect jumps).
    /// `None` if the target is statically known.
    pub target_local: Option<u32>,
    /// The kind of control-flow transfer.
    pub kind: JumpKind,
}

impl JumpInfo {
    /// Convenience constructor for a direct (static-target) jump.
    pub fn direct(source_pc: u64, target_pc: u64, kind: JumpKind) -> Self {
        Self {
            source_pc,
            target_pc: Some(target_pc),
            target_local: None,
            kind,
        }
    }

    /// Convenience constructor for an indirect (dynamic-target) jump.
    pub fn indirect(source_pc: u64, target_local: u32, kind: JumpKind) -> Self {
        Self {
            source_pc,
            target_pc: None,
            target_local: Some(target_local),
            kind,
        }
    }
}

// ── JumpTrap trait ────────────────────────────────────────────────────────────

/// Fires immediately before each control-flow transfer.
///
/// See the [module documentation](self) for the full description of when this
/// fires and what the trap can do.
///
/// # Type parameters
///
/// * `Context` — the recompiler's user context type.
/// * `E` — the error type returned by the recompiler's instruction sink.
pub trait JumpTrap<Context, E>: LocalDeclarator {
    /// Called immediately before a control-flow transfer is emitted.
    ///
    /// Return [`TrapAction::Continue`] to let the jump proceed normally, or
    /// [`TrapAction::Skip`] to suppress it and invoke
    /// [`skip_snippet`](Self::skip_snippet) instead.
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E>,
    ) -> Result<TrapAction, E>;

    /// Code to emit in place of the suppressed jump when this trap returns
    /// [`TrapAction::Skip`].
    ///
    /// The default emits `unreachable`.  Override to redirect to a violation
    /// handler or emit a diagnostic sequence.
    fn skip_snippet(
        &self,
        info: &JumpInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E>,
    ) -> Result<(), E> {
        let _ = info;
        skip_ctx.emit(ctx, &Instruction::Unreachable)
    }
}

// ── Blanket impl for FnMut closures ──────────────────────────────────────────

impl<Context, E, Fn> JumpTrap<Context, E> for Fn
where
    Fn: FnMut(&JumpInfo, &mut Context, &mut TrapContext<Context, E>) -> Result<TrapAction, E>
        + LocalDeclarator,
{
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E>,
    ) -> Result<TrapAction, E> {
        self(info, ctx, trap_ctx)
    }
}

/// `Box<dyn JumpTrap<…>>` delegates to the inner value.
impl<Context, E> LocalDeclarator for Box<dyn JumpTrap<Context, E> + '_> {
    fn declare_params(&mut self, cell: CellIdx, params: &mut LocalLayout) {
        (**self).declare_params(cell, params);
    }

    fn declare_locals(&mut self, cell: CellIdx, locals: &mut LocalLayout) {
        (**self).declare_locals(cell, locals);
    }
}

impl<Context, E> JumpTrap<Context, E> for Box<dyn JumpTrap<Context, E> + '_> {
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E>,
    ) -> Result<TrapAction, E> {
        (**self).on_jump(info, ctx, trap_ctx)
    }

    fn skip_snippet(
        &self,
        info: &JumpInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E>,
    ) -> Result<(), E> {
        (**self).skip_snippet(info, ctx, skip_ctx)
    }
}
