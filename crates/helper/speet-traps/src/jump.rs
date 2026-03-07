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
//! ## `TrapAction::Skip`
//!
//! Returning [`TrapAction::Skip`] suppresses the original jump.  The
//! recompiler instead calls [`JumpTrap::skip_snippet`] which may emit any
//! code — typically a jump to a violation handler.  If `skip_snippet` emits
//! no instructions, `unreachable` is used as the sole terminator so the wasm
//! function remains valid.
//!
//! ## Blanket impl for closures
//!
//! Any `FnMut(&JumpInfo, &mut Context, &mut TrapContext<…>) -> Result<TrapAction, E>`
//! closure automatically implements `JumpTrap`.

use alloc::{boxed::Box, vec::Vec};
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;

use crate::context::TrapContext;
use crate::insn::TrapAction;
use crate::layout::ExtraParams;
use crate::locals::ExtraLocals;

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

/// Metadata about the control-flow transfer being emitted.
///
/// Passed by the recompiler to [`TrapConfig::on_jump`] and forwarded to the
/// active [`JumpTrap`].
#[derive(Clone, Debug)]
pub struct JumpInfo {
    /// Guest PC of the instruction performing the jump.
    pub source_pc: u64,

    /// Statically-known target guest PC, if available.
    ///
    /// `None` for indirect jumps (target computed at runtime).
    pub target_pc: Option<u64>,

    /// Wasm local index holding the **runtime** target value.
    ///
    /// * For **indirect** jumps this is always `Some(local)`, where `local`
    ///   holds the runtime target (typically a guest PC or function index)
    ///   as an `i32` or `i64` depending on the address width.  The recompiler
    ///   ensures this local is populated with a `local.tee` before calling
    ///   `on_jump`.
    ///
    /// * For **direct** jumps this is `None`; the static target is in
    ///   `target_pc`.
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
/// Same as [`InstructionTrap`](crate::insn::InstructionTrap).
pub trait JumpTrap<Context, E, F: InstructionSink<Context, E>> {
    /// Called immediately before a control-flow transfer is emitted.
    ///
    /// Return [`TrapAction::Continue`] to let the jump proceed normally, or
    /// [`TrapAction::Skip`] to suppress it and invoke
    /// [`skip_snippet`](Self::skip_snippet) instead.
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E>;

    /// Extra wasm locals needed per function.  Default: none.
    fn extra_locals(&self) -> ExtraLocals {
        ExtraLocals::none()
    }

    /// Extra wasm **parameters** needed per function group.
    ///
    /// Parameters survive `return_call` — use them for state that must carry
    /// over from one translated-instruction function to the next.
    /// Default: none.
    fn extra_params(&self) -> ExtraParams {
        ExtraParams::none()
    }

    /// Code to emit in place of the suppressed jump when this trap returns
    /// [`TrapAction::Skip`].
    ///
    /// The default emits `unreachable`.  Override to redirect to a violation
    /// handler or emit a diagnostic sequence.
    fn skip_snippet(
        &self,
        info: &JumpInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<(), E> {
        let _ = info;
        skip_ctx.emit(ctx, &Instruction::Unreachable)
    }
}

// ── Blanket impl for FnMut closures ──────────────────────────────────────────

impl<Context, E, F, Fn> JumpTrap<Context, E, F> for Fn
where
    F: InstructionSink<Context, E>,
    Fn: FnMut(&JumpInfo, &mut Context, &mut TrapContext<Context, E, F>)
            -> Result<TrapAction, E>,
{
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        self(info, ctx, trap_ctx)
    }
}

// ── Vec<Box<dyn JumpTrap>> ────────────────────────────────────────────────────

/// `Vec<Box<dyn JumpTrap<…>>>` runs each element in order, short-circuiting
/// on the first [`TrapAction::Skip`].
///
/// See the corresponding note on `Vec<Box<dyn InstructionTrap<…>>>` regarding
/// `extra_locals`: the vec impl returns [`ExtraLocals::none`]; per-element
/// locals are managed by [`TrapConfig`].
impl<Context, E, F> JumpTrap<Context, E, F>
    for Vec<Box<dyn JumpTrap<Context, E, F> + '_>>
where
    F: InstructionSink<Context, E>,
{
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        for trap in self.iter_mut() {
            if trap.on_jump(info, ctx, trap_ctx)? == TrapAction::Skip {
                return Ok(TrapAction::Skip);
            }
        }
        Ok(TrapAction::Continue)
    }
}

/// `Box<dyn JumpTrap<…>>` delegates to the inner value.
impl<Context, E, F> JumpTrap<Context, E, F>
    for Box<dyn JumpTrap<Context, E, F> + '_>
where
    F: InstructionSink<Context, E>,
{
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        (**self).on_jump(info, ctx, trap_ctx)
    }

    fn extra_locals(&self) -> ExtraLocals {
        (**self).extra_locals()
    }

    fn extra_params(&self) -> ExtraParams {
        (**self).extra_params()
    }

    fn skip_snippet(
        &self,
        info: &JumpInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<(), E> {
        (**self).skip_snippet(info, ctx, skip_ctx)
    }
}
