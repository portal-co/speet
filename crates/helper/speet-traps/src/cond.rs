//! [`ConditionTrap`] — low-level conditional-branch hook.
//!
//! A `ConditionTrap` fires immediately *after* the condition `i32` has been
//! pushed onto the WASM stack by the condition snippet, but *before* the `if`
//! (or `br_if`) instruction is emitted.
//!
//! ## Difference from `JumpTrap`
//!
//! [`JumpTrap`](crate::jump::JumpTrap) fires before the condition snippet runs,
//! so it cannot observe or transform the condition value.  `ConditionTrap`
//! fires after the condition is on the stack, enabling:
//!
//! - **Logging** — `local.tee` the condition and call a WASM import.
//! - **Flipping** — emit `i32.eqz` to force the opposite branch.
//! - **Runtime backtracking** — call a host `decide(i32) -> i32` import that
//!   can override the branch decision for replay/fuzzing.
//!
//! ## Contract
//!
//! The trap's [`on_condition`](ConditionTrap::on_condition) method may emit
//! any sequence of WASM instructions, but **must** leave exactly one `i32`
//! value on the stack when it returns.  That value becomes the new condition
//! for the `if`/`br_if`.  An identity implementation emits nothing; a flip
//! implementation emits a single `i32.eqz`.
//!
//! ## Integration with yecta
//!
//! The arch recompiler creates a [`ConditionHookWrapper`] from a
//! `&dyn ConditionTrap` and a [`ConditionInfo`], then attaches it to
//! [`yecta::JumpCallParams::condition_hook`].  yecta calls the wrapper
//! (as a [`yecta::Snippet`]) between the condition snippet and the `if`.
//!
//! ## Integration with the WASM frontend
//!
//! `speet-wasm`'s `WasmFrontend` stores an optional `Box<dyn ConditionTrap>`
//! and calls `on_condition` before emitting `Operator::If` and
//! `Operator::BrIf`.

use alloc::boxed::Box;
use wasm_encoder::Instruction;
use wax_core::build::{InstructionOperatorSink, InstructionOperatorSource, InstructionSink, InstructionSource};
use yecta::layout::CellIdx;
use yecta::{LocalDeclarator, LocalLayout};

use crate::context::TrapContext;

// ── ConditionInfo ─────────────────────────────────────────────────────────────

/// Metadata passed to [`ConditionTrap::on_condition`].
pub struct ConditionInfo {
    /// Guest PC of the instruction whose condition is being evaluated.
    /// `0` for managed-bytecode contexts (WASM frontend) where the guest PC
    /// is not tracked.
    pub source_pc: u64,
    /// Static target address of the taken branch, if statically known.
    pub target_pc: Option<u64>,
}

// ── ConditionTrap trait ───────────────────────────────────────────────────────

/// Fires after the condition `i32` is on the WASM stack, before `if`/`br_if`.
///
/// See the [module documentation](self) for the full description of when this
/// fires and what the trap can do.
///
/// # Type parameters
///
/// * `Context` — the recompiler's user context type.
/// * `E` — the error type returned by the recompiler's instruction sink.
///
/// # `&self` vs `&mut self`
///
/// `on_condition` takes `&self` (not `&mut self`).  All local-index state is
/// fixed at declare-time; no Rust-side mutation is needed during emission.
/// This allows the trap to be wrapped as a [`yecta::Snippet`] via
/// [`ConditionHookWrapper`].  If you need mutable Rust state during emission,
/// use a `RefCell` inside your trap struct.
pub trait ConditionTrap<Context, E>: LocalDeclarator {
    /// Called after the condition `i32` is on the WASM stack.
    ///
    /// Emit WASM instructions by calling `go(ctx, &instruction)` for each one.
    /// The net effect must be that exactly one `i32` remains on the stack.
    fn on_condition(
        &self,
        info: &ConditionInfo,
        ctx: &mut Context,
        go: &mut (dyn FnMut(&mut Context, &Instruction<'_>) -> Result<(), E> + '_),
    ) -> Result<(), E>;
}

// ── ConditionHookWrapper ──────────────────────────────────────────────────────

/// Bridges a [`ConditionTrap`] into yecta's [`Snippet`](yecta::Snippet) API.
///
/// Create one per conditional-jump site via
/// [`TrapConfig::make_condition_hook`](crate::config::TrapConfig::make_condition_hook),
/// then pass it to
/// [`JumpCallParams::with_condition_hook`](yecta::JumpCallParams::with_condition_hook).
pub struct ConditionHookWrapper<'t, Context, E> {
    /// Metadata about the branch site.
    pub info: ConditionInfo,
    /// The installed condition trap.
    pub trap: &'t (dyn ConditionTrap<Context, E> + 't),
}

impl<Context, E> InstructionOperatorSource<Context, E> for ConditionHookWrapper<'_, Context, E> {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        self.trap
            .on_condition(&self.info, ctx, &mut |c, i| sink.instruction(c, i))
    }
}

impl<Context, E> InstructionSource<Context, E> for ConditionHookWrapper<'_, Context, E> {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        self.trap
            .on_condition(&self.info, ctx, &mut |c, i| sink.instruction(c, i))
    }
}

// ── Blanket impl for closures ─────────────────────────────────────────────────

/// A `FnMut` closure that takes `(info, ctx, go)` automatically implements
/// `ConditionTrap` if it also implements `LocalDeclarator`.
///
/// In practice this means you can write a zero-local condition trap as a
/// closure that calls `TrapContext::emit` directly, without declaring a struct.
impl<Context, E, F> ConditionTrap<Context, E> for F
where
    F: for<'go> Fn(
            &ConditionInfo,
            &mut Context,
            &mut (dyn FnMut(&mut Context, &Instruction<'_>) -> Result<(), E> + 'go),
        ) -> Result<(), E>
        + LocalDeclarator,
{
    fn on_condition(
        &self,
        info: &ConditionInfo,
        ctx: &mut Context,
        go: &mut (dyn FnMut(&mut Context, &Instruction<'_>) -> Result<(), E> + '_),
    ) -> Result<(), E> {
        self(info, ctx, go)
    }
}

// ── Box<dyn ConditionTrap> delegation ────────────────────────────────────────

impl<Context, E> LocalDeclarator for Box<dyn ConditionTrap<Context, E> + '_> {
    fn declare_params(&mut self, cell: CellIdx, params: &mut LocalLayout) {
        (**self).declare_params(cell, params);
    }

    fn declare_locals(&mut self, cell: CellIdx, locals: &mut LocalLayout) {
        (**self).declare_locals(cell, locals);
    }
}

impl<Context, E> ConditionTrap<Context, E> for Box<dyn ConditionTrap<Context, E> + '_> {
    fn on_condition(
        &self,
        info: &ConditionInfo,
        ctx: &mut Context,
        go: &mut (dyn FnMut(&mut Context, &Instruction<'_>) -> Result<(), E> + '_),
    ) -> Result<(), E> {
        (**self).on_condition(info, ctx, go)
    }
}

// ── ConditionTrap impl for TrapContext passthrough ────────────────────────────

/// Helper used by `TrapConfig::on_condition_direct`: fire the condition trap
/// using a [`TrapContext`] rather than the raw `go` closure.
///
/// This is the integration point for the WASM frontend, which already has a
/// `TrapContext`-compatible `EmitSink`.
pub(crate) fn fire_via_trap_ctx<Context, E>(
    trap: &dyn ConditionTrap<Context, E>,
    info: &ConditionInfo,
    ctx: &mut Context,
    trap_ctx: &mut TrapContext<'_, Context, E>,
) -> Result<(), E> {
    trap.on_condition(info, ctx, &mut |c, i| trap_ctx.emit(c, i))
}
