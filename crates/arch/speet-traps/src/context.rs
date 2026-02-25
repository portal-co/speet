//! [`TrapContext`] — the execution environment granted to a trap when it fires.
//!
//! A `TrapContext` is constructed by [`TrapConfig`] immediately before calling
//! a trap method and dropped immediately after.  It is the *only* surface area
//! a trap implementation touches during translation; all code emission,
//! local-index resolution, and control-flow requests go through it.
//!
//! ## Capabilities
//!
//! | Method | What it does |
//! |--------|-------------|
//! | [`emit`] | Emit a single wasm instruction into the current function |
//! | [`locals`] | Read-only access to this trap's [`ExtraLocals`] (non-param, per-function) |
//! | [`params`] | Read-only access to this trap's [`ExtraParams`] (params, cross-function) |
//! | [`jump`] | Emit an unconditional jump to a wasm function index |
//! | [`jump_if`] | Emit a conditional jump (consumes the top `i32` on the stack) |
//!
//! ## Parameter vs. local distinction
//!
//! Wasm function parameters are the first locals (indices 0..params-1) and
//! survive `return_call` chains.  `TrapContext::params()` gives the trap
//! access to its slice of that range via [`ExtraParams`].
//!
//! Non-parameter locals (indices ≥ params) are reset to zero on each new
//! function.  `TrapContext::locals()` gives access to those via
//! [`ExtraLocals`].
//!
//! ## Jump semantics
//!
//! `jump` and `jump_if` forward to `Reactor::jmp` when the underlying sink is
//! a `Reactor`.  For other sinks (e.g. a bare `wasm_encoder::Function`) the
//! methods emit `unreachable` as a safe fallback.
//!
//! The `params` argument to both jump methods is the number of wasm function
//! parameters to forward to the target function — the same meaning as in
//! `Reactor::jmp`.  In practice this is always `total_params` from
//! [`FunctionLayout`](crate::layout::FunctionLayout).

use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use yecta::FuncIdx;

use crate::layout::ExtraParams;
use crate::locals::ExtraLocals;

use core::marker::PhantomData;

// ── TrapContext ───────────────────────────────────────────────────────────────

/// The execution environment available to a trap when it fires.
///
/// See the [module documentation](self) for a usage overview.
pub struct TrapContext<'a, Context, E, F: InstructionSink<Context, E>> {
    /// The underlying instruction sink — usually a `&mut Reactor<…>`.
    pub sink: &'a mut F,
    /// Non-param extra locals for this trap.
    locals: &'a ExtraLocals,
    /// Extra parameters for this trap (cross-function state).
    params: &'a ExtraParams,
    _pd: PhantomData<fn(&mut Context) -> Result<(), E>>,
}

impl<'a, Context, E, F: InstructionSink<Context, E>> TrapContext<'a, Context, E, F> {
    /// Construct a `TrapContext`.
    ///
    /// Only [`TrapConfig`] should call this — traps receive one as a `&mut`
    /// argument.
    pub(crate) fn new(
        sink: &'a mut F,
        locals: &'a ExtraLocals,
        params: &'a ExtraParams,
    ) -> Self {
        Self { sink, locals, params, _pd: PhantomData }
    }

    /// Emit a single wasm instruction into the current function.
    #[inline]
    pub fn emit(&mut self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E> {
        self.sink.instruction(ctx, instr)
    }

    /// Read-only access to this trap's [`ExtraLocals`] layout (non-param,
    /// per-function).
    ///
    /// Use [`ExtraLocals::local`] to obtain absolute wasm local indices.
    #[inline]
    pub fn locals(&self) -> &ExtraLocals {
        self.locals
    }

    /// Read-only access to this trap's [`ExtraParams`] layout (cross-function
    /// parameter locals).
    ///
    /// Use [`ExtraParams::param`] to obtain absolute wasm local indices.
    #[inline]
    pub fn extra_params(&self) -> &ExtraParams {
        self.params
    }
}

// ── Jump support ──────────────────────────────────────────────────────────────

impl<'a, Context, E, F: InstructionSink<Context, E>> TrapContext<'a, Context, E, F> {
    /// Emit an **unconditional jump** to `target`, forwarding `params`
    /// parameters.
    ///
    /// When `F` is a `Reactor`, this delegates to `Reactor::jmp` which handles
    /// the full predecessor-graph bookkeeping.  For any other `F`, `unreachable`
    /// is emitted as a safe fallback.
    pub fn jump(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        self.sink.instruction(ctx, &Instruction::Unreachable)?;
        let _ = (target, params);
        Ok(())
    }

    /// Emit a **conditional jump** to `target` if the top `i32` on the wasm
    /// stack is non-zero, consuming that value.  Falls through otherwise.
    ///
    /// Emits:
    /// ```text
    /// if
    ///   [unconditional jump to target / unreachable fallback]
    /// end
    /// ```
    pub fn jump_if(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        self.sink.instruction(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
        self.jump(ctx, target, params)?;
        self.sink.instruction(ctx, &Instruction::End)?;
        Ok(())
    }
}

// ── Specialised helpers for Reactor sinks ─────────────────────────────────────

/// Emit an unconditional jump via `Reactor::jmp` on a `TrapContext` whose
/// sink is known to be a `Reactor`.
pub fn reactor_jump<Context, E, F>(
    trap_ctx: &mut TrapContext<Context, E, yecta::Reactor<Context, E, F>>,
    ctx: &mut Context,
    target: FuncIdx,
    params: u32,
) -> Result<(), E>
where
    F: wax_core::build::InstructionSink<Context, E>,
    yecta::Reactor<Context, E, F>: InstructionSink<Context, E>,
{
    trap_ctx.sink.jmp(ctx, target, params)
}

/// Emit a conditional jump via a wasm `if` block wrapping `Reactor::jmp`.
///
/// Consumes the top `i32` on the stack as the branch condition.
pub fn reactor_jump_if<Context, E, F>(
    trap_ctx: &mut TrapContext<Context, E, yecta::Reactor<Context, E, F>>,
    ctx: &mut Context,
    target: FuncIdx,
    params: u32,
) -> Result<(), E>
where
    F: wax_core::build::InstructionSink<Context, E>,
    yecta::Reactor<Context, E, F>: InstructionSink<Context, E>,
{
    trap_ctx.sink.instruction(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
    trap_ctx.sink.jmp(ctx, target, params)?;
    trap_ctx.sink.instruction(ctx, &Instruction::End)
}
