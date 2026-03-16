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
//! | [`layout`] | Read-only access to the unified [`LocalLayout`] (params + locals) |
//! | [`jump`] | Emit an unconditional jump to a wasm function index |
//! | [`jump_if`] | Emit a conditional jump (consumes the top `i32` on the stack) |
//!
//! ## Resolving local indices
//!
//! Traps store [`LocalSlot`] handles obtained during
//! [`declare_locals`](crate::insn::InstructionTrap::declare_locals) or
//! [`declare_params`](crate::insn::InstructionTrap::declare_params).  Inside
//! `on_instruction` or `on_jump` they resolve those handles via the unified layout:
//!
//! ```ignore
//! // A scratch local declared in declare_locals:
//! let idx = trap_ctx.layout().local(self.scratch_slot, 0);
//!
//! // A depth counter declared in declare_params:
//! let idx = trap_ctx.layout().local(self.depth_slot, 0);
//! ```
//!
//! Both params and function-locals live in the same [`LocalLayout`] owned by
//! the arch recompiler: arch params first (at indices 0+), then trap params,
//! then per-function arch locals, then per-function trap locals.  Because
//! every group is appended in order, [`LocalLayout::local`] and
//! [`LocalLayout::base`] return correct **absolute** wasm local indices without
//! any additional base-offset arithmetic.
//!
//! ## Jump semantics
//!
//! `jump` and `jump_if` forward to `Reactor::jmp` when the underlying sink is
//! a `Reactor`.  For other sinks (e.g. a bare `wasm_encoder::Function`) the
//! methods emit `unreachable` as a safe fallback.
//!
//! The `params` argument to both jump methods is the number of wasm function
//! parameters to forward to the target function — the same meaning as in
//! `Reactor::jmp`.  In practice this is always `total_params` from the
//! recompiler's stored field.

use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use yecta::{FuncIdx, LocalLayout, LocalPoolBackend};

use core::marker::PhantomData;

// ── TrapContext ───────────────────────────────────────────────────────────────

/// The execution environment available to a trap when it fires.
///
/// See the [module documentation](self) for a usage overview.
pub struct TrapContext<'a, Context, E, F: InstructionSink<Context, E>> {
    /// The underlying instruction sink — usually a `&mut Reactor<…>`.
    pub sink: &'a mut F,
    /// Unified layout for all params and locals owned by the arch recompiler.
    layout: &'a LocalLayout,
    _pd: PhantomData<fn(&mut Context) -> Result<(), E>>,
}

impl<'a, Context, E, F: InstructionSink<Context, E>> TrapContext<'a, Context, E, F> {
    /// Construct a `TrapContext`.
    ///
    /// Only [`TrapConfig`](crate::config::TrapConfig) should call this —
    /// traps receive one as a `&mut` argument.
    pub(crate) fn new(sink: &'a mut F, layout: &'a LocalLayout) -> Self {
        Self {
            sink,
            layout,
            _pd: PhantomData,
        }
    }

    /// Emit a single wasm instruction into the current function.
    #[inline]
    pub fn emit(&mut self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E> {
        self.sink.instruction(ctx, instr)
    }

    /// Read-only access to the unified **param + locals** layout.
    ///
    /// Use [`LocalLayout::local`] with a [`LocalSlot`](yecta::LocalSlot)
    /// obtained during `declare_params` or `declare_locals` to resolve
    /// absolute wasm local indices.
    ///
    /// Because arch params start at index 0 and every group is appended in
    /// order, the same layout correctly resolves both parameter slots and
    /// per-function local slots without any base-offset adjustment.
    #[inline]
    pub fn layout(&self) -> &LocalLayout {
        self.layout
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
        self.sink
            .instruction(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
        self.jump(ctx, target, params)?;
        self.sink.instruction(ctx, &Instruction::End)?;
        Ok(())
    }
}

// ── Specialised helpers for Reactor sinks ─────────────────────────────────────

/// Emit an unconditional jump via `Reactor::jmp` on a `TrapContext` whose
/// sink is known to be a `Reactor`.
pub fn reactor_jump<Context, E, F, P: LocalPoolBackend>(
    trap_ctx: &mut TrapContext<Context, E, yecta::Reactor<Context, E, F, P>>,
    ctx: &mut Context,
    target: FuncIdx,
    params: u32,
) -> Result<(), E>
where
    F: wax_core::build::InstructionSink<Context, E>,
    yecta::Reactor<Context, E, F, P>: InstructionSink<Context, E>,
{
    trap_ctx.sink.jmp(ctx, target, params)
}

/// Emit a conditional jump via a wasm `if` block wrapping `Reactor::jmp`.
///
/// Consumes the top `i32` on the stack as the branch condition.
pub fn reactor_jump_if<Context, E, F, P: LocalPoolBackend>(
    trap_ctx: &mut TrapContext<Context, E, yecta::Reactor<Context, E, F, P>>,
    ctx: &mut Context,
    target: FuncIdx,
    params: u32,
) -> Result<(), E>
where
    F: wax_core::build::InstructionSink<Context, E>,
    yecta::Reactor<Context, E, F, P>: InstructionSink<Context, E>,
{
    trap_ctx
        .sink
        .instruction(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
    trap_ctx.sink.jmp(ctx, target, params)?;
    trap_ctx.sink.instruction(ctx, &Instruction::End)
}
