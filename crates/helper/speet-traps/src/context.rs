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
//! | [`layout`] | Read-only access to the unified layout (params + locals) |
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
//! Both params and function-locals live in the same layout owned by
//! the arch recompiler: arch params first (at indices 0+), then trap params,
//! then per-function arch locals, then per-function trap locals.  Because
//! every group is appended in order, [`LocalAllocator::local`] and
//! [`LocalAllocator::base`] return correct **absolute** wasm local indices without
//! any additional base-offset arithmetic.
//!
//! ## Jump semantics
//!
//! `jump` and `jump_if` call [`EmitSink::emit_jmp`] on the underlying sink.
//! For a `Reactor` sink this delegates to `Reactor::jmp`, which records the
//! target as a successor and emits the full `local.get … return_call` sequence
//! with predecessor-graph bookkeeping.
//!
//! The `params` argument to both jump methods is the number of wasm function
//! parameters to forward to the target function — the same meaning as in
//! `Reactor::jmp`.  In practice this is always `total_params` from the
//! recompiler's stored field.

use wasm_encoder::Instruction;
use yecta::{EmitSink, FuncIdx, LocalAllocator};

// ── TrapContext ───────────────────────────────────────────────────────────────

/// The execution environment available to a trap when it fires.
///
/// See the [module documentation](self) for a usage overview.
pub struct TrapContext<'a, Context, E> {
    /// The underlying emission sink — erased to [`EmitSink`] so traps need
    /// not be generic over the concrete sink type.
    sink: &'a mut dyn EmitSink<Context, E>,
    /// Unified layout for all params and locals owned by the arch recompiler.
    layout: &'a dyn LocalAllocator,
}

impl<'a, Context, E> TrapContext<'a, Context, E> {
    /// Construct a `TrapContext`.
    ///
    /// Only [`TrapConfig`](crate::config::TrapConfig) should call this —
    /// traps receive one as a `&mut` argument.
    pub(crate) fn new(
        sink: &'a mut dyn EmitSink<Context, E>,
        layout: &'a dyn LocalAllocator,
    ) -> Self {
        Self { sink, layout }
    }

    /// Emit a single wasm instruction into the current function.
    #[inline]
    pub fn emit(&mut self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E> {
        self.sink.emit(ctx, instr)
    }

    /// Read-only access to the unified **param + locals** layout.
    ///
    /// Use [`LocalAllocator::local`] with a [`LocalSlot`](yecta::LocalSlot)
    /// obtained during `declare_params` or `declare_locals` to resolve
    /// absolute wasm local indices.
    ///
    /// Because arch params start at index 0 and every group is appended in
    /// order, the same layout correctly resolves both parameter slots and
    /// per-function local slots without any base-offset adjustment.
    #[inline]
    pub fn layout(&self) -> &dyn LocalAllocator {
        self.layout
    }

    /// Emit an **unconditional jump** to `target`, forwarding `params`
    /// parameters.
    ///
    /// Delegates to [`EmitSink::emit_jmp`].  For a `Reactor` sink this calls
    /// `Reactor::jmp` with full predecessor-graph bookkeeping.
    #[inline]
    pub fn jump(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        self.sink.emit_jmp(ctx, target, params)
    }

    /// Emit a **conditional jump** to `target` if the top `i32` on the wasm
    /// stack is non-zero, consuming that value.  Falls through otherwise.
    ///
    /// Emits:
    /// ```text
    /// if
    ///   [unconditional jump to target]
    /// end
    /// ```
    pub fn jump_if(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        self.sink
            .emit(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
        self.sink.emit_jmp(ctx, target, params)?;
        self.sink.emit(ctx, &Instruction::End)
    }
}
