//! [`InterpBuildCtx`] — configuration passed to [`InterpBodyBuilder::build_interp`].
//!
//! Also provides adapter types that bridge a plain [`InstructionSink`] to the
//! [`EmitSink`] and [`MemorySink`] interfaces expected by `TrapConfig` and
//! `MemoryAccess` helpers.

use alloc::vec::Vec;
use speet_link_core::OobConfig;
use speet_memory::MemoryAccess;
use speet_traps::{InstructionTrap, JumpTrap};
use wasm_encoder::{Instruction, ValType};
use wax_core::build::InstructionSink;
use yecta::{EmitSink, FuncIdx, LocalLayout};

// ── InterpBuildCtx ────────────────────────────────────────────────────────────

/// Configuration passed to [`InterpBodyBuilder::build_interp`] at code-gen time.
///
/// Carries the resolved WASM indices needed to emit correct tail-calls and
/// memory-access code, plus optional inherited memory and trap configuration.
///
/// Memory, traps, and layout are held here rather than on the sink because they
/// are used at code-gen time to emit WASM sequences — mirroring how arch
/// recompilers hold them.
pub struct InterpBuildCtx<'a, Context, E> {
    /// Absolute WASM function index of the dispatch function itself (slot 1).
    ///
    /// Opcode handlers that encounter an unrecognised opcode can re-dispatch
    /// via `return_call dispatch_func_idx`.
    pub dispatch_func_idx: u32,

    /// Absolute WASM function indices of the N handler functions (slots 2..2+N).
    pub handler_func_indices: Vec<u32>,

    /// OOB config — `lookup_stub_func_idx` for re-entering compiled code.
    pub oob: &'a OobConfig,

    /// WASM type index of the shared function type (arch regs + `target_pc: i64` → regs).
    pub type_idx: u32,

    /// WASM table index for `call_indirect` dispatch.
    pub table_idx: u32,

    /// Memory index for reading guest instruction bytes.
    pub insn_mem_idx: u32,

    /// Optional guest memory access — same configuration as the recompiler.
    ///
    /// When present, LOAD/STORE handlers should call `emit_load` / `emit_store_*`
    /// via a [`FlatMemorySink`] wrapping their current handler sink, inheriting
    /// the same address mapper the recompiler uses.
    pub memory: Option<&'a mut dyn MemoryAccess<Context, E>>,

    /// Optional per-instruction trap — same hook the recompiler fires.
    ///
    /// Builders that want to fire it should wrap their sink in a
    /// [`FlatEmitSink`] and call
    /// `trap.on_instruction(info, ctx, &mut flat_sink, layout)`.
    pub insn_trap: Option<&'a mut dyn InstructionTrap<Context, E>>,

    /// Optional per-jump trap — same hook the recompiler fires.
    pub jump_trap: Option<&'a mut dyn JumpTrap<Context, E>>,

    /// Finalised param/local layout (arch regs + trap params).
    ///
    /// Read-only: the interpreter inherits the layout established by the
    /// recompiler.  No new locals are appended here.
    pub layout: &'a LocalLayout,
}

// ── FlatEmitSink ─────────────────────────────────────────────────────────────

/// [`EmitSink`] adapter for a plain [`InstructionSink`].
///
/// Bridges the trap infrastructure (which expects `&mut dyn EmitSink`) to a
/// bare instruction sink, without requiring a full reactor.
///
/// `emit_jmp` emits `local.get 0..params-1` followed by `return_call target`,
/// matching the flat tail-call semantics of the reactor without needing the
/// predecessor-graph bookkeeping.
pub struct FlatEmitSink<'a, Context, E> {
    pub sink: &'a mut dyn InstructionSink<Context, E>,
}

impl<'a, Context, E> FlatEmitSink<'a, Context, E> {
    pub fn new(sink: &'a mut dyn InstructionSink<Context, E>) -> Self {
        Self { sink }
    }
}

impl<Context, E> EmitSink<Context, E> for FlatEmitSink<'_, Context, E> {
    fn emit(&mut self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E> {
        self.sink.instruction(ctx, instr)
    }

    fn emit_jmp(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        for i in 0..params {
            self.sink.instruction(ctx, &Instruction::LocalGet(i))?;
        }
        self.sink.instruction(ctx, &Instruction::ReturnCall(target.0))
    }
}

// ── FlatMemorySink ────────────────────────────────────────────────────────────

/// [`MemorySink`] adapter for a plain [`InstructionSink`].
///
/// Wraps a bare instruction sink with eager (non-deferred) store semantics,
/// suitable for interpreter handlers that do not require the reactor's
/// lazy-store coalescing.
///
/// Use this to call `MemoryAccess::emit_load` / `emit_store_*` from within
/// an interpreter handler function body.
pub struct FlatMemorySink<'a, Context, E> {
    pub sink: &'a mut dyn InstructionSink<Context, E>,
}

impl<'a, Context, E> FlatMemorySink<'a, Context, E> {
    pub fn new(sink: &'a mut dyn InstructionSink<Context, E>) -> Self {
        Self { sink }
    }
}

impl<Context, E> InstructionSink<Context, E> for FlatMemorySink<'_, Context, E> {
    fn instruction(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        self.sink.instruction(ctx, instruction)
    }
}

impl<Context, E> speet_ordering::MemorySink<Context, E> for FlatMemorySink<'_, Context, E> {
    fn feed_store(
        &mut self,
        ctx: &mut Context,
        _addr_type: ValType,
        instr: Instruction<'static>,
    ) -> Result<(), E> {
        self.sink.instruction(ctx, &instr)
    }

    fn feed_load(
        &mut self,
        ctx: &mut Context,
        _addr_local: u32,
        _addr_type: ValType,
        instr: Instruction<'static>,
    ) -> Result<(), E> {
        // No alias-check flush needed in the interpreter's flat execution model.
        self.sink.instruction(ctx, &instr)
    }

    fn flush_all(&mut self, _ctx: &mut Context) -> Result<(), E> {
        Ok(())
    }
}
