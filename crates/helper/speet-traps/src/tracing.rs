//! Tracing traps — observation without policy enforcement.
//!
//! These traps emit instrumentation code into every translated function but
//! never suppress or redirect control flow.  They are suitable for profiling,
//! coverage, and debugging.
//!
//! | Type | Trait | What it measures |
//! |------|-------|-----------------|
//! | [`CounterTrap`] | [`InstructionTrap`] | Increments a wasm global for each instruction matching an [`InsnClass`] mask |
//! | [`TraceLogTrap`] | [`JumpTrap`] | Calls a wasm import before every control-flow transfer |
//!
//! ## Composition
//!
//! Tracing traps always return [`TrapAction::Continue`] so they compose freely
//! with security traps via [`ChainedTrap`](crate::ChainedTrap):
//!
//! ```ignore
//! let trap = ChainedTrap::new(
//!     CounterTrap { global_idx: 0, mask: InsnClass(u32::MAX) },
//!     RopDetectTrap::new(handler, params),
//! );
//! linker.traps.set_instruction_trap(&mut trap);
//! ```

use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;

use crate::context::TrapContext;
use crate::insn::{InsnClass, InstructionInfo, InstructionTrap, TrapAction};
use crate::jump::{JumpInfo, JumpKind, JumpTrap};

// ── CounterTrap ───────────────────────────────────────────────────────────────

/// Increment a wasm global counter for each instruction matching a class mask.
///
/// Useful for profiling (count memory accesses, count branches, etc.) without
/// any per-function locals.  The counter is a wasm global of type `i32` at
/// index `global_idx`.
///
/// The trap emits:
/// ```text
/// global.get global_idx
/// i32.const 1
/// i32.add
/// global.set global_idx
/// ```
/// for every instruction whose `class` field has any bit in common with
/// `mask`.  Pass `InsnClass(u32::MAX)` to count every instruction regardless
/// of class.
pub struct CounterTrap {
    /// Wasm global index to increment.
    pub global_idx: u32,
    /// Instruction class mask — increment if `info.class.0 & mask.0 != 0`.
    /// Use `InsnClass(u32::MAX)` to count every instruction.
    pub mask: InsnClass,
}

impl<Context, E, F: InstructionSink<Context, E>> InstructionTrap<Context, E, F> for CounterTrap {
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        if self.mask == InsnClass::OTHER || info.class.contains(self.mask) {
            trap_ctx.emit(ctx, &Instruction::GlobalGet(self.global_idx))?;
            trap_ctx.emit(ctx, &Instruction::I32Const(1))?;
            trap_ctx.emit(ctx, &Instruction::I32Add)?;
            trap_ctx.emit(ctx, &Instruction::GlobalSet(self.global_idx))?;
        }
        Ok(TrapAction::Continue)
    }
}

// ── TraceLogTrap ──────────────────────────────────────────────────────────────

/// Emit a call to a wasm import before each control-flow transfer.
///
/// Before each jump the trap emits a call to the import at `log_func_idx`
/// with three `i32` arguments:
///
/// ```text
/// i32: source_pc  (truncated to 32 bits)
/// i32: target_pc  (truncated, or 0 for indirect / unknown targets)
/// i32: JumpKind   (discriminant — see below)
/// ```
///
/// The import must have wasm type `(i32, i32, i32) -> ()`.
///
/// `JumpKind` discriminants:
///
/// | Value | Kind |
/// |------:|------|
/// | 0 | `DirectJump` |
/// | 1 | `ConditionalBranch` |
/// | 2 | `Call` |
/// | 3 | `Return` |
/// | 4 | `IndirectJump` |
/// | 5 | `IndirectCall` |
/// | 6 | `Syscall` |
///
/// This trap always returns [`TrapAction::Continue`] — it logs but does not
/// redirect.  For active enforcement see [`security`](crate::security) and
/// [`hardening`](crate::hardening).
pub struct TraceLogTrap {
    /// Index of the wasm function import to call before each jump.
    pub log_func_idx: u32,
}

impl TraceLogTrap {
    fn kind_to_i32(kind: JumpKind) -> i32 {
        match kind {
            JumpKind::DirectJump => 0,
            JumpKind::ConditionalBranch => 1,
            JumpKind::Call => 2,
            JumpKind::Return => 3,
            JumpKind::IndirectJump => 4,
            JumpKind::IndirectCall => 5,
            JumpKind::Syscall => 6,
        }
    }
}

impl<Context, E, F: InstructionSink<Context, E>> JumpTrap<Context, E, F> for TraceLogTrap {
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        let target_i32 = info.target_pc.unwrap_or(0) as i32;
        trap_ctx.emit(ctx, &Instruction::I32Const(info.source_pc as i32))?;
        trap_ctx.emit(ctx, &Instruction::I32Const(target_i32))?;
        trap_ctx.emit(ctx, &Instruction::I32Const(Self::kind_to_i32(info.kind)))?;
        trap_ctx.emit(ctx, &Instruction::Call(self.log_func_idx))?;
        Ok(TrapAction::Continue)
    }
}
