//! [`UnknownTargetTrap`] — the debug seam's handler-sharing mechanism
//! (log / snapshot / panic on an unknown instruction, syscall, or library
//! call), function-reference-based rather than a fixed baked-in import.
//!
//! Fires for [`JumpKind::IndirectJump`], [`JumpKind::IndirectCall`], and
//! [`JumpKind::Syscall`] — the three kinds whose target isn't statically
//! known at compile time, and thus the three where "is this actually a
//! target we recognize?" can only be decided at runtime. Every other
//! [`JumpKind`] passes through untouched (this trap emits nothing for
//! them).
//!
//! ## Why function references, not a fixed import
//!
//! [`TraceLogTrap`](crate::tracing::TraceLogTrap) is the closest existing
//! precedent: it emits a `call` to a *fixed* wasm import, chosen once when
//! the trap is constructed and shared by every compiled function. That
//! doesn't fit the debug seam's actual requirement — "so an agent can
//! debug test processes running under JIT and recompilation" means
//! *different* guest processes, or even different compile requests for the
//! same process, may want *different* handlers (or none at all) without a
//! recompile. So `UnknownTargetTrap` instead declares an extra WASM
//! **parameter** of type `(ref null $handler)` (via `declare_params` — see
//! `speet-traps`' established param-declaration protocol, under which
//! trap-declared params are genuine WASM function parameters and thus
//! persist across `return_call` chains, exactly like an architectural
//! register would). *Which* function actually runs is decided by whoever
//! calls the compiled function and supplies that argument — swapping the
//! handler needs no recompile, just a different value at the call site.
//! This is what "share this handling via WASM function references" means
//! concretely: the handler is data, passed like any other argument, not
//! baked into the compiled code.
//!
//! ## What the handler does
//!
//! The handler decides everything — log, dump a snapshot, or panic
//! (`unreachable`, or any other genuinely-halting effect) — entirely on
//! its own side; this trap's only job is the unconditional call-through-ref
//! before the jump/call/syscall proceeds, exactly like `TraceLogTrap`'s
//! call-through-import. [`JumpTrap::on_jump`] therefore always returns
//! [`TrapAction::Continue`]: this trap observes, it does not redirect.
//! A daemon (see `os-daemon`'s debug-session mechanism) controls policy by
//! controlling *what the handler function does*, not by talking to this
//! trap directly.
//!
//! ## Handler function type
//!
//! The handler's WASM function type — caller-supplied as
//! `handler_type_idx`, since the type section is module-global and other
//! consumers may share the same type index — must be `(i64 source_pc, i64
//! target_or_syscall, i32 kind) -> ()`. `kind`'s encoding matches
//! `TraceLogTrap::kind_to_i32`'s discriminant table (0..6); this trap only
//! ever fires with `kind` in `{4 (IndirectJump), 5 (IndirectCall), 6
//! (Syscall)}`.

use wasm_encoder::{HeapType, Instruction, RefType, ValType};
use yecta::layout::CellIdx;
use yecta::{LocalAllocator, LocalDeclarator, LocalLayout, LocalSlot};

use crate::context::TrapContext;
use crate::insn::TrapAction;
use crate::jump::{JumpInfo, JumpKind, JumpTrap};

/// See the [module documentation](self) for the full design.
pub struct UnknownTargetTrap {
    /// WASM type index the handler funcref must match: `(i64, i64, i32) -> ()`.
    pub handler_type_idx: u32,
    /// Filled in by [`declare_params`](LocalDeclarator::declare_params):
    /// the local slot holding the `(ref null $handler)` parameter.
    handler_slot: LocalSlot,
}

impl UnknownTargetTrap {
    /// `handler_type_idx` must name a WASM function type `(i64, i64, i32) ->
    /// ()` already present in (or about to be added to) the module's type
    /// section — this trap does not declare the type itself.
    pub fn new(handler_type_idx: u32) -> Self {
        Self { handler_type_idx, handler_slot: LocalSlot::default() }
    }

    /// `Some(discriminant)` for the three kinds this trap actually fires
    /// on, matching `TraceLogTrap::kind_to_i32`'s table; `None` for every
    /// other kind (statically-known targets, where "is this unknown?"
    /// can't even arise).
    fn kind_to_i32(kind: JumpKind) -> Option<i32> {
        match kind {
            JumpKind::IndirectJump => Some(4),
            JumpKind::IndirectCall => Some(5),
            JumpKind::Syscall => Some(6),
            JumpKind::DirectJump
            | JumpKind::ConditionalBranch
            | JumpKind::Call
            | JumpKind::Return => None,
        }
    }

    fn handler_ref_type(&self) -> ValType {
        ValType::Ref(RefType { nullable: true, heap_type: HeapType::Concrete(self.handler_type_idx) })
    }
}

impl LocalDeclarator for UnknownTargetTrap {
    fn declare_params(&mut self, _cell: CellIdx, params: &mut LocalLayout) {
        self.handler_slot = params.append(1, self.handler_ref_type());
    }
}

impl<Context, E> JumpTrap<Context, E> for UnknownTargetTrap {
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E>,
    ) -> Result<TrapAction, E> {
        let Some(kind_i32) = Self::kind_to_i32(info.kind) else {
            return Ok(TrapAction::Continue);
        };
        let handler_local = trap_ctx.layout().local(self.handler_slot, 0);

        trap_ctx.emit(ctx, &Instruction::I64Const(info.source_pc as i64))?;
        match info.target_local {
            Some(local) => trap_ctx.emit(ctx, &Instruction::LocalGet(local))?,
            None => trap_ctx.emit(ctx, &Instruction::I64Const(info.target_pc.unwrap_or(0) as i64))?,
        }
        trap_ctx.emit(ctx, &Instruction::I32Const(kind_i32))?;
        trap_ctx.emit(ctx, &Instruction::LocalGet(handler_local))?;
        trap_ctx.emit(ctx, &Instruction::CallRef(self.handler_type_idx))?;

        Ok(TrapAction::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrapConfig;
    use yecta::{EmitSink, FuncIdx, LocalAllocator};

    /// Minimal `EmitSink` collecting emitted instructions for inspection —
    /// this trap's contract is "what code gets emitted", so tests assert on
    /// the emitted sequence rather than executing it (no existing
    /// precedent in this crate builds a full module + engine harness just
    /// to test a `JumpTrap`).
    #[derive(Default)]
    struct RecordingSink(alloc::vec::Vec<alloc::string::String>);

    impl EmitSink<(), core::convert::Infallible> for RecordingSink {
        fn emit(&mut self, _ctx: &mut (), instr: &Instruction<'_>) -> Result<(), core::convert::Infallible> {
            self.0.push(alloc::format!("{instr:?}"));
            Ok(())
        }
        fn emit_jmp(&mut self, _ctx: &mut (), _target: FuncIdx, _params: u32) -> Result<(), core::convert::Infallible> {
            unreachable!("UnknownTargetTrap never calls emit_jmp")
        }
    }

    fn setup() -> (UnknownTargetTrap, LocalLayout, CellIdx) {
        let mut trap = UnknownTargetTrap::new(7 /* arbitrary handler type idx */);
        let mut layout = LocalLayout::empty();
        // Pretend one architectural i64 register param came first, like a
        // real recompiler's arch params would.
        layout.append(1, ValType::I64);
        let cell = CellIdx(0);
        trap.declare_params(cell, &mut layout);
        (trap, layout, cell)
    }

    #[test]
    fn declare_params_appends_exactly_one_ref_slot() {
        let (trap, layout, _cell) = setup();
        assert_eq!(layout.total_locals(), 2, "1 arch param + 1 handler param");
        assert_eq!(layout.val_type(trap.handler_slot), trap.handler_ref_type());
    }

    #[test]
    fn direct_jump_emits_nothing_and_continues() {
        let (mut trap, layout, cell) = setup();
        let mut sink = RecordingSink::default();
        let info = JumpInfo::direct(0x1000, 0x1004, JumpKind::DirectJump);
        let action = crate::jump::fire_jump_trap(&mut trap, &info, &mut (), &mut sink, &layout, cell)
            .unwrap();
        assert_eq!(action, TrapAction::Continue);
        assert!(sink.0.is_empty(), "statically-known-target kinds must emit nothing");
    }

    #[test]
    fn indirect_call_emits_call_through_handler_ref_and_continues() {
        let (mut trap, layout, cell) = setup();
        let mut sink = RecordingSink::default();
        // target_local = 5 (the runtime-computed call target's local).
        let info = JumpInfo::indirect(0x2000, 5, JumpKind::IndirectCall);
        let action = crate::jump::fire_jump_trap(&mut trap, &info, &mut (), &mut sink, &layout, cell)
            .unwrap();
        assert_eq!(action, TrapAction::Continue, "trap must never itself redirect control flow");

        let handler_local = layout.local(trap.handler_slot, 0);
        assert_eq!(
            sink.0,
            alloc::vec![
                alloc::format!("{:?}", Instruction::I64Const(0x2000)),
                alloc::format!("{:?}", Instruction::LocalGet(5)),
                alloc::format!("{:?}", Instruction::I32Const(5)), // IndirectCall discriminant
                alloc::format!("{:?}", Instruction::LocalGet(handler_local)),
                alloc::format!("{:?}", Instruction::CallRef(7)),
            ]
        );
    }

    #[test]
    fn syscall_uses_target_pc_fallback_when_no_target_local() {
        let (mut trap, layout, cell) = setup();
        let mut sink = RecordingSink::default();
        let info = JumpInfo { source_pc: 0x3000, target_pc: None, target_local: None, kind: JumpKind::Syscall };
        crate::jump::fire_jump_trap(&mut trap, &info, &mut (), &mut sink, &layout, cell).unwrap();
        assert_eq!(
            sink.0,
            alloc::vec![
                alloc::format!("{:?}", Instruction::I64Const(0x3000)),
                alloc::format!("{:?}", Instruction::I64Const(0)), // no target_local -> falls back to target_pc.unwrap_or(0)
                alloc::format!("{:?}", Instruction::I32Const(6)), // Syscall discriminant
                alloc::format!("{:?}", Instruction::LocalGet(layout.local(trap.handler_slot, 0))),
                alloc::format!("{:?}", Instruction::CallRef(7)),
            ]
        );
    }

    #[test]
    fn composes_via_trap_config_like_any_other_jump_trap() {
        // Proves this trap plugs into TrapConfig's normal install/declare/fire
        // protocol, not just fire_jump_trap directly.
        let mut trap = UnknownTargetTrap::new(7);
        let mut config: TrapConfig<'_, '_, (), core::convert::Infallible> = TrapConfig::new();
        config.set_jump_trap(&mut trap);

        let mut layout = LocalLayout::empty();
        layout.append(1, ValType::I64);
        config.declare_params(CellIdx(0), &mut layout);
        assert_eq!(layout.total_locals(), 2);

        let mut sink = RecordingSink::default();
        let info = JumpInfo::indirect(0x4000, 3, JumpKind::IndirectJump);
        let action = config.on_jump(&info, &mut (), &mut sink, &layout, CellIdx(0)).unwrap();
        assert_eq!(action, TrapAction::Continue);
        assert_eq!(sink.0.len(), 5, "one full call-through-ref sequence");
    }
}
