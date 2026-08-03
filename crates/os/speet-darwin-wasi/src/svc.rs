//! Route guest `svc` sites to a link-time Darwin number → Unix handler dispatch.

use speet_reach::PcSlotMap;
use wasm_encoder::Instruction;
use yecta::SlotAssigner;

/// AArch64 register locals in the default speet layout (x0–x30 as i64).
pub const X0_LOCAL: u32 = 0;
pub const X1_LOCAL: u32 = 1;
pub const X2_LOCAL: u32 = 2;
/// Syscall number register (Darwin/BSD convention).
pub const X16_LOCAL: u32 = 16;

/// Absolute WASM indices of shared Unix guest handlers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandlerIndices {
    pub read: u32,
    pub write: u32,
    pub close: u32,
    pub exit: u32,
}

/// Build a `(num, a0, a1, a2) -> i64` dispatch for Darwin/BSD numbers.
pub fn build_syscall_dispatch(handlers: HandlerIndices) -> wasm_encoder::Function {
    use os_darwin_wasi::sysno;
    use wasm_encoder::{BlockType, Function, Instruction as I, ValType};
    let mut f = Function::new([]);
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(sysno::READ as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::If(BlockType::Result(ValType::I64)));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::LocalGet(2));
    f.instruction(&I::LocalGet(3));
    f.instruction(&I::Call(handlers.read));
    f.instruction(&I::Else);
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(sysno::WRITE as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::If(BlockType::Result(ValType::I64)));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::LocalGet(2));
    f.instruction(&I::LocalGet(3));
    f.instruction(&I::Call(handlers.write));
    f.instruction(&I::Else);
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(sysno::CLOSE as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::If(BlockType::Result(ValType::I64)));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::Call(handlers.close));
    f.instruction(&I::Else);
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(sysno::EXIT as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::If(BlockType::Result(ValType::I64)));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::Call(handlers.exit));
    f.instruction(&I::Unreachable);
    f.instruction(&I::Else);
    f.instruction(&I::I64Const(-38));
    f.instruction(&I::End);
    f.instruction(&I::End);
    f.instruction(&I::End);
    f.instruction(&I::End);
    f.instruction(&I::End);
    f
}

/// [`speet_aarch64::SvcCallback`] that calls a synthetic Darwin dispatch function.
pub struct DarwinWasiSvc {
    pub syscall_dispatch_idx: u32,
    pub slots: PcSlotMap,
    pub base_func_offset: u32,
    pub halt_stub_idx: u32,
    pub num_params: u32,
}

impl DarwinWasiSvc {
    fn emit<Context, E>(
        &self,
        svc: &speet_aarch64::SvcInfo,
        ctx: &mut Context,
        cb: &mut speet_aarch64::CallbackContext<'_, Context, E>,
    ) -> Result<(), E> {
        let next_pc = svc.pc + 4;
        let next_pc_func = self
            .slots
            .slot_for_pc(next_pc)
            .map(|slot| self.base_func_offset + slot)
            .unwrap_or(self.halt_stub_idx);

        cb.emit(ctx, &Instruction::LocalGet(X16_LOCAL))?;
        cb.emit(ctx, &Instruction::LocalGet(X0_LOCAL))?;
        cb.emit(ctx, &Instruction::LocalGet(X1_LOCAL))?;
        cb.emit(ctx, &Instruction::LocalGet(X2_LOCAL))?;
        cb.emit_call(ctx, self.syscall_dispatch_idx)?;
        cb.emit(ctx, &Instruction::LocalSet(X0_LOCAL))?;

        for p in 0..self.num_params {
            cb.emit(ctx, &Instruction::LocalGet(p))?;
        }
        cb.emit(ctx, &Instruction::ReturnCall(next_pc_func))?;
        Ok(())
    }
}

impl<Context, E> speet_aarch64::SvcCallback<Context, E> for DarwinWasiSvc {
    fn call(
        &mut self,
        svc: &speet_aarch64::SvcInfo,
        ctx: &mut Context,
        cb: &mut speet_aarch64::CallbackContext<'_, Context, E>,
    ) {
        let _ = self.emit(svc, ctx, cb);
    }
}
