//! Route guest `svc` sites to the linked [`syscall_dispatch`](super::EXPORT_SYSCALL_DISPATCH) function.

use speet_reach::PcSlotMap;
use wasm_encoder::Instruction;
use yecta::SlotAssigner;

/// AArch64 register locals in the default speet layout (x0–x30 as i64).
pub const X0_LOCAL: u32 = 0;
pub const X1_LOCAL: u32 = 1;
pub const X2_LOCAL: u32 = 2;
/// Syscall number register (Darwin/BSD convention).
pub const X16_LOCAL: u32 = 16;

/// [`speet_aarch64::SvcCallback`] that calls the merged Darwin-WASI guest module.
pub struct DarwinWasiSvc {
    /// Absolute WASM index of [`super::EXPORT_SYSCALL_DISPATCH`].
    pub syscall_dispatch_idx: u32,
    /// PC → sequential slot map for the translated `.text` blob.
    pub slots: PcSlotMap,
    /// WASM index of the first translated guest function.
    pub base_func_offset: u32,
    /// Halt stub index when fall-through PC has no translated slot.
    pub halt_stub_idx: u32,
    /// Full translated-function param count (register file width).
    pub num_params: u32,
}

impl DarwinWasiSvc {
    fn emit<Context, E>(
        &self,
        svc: &speet_aarch64::SvcInfo,
        ctx: &mut Context,
        cb: &mut speet_aarch64::CallbackContext<'_, Context, E>,
    ) -> Result<(), E> {
        // AArch64 `svc` is always 4 bytes wide.
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
