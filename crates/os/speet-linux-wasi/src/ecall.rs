//! Route guest `ecall` sites to the linked [`syscall_dispatch`](super::EXPORT_SYSCALL_DISPATCH) function.

use speet_reach::PcSlotMap;
use wasm_encoder::Instruction;
use yecta::SlotAssigner;

/// RV64 register locals in the default speet layout (x0–x31 as i64).
pub const A0_LOCAL: u32 = 10;
pub const A1_LOCAL: u32 = 11;
pub const A2_LOCAL: u32 = 12;
pub const A7_LOCAL: u32 = 17;

/// [`speet_riscv::EcallCallback`] that calls the merged Linux-WASI guest module.
pub struct LinuxWasiEcall {
    /// Absolute WASM index of [`super::EXPORT_SYSCALL_DISPATCH`].
    pub syscall_dispatch_idx: u32,
    /// PC → sequential slot map for the translated `.text` blob.
    pub slots: PcSlotMap,
    /// WASM index of the first translated guest function (`n_imports`).
    pub base_func_offset: u32,
    /// Halt stub index when fall-through PC has no translated slot (e.g. final `exit`).
    pub halt_stub_idx: u32,
    /// Full translated-function param count (register file width).
    pub num_params: u32,
}

impl LinuxWasiEcall {
    fn emit<Context, E>(
        &self,
        ecall: &speet_riscv::EcallInfo,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) -> Result<(), E> {
        // RV64 `ecall` is always 4 bytes wide.
        let next_pc = ecall.pc as u64 + 4;
        let next_pc_func = self
            .slots
            .slot_for_pc(next_pc)
            .map(|slot| self.base_func_offset + slot)
            .unwrap_or(self.halt_stub_idx);

        cb.emit(ctx, &Instruction::LocalGet(A7_LOCAL))?;
        cb.emit(ctx, &Instruction::LocalGet(A0_LOCAL))?;
        cb.emit(ctx, &Instruction::LocalGet(A1_LOCAL))?;
        cb.emit(ctx, &Instruction::LocalGet(A2_LOCAL))?;
        cb.emit_call(ctx, self.syscall_dispatch_idx)?;
        cb.emit(ctx, &Instruction::LocalSet(A0_LOCAL))?;

        for p in 0..self.num_params {
            cb.emit(ctx, &Instruction::LocalGet(p))?;
        }
        cb.emit(ctx, &Instruction::ReturnCall(next_pc_func))?;
        Ok(())
    }
}

impl<Context, E> speet_riscv::EcallCallback<Context, E> for LinuxWasiEcall {
    fn call(
        &mut self,
        ecall: &speet_riscv::EcallInfo,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) {
        let _ = self.emit(ecall, ctx, cb);
    }
}
