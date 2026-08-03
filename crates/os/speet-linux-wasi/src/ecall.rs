//! Route guest `ecall` sites to a link-time Linux number → Unix handler dispatch.

use speet_reach::PcSlotMap;
use wasm_encoder::Instruction;
use yecta::SlotAssigner;

/// RV64 register locals in the default speet layout (x0–x31 as i64).
pub const A0_LOCAL: u32 = 10;
pub const A1_LOCAL: u32 = 11;
pub const A2_LOCAL: u32 = 12;
pub const A7_LOCAL: u32 = 17;

/// Absolute WASM indices of shared Unix guest handlers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandlerIndices {
    pub read: u32,
    pub write: u32,
    pub close: u32,
    pub exit: u32,
}

/// Linux syscall numbers for one ABI family.
#[derive(Clone, Copy, Debug)]
pub struct LinuxSyscallNums {
    pub read: u64,
    pub write: u64,
    pub close: u64,
    pub exit: u64,
    pub exit_group: u64,
}

impl Default for LinuxSyscallNums {
    fn default() -> Self {
        Self::for_abi(os_linux_wasi::LinuxAbi::Rv64)
    }
}

impl LinuxSyscallNums {
    /// Linux syscall numbers for a particular register ABI.
    pub fn for_abi(abi: os_linux_wasi::LinuxAbi) -> Self {
        use os_linux_wasi::sysno;
        let (read, write, close, exit, exit_group) = match abi {
            os_linux_wasi::LinuxAbi::Rv64 | os_linux_wasi::LinuxAbi::AArch64 => (
                sysno::generic::READ,
                sysno::generic::WRITE,
                sysno::generic::CLOSE,
                sysno::generic::EXIT,
                sysno::generic::EXIT_GROUP,
            ),
            os_linux_wasi::LinuxAbi::X86_64 => (
                sysno::x86_64::READ,
                sysno::x86_64::WRITE,
                sysno::x86_64::CLOSE,
                sysno::x86_64::EXIT,
                sysno::x86_64::EXIT_GROUP,
            ),
            os_linux_wasi::LinuxAbi::MipsN64 => (
                sysno::mips_n64::READ,
                sysno::mips_n64::WRITE,
                sysno::mips_n64::CLOSE,
                sysno::mips_n64::EXIT,
                sysno::mips_n64::EXIT_GROUP,
            ),
        };
        Self {
            read,
            write,
            close,
            exit,
            exit_group,
        }
    }
}

/// Build a `(num, a0, a1, a2) -> i64` dispatch function for the given numbers.
pub fn build_syscall_dispatch(handlers: HandlerIndices, nums: LinuxSyscallNums) -> wasm_encoder::Function {
    use wasm_encoder::{BlockType, Function, Instruction as I, ValType};
    // params: num, a0, a1, a2 — already function params (locals 0..3)
    let mut f = Function::new([]);
    // if num == read
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(nums.read as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::If(BlockType::Result(ValType::I64)));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::LocalGet(2));
    f.instruction(&I::LocalGet(3));
    f.instruction(&I::Call(handlers.read));
    f.instruction(&I::Else);
    // write
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(nums.write as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::If(BlockType::Result(ValType::I64)));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::LocalGet(2));
    f.instruction(&I::LocalGet(3));
    f.instruction(&I::Call(handlers.write));
    f.instruction(&I::Else);
    // close
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(nums.close as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::If(BlockType::Result(ValType::I64)));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::Call(handlers.close));
    f.instruction(&I::Else);
    // exit / exit_group
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(nums.exit as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I64Const(nums.exit_group as i64));
    f.instruction(&I::I64Eq);
    f.instruction(&I::I32Or);
    f.instruction(&I::If(BlockType::Result(ValType::I64)));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::Call(handlers.exit));
    f.instruction(&I::Unreachable);
    f.instruction(&I::Else);
    f.instruction(&I::I64Const(-38));
    f.instruction(&I::End); // exit if
    f.instruction(&I::End); // close if
    f.instruction(&I::End); // write if
    f.instruction(&I::End); // read if
    f.instruction(&I::End); // function body
    f
}

/// [`speet_riscv::EcallCallback`] that calls a synthetic Linux dispatch function.
pub struct LinuxWasiEcall {
    /// Absolute WASM index of the synthetic `syscall_dispatch`.
    pub syscall_dispatch_idx: u32,
    pub num_local: u32,
    pub a0_local: u32,
    pub a1_local: u32,
    pub a2_local: u32,
    pub slots: PcSlotMap,
    pub base_func_offset: u32,
    pub halt_stub_idx: u32,
    pub num_params: u32,
    pub insn_width: u64,
}

impl LinuxWasiEcall {
    pub fn rv64(
        syscall_dispatch_idx: u32,
        slots: PcSlotMap,
        base_func_offset: u32,
        halt_stub_idx: u32,
        num_params: u32,
    ) -> Self {
        Self {
            syscall_dispatch_idx,
            num_local: A7_LOCAL,
            a0_local: A0_LOCAL,
            a1_local: A1_LOCAL,
            a2_local: A2_LOCAL,
            slots,
            base_func_offset,
            halt_stub_idx,
            num_params,
            insn_width: 4,
        }
    }

    fn emit<Context, E>(
        &self,
        ecall: &speet_riscv::EcallInfo,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) -> Result<(), E> {
        let next_pc = ecall.pc as u64 + self.insn_width;
        let next_pc_func = self
            .slots
            .slot_for_pc(next_pc)
            .map(|slot| self.base_func_offset + slot)
            .unwrap_or(self.halt_stub_idx);

        cb.emit(ctx, &Instruction::LocalGet(self.num_local))?;
        cb.emit(ctx, &Instruction::LocalGet(self.a0_local))?;
        cb.emit(ctx, &Instruction::LocalGet(self.a1_local))?;
        cb.emit(ctx, &Instruction::LocalGet(self.a2_local))?;
        cb.emit_call(ctx, self.syscall_dispatch_idx)?;
        cb.emit(ctx, &Instruction::LocalSet(self.a0_local))?;

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

/// [`speet_aarch64::SvcCallback`] for Linux `svc #0`: x8 is the syscall
/// number and x0–x2 are its first three arguments. The same callback hook is
/// also used by Darwin, whose embedder instead reads x16.
pub struct LinuxWasiSvc {
    pub syscall_dispatch_idx: u32,
    pub slots: PcSlotMap,
    pub base_func_offset: u32,
    pub halt_stub_idx: u32,
    pub num_params: u32,
}

impl<Context, E> speet_aarch64::SvcCallback<Context, E> for LinuxWasiSvc {
    fn call(
        &mut self,
        svc: &speet_aarch64::SvcInfo,
        ctx: &mut Context,
        cb: &mut speet_aarch64::CallbackContext<'_, Context, E>,
    ) {
        let next = self.slots.slot_for_pc(svc.pc + 4)
            .map(|slot| self.base_func_offset + slot)
            .unwrap_or(self.halt_stub_idx);
        let _ = (|| -> Result<(), E> {
            cb.emit(ctx, &Instruction::LocalGet(8))?;
            cb.emit(ctx, &Instruction::LocalGet(0))?;
            cb.emit(ctx, &Instruction::LocalGet(1))?;
            cb.emit(ctx, &Instruction::LocalGet(2))?;
            cb.emit_call(ctx, self.syscall_dispatch_idx)?;
            cb.emit(ctx, &Instruction::LocalSet(0))?;
            for p in 0..self.num_params {
                cb.emit(ctx, &Instruction::LocalGet(p))?;
            }
            cb.emit(ctx, &Instruction::ReturnCall(next))?;
            Ok(())
        })();
    }
}

/// [`speet_x86_64::SyscallCallback`] for Linux: rax is the number and
/// rdi/rsi/rdx are the first three arguments.
pub struct LinuxWasiSyscall {
    pub syscall_dispatch_idx: u32,
    pub slots: PcSlotMap,
    pub base_func_offset: u32,
    pub halt_stub_idx: u32,
    pub num_params: u32,
}

impl<Context, E> speet_x86_64::SyscallCallback<Context, E> for LinuxWasiSyscall {
    fn call(
        &mut self,
        syscall: &speet_x86_64::SyscallInfo,
        ctx: &mut Context,
        cb: &mut speet_x86_64::CallbackContext<'_, Context, E>,
    ) {
        let next = self.slots.slot_for_pc(syscall.pc + syscall.width as u64)
            .map(|slot| self.base_func_offset + slot)
            .unwrap_or(self.halt_stub_idx);
        let _ = (|| -> Result<(), E> {
            cb.emit(ctx, &Instruction::LocalGet(0))?;
            cb.emit(ctx, &Instruction::LocalGet(7))?;
            cb.emit(ctx, &Instruction::LocalGet(6))?;
            cb.emit(ctx, &Instruction::LocalGet(2))?;
            cb.emit_call(ctx, self.syscall_dispatch_idx)?;
            cb.emit(ctx, &Instruction::LocalSet(0))?;
            for p in 0..self.num_params {
                cb.emit(ctx, &Instruction::LocalGet(p))?;
            }
            cb.emit(ctx, &Instruction::ReturnCall(next))?;
            Ok(())
        })();
    }
}

/// [`speet_mips::SyscallCallback`] for Linux N64: v0 ($2) is the number and
/// a0–a2 are $4–$6.
pub struct LinuxWasiMipsSyscall {
    pub syscall_dispatch_idx: u32,
    pub slots: PcSlotMap,
    pub base_func_offset: u32,
    pub halt_stub_idx: u32,
    pub num_params: u32,
}

impl<Context, E, F> speet_mips::SyscallCallback<Context, E, F> for LinuxWasiMipsSyscall
where
    F: wax_core::build::InstructionSink<Context, E>,
{
    fn call(
        &mut self,
        syscall: &speet_mips::SyscallInfo,
        ctx: &mut Context,
        cb: &mut speet_mips::CallbackContext<'_, Context, E>,
    ) {
        let next = self.slots.slot_for_pc(syscall.pc as u64 + 4)
            .map(|slot| self.base_func_offset + slot)
            .unwrap_or(self.halt_stub_idx);
        let _ = (|| -> Result<(), E> {
            cb.emit(ctx, &Instruction::LocalGet(2))?;
            cb.emit(ctx, &Instruction::LocalGet(4))?;
            cb.emit(ctx, &Instruction::LocalGet(5))?;
            cb.emit(ctx, &Instruction::LocalGet(6))?;
            cb.emit_call(ctx, self.syscall_dispatch_idx)?;
            cb.emit(ctx, &Instruction::LocalSet(2))?;
            for p in 0..self.num_params {
                cb.emit(ctx, &Instruction::LocalGet(p))?;
            }
            cb.emit(ctx, &Instruction::ReturnCall(next))?;
            Ok(())
        })();
    }
}
