//! `speet-syscall` — WASM inline syscall dispatch for static recompilers.
//!
//! The generic syscall-table data model now lives in `os-syscall-emit`; this
//! crate re-exports it and provides the WASM-specific [`WasmSyscallDispatcher`]
//! that renders inline `br_table` dispatch at guest `ecall` sites.

#![no_std]
extern crate alloc;

pub use os_syscall_emit::{MemoryStore, ParamSource, SavePair, SyscallEntry, SyscallTable};

use alloc::borrow::Cow;
use alloc::vec::Vec;
use wasm_encoder::Instruction;

/// An [`EcallCallback`](speet_riscv::EcallCallback)-compatible type that
/// emits inline `br_table` syscall dispatch at guest `ecall` sites.
///
/// See the [`os-syscall-emit`] docs for the data model; this crate owns the
/// WASM rendering.
pub struct WasmSyscallDispatcher<'t> {
    /// The dispatch table.  Entries must be sorted by syscall number.
    pub table: &'t SyscallTable,

    /// WASM local index of the register holding the syscall number.
    ///
    /// For RISC-V: `a7 = x17`, which occupies local 17 in the default layout
    /// where locals 0–31 map to `x0`–`x31`.
    pub syscall_num_local: u32,

    /// If `true`, the syscall-number local has type `i64` and must be wrapped
    /// to `i32` before the `br_table` index computation.
    pub syscall_num_is_i64: bool,

    /// Total number of WASM parameters in the current chain function.
    pub num_params: u32,

    /// Absolute WASM function index of the function representing the
    /// instruction immediately following this `ecall` (i.e. PC + inst_len).
    pub next_pc_func: u32,
}

impl<'t, Context, E> speet_riscv::EcallCallback<Context, E> for WasmSyscallDispatcher<'t> {
    fn call(
        &mut self,
        _ecall: &speet_riscv::EcallInfo,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) {
        let _ = self.emit(ctx, cb);
    }
}

impl<'t> WasmSyscallDispatcher<'t> {
    fn emit<Context, E>(
        &self,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) -> Result<(), E> {
        let entries = self.table.entries();

        if entries.is_empty() {
            cb.emit(ctx, &Instruction::Unreachable)?;
            return Ok(());
        }

        let min_num = entries.first().unwrap().0;
        let max_num = entries.last().unwrap().0;
        let range = (max_num - min_num + 1) as usize;
        let n = entries.len();
        let unknown_depth = n as u32;

        let mut br_targets: Vec<u32> = alloc::vec![unknown_depth; range];
        for (arm_idx, (syscall_num, _)) in entries.iter().enumerate() {
            let offset = (syscall_num - min_num) as usize;
            br_targets[offset] = arm_idx as u32;
        }

        // Outer blocks: $exit, $unknown, then arm blocks.
        cb.emit(ctx, &Instruction::Block(wasm_encoder::BlockType::Empty))?;
        cb.emit(ctx, &Instruction::Block(wasm_encoder::BlockType::Empty))?;
        for _ in 0..n {
            cb.emit(ctx, &Instruction::Block(wasm_encoder::BlockType::Empty))?;
        }

        // Load syscall number as i32.
        cb.emit(ctx, &Instruction::LocalGet(self.syscall_num_local))?;
        if self.syscall_num_is_i64 {
            cb.emit(ctx, &Instruction::I32WrapI64)?;
        }
        if min_num > 0 {
            cb.emit(ctx, &Instruction::I32Const(min_num as i32))?;
            cb.emit(ctx, &Instruction::I32Sub)?;
        }
        cb.emit(
            ctx,
            &Instruction::BrTable(Cow::Owned(br_targets), unknown_depth),
        )?;

        // Handler arms.
        for (arm_idx, (_, entry)) in entries.iter().enumerate() {
            cb.emit(ctx, &Instruction::End)?; // close arm block

            for save in &entry.saves {
                cb.emit(ctx, &Instruction::LocalGet(save.local_idx))?;
                cb.emit(ctx, &Instruction::GlobalSet(save.global_idx))?;
            }

            for store in &entry.memory_stores {
                cb.emit(ctx, &Instruction::I32Const(store.addr as i32))?;
                cb.emit(ctx, &Instruction::LocalGet(store.value_local))?;
                if store.value_is_i64 {
                    cb.emit(ctx, &Instruction::I32WrapI64)?;
                }
                cb.emit(ctx, &Instruction::I32Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }))?;
            }

            for source in &entry.param_map {
                match source {
                    ParamSource::LocalI64AsI32(local) => {
                        cb.emit(ctx, &Instruction::LocalGet(*local))?;
                        cb.emit(ctx, &Instruction::I32WrapI64)?;
                    }
                    ParamSource::LocalI32(local) => {
                        cb.emit(ctx, &Instruction::LocalGet(*local))?;
                    }
                    ParamSource::ConstI32(v) => {
                        cb.emit(ctx, &Instruction::I32Const(*v))?;
                    }
                    ParamSource::ConstI64(v) => {
                        cb.emit(ctx, &Instruction::I64Const(*v))?;
                    }
                }
            }

            cb.emit_call(ctx, entry.func_idx)?;

            if let Some(result_local) = entry.result_local {
                if entry.negate_nonzero_result || entry.load_mem_on_success.is_some() {
                    cb.emit(ctx, &Instruction::I64ExtendI32S)?;
                    cb.emit(ctx, &Instruction::LocalTee(result_local))?;
                    cb.emit(ctx, &Instruction::I64Const(0))?;
                    cb.emit(ctx, &Instruction::I64Ne)?;
                    cb.emit(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
                    if entry.negate_nonzero_result {
                        cb.emit(ctx, &Instruction::I64Const(0))?;
                        cb.emit(ctx, &Instruction::LocalGet(result_local))?;
                        cb.emit(ctx, &Instruction::I64Sub)?;
                        cb.emit(ctx, &Instruction::LocalSet(result_local))?;
                    }
                    if let Some(mem_offset) = entry.load_mem_on_success {
                        cb.emit(ctx, &Instruction::Else)?;
                        cb.emit(ctx, &Instruction::I32Const(mem_offset as i32))?;
                        cb.emit(ctx, &Instruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                        cb.emit(ctx, &Instruction::I64ExtendI32S)?;
                        cb.emit(ctx, &Instruction::LocalSet(result_local))?;
                    }
                    cb.emit(ctx, &Instruction::End)?;
                } else {
                    cb.emit(ctx, &Instruction::I64ExtendI32S)?;
                    cb.emit(ctx, &Instruction::LocalSet(result_local))?;
                }
            } else if entry.has_return {
                cb.emit(ctx, &Instruction::Drop)?;
            }

            if entry.terminates {
                cb.emit(ctx, &Instruction::Unreachable)?;
            } else {
                self.emit_continue_to_next_pc(ctx, cb)?;
            }
        }

        cb.emit(ctx, &Instruction::End)?; // close $unknown
        cb.emit(ctx, &Instruction::Unreachable)?;
        cb.emit(ctx, &Instruction::End)?; // close $exit

        Ok(())
    }

    fn emit_continue_to_next_pc<Context, E>(
        &self,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) -> Result<(), E> {
        for p in 0..self.num_params {
            cb.emit(ctx, &Instruction::LocalGet(p))?;
        }
        cb.emit(ctx, &Instruction::ReturnCall(self.next_pc_func))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::format;
    use core::convert::Infallible;
    use speet_riscv::CallbackContext;

    struct MockSink {
        instructions: Vec<String>,
    }

    impl wax_core::build::InstructionSink<(), Infallible> for MockSink {
        fn instruction(&mut self, _ctx: &mut (), instruction: &Instruction<'_>) -> Result<(), Infallible> {
            self.instructions.push(format!("{:?}", instruction));
            Ok(())
        }
    }

    #[test]
    fn test_syscall_dispatcher_emit() {
        let entries = alloc::vec![
            (10, SyscallEntry {
                func_idx: 100,
                param_map: alloc::vec![ParamSource::LocalI64AsI32(1)],
                saves: alloc::vec![],
                result_local: Some(2),
                negate_nonzero_result: false,
                has_return: true,
                terminates: false,
                memory_stores: alloc::vec![],
                load_mem_on_success: None,
            }),
            (20, SyscallEntry {
                func_idx: 200,
                param_map: alloc::vec![ParamSource::ConstI32(42)],
                saves: alloc::vec![],
                result_local: None,
                negate_nonzero_result: false,
                has_return: false,
                terminates: false,
                memory_stores: alloc::vec![],
                load_mem_on_success: None,
            }),
        ];
        let table = SyscallTable::new(entries);

        let dispatcher = WasmSyscallDispatcher {
            table: &table,
            syscall_num_local: 10,
            syscall_num_is_i64: true,
            num_params: 5,
            next_pc_func: 300,
        };

        let mut sink = MockSink { instructions: Vec::new() };
        {
            let mut cb = CallbackContext::new(&mut sink);
            dispatcher.emit(&mut (), &mut cb).unwrap();
        }

        let insts = &sink.instructions;
        assert!(insts.iter().any(|i| i.contains("LocalGet(10)")));
        assert!(insts.iter().any(|i| i.contains("I32WrapI64")));
        assert!(insts.iter().any(|i| i.contains("BrTable")));
        assert!(insts.iter().any(|i| i.contains("Call(100)")));
        assert!(insts.iter().any(|i| i.contains("Call(200)")));
        assert!(insts.iter().any(|i| i.contains("ReturnCall(300)")));
        assert_eq!(
            insts.iter().filter(|i| i.contains("ReturnCall(300)")).count(),
            2
        );
    }
}