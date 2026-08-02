//! Encapsulated per-syscall handler functions for the Phase 2 migration.
//!
//! Guest `ecall` sites will call into these module functions instead of expanding
//! inline `br_table` dispatch. Scratch copies route through `__speet_host_mem_*`
//! imports pre-shimming (lowered to memory index 1 by [`speet_recompile::host_mem_shim`]).
//!
//! Full linker integration (reserve handler slots, wire dispatch stub) is follow-up
//! work; this module owns the standalone WASM bodies.

extern crate alloc;

use alloc::vec::Vec;
use os_syscall_emit::{MemoryStore, ParamSource, SyscallEntry};
use wasm_encoder::{Function, Instruction, MemArg};

/// Import indices for the pre-shimming host-memory surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemImportIndices {
    pub store_i32: u32,
    pub load_i32: u32,
}

impl HostMemImportIndices {
    fn emit_store_i32(&self, out: &mut Function, addr: u32, value_local: u32, value_is_i64: bool) {
        out.instruction(&Instruction::I32Const(addr as i32));
        out.instruction(&Instruction::LocalGet(value_local));
        if value_is_i64 {
            out.instruction(&Instruction::I32WrapI64);
        }
        out.instruction(&Instruction::Call(self.store_i32));
    }

    fn emit_load_i32_as_i64(&self, out: &mut Function, addr: u32, dest_local: u32) {
        out.instruction(&Instruction::I32Const(addr as i32));
        out.instruction(&Instruction::Call(self.load_i32));
        out.instruction(&Instruction::I64ExtendI32S);
        out.instruction(&Instruction::LocalSet(dest_local));
    }
}

/// Plan mapping Linux syscall numbers to handler function indices inside a linked module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HandlerModulePlan {
    pub handlers: Vec<(u64, u32)>,
}

impl HandlerModulePlan {
    pub fn from_table(table: &os_syscall_emit::SyscallTable, base_func_idx: u32) -> Self {
        let handlers = table
            .entries()
            .iter()
            .enumerate()
            .map(|(i, (num, _))| (*num, base_func_idx + i as u32))
            .collect();
        Self { handlers }
    }
}

/// Emit one handler WASM function body matching a [`SyscallEntry`].
///
/// When `host_mem` is `Some`, scratch [`MemoryStore`] ops use `__speet_host_mem`
/// imports; otherwise they target guest memory index 0 (legacy inline path).
pub fn emit_handler_function(
    entry: &SyscallEntry,
    host_mem: Option<HostMemImportIndices>,
) -> Function {
    let mut f = Function::new([]);
    emit_handler_body(&mut f, entry, host_mem);
    f.instruction(&Instruction::End);
    f
}

fn emit_handler_body(f: &mut Function, entry: &SyscallEntry, host_mem: Option<HostMemImportIndices>) {
    for save in &entry.saves {
        f.instruction(&Instruction::LocalGet(save.local_idx));
        f.instruction(&Instruction::GlobalSet(save.global_idx));
    }

    for store in &entry.memory_stores {
        if let Some(hm) = host_mem {
            hm.emit_store_i32(f, store.addr, store.value_local, store.value_is_i64);
        } else {
            f.instruction(&Instruction::I32Const(store.addr as i32));
            f.instruction(&Instruction::LocalGet(store.value_local));
            if store.value_is_i64 {
                f.instruction(&Instruction::I32WrapI64);
            }
            f.instruction(&Instruction::I32Store(MemArg {
                offset: 0,
                align: 2,
                memory_index: 0,
            }));
        }
    }

    for source in &entry.param_map {
        match source {
            ParamSource::LocalI64AsI32(local) => {
                f.instruction(&Instruction::LocalGet(*local));
                f.instruction(&Instruction::I32WrapI64);
            }
            ParamSource::LocalI32(local) => {
                f.instruction(&Instruction::LocalGet(*local));
            }
            ParamSource::ConstI32(v) => {
                f.instruction(&Instruction::I32Const(*v));
            }
            ParamSource::ConstI64(v) => {
                f.instruction(&Instruction::I64Const(*v));
            }
        }
    }

    f.instruction(&Instruction::Call(entry.func_idx));

    if let Some(result_local) = entry.result_local {
        if entry.negate_nonzero_result || entry.load_mem_on_success.is_some() {
            f.instruction(&Instruction::I64ExtendI32S);
            f.instruction(&Instruction::LocalTee(result_local));
            f.instruction(&Instruction::I64Const(0));
            f.instruction(&Instruction::I64Ne);
            f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
            if entry.negate_nonzero_result {
                f.instruction(&Instruction::I64Const(0));
                f.instruction(&Instruction::LocalGet(result_local));
                f.instruction(&Instruction::I64Sub);
                f.instruction(&Instruction::LocalSet(result_local));
            }
            if let Some(mem_offset) = entry.load_mem_on_success {
                f.instruction(&Instruction::Else);
                if let Some(hm) = host_mem {
                    hm.emit_load_i32_as_i64(f, mem_offset, result_local);
                } else {
                    f.instruction(&Instruction::I32Const(mem_offset as i32));
                    f.instruction(&Instruction::I32Load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }));
                    f.instruction(&Instruction::I64ExtendI32S);
                    f.instruction(&Instruction::LocalSet(result_local));
                }
            }
            f.instruction(&Instruction::End);
        } else {
            f.instruction(&Instruction::I64ExtendI32S);
            f.instruction(&Instruction::LocalSet(result_local));
        }
    } else if entry.has_return {
        f.instruction(&Instruction::Drop);
    }

    if entry.terminates {
        f.instruction(&Instruction::Unreachable);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use os_linux_wasi::{LinuxToWasi, WasiImports};
    use wasmparser::{FunctionBody, Operator, Parser, Payload};

    #[test]
    fn handler_plan_covers_linux_to_wasi_table() {
        let imports = WasiImports {
            fd_write: 0,
            fd_read: 1,
            fd_close: 2,
            proc_exit: 3,
        };
        let table = LinuxToWasi::new(imports).build_table(|n| n as u32);
        let plan = HandlerModulePlan::from_table(&table, 10);
        assert_eq!(plan.handlers.len(), table.entries().len());
        assert!(plan.handlers.iter().any(|(n, _)| *n == 64));
    }

    #[test]
    fn close_handler_calls_wasi_import_without_inline_br_table() {
        let entry = SyscallEntry {
            func_idx: 2,
            param_map: alloc::vec![ParamSource::LocalI64AsI32(10)],
            saves: alloc::vec![],
            result_local: Some(10),
            negate_nonzero_result: true,
            has_return: true,
            terminates: false,
            memory_stores: alloc::vec![],
            load_mem_on_success: None,
        };
        let f = emit_handler_function(&entry, None);
        let raw = f.into_raw_body();
        let body = FunctionBody::new(wasmparser::BinaryReader::new(&raw, 0));
        let mut ops = body.get_operators_reader().unwrap();
        assert!(matches!(ops.read().unwrap(), Operator::LocalGet { local_index: 10 }));
        assert!(matches!(ops.read().unwrap(), Operator::I32WrapI64));
        assert!(matches!(
            ops.read().unwrap(),
            Operator::Call { function_index: 2 }
        ));
    }
}
