//! Patch `unreachable` opcodes to log guest PC via `env.__speet_log_unreachable`.

use wasm_encoder::{Function, Instruction};
use wasmparser::{FunctionBody, Operator};

/// How to read the guest PC inside a translated WASM function body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuestPcRef {
    /// x86_64: PC lives in local 16 (`i64`).
    LocalI64(u32),
    /// aarch64: PC is function param 31 (`i64`).
    ParamI64(u32),
}

impl GuestPcRef {
    pub fn for_arch(arch: binary_io::BinArch) -> Self {
        match arch {
            binary_io::BinArch::X86_64 => GuestPcRef::LocalI64(16),
            binary_io::BinArch::AArch64 => GuestPcRef::ParamI64(31),
        }
    }

    fn emit_load_pc(&self, out: &mut Function) {
        match *self {
            GuestPcRef::LocalI64(idx) => {
                out.instruction(&Instruction::LocalGet(idx));
            }
            GuestPcRef::ParamI64(idx) => {
                out.instruction(&Instruction::LocalGet(idx));
            }
        }
        // Stub takes i32; guest PCs fit in i32 for same-platform v1 guests.
        out.instruction(&Instruction::I32WrapI64);
    }
}

/// Module import index of `env.__speet_log_unreachable` (must match assembly order).
pub const LOG_UNREACHABLE_IMPORT_IDX: u32 = 3;

/// Patch every translated function before integrated module assembly.
pub fn instrument_unreachable_logging(fns: &mut [Function], pc_ref: GuestPcRef) {
    for f in fns.iter_mut() {
        let raw = std::mem::replace(f, Function::new([])).into_raw_body();
        *f = patch_function_body_bytes(&raw, pc_ref)
            .unwrap_or_else(|e| panic!("instrument unreachable: {e}"));
    }
}

fn patch_function_body_bytes(raw: &[u8], pc_ref: GuestPcRef) -> Result<Function, String> {
    let reader = wasmparser::BinaryReader::new(raw, 0);
    let body = FunctionBody::new(reader);
    patch_function_body(body, pc_ref)
}

fn patch_function_body(body: FunctionBody<'_>, pc_ref: GuestPcRef) -> Result<Function, String> {
    let locals_reader = body.get_locals_reader().map_err(|e| e.to_string())?;
    let mut locals = Vec::new();
    let mut lr = locals_reader;
    for _ in 0..lr.get_count() {
        let (n, ty) = lr.read().map_err(|e| e.to_string())?;
        locals.push((n, val_type(ty)));
    }

    let mut ops_reader = body.get_operators_reader().map_err(|e| e.to_string())?;
    let mut out = Function::new(locals);
    while !ops_reader.eof() {
        let op = ops_reader.read().map_err(|e| e.to_string())?;
        match op {
            Operator::Unreachable => {
                pc_ref.emit_load_pc(&mut out);
                out.instruction(&Instruction::Call(LOG_UNREACHABLE_IMPORT_IDX));
                out.instruction(&Instruction::Unreachable);
            }
            other => {
                let instr: Instruction<'_> =
                    other.try_into().map_err(|_| "reencode operator failed".to_string())?;
                out.instruction(&instr);
            }
        }
    }
    Ok(out)
}

fn val_type(ty: wasmparser::ValType) -> wasm_encoder::ValType {
    match ty {
        wasmparser::ValType::I32 => wasm_encoder::ValType::I32,
        wasmparser::ValType::I64 => wasm_encoder::ValType::I64,
        wasmparser::ValType::F32 => wasm_encoder::ValType::F32,
        wasmparser::ValType::F64 => wasm_encoder::ValType::F64,
        wasmparser::ValType::V128 => wasm_encoder::ValType::V128,
        other => panic!("unsupported valtype: {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_encoder::ValType;

    #[test]
    fn patches_unreachable_with_pc_load_and_call() {
        let mut f = Function::new([]);
        f.instruction(&Instruction::I32Const(0x1000));
        f.instruction(&Instruction::LocalSet(16));
        f.instruction(&Instruction::Unreachable);
        f.instruction(&Instruction::End);

        instrument_unreachable_logging(core::slice::from_mut(&mut f), GuestPcRef::LocalI64(16));

        let raw = f.clone().into_raw_body();
        let reader = wasmparser::BinaryReader::new(&raw, 0);
        let body = FunctionBody::new(reader);
        let mut ops = body.get_operators_reader().unwrap();
        assert!(matches!(ops.read().unwrap(), Operator::I32Const { value: 0x1000 }));
        assert!(matches!(
            ops.read().unwrap(),
            Operator::LocalSet { local_index: 16 }
        ));
        assert!(matches!(ops.read().unwrap(), Operator::LocalGet { local_index: 16 }));
        assert!(matches!(ops.read().unwrap(), Operator::I32WrapI64));
        assert!(matches!(
            ops.read().unwrap(),
            Operator::Call { function_index: 3 }
        ));
        assert!(matches!(ops.read().unwrap(), Operator::Unreachable));
    }
}
