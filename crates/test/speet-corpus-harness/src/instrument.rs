//! Rewrite `unreachable` opcodes to call `env.__speet_unreachable_trap` first.

use wasm_encoder::{Function, Instruction};
use wasmparser::{FunctionBody, Operator};

/// Patch every translated function before module assembly.
///
/// `trap_import_idx` must be the WASM import index of
/// `env.__speet_unreachable_trap` — derive it via
/// `ImportManifest::corpus_harness().index_of("env", "__speet_unreachable_trap")`.
pub fn instrument_functions(fns: &mut [Function], import_count: u32, trap_import_idx: u32) {
    for (i, f) in fns.iter_mut().enumerate() {
        let module_func_idx = import_count + i as u32;
        let raw = std::mem::replace(f, Function::new([])).into_raw_body();
        *f = patch_function_body_bytes(&raw, module_func_idx, trap_import_idx)
            .unwrap_or_else(|e| panic!("instrument func {module_func_idx}: {e}"));
    }
}

fn patch_function_body_bytes(
    raw: &[u8],
    module_func_idx: u32,
    trap_import_idx: u32,
) -> Result<Function, String> {
    let reader = wasmparser::BinaryReader::new(raw, 0);
    let body = FunctionBody::new(reader);
    patch_function_body(body, module_func_idx, trap_import_idx)
}

fn patch_function_body(
    body: FunctionBody<'_>,
    module_func_idx: u32,
    trap_import_idx: u32,
) -> Result<Function, String> {
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
                out.instruction(&Instruction::I32Const(module_func_idx as i32));
                out.instruction(&Instruction::Call(trap_import_idx));
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
