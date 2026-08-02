//! Extract and import-remap handler bodies from [`super::CANONICAL_GUEST_WASM`].

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use wasm_encoder::{Function, Instruction};
use wasmparser::{FunctionBody, Operator, Parser, Payload, TypeRef};

use super::WasiImports;
use super::{
    EXPORT_HANDLER_CLOSE, EXPORT_HANDLER_EXIT, EXPORT_HANDLER_READ, EXPORT_HANDLER_WRITE,
    EXPORT_SYSCALL_DISPATCH,
};

/// One guest handler function ready to append after translated RV64 code.
pub struct GuestHandlerFunctions {
    /// Defined functions in order: close, exit, read, write, dispatch.
    pub functions: Vec<Function>,
    /// Type-section indices for each function in `functions` (guest-module relative).
    pub type_indices: Vec<u32>,
    /// Index of [`EXPORT_SYSCALL_DISPATCH`] within `functions`.
    pub syscall_dispatch_offset: u32,
}

/// Parse the embedded guest module and rewrite calls to megabinary indices.
pub fn extract_guest_handlers(
    wasi: &WasiImports,
    handler_base: u32,
) -> Result<GuestHandlerFunctions, String> {
    let guest = super::CANONICAL_GUEST_WASM;
    let module = parse_guest_module(guest)?;
    let import_remap = guest_import_remap(&module, wasi)?;
    let defined_remap = guest_defined_remap(&module, handler_base);

    let order = [
        EXPORT_HANDLER_CLOSE,
        EXPORT_HANDLER_EXIT,
        EXPORT_HANDLER_READ,
        EXPORT_HANDLER_WRITE,
        EXPORT_SYSCALL_DISPATCH,
    ];

    let mut export_to_defined: BTreeMap<&str, u32> = BTreeMap::new();
    for name in order {
        let abs_idx = *module
            .exports
            .get(name)
            .ok_or_else(|| format!("guest export missing: {name}"))?;
        let defined_idx = abs_idx
            .checked_sub(module.n_import_funcs)
            .ok_or_else(|| format!("export {name} is not a defined function (idx {abs_idx})"))?;
        export_to_defined.insert(name, defined_idx);
    }

    let mut defined_bodies: BTreeMap<u32, (Function, u32)> = BTreeMap::new();
    let mut code_idx = 0u32;
    for payload in Parser::new(0).parse_all(guest) {
        if let Payload::CodeSectionEntry(body) = payload.map_err(|e| format!("{e:?}"))? {
            if let Some((_, defined_idx)) = export_to_defined
                .iter()
                .find(|(_, idx)| **idx == code_idx)
            {
                let ty_idx = module.func_type_indices[code_idx as usize];
                let remapped = rewrite_body(body, &import_remap, &defined_remap)?;
                defined_bodies.insert(*defined_idx, (remapped, ty_idx));
            }
            code_idx += 1;
        }
    }

    let mut functions = Vec::with_capacity(order.len());
    let mut type_indices = Vec::with_capacity(order.len());
    let mut syscall_dispatch_offset = 0u32;

    for (out_idx, name) in order.iter().enumerate() {
        let defined_idx = export_to_defined[name];
        let (func, ty_idx) = defined_bodies
            .remove(&defined_idx)
            .ok_or_else(|| format!("missing code for guest handler {name}"))?;
        functions.push(func);
        type_indices.push(ty_idx);
        if *name == EXPORT_SYSCALL_DISPATCH {
            syscall_dispatch_offset = out_idx as u32;
        }
    }

    Ok(GuestHandlerFunctions {
        functions,
        type_indices,
        syscall_dispatch_offset,
    })
}

struct ParsedGuestModule {
    n_import_funcs: u32,
    func_type_indices: Vec<u32>,
    exports: BTreeMap<String, u32>,
}

fn guest_import_remap(
    _module: &ParsedGuestModule,
    wasi: &WasiImports,
) -> Result<BTreeMap<u32, u32>, String> {
    let guest = super::CANONICAL_GUEST_WASM;
    let mut guest_imports: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut import_func_idx = 0u32;
    for payload in Parser::new(0).parse_all(guest) {
        if let Payload::ImportSection(reader) = payload.map_err(|e| format!("{e:?}"))? {
            for imp in reader {
                let imp = imp.map_err(|e| format!("{e:?}"))?;
                if let TypeRef::Func(_) = imp.ty {
                    guest_imports.insert(
                        (imp.module.to_string(), imp.name.to_string()),
                        import_func_idx,
                    );
                    import_func_idx += 1;
                }
            }
        }
    }

    let megabinary = [
        ("wasi_snapshot_preview1", "fd_write", wasi.fd_write),
        ("wasi_snapshot_preview1", "fd_read", wasi.fd_read),
        ("wasi_snapshot_preview1", "fd_close", wasi.fd_close),
        ("wasi_snapshot_preview1", "proc_exit", wasi.proc_exit),
    ];

    let mut remap = BTreeMap::new();
    for (module_name, name, meg_idx) in megabinary {
        let guest_idx = guest_imports
            .get(&(module_name.into(), name.into()))
            .ok_or_else(|| format!("guest module missing import {module_name}::{name}"))?;
        remap.insert(*guest_idx, meg_idx);
    }
    Ok(remap)
}

fn guest_defined_remap(module: &ParsedGuestModule, handler_base: u32) -> BTreeMap<u32, u32> {
    let mut remap = BTreeMap::new();
    for defined_idx in 0..module.func_type_indices.len() as u32 {
        let guest_abs = module.n_import_funcs + defined_idx;
        remap.insert(guest_abs, handler_base + defined_idx);
    }
    remap
}

fn parse_guest_module(guest: &[u8]) -> Result<ParsedGuestModule, String> {
    let mut func_type_indices = Vec::new();
    let mut exports = BTreeMap::new();
    let mut n_import_funcs = 0u32;

    for payload in Parser::new(0).parse_all(guest) {
        match payload.map_err(|e| format!("{e:?}"))? {
            Payload::ImportSection(reader) => {
                for imp in reader {
                    if matches!(
                        imp.map_err(|e| format!("{e:?}"))?.ty,
                        TypeRef::Func(_)
                    ) {
                        n_import_funcs += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for ty in reader {
                    func_type_indices.push(ty.map_err(|e| format!("{e:?}"))?);
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.map_err(|e| format!("{e:?}"))?;
                    if export.kind == wasmparser::ExternalKind::Func {
                        exports.insert(export.name.to_string(), export.index);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(ParsedGuestModule {
        n_import_funcs,
        func_type_indices,
        exports,
    })
}

fn rewrite_body(
    body: FunctionBody<'_>,
    import_remap: &BTreeMap<u32, u32>,
    defined_remap: &BTreeMap<u32, u32>,
) -> Result<Function, String> {
    let locals_reader = body
        .get_locals_reader()
        .map_err(|e| format!("{e:?}"))?;
    let mut locals = Vec::new();
    let mut lr = locals_reader;
    for _ in 0..lr.get_count() {
        let (n, ty) = lr.read().map_err(|e| format!("{e:?}"))?;
        locals.push((n, val_type(ty)));
    }

    let mut ops = body
        .get_operators_reader()
        .map_err(|e| format!("{e:?}"))?;
    let mut out = Function::new(locals);
    while !ops.eof() {
        let op = ops.read().map_err(|e| format!("{e:?}"))?;
        match op {
            Operator::Call { function_index } => {
                let idx = remap_func(function_index, import_remap, defined_remap);
                out.instruction(&Instruction::Call(idx));
            }
            Operator::ReturnCall { function_index } => {
                let idx = remap_func(function_index, import_remap, defined_remap);
                out.instruction(&Instruction::ReturnCall(idx));
            }
            other => {
                let instr: Instruction<'_> = other
                    .try_into()
                    .map_err(|_| "reencode operator failed".to_string())?;
                out.instruction(&instr);
            }
        }
    }
    Ok(out)
}

fn remap_func(
    idx: u32,
    import_remap: &BTreeMap<u32, u32>,
    defined_remap: &BTreeMap<u32, u32>,
) -> u32 {
    if let Some(&new_idx) = import_remap.get(&idx) {
        return new_idx;
    }
    if let Some(&new_idx) = defined_remap.get(&idx) {
        return new_idx;
    }
    idx
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

    #[test]
    fn extracts_five_handlers_with_dispatch_last() {
        let wasi = WasiImports {
            fd_write: 0,
            fd_read: 1,
            fd_close: 2,
            proc_exit: 3,
        };
        let guest = extract_guest_handlers(&wasi, 10).expect("extract");
        assert_eq!(guest.functions.len(), 5);
        assert_eq!(guest.syscall_dispatch_offset, 4);
    }
}
