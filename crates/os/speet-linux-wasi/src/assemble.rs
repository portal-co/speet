//! Assemble a Linux→WASI megabinary: WASI imports + translated RV64 + guest handlers.

use alloc::vec::Vec;
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, ImportSection, Instruction, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TypeSection, ValType,
};
use wasmparser::{Parser, Payload};

use crate::manifest::wasi_preview1_manifest;
use crate::merge::{extract_guest_handlers, GuestHandlerFunctions};
use crate::translate::WasiTranslation;

/// Host scratch memory index (guest linear memory stays at 0).
pub const HOST_MEMORY_INDEX: u32 = 1;

/// Build a runnable WASM module from [`translate_rv64_wasi`] output.
pub fn assemble_wasi_module(t: &WasiTranslation) -> Vec<u8> {
    let manifest = wasi_preview1_manifest();
    let n_imports = manifest.func_imports.len() as u32;
    let n_translated = t.fns.len() as u32;
    let guest =
        extract_guest_handlers(&t.wasi, n_imports + n_translated).expect("guest handler extract");
    let guest_types = guest_type_section_entries();
    let guest_type_base = 1 + n_imports;

    let mut types = TypeSection::new();
    types
        .ty()
        .function(t.params.clone(), t.params.clone());
    for imp in &manifest.func_imports {
        types.ty().function(
            imp.params.iter().copied().map(wasm_val_type),
            imp.results.iter().copied().map(wasm_val_type),
        );
    }
    for (params, results) in &guest_types {
        types.ty().function(params.clone(), results.clone());
    }

    let mut imports = ImportSection::new();
    for (i, imp) in manifest.func_imports.iter().enumerate() {
        imports.import(
            &imp.module,
            &imp.name,
            EntityType::Function((i + 1) as u32),
        );
    }

    let n_register_params = t.params.len() as u32;
    let halt_stub = build_halt_stub(n_register_params);

    let mut funcs = FunctionSection::new();
    for _ in 0..n_translated {
        funcs.function(0);
    }
    for ty in &guest.megabinary_type_indices(guest_type_base) {
        funcs.function(*ty);
    }
    funcs.function(0); // halt stub

    let table_size = n_imports + n_translated + 1;
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: table_size as u64,
        maximum: Some(table_size as u64),
        table64: true,
        shared: false,
    });

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    mems.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("_start", ExportKind::Func, n_imports + t.entry_func_idx);

    let indices: Vec<u32> = (n_imports..n_imports + n_translated + 1).collect();
    let mut elems = ElementSection::new();
    elems.active(
        Some(0),
        &ConstExpr::i64_const(n_imports as i64),
        Elements::Functions(alloc::borrow::Cow::Borrowed(&indices)),
    );

    let mut code = CodeSection::new();
    for f in &t.fns {
        code.function(f);
    }
    for f in &guest.functions {
        code.function(f);
    }
    code.function(&halt_stub);

    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    module.section(&tables);
    module.section(&mems);
    module.section(&exports);
    module.section(&elems);
    module.section(&code);
    module.finish()
}

/// Translate RV64 Linux `.text` and assemble a WASI megabinary.
pub fn recompile_rv64_wasi_to_wasm(text: &[u8], start_addr: u64) -> Vec<u8> {
    let t = crate::translate::translate_rv64_wasi(text, start_addr);
    assemble_wasi_module(&t)
}

fn build_halt_stub(n_params: u32) -> Function {
    let mut f = Function::new([]);
    for p in 0..n_params {
        f.instruction(&Instruction::LocalGet(p));
    }
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);
    f
}

fn guest_type_section_entries() -> Vec<(Vec<ValType>, Vec<ValType>)> {
    let mut types = Vec::new();
    for payload in Parser::new(0).parse_all(crate::CANONICAL_GUEST_WASM) {
        if let Payload::TypeSection(reader) = payload.expect("guest wasm parse") {
            for group in reader {
                let group = group.expect("guest type group");
                for sub_ty in group.types() {
                    if let wasmparser::CompositeInnerType::Func(ref ft) =
                        sub_ty.composite_type.inner
                    {
                        types.push((
                            ft.params()
                                .iter()
                                .copied()
                                .map(val_type)
                                .collect(),
                            ft.results()
                                .iter()
                                .copied()
                                .map(val_type)
                                .collect(),
                        ));
                    }
                }
            }
        }
    }
    types
}

fn wasm_val_type(t: speet_host_api::WasmValType) -> ValType {
    match t {
        speet_host_api::WasmValType::I32 => ValType::I32,
        speet_host_api::WasmValType::I64 => ValType::I64,
        speet_host_api::WasmValType::F32 => ValType::F32,
        speet_host_api::WasmValType::F64 => ValType::F64,
    }
}

fn val_type(ty: wasmparser::ValType) -> ValType {
    match ty {
        wasmparser::ValType::I32 => ValType::I32,
        wasmparser::ValType::I64 => ValType::I64,
        wasmparser::ValType::F32 => ValType::F32,
        wasmparser::ValType::F64 => ValType::F64,
        wasmparser::ValType::V128 => ValType::V128,
        other => panic!("unsupported valtype: {other:?}"),
    }
}

impl GuestHandlerFunctions {
    fn megabinary_type_indices(&self, guest_type_base: u32) -> Vec<u32> {
        self.type_indices
            .iter()
            .map(|idx| guest_type_base + idx)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::translate_rv64_wasi;

    const WRITE_EXIT: &[u8] = &[
        0x13, 0x05, 0x10, 0x00, // addi a0, x0, 1
        0x93, 0x05, 0x80, 0x20, // addi a1, x0, 520
        0x13, 0x06, 0x60, 0x00, // addi a2, x0, 6
        0x93, 0x08, 0x00, 0x04, // addi a7, x0, 64
        0x73, 0x00, 0x00, 0x00, // ecall (write)
        0x13, 0x05, 0x00, 0x00, // addi a0, x0, 0
        0x93, 0x08, 0xd0, 0x05, // addi a7, x0, 93
        0x73, 0x00, 0x00, 0x00, // ecall (exit)
    ];

    #[test]
    fn wasi_module_validates() {
        let wasm = recompile_rv64_wasi_to_wasm(WRITE_EXIT, 0x1000);
        wasmparser::validate(&wasm).expect("assembled WASI module");
    }

    #[test]
    fn syscall_dispatch_lands_after_handlers() {
        let t = translate_rv64_wasi(WRITE_EXIT, 0x1000);
        let n_imports = wasi_preview1_manifest().func_imports.len() as u32;
        assert_eq!(
            t.syscall_dispatch_idx,
            n_imports + t.fns.len() as u32 + 4
        );
    }
}
