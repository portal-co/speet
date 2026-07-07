//! Assemble translated speet functions into a runnable WASM module.

use std::borrow::Cow;

use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, ExportKind, ExportSection, Function,
    FunctionSection, ImportSection, MemorySection, MemoryType, Module, RefType, TableSection,
    TableType, TypeSection, ValType,
};

/// `env.__speet_hint`, `env.write`, `env.exit`, `env.__speet_unreachable_trap`.
pub const N_CORPUS_IMPORTS: u32 = 4;
/// Module function index of `env.__speet_unreachable_trap`.
pub const TRAP_IMPORT_IDX: u32 = 3;

/// Build a runnable module exporting `_start` at `entry_func_idx`.
pub fn assemble_corpus_module(
    fns: &[Function],
    params: &[ValType],
    entry_func_idx: u32,
) -> Vec<u8> {
    let mut types = TypeSection::new();
    types.ty().function(params.to_vec(), []);
    types.ty().function([ValType::I32], []);
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32], [ValType::I32]);
    types.ty().function([ValType::I32], []); // unreachable trap

    let mut imports = ImportSection::new();
    imports.import("env", "__speet_hint", wasm_encoder::EntityType::Function(1));
    imports.import("env", "write", wasm_encoder::EntityType::Function(2));
    imports.import("env", "exit", wasm_encoder::EntityType::Function(1));
    imports.import(
        "env",
        "__speet_unreachable_trap",
        wasm_encoder::EntityType::Function(3),
    );

    let mut funcs = FunctionSection::new();
    for _ in fns {
        funcs.function(0);
    }

    let total = fns.len() as u32;
    let table_size = N_CORPUS_IMPORTS + total;
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
        minimum: 64,
        maximum: None,
        memory64: true,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("_start", ExportKind::Func, entry_func_idx);

    let indices: Vec<u32> = (N_CORPUS_IMPORTS..N_CORPUS_IMPORTS + total).collect();
    let mut elems = ElementSection::new();
    elems.active(
        Some(0),
        &ConstExpr::i64_const(N_CORPUS_IMPORTS as i64),
        Elements::Functions(Cow::Borrowed(&indices)),
    );

    let mut code = CodeSection::new();
    for f in fns {
        code.function(f);
    }

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
