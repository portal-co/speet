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
///
/// Type 0 (the register-file type every translated function shares) is
/// **symmetric** — `(registers) -> (registers)`, never `-> ()` — per
/// `docs/guides/thin-runtime-genericity.md` principle 4 in the `speet`
/// repo: "every speet-emitted function has type `(registers) -> (registers)`
/// — no exceptions". A module appends one **halt stub** past the last
/// translated function (see `speet_recompile::frontend::{build_halt_stub,
/// halt_addr}`) so a guest `ret`/indirect return past the end of the
/// translated set — e.g. `main` returning with no crt0 chain, the common
/// case for this harness's `arith`/`pairs`/`frame` corpus programs — has
/// somewhere to land that surfaces the live register file as the call's
/// WASM results, instead of the previous `-> ()` type silently discarding
/// the final return value.
pub fn assemble_corpus_module(
    fns: &[Function],
    params: &[ValType],
    entry_func_idx: u32,
) -> Vec<u8> {
    let mut types = TypeSection::new();
    types.ty().function(params.to_vec(), params.to_vec());
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

    let halt_stub = speet_recompile::frontend::build_halt_stub(params.len() as u32);

    let mut funcs = FunctionSection::new();
    for _ in fns {
        funcs.function(0);
    }
    funcs.function(0); // halt stub: also type 0, (registers) -> (registers)

    let total = fns.len() as u32;
    let table_size = N_CORPUS_IMPORTS + total + 1;
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

    let indices: Vec<u32> = (N_CORPUS_IMPORTS..N_CORPUS_IMPORTS + total + 1).collect();
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
