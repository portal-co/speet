//! Integration tests for the JIT-aware lookup stub (Phase 1 of the dynamic-JIT plan).
//!
//! Three things are verified:
//!
//! 1. `emit_lookup_stub_dispatch` with `oob.jit: None` reproduces
//!    `emit_lookup_stub`'s output byte-for-byte (a pure encoding comparison,
//!    no execution needed).
//! 2. The three-tier dispatch (static table -> dynamic table -> hit-counted
//!    interpreter fallback with fire-and-forget JIT request) behaves
//!    correctly, by assembling a real WASM module and executing it in
//!    `wasmi`.
//! 3. `emit_lookup_stub` itself produces valid, executable WASM (a
//!    regression test — see that function's doc comment for the bug this
//!    guards against).

use speet_interp::{
    emit_jit_lookup_stub, emit_lookup_stub, emit_lookup_stub_dispatch, jit_lookup_stub_extra_locals,
};
use speet_link_core::{DispatchMode, IndexSlot, JitConfig, OobConfig};
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, Elements, EntityType, ExportKind,
    ExportSection, Function, FunctionSection, ImportSection, Instruction, MemorySection,
    MemoryType, Module, RefType, TableSection, TableType, TypeSection, ValType,
};

type E = core::convert::Infallible;

fn base_oob(interp_func_idx: u32, jit: Option<JitConfig>) -> OobConfig {
    OobConfig {
        lookup_stub_func_idx: 0,
        interp_func_idx,
        dispatch_table_slot: IndexSlot(0),
        jit,
    }
}

// ── Test 1: `jit: None` reproduces `emit_lookup_stub` byte-for-byte ────────

#[test]
fn none_reproduces_emit_lookup_stub_byte_for_byte() {
    let oob = base_oob(3, None);
    let params = [ValType::I64, ValType::I64];

    let mut direct = Function::new([
        (1, ValType::I32),
        (1, ValType::I32),
        (1, ValType::I32),
        (1, ValType::I64),
        (1, ValType::I32),
    ]);
    emit_lookup_stub::<(), E>(&mut direct, &mut (), &oob, &params, 0, 0, 0, 1).unwrap();

    let mut via_dispatch = Function::new([
        (1, ValType::I32),
        (1, ValType::I32),
        (1, ValType::I32),
        (1, ValType::I64),
        (1, ValType::I32),
    ]);
    emit_lookup_stub_dispatch::<(), E>(&mut via_dispatch, &mut (), &oob, &params, 0, 0, 0, None, 0, 1)
        .unwrap();

    assert_eq!(direct, via_dispatch);
}

// ── Test 2: three-tier dispatch, executed end-to-end in wasmi ──────────────

struct Built {
    wasm: Vec<u8>,
}

/// Assemble a minimal module exercising the JIT-aware lookup stub:
///
/// * Static table (table 0) has one entry: pc=100 -> the "identity" function.
/// * Dynamic table (table 1, capacity 4) has one seeded entry: pc=200 -> the
///   "dyn_func" function (reg0 + 42), simulating a JIT backend that already
///   compiled and linked a function for that PC.
/// * Hit-count table (capacity 4) starts empty; `hit_threshold` is 2.
/// * Anything else falls to the "interp" function (reg0 + 999).
fn build_module(hit_threshold: u32) -> Built {
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64, ValType::I64], [ValType::I64]); // type 0: regfn
    types.ty().function([ValType::I64], []); // type 1: jit_request

    let mut imports = ImportSection::new();
    imports.import("env", "jit_request", EntityType::Function(1)); // func idx 0

    let mut functions = FunctionSection::new();
    functions.function(0); // func 1: identity
    functions.function(0); // func 2: jit_lookup (under test)
    functions.function(0); // func 3: interp
    functions.function(0); // func 4: dyn_func

    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: 1,
        maximum: Some(1),
        table64: false,
        shared: false,
    }); // table 0: static dispatch
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: 4,
        maximum: Some(4),
        table64: false,
        shared: false,
    }); // table 1: dynamic dispatch

    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // mem 0: static PC table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // mem 1: dynamic dispatch side table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // mem 2: hit-count side table

    let mut exports = ExportSection::new();
    exports.export("lookup_stub", ExportKind::Func, 2);
    exports.export("memory", ExportKind::Memory, 0);

    let mut elements = ElementSection::new();
    let offset0 = ConstExpr::i32_const(0);
    elements.active(Some(0), &offset0, Elements::Functions(std::borrow::Cow::Borrowed(&[1])));
    elements.active(Some(1), &offset0, Elements::Functions(std::borrow::Cow::Borrowed(&[4])));

    let mut data = DataSection::new();
    // mem0: static PC table, one entry: (pc=100, table_slot=0)
    let mut static_table = Vec::new();
    static_table.extend_from_slice(&100u64.to_le_bytes());
    static_table.extend_from_slice(&0u32.to_le_bytes());
    data.active(0, &ConstExpr::i32_const(0), static_table);

    // mem1: dynamic dispatch side table, capacity 4, one occupied entry:
    // (pc=200, table_slot=0) -> table1[0] = dyn_func; rest empty (-1 sentinel).
    let mut dyn_table = Vec::new();
    dyn_table.extend_from_slice(&200u64.to_le_bytes());
    dyn_table.extend_from_slice(&0u32.to_le_bytes());
    for _ in 0..3 {
        dyn_table.extend_from_slice(&0u64.to_le_bytes());
        dyn_table.extend_from_slice(&(-1i32).to_le_bytes());
    }
    data.active(1, &ConstExpr::i32_const(0), dyn_table);
    // mem2 (hit-count table): left zero-initialised, no data segment needed.

    let jit = JitConfig {
        dyn_dispatch_table_slot: IndexSlot(0),
        dispatch_mode: DispatchMode::TableIndirect,
        dyn_funcref_table_slot: None,
        dyn_table_mem_idx: 1,
        dyn_table_mem_offset: 0,
        dyn_table_capacity: 4,
        hit_table_mem_idx: 2,
        hit_table_mem_offset: 0,
        hit_table_capacity: 4,
        hit_threshold,
        jit_request_func_idx: Some(0),
    };
    let oob = base_oob(3, Some(jit.clone()));
    let params = [ValType::I64, ValType::I64];

    let mut identity_fn = Function::new([]);
    identity_fn.instruction(&Instruction::LocalGet(0));
    identity_fn.instruction(&Instruction::End);

    let mut lookup_fn = {
        let mut locals: Vec<(u32, ValType)> = vec![
            (1, ValType::I32),
            (1, ValType::I32),
            (1, ValType::I32),
            (1, ValType::I64),
            (1, ValType::I32),
        ];
        locals.extend(jit_lookup_stub_extra_locals());
        Function::new(locals)
    };
    emit_jit_lookup_stub::<(), E>(
        &mut lookup_fn,
        &mut (),
        &oob,
        &jit,
        &params,
        /* type_idx */ 0,
        /* table_idx */ 0,
        /* dyn_table_idx */ 1,
        /* dyn_funcref_table_idx */ None,
        /* data_mem_idx */ 0,
        /* n_entries */ 1,
    )
    .unwrap();

    let mut interp_fn = Function::new([]);
    interp_fn.instruction(&Instruction::LocalGet(0));
    interp_fn.instruction(&Instruction::I64Const(999));
    interp_fn.instruction(&Instruction::I64Add);
    interp_fn.instruction(&Instruction::End);

    let mut dyn_fn = Function::new([]);
    dyn_fn.instruction(&Instruction::LocalGet(0));
    dyn_fn.instruction(&Instruction::I64Const(42));
    dyn_fn.instruction(&Instruction::I64Add);
    dyn_fn.instruction(&Instruction::End);

    let mut code = CodeSection::new();
    code.function(&identity_fn);
    code.function(&lookup_fn);
    code.function(&interp_fn);
    code.function(&dyn_fn);

    let mut module = Module::new();
    module
        .section(&types)
        .section(&imports)
        .section(&functions)
        .section(&tables)
        .section(&memories)
        .section(&exports)
        .section(&elements)
        .section(&code)
        .section(&data);

    Built { wasm: module.finish() }
}

#[derive(Default)]
struct HostState {
    jit_requests: Vec<i64>,
}

fn instantiate(wasm: &[u8]) -> (wasmi::Store<HostState>, wasmi::TypedFunc<(i64, i64), i64>) {
    let mut config = wasmi::Config::default();
    config.wasm_tail_call(true);
    config.wasm_multi_memory(true);
    let engine = wasmi::Engine::new(&config);
    let mut store = wasmi::Store::new(&engine, HostState::default());
    let mut linker: wasmi::Linker<HostState> = wasmi::Linker::new(&engine);
    linker
        .func_wrap("env", "jit_request", |mut caller: wasmi::Caller<'_, HostState>, pc: i64| {
            caller.data_mut().jit_requests.push(pc);
        })
        .unwrap();
    let module = wasmi::Module::new(&engine, wasm).expect("module should validate");
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("module should instantiate");
    let func = instance
        .get_typed_func::<(i64, i64), i64>(&store, "lookup_stub")
        .expect("lookup_stub export should exist with the expected type");
    (store, func)
}

#[test]
fn static_table_hit_dispatches_to_compiled_function() {
    let built = build_module(2);
    let (mut store, func) = instantiate(&built.wasm);
    let result = func.call(&mut store, (7, 100)).unwrap();
    assert_eq!(result, 7, "pc=100 is in the static table and should hit the identity function");
    assert!(store.data().jit_requests.is_empty());
}

#[test]
fn dynamic_table_hit_dispatches_to_jitted_function() {
    let built = build_module(2);
    let (mut store, func) = instantiate(&built.wasm);
    let result = func.call(&mut store, (7, 200)).unwrap();
    assert_eq!(result, 7 + 42, "pc=200 is in the dynamic table and should hit dyn_func");
    assert!(store.data().jit_requests.is_empty());
}

#[test]
fn miss_falls_through_to_interpreter_and_counts_hits() {
    let built = build_module(2);
    let (mut store, func) = instantiate(&built.wasm);

    // First miss: count becomes 1, below hit_threshold=2 -> no JIT request yet.
    let r1 = func.call(&mut store, (7, 300)).unwrap();
    assert_eq!(r1, 7 + 999, "unknown pc should fall through to the interpreter");
    assert!(store.data().jit_requests.is_empty(), "hit_threshold not yet reached");

    // Second miss on the same PC: count becomes 2, crosses hit_threshold=2.
    let r2 = func.call(&mut store, (7, 300)).unwrap();
    assert_eq!(r2, 7 + 999, "still falls through to the interpreter regardless of JIT request");
    assert_eq!(
        store.data().jit_requests,
        vec![300],
        "crossing hit_threshold should fire exactly one fire-and-forget JIT request for pc=300"
    );
}

#[test]
fn zero_threshold_never_requests_jit() {
    let built = build_module(0);
    let (mut store, func) = instantiate(&built.wasm);
    for _ in 0..5 {
        let r = func.call(&mut store, (1, 300)).unwrap();
        assert_eq!(r, 1 + 999);
    }
    assert!(store.data().jit_requests.is_empty(), "hit_threshold=0 must disable JIT requests");
}

// ── Test 3: `emit_lookup_stub` itself produces valid, executable WASM ──────
//
// Regression test for the bug this file's module doc used to work around:
// the binary-search loop's `BrIf(1)` break wasn't wrapped in a `Block`, so
// it targeted the function's own implicit (non-empty) result frame instead
// of "after the loop" and failed WASM validation whenever `params`/results
// were non-empty -- i.e. always, per `OobConfig`'s own calling convention.
// `none_reproduces_emit_lookup_stub_byte_for_byte` above only ever compared
// `emit_lookup_stub`'s output against itself (via `emit_lookup_stub_dispatch`
// with `jit: None`), so it could not have caught this: both sides were
// wrong in the same way. This test instead builds a real module and
// validates + executes it.

fn build_static_only_module() -> Vec<u8> {
    let oob = base_oob(2, None);
    let params = [ValType::I64, ValType::I64];

    let mut types = TypeSection::new();
    types.ty().function(params, [ValType::I64]);

    let mut functions = FunctionSection::new();
    functions.function(0); // func 0: identity
    functions.function(0); // func 1: lookup_stub (under test)
    functions.function(0); // func 2: interp

    let mut tables = TableSection::new();
    tables.table(TableType { element_type: RefType::FUNCREF, minimum: 1, maximum: Some(1), table64: false, shared: false });

    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None });

    let mut exports = ExportSection::new();
    exports.export("lookup_stub", ExportKind::Func, 1);

    let mut elements = ElementSection::new();
    elements.active(Some(0), &ConstExpr::i32_const(0), Elements::Functions(std::borrow::Cow::Borrowed(&[0])));

    let mut data = DataSection::new();
    let mut static_table = Vec::new();
    static_table.extend_from_slice(&100u64.to_le_bytes());
    static_table.extend_from_slice(&0u32.to_le_bytes());
    data.active(0, &ConstExpr::i32_const(0), static_table);

    let mut identity_fn = Function::new([]);
    identity_fn.instruction(&Instruction::LocalGet(0));
    identity_fn.instruction(&Instruction::End);

    let mut lookup_fn = Function::new([
        (1, ValType::I32), (1, ValType::I32), (1, ValType::I32), (1, ValType::I64), (1, ValType::I32),
    ]);
    let params_slice = [ValType::I64, ValType::I64];
    emit_lookup_stub::<(), E>(&mut lookup_fn, &mut (), &oob, &params_slice, 0, 0, 0, 1).unwrap();

    let mut interp_fn = Function::new([]);
    interp_fn.instruction(&Instruction::LocalGet(0));
    interp_fn.instruction(&Instruction::I64Const(999));
    interp_fn.instruction(&Instruction::I64Add);
    interp_fn.instruction(&Instruction::End);

    let mut code = CodeSection::new();
    code.function(&identity_fn);
    code.function(&lookup_fn);
    code.function(&interp_fn);

    let mut module = Module::new();
    module
        .section(&types)
        .section(&functions)
        .section(&tables)
        .section(&memories)
        .section(&exports)
        .section(&elements)
        .section(&code)
        .section(&data);
    module.finish()
}

#[test]
fn emit_lookup_stub_output_validates_and_executes() {
    let wasm = build_static_only_module();

    let mut config = wasmi::Config::default();
    config.wasm_tail_call(true);
    let engine = wasmi::Engine::new(&config);
    let mut store = wasmi::Store::new(&engine, ());
    let linker: wasmi::Linker<()> = wasmi::Linker::new(&engine);
    let module = wasmi::Module::new(&engine, &wasm[..]).expect("emit_lookup_stub output must be valid WASM");
    let instance = linker.instantiate_and_start(&mut store, &module).expect("should instantiate");
    let func = instance
        .get_typed_func::<(i64, i64), i64>(&store, "lookup_stub")
        .expect("lookup_stub export should exist with the expected type");

    assert_eq!(func.call(&mut store, (7, 100)).unwrap(), 7, "static hit should dispatch to the identity function");
    assert_eq!(func.call(&mut store, (7, 999)).unwrap(), 7 + 999, "miss should fall through to the interpreter");
}
