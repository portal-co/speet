//! `DispatchMode::FunctionRef`-mode sibling of `end_to_end.rs`: the same
//! proof (a guest RISC-V instruction, never known statically, is compiled
//! by vane's real RV64 frontend, linked into a live dynamic dispatch table,
//! and reached via a tail call from the lookup stub), but the dynamic
//! table's hit path dispatches via `table.get` + `return_call_ref` against
//! a typed funcref table, executed under `wasmtime` — `wasmi` has no
//! function-references support, so this mode can only run here.
//!
//! Gated behind the `wasmtime` feature; an empty (zero-test) compilation
//! unit otherwise.

#![cfg(feature = "wasmtime")]

use speet_dynamic_jit::compile_pc;
use speet_dynamic_jit::wasmtime_link::link_jit_function_funcref;
use speet_interp::emit_jit_lookup_stub;
use speet_link_core::{DispatchMode, IndexSlot, JitConfig, OobConfig};
use vane_riscv::Mem;
// Aliased to speet's own wasm-encoder version (0.240.0) — see end_to_end.rs's
// comment on this same alias for why the two wasm-encoder versions never
// need to interoperate as typed values here.
use wasm_encoder_240 as wasm_encoder;
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, HeapType, ImportSection, Instruction, MemorySection, MemoryType,
    Module, RefType, TableSection, TableType, TypeSection, ValType,
};

const NUM_REGS: u32 = 2; // x0, x1 — enough for `ret` (jalr x0, 0(x1))
const TARGET_PC_LOCAL: u32 = NUM_REGS;

fn regfn_params() -> Vec<ValType> {
    vec![ValType::I64; (TARGET_PC_LOCAL + 1) as usize]
}
fn regfn_results() -> Vec<ValType> {
    vec![ValType::I64; NUM_REGS as usize]
}

/// Builds the "main" module, `DispatchMode::FunctionRef` sibling of
/// `end_to_end.rs::build_main_module`. Table 1 is a *typed* funcref table
/// `(ref null 0)` (heap type `Concrete(0)`, `0` being `regfn`'s own type
/// index) instead of an untyped `funcref` table — the dynamic tier's only
/// consumer under `FunctionRef` mode.
fn build_main_module(jit: &JitConfig, guest_pc: u64, guest_bytes: &[u8]) -> Vec<u8> {
    let oob = OobConfig {
        lookup_stub_func_idx: 2,
        interp_func_idx: 3,
        dispatch_table_slot: IndexSlot(0),
        jit: Some(jit.clone()),
    };

    let mut types = TypeSection::new();
    types.ty().function(regfn_params(), regfn_results()); // type 0: regfn
    types.ty().function(vec![ValType::I64; NUM_REGS as usize], vec![ValType::I64; NUM_REGS as usize]); // type 1: ecall
    types.ty().function([ValType::I64], []); // type 2: jit_invalidate

    let mut imports = ImportSection::new();
    imports.import("env", "ecall", EntityType::Function(1)); // func 0
    imports.import("env", "jit_invalidate", EntityType::Function(2)); // func 1

    let mut functions = FunctionSection::new();
    functions.function(0); // func 2: lookup_stub
    functions.function(0); // func 3: interp_stub

    let mut tables = TableSection::new();
    tables.table(TableType { element_type: RefType::FUNCREF, minimum: 1, maximum: Some(1), table64: false, shared: false }); // table 0: static (empty)
    tables.table(TableType {
        element_type: RefType { nullable: true, heap_type: HeapType::Concrete(0) }, // (ref null $regfn)
        minimum: jit.dyn_table_capacity as u64,
        maximum: Some(jit.dyn_table_capacity as u64),
        table64: false,
        shared: false,
    }); // table 1: typed dynamic dispatch

    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 0: static PC table (empty, n_entries=0)
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 1: dyn table side-table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 2: hit-count table (unused, jit_request disabled)
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 3: guest memory (shared with the JIT'd module)

    let mut exports = ExportSection::new();
    exports.export("lookup_stub", ExportKind::Func, 2);
    exports.export("dyn_funcref_table", ExportKind::Table, 1);
    exports.export("dyn_table_mem", ExportKind::Memory, 1);
    exports.export("guest_memory", ExportKind::Memory, 3);

    // No static element entries; table 1 starts empty too — link_jit_function_funcref
    // populates it via a host-side Table::set, the whole point of this test.
    let elements = ElementSection::new();

    let mut data = DataSection::new();
    let mut dyn_table_init = Vec::new();
    for _ in 0..jit.dyn_table_capacity {
        dyn_table_init.extend_from_slice(&0u64.to_le_bytes());
        dyn_table_init.extend_from_slice(&(-1i32).to_le_bytes());
    }
    data.active(jit.dyn_table_mem_idx, &ConstExpr::i32_const(jit.dyn_table_mem_offset as i32), dyn_table_init);
    data.active(3, &ConstExpr::i32_const(guest_pc as i32), guest_bytes.to_vec());

    let mut lookup_fn = {
        let mut locals: Vec<(u32, ValType)> = vec![
            (1, ValType::I32), (1, ValType::I32), (1, ValType::I32), (1, ValType::I64), (1, ValType::I32),
        ];
        locals.extend(speet_interp::jit_lookup_stub_extra_locals());
        Function::new(locals)
    };
    emit_jit_lookup_stub::<(), core::convert::Infallible>(
        &mut lookup_fn,
        &mut (),
        &oob,
        jit,
        &regfn_params(),
        /* type_idx */ 0,
        /* static table_idx */ 0,
        /* dyn table_idx (unused under FunctionRef) */ 0,
        /* dyn_funcref_table_idx */ Some(1),
        /* data_mem_idx (static) */ 0,
        /* n_entries */ 0,
    )
    .unwrap();

    // interp_stub: r0 += 0xBAD (tags that this path ran), r1 unchanged.
    let mut interp_fn = Function::new([]);
    interp_fn.instruction(&Instruction::LocalGet(0));
    interp_fn.instruction(&Instruction::I64Const(0xBAD));
    interp_fn.instruction(&Instruction::I64Add);
    interp_fn.instruction(&Instruction::LocalGet(1));
    interp_fn.instruction(&Instruction::End);

    let mut code = CodeSection::new();
    code.function(&lookup_fn);
    code.function(&interp_fn);

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
    module.finish()
}

#[test]
fn oob_miss_then_dynamic_jit_compile_reaches_and_executes_via_return_call_ref() {
    let jit_config = JitConfig {
        dyn_dispatch_table_slot: IndexSlot(0),
        dispatch_mode: DispatchMode::FunctionRef,
        dyn_funcref_table_slot: Some(IndexSlot(1)),
        dyn_table_mem_idx: 1,
        dyn_table_mem_offset: 0,
        dyn_table_capacity: 4,
        hit_table_mem_idx: 2,
        hit_table_mem_offset: 0,
        hit_table_capacity: 4,
        hit_threshold: 0,
        jit_request_func_idx: None,
    };

    const SEED: i64 = 5;
    const TARGET_PC: u64 = 0x1000;
    const RET_TARGET: i64 = 0x9999; // where `ret` (jalr x0, 0(x1)) jumps to

    let guest_bytes = 0x0000_8067u32.to_le_bytes(); // `ret`
    let main_wasm = build_main_module(&jit_config, TARGET_PC, &guest_bytes);

    let mut config = wasmtime::Config::default();
    config.wasm_function_references(true); // wasm_tail_call is already true by default
    let engine = wasmtime::Engine::new(&config).expect("engine should build");
    let mut store = wasmtime::Store::new(&engine, ());
    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    linker.func_wrap("env", "ecall", |r0: i64, r1: i64| (r0, r1 + 1000)).unwrap();
    linker
        .func_wrap("env", "jit_invalidate", |_pc: i64| -> () {
            panic!("CheckCode must not invalidate: compiled from the exact live guest bytes");
        })
        .unwrap();

    let main_module = wasmtime::Module::new(&engine, &main_wasm[..]).expect("main module should validate");
    let main_instance = linker.instantiate(&mut store, &main_module).expect("main module should instantiate");
    let lookup_stub = main_instance.get_func(&mut store, "lookup_stub").unwrap();
    let dyn_funcref_table = main_instance.get_table(&mut store, "dyn_funcref_table").unwrap();
    let dyn_table_mem = main_instance.get_memory(&mut store, "dyn_table_mem").unwrap();
    let guest_memory = main_instance.get_memory(&mut store, "guest_memory").unwrap();

    let call_lookup = |store: &mut wasmtime::Store<()>, r0: i64, r1: i64, tpc: i64| -> (i64, i64) {
        let params = [wasmtime::Val::I64(r0), wasmtime::Val::I64(r1), wasmtime::Val::I64(tpc)];
        let mut results = [wasmtime::Val::I64(0), wasmtime::Val::I64(0)];
        lookup_stub.call(&mut *store, &params, &mut results).expect("call should not trap");
        let get = |v: &wasmtime::Val| match v {
            wasmtime::Val::I64(x) => *x,
            other => panic!("unexpected result type: {other:?}"),
        };
        (get(&results[0]), get(&results[1]))
    };

    // Before compiling anything: OOB miss on TARGET_PC falls through to the interpreter.
    let (r0, r1) = call_lookup(&mut store, SEED, RET_TARGET, TARGET_PC as i64);
    assert_eq!((r0, r1), (SEED + 0xBAD, RET_TARGET), "unknown PC must fall through to the interpreter");

    // Write a real `ret` instruction into guest memory and compile it with
    // vane's real RV64 frontend — `compile_pc` itself is dispatch-mode-
    // agnostic (its own TailCall lowering uses a plain function-index
    // ReturnCall to the lookup stub, not a table), so this step is
    // byte-for-byte identical to end_to_end.rs's.
    let mut mem = Mem::default();
    for (i, b) in guest_bytes.into_iter().enumerate() {
        mem.write_byte(TARGET_PC + i as u64, b);
    }
    let jit_wasm = compile_pc(&mem, TARGET_PC, NUM_REGS);

    linker.define(&store, "env", "lookup_stub", lookup_stub).unwrap();
    linker.define(&store, "env", "guest_memory", guest_memory).unwrap();
    let slot = link_jit_function_funcref(
        &mut store,
        &linker,
        &engine,
        &jit_config,
        &dyn_funcref_table,
        &dyn_table_mem,
        TARGET_PC,
        &jit_wasm,
    )
    .expect("linking the JIT'd function should succeed");
    assert_eq!(slot, 0, "first link should land in the first empty dynamic slot");

    // Same call as before, same PC — now it must hit the typed dynamic
    // table via table.get + return_call_ref, execute the real vane-compiled
    // `ret`, which tail-calls back into lookup_stub with target_pc = r1.
    let (r0, r1) = call_lookup(&mut store, SEED, RET_TARGET, TARGET_PC as i64);
    assert_eq!(
        (r0, r1),
        (SEED + 0xBAD, RET_TARGET),
        "dynamically JIT'd `ret` must be reached via return_call_ref and tail-call back to lookup_stub \
         with r1 as the new target_pc, which misses again and falls through to the interpreter"
    );
}
