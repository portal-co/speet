//! `DispatchMode::FunctionRef`-mode sibling of `end_to_end_aarch64.rs`: the
//! same proof (a guest AArch64 `ret` instruction, compiled by vane's real
//! AArch64 frontend, linked into a live dynamic dispatch table, reached via
//! a tail call from the lookup stub, all 73 registers threaded correctly),
//! but the dynamic table's hit path dispatches via `table.get` +
//! `return_call_ref` against a typed funcref table, executed under
//! `wasmtime` — `wasmi` has no function-references support, so this mode
//! can only run here. Completes AArch64's dispatch-mode coverage to match
//! RISC-V's `end_to_end_funcref.rs`.
//!
//! Gated behind the `wasmtime` feature; an empty (zero-test) compilation
//! unit otherwise.

#![cfg(feature = "wasmtime")]

use speet_dynamic_jit::compile_pc_aarch64;
use speet_dynamic_jit::wasmtime_link::link_jit_function_funcref;
use speet_interp::emit_jit_lookup_stub;
use speet_link_core::{DispatchMode, IndexSlot, JitConfig, OobConfig};
use vane_aarch64::Mem;
use wasm_encoder_240 as wasm_encoder;
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, HeapType, ImportSection, Instruction, MemorySection, MemoryType,
    Module, RefType, TableSection, TableType, TypeSection, ValType,
};

const NUM_REGS: u32 = 73;
const TARGET_PC_LOCAL: u32 = NUM_REGS;

fn regfn_params() -> Vec<ValType> {
    vec![ValType::I64; (TARGET_PC_LOCAL + 1) as usize]
}
fn regfn_results() -> Vec<ValType> {
    vec![ValType::I64; NUM_REGS as usize]
}

/// `DispatchMode::FunctionRef` sibling of `end_to_end_aarch64.rs::build_main_module`.
/// Table 1 is a typed funcref table `(ref null 0)` instead of an untyped one.
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
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 0: static PC table (empty)
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 1: dyn table side-table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 2: hit-count table (unused)
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 3: guest memory

    let mut exports = ExportSection::new();
    exports.export("lookup_stub", ExportKind::Func, 2);
    exports.export("dyn_funcref_table", ExportKind::Table, 1);
    exports.export("dyn_table_mem", ExportKind::Memory, 1);
    exports.export("guest_memory", ExportKind::Memory, 3);

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

    // interp_stub: result[0] += 0xBAD; every other result passes through.
    let mut interp_fn = Function::new([]);
    interp_fn.instruction(&Instruction::LocalGet(0));
    interp_fn.instruction(&Instruction::I64Const(0xBAD));
    interp_fn.instruction(&Instruction::I64Add);
    for i in 1..NUM_REGS {
        interp_fn.instruction(&Instruction::LocalGet(i));
    }
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
fn oob_miss_then_dynamic_jit_compile_reaches_and_executes_aarch64_via_return_call_ref() {
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
    const RET_TARGET: i64 = 0x9999;

    let guest_bytes = 0xD65F_03C0u32.to_le_bytes(); // `ret` (x30)
    let main_wasm = build_main_module(&jit_config, TARGET_PC, &guest_bytes);

    let mut config = wasmtime::Config::default();
    config.wasm_function_references(true);
    let engine = wasmtime::Engine::new(&config).expect("engine should build");
    let mut store = wasmtime::Store::new(&engine, ());
    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    let ecall_ty = wasmtime::FuncType::new(
        &engine,
        vec![wasmtime::ValType::I64; NUM_REGS as usize],
        vec![wasmtime::ValType::I64; NUM_REGS as usize],
    );
    linker
        .func_new("env", "ecall", ecall_ty, |_caller, params: &[wasmtime::Val], results: &mut [wasmtime::Val]| {
            results.clone_from_slice(params);
            Ok(())
        })
        .unwrap();
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

    let call_lookup = |store: &mut wasmtime::Store<()>, regs: &[i64; NUM_REGS as usize], target_pc: i64| -> Vec<i64> {
        let mut params: Vec<wasmtime::Val> = regs.iter().map(|&r| wasmtime::Val::I64(r)).collect();
        params.push(wasmtime::Val::I64(target_pc));
        let mut results = vec![wasmtime::Val::I64(0); NUM_REGS as usize];
        lookup_stub.call(&mut *store, &params, &mut results).expect("call should not trap");
        results
            .into_iter()
            .map(|v| match v {
                wasmtime::Val::I64(x) => x,
                other => panic!("unexpected result type: {other:?}"),
            })
            .collect()
    };

    let mut seed = [0i64; NUM_REGS as usize];
    seed[0] = SEED;
    seed[30] = RET_TARGET;

    let results = call_lookup(&mut store, &seed, TARGET_PC as i64);
    assert_eq!(results[0], SEED + 0xBAD, "unknown PC must fall through to the interpreter");

    let mut mem = Mem::default();
    for (i, b) in guest_bytes.into_iter().enumerate() {
        mem.write_byte(TARGET_PC + i as u64, b);
    }
    let jit_wasm = compile_pc_aarch64(&mem, TARGET_PC);

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

    let results = call_lookup(&mut store, &seed, TARGET_PC as i64);
    assert_eq!(
        results[0],
        SEED + 0xBAD,
        "dynamically JIT'd AArch64 `ret` must be reached via return_call_ref and tail-call back to \
         lookup_stub with x30 as the new target_pc, missing again and falling through to the interpreter"
    );
    for (i, &r) in results.iter().enumerate() {
        if i != 0 {
            assert_eq!(r, seed[i], "register x{i} must be threaded through the tail-call chain unchanged");
        }
    }
}
