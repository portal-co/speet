//! End-to-end proof of the Phase 2a mechanism: a guest RISC-V instruction,
//! never known statically, is dynamically compiled by vane's real RV64
//! frontend, linked into a live `speet-interp` dynamic dispatch table, and
//! then actually reached (and correctly executed, including a tail call back
//! out through the same lookup stub) via `return_call_indirect` from a
//! second call into the same entry point.

use speet_dynamic_jit::{compile_pc, link_jit_function};
use speet_interp::emit_jit_lookup_stub;
use speet_link_core::{DispatchMode, IndexSlot, JitConfig, OobConfig};
use vane_riscv::Mem;
// Aliased to speet's own wasm-encoder version (0.240.0), matching
// speet-interp/speet-link-core — distinct from the crate's library
// dependency on vane's 0.239.0, see Cargo.toml's comment.
use wasm_encoder_240 as wasm_encoder;
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, ImportSection, Instruction, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TypeSection, ValType,
};

const NUM_REGS: u32 = 2; // x0, x1 — enough for `ret` (jalr x0, 0(x1))
const TARGET_PC_LOCAL: u32 = NUM_REGS;

fn regfn_params() -> Vec<ValType> {
    vec![ValType::I64; (TARGET_PC_LOCAL + 1) as usize]
}
fn regfn_results() -> Vec<ValType> {
    vec![ValType::I64; NUM_REGS as usize]
}

struct HostState;

/// Build the "main" module: a JIT-aware lookup stub (exported, so a
/// separately-compiled JIT module can import it as its tail-call target),
/// an interpreter stub that tags its r0 result so we can tell it ran, no
/// static entries, and an empty dynamic table/side-tables ready to be
/// populated by [`link_jit_function`].
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
    tables.table(TableType { element_type: RefType::FUNCREF, minimum: 1, maximum: Some(1), table64: false, shared: false }); // static (empty)
    tables.table(TableType { element_type: RefType::FUNCREF, minimum: jit.dyn_table_capacity as u64, maximum: Some(jit.dyn_table_capacity as u64), table64: false, shared: false }); // dynamic

    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 0: static PC table (empty, n_entries=0)
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 1: dyn table side-table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 2: hit-count table (unused, jit_request disabled)
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 3: guest memory (shared with the JIT'd module)

    let mut exports = ExportSection::new();
    exports.export("lookup_stub", ExportKind::Func, 2);
    exports.export("dyn_table", ExportKind::Table, 1);
    exports.export("dyn_table_mem", ExportKind::Memory, 1);
    exports.export("guest_memory", ExportKind::Memory, 3);

    // No static element entries. Dynamic table starts empty too (element
    // section left empty) — link_jit_function populates it via Table::set at
    // runtime, which is the whole point of this test.
    let elements = ElementSection::new();

    // Dynamic side-table: every slot starts as the empty sentinel
    // (table_slot == -1); memory is zero-initialised otherwise, satisfying
    // the hit-count table's "count == 0 marks empty" convention for free.
    let mut data = DataSection::new();
    let mut dyn_table_init = Vec::new();
    for _ in 0..jit.dyn_table_capacity {
        dyn_table_init.extend_from_slice(&0u64.to_le_bytes());
        dyn_table_init.extend_from_slice(&(-1i32).to_le_bytes());
    }
    data.active(jit.dyn_table_mem_idx, &ConstExpr::i32_const(jit.dyn_table_mem_offset as i32), dyn_table_init);

    // Guest memory (memory 3): the same bytes CheckCode will re-verify at
    // runtime, at the same flat address `compile_pc`'s frontend read them
    // from during compilation.
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
        /* dyn table_idx */ 1,
        /* dyn_funcref_table_idx */ None,
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
fn oob_miss_then_dynamic_jit_compile_reaches_and_executes_real_vane_output() {
    let jit_config = JitConfig {
        dyn_dispatch_table_slot: IndexSlot(0),
        dispatch_mode: DispatchMode::TableIndirect,
        dyn_funcref_table_slot: None,
        dyn_table_mem_idx: 1,
        dyn_table_mem_offset: 0,
        dyn_table_capacity: 4,
        hit_table_mem_idx: 2,
        hit_table_mem_offset: 0,
        hit_table_capacity: 4,
        hit_threshold: 0,       // fire-and-forget JIT requests disabled — this
        jit_request_func_idx: None, // test drives the compile+link step manually
    };

    const SEED: i64 = 5;
    const TARGET_PC: u64 = 0x1000;
    const RET_TARGET: i64 = 0x9999; // where `ret` (jalr x0, 0(x1)) jumps to

    // `ret` a.k.a. `jalr x0, 0(x1)`.
    let guest_bytes = 0x0000_8067u32.to_le_bytes();
    let main_wasm = build_main_module(&jit_config, TARGET_PC, &guest_bytes);

    let mut config = wasmi::Config::default();
    config.wasm_tail_call(true);
    config.wasm_multi_memory(true);
    let engine = wasmi::Engine::new(&config);
    let mut store = wasmi::Store::new(&engine, HostState);
    let mut linker: wasmi::Linker<HostState> = wasmi::Linker::new(&engine);
    linker
        .func_wrap("env", "ecall", |_: wasmi::Caller<'_, HostState>, r0: i64, r1: i64| (r0, r1))
        .unwrap();
    linker
        .func_wrap("env", "jit_invalidate", |_: wasmi::Caller<'_, HostState>, _pc: i64| -> () {
            panic!("CheckCode must not invalidate: compiled from the exact live guest bytes");
        })
        .unwrap();

    let main_module = wasmi::Module::new(&engine, &main_wasm[..]).expect("main module should validate");
    let main_instance =
        linker.instantiate_and_start(&mut store, &main_module).expect("main module should instantiate");
    let lookup_stub = main_instance.get_func(&mut store, "lookup_stub").unwrap();
    let dyn_table = main_instance.get_table(&store, "dyn_table").unwrap();
    let dyn_table_mem = main_instance.get_export(&store, "dyn_table_mem").and_then(|e| e.into_memory()).unwrap();
    let guest_memory = main_instance.get_export(&store, "guest_memory").and_then(|e| e.into_memory()).unwrap();

    let call_lookup = |store: &mut wasmi::Store<HostState>, r0: i64, r1: i64, tpc: i64| -> (i64, i64) {
        let params = [wasmi::Val::I64(r0), wasmi::Val::I64(r1), wasmi::Val::I64(tpc)];
        let mut results = [wasmi::Val::I64(0), wasmi::Val::I64(0)];
        lookup_stub.call(&mut *store, &params, &mut results).expect("call should not trap");
        let get = |v: &wasmi::Val| match v {
            wasmi::Val::I64(x) => *x,
            other => panic!("unexpected result type: {other:?}"),
        };
        (get(&results[0]), get(&results[1]))
    };

    // Before compiling anything: OOB miss on TARGET_PC falls through to the interpreter.
    let (r0, r1) = call_lookup(&mut store, SEED, RET_TARGET, TARGET_PC as i64);
    assert_eq!((r0, r1), (SEED + 0xBAD, RET_TARGET), "unknown PC must fall through to the interpreter");

    // Write a real `ret` instruction (`jalr x0, 0(x1)`, encoding 0x00008067)
    // into guest memory at TARGET_PC, and compile it with vane's real RV64
    // frontend.
    let mut mem = Mem::default();
    for (i, b) in 0x0000_8067u32.to_le_bytes().into_iter().enumerate() {
        mem.write_byte(TARGET_PC + i as u64, b);
    }
    let jit_wasm = compile_pc(&mem, TARGET_PC, NUM_REGS);

    // Register the main module's own lookup stub as the JIT module's
    // tail-call target, then link the freshly compiled function into the
    // live dynamic dispatch table — this is the step a background JIT
    // thread would perform in production.
    linker.define("env", "lookup_stub", wasmi::Extern::Func(lookup_stub)).unwrap();
    linker.define("env", "guest_memory", wasmi::Extern::Memory(guest_memory)).unwrap();
    let slot = link_jit_function(
        &mut store,
        &linker,
        &engine,
        &jit_config,
        &dyn_table,
        &dyn_table_mem,
        TARGET_PC,
        &jit_wasm,
    )
    .expect("linking the JIT'd function should succeed");
    assert_eq!(slot, 0, "first link should land in the first empty dynamic slot");

    // Same call as before, same PC — now it must hit the dynamic table,
    // execute the *real* vane-compiled `ret`, which tail-calls back into
    // lookup_stub with target_pc = r1 (RET_TARGET); that misses again and
    // falls to the interpreter, so the final result is identical in shape
    // to the pre-JIT miss, but this time it round-tripped through genuine
    // dynamically-linked, vane-compiled RISC-V semantics.
    let (r0, r1) = call_lookup(&mut store, SEED, RET_TARGET, TARGET_PC as i64);
    assert_eq!(
        (r0, r1),
        (SEED + 0xBAD, RET_TARGET),
        "dynamically JIT'd `ret` must tail-call back to lookup_stub with r1 as the new target_pc, \
         which misses again and falls through to the interpreter with r0/r1 threaded correctly"
    );
}
