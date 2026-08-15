//! Proves Phase 5's SMC-invalidation path end-to-end: a JIT'd function's
//! `CheckCode` mismatch (guest bytes changed since it was compiled) evicts
//! its own dynamic-dispatch-table entry via `invalidate_dyn_entry`, and a
//! second dispatch to the same PC correctly misses the dynamic table
//! afterward instead of looping back into the now-stale entry.

use speet_dynamic_jit::{compile_pc, invalidate_dyn_entry, link_jit_function};
use speet_interp::emit_jit_lookup_stub;
use speet_link_core::{DispatchMode, IndexSlot, JitConfig, OobConfig};
use vane_riscv::Mem;
use wasm_encoder_240 as wasm_encoder;
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, ImportSection, Instruction, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TypeSection, ValType,
};

const NUM_REGS: u32 = 2;
const TARGET_PC_LOCAL: u32 = NUM_REGS;

fn regfn_params() -> Vec<ValType> {
    vec![ValType::I64; (TARGET_PC_LOCAL + 1) as usize]
}
fn regfn_results() -> Vec<ValType> {
    vec![ValType::I64; NUM_REGS as usize]
}

struct HostState;

fn build_main_module(jit: &JitConfig, guest_pc: u64, guest_bytes: &[u8]) -> Vec<u8> {
    let oob = OobConfig {
        lookup_stub_func_idx: 2,
        interp_func_idx: 3,
        dispatch_table_slot: IndexSlot(0),
        jit: Some(jit.clone()),
    };

    let mut types = TypeSection::new();
    types.ty().function(regfn_params(), regfn_results());
    types.ty().function(vec![ValType::I64; NUM_REGS as usize], vec![ValType::I64; NUM_REGS as usize]);
    types.ty().function([ValType::I64], []);

    let mut imports = ImportSection::new();
    imports.import("env", "ecall", EntityType::Function(1));
    imports.import("env", "jit_invalidate", EntityType::Function(2));

    let mut functions = FunctionSection::new();
    functions.function(0); // lookup_stub
    functions.function(0); // interp_stub

    let mut tables = TableSection::new();
    tables.table(TableType { element_type: RefType::FUNCREF, minimum: 1, maximum: Some(1), table64: false, shared: false });
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: jit.dyn_table_capacity as u64,
        maximum: Some(jit.dyn_table_capacity as u64),
        table64: false,
        shared: false,
    });

    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 0: static table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 1: dyn side-table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 2: hit table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 3: guest memory

    let mut exports = ExportSection::new();
    exports.export("lookup_stub", ExportKind::Func, 2);
    exports.export("dyn_table", ExportKind::Table, 1);
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
        &mut lookup_fn, &mut (), &oob, jit, &regfn_params(), 0, 0, 1, None, 0, 0,
    )
    .unwrap();

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
fn check_code_mismatch_evicts_stale_dynamic_entry_and_falls_through_cleanly() {
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
        hit_threshold: 0,
        jit_request_func_idx: None,
    };

    const SEED: i64 = 5;
    const TARGET_PC: u64 = 0x1000;
    const RET_TARGET: i64 = 0x9999;
    let ret_bytes = 0x0000_8067u32.to_le_bytes(); // `ret` / jalr x0, 0(x1)

    let main_wasm = build_main_module(&jit_config, TARGET_PC, &ret_bytes);

    let mut config = wasmi::Config::default();
    config.wasm_tail_call(true);
    config.wasm_multi_memory(true);
    let engine = wasmi::Engine::new(&config);
    let mut store = wasmi::Store::new(&engine, HostState);
    let mut linker: wasmi::Linker<HostState> = wasmi::Linker::new(&engine);
    linker.func_wrap("env", "ecall", |_: wasmi::Caller<'_, HostState>, r0: i64, r1: i64| (r0, r1)).unwrap();
    // Stub for the main module's own (unused) import -- real behavior gets
    // wired in below, once dyn_table_mem is available.
    linker.func_wrap("env", "jit_invalidate", |_: wasmi::Caller<'_, HostState>, _pc: i64| -> () {}).unwrap();

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

    // Compile and link the real `ret` at TARGET_PC.
    let mut mem = Mem::default();
    for (i, b) in ret_bytes.into_iter().enumerate() {
        mem.write_byte(TARGET_PC + i as u64, b);
    }
    let jit_wasm = compile_pc(&mem, TARGET_PC, NUM_REGS);

    // A fresh linker for instantiating the JIT'd module: wasmi::Linker
    // rejects redefining a name once `func_wrap`'d (unlike os_daemon's own
    // registries, which explicitly allow replacement), so the real
    // `jit_invalidate` -- which needs `dyn_table_mem`, only available after
    // the main module is already instantiated -- can't just overwrite the
    // stub used above. `ecall` is re-registered identically; `lookup_stub`/
    // `guest_memory` point back at the main instance's own exports, exactly
    // as `linker.define` does for them elsewhere in this crate's tests.
    let mut jit_linker: wasmi::Linker<HostState> = wasmi::Linker::new(&engine);
    jit_linker.func_wrap("env", "ecall", |_: wasmi::Caller<'_, HostState>, r0: i64, r1: i64| (r0, r1)).unwrap();
    jit_linker.define("env", "lookup_stub", wasmi::Extern::Func(lookup_stub)).unwrap();
    jit_linker.define("env", "guest_memory", wasmi::Extern::Memory(guest_memory)).unwrap();
    let jit_config_for_invalidate = jit_config.clone();
    jit_linker
        .func_wrap(
            "env",
            "jit_invalidate",
            move |mut caller: wasmi::Caller<'_, HostState>, pc: i64| {
                invalidate_dyn_entry(&mut caller, &jit_config_for_invalidate, &dyn_table_mem, pc as u64);
            },
        )
        .unwrap();

    link_jit_function(&mut store, &jit_linker, &engine, &jit_config, &dyn_table, &dyn_table_mem, TARGET_PC, &jit_wasm)
        .expect("linking the JIT'd function should succeed");

    // Sanity: the dynamic table entry exists before any SMC happens. (Can't
    // distinguish "hit the dynamic table, then `ret` tail-called onward" from
    // "missed and fell straight to the interpreter" by final register values
    // alone here -- `ret` never returns its target directly, it always
    // chains into another dispatch, which also misses in this minimal
    // fixture and lands on the same interpreter fallback shape either way.
    // Checking the side-table memory directly is the real signal.)
    {
        let data = dyn_table_mem.data(&store);
        let pc = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let table_slot = i32::from_le_bytes(data[8..12].try_into().unwrap());
        assert_eq!((pc, table_slot), (TARGET_PC, 0), "dynamic entry must exist before SMC");
    }

    // Self-modifying code: guest memory at TARGET_PC now holds something
    // else (`nop` / `addi x0, x0, 0`), but the dynamic table entry is still
    // the *old* compiled `ret`.
    for (i, b) in 0x0000_0013u32.to_le_bytes().into_iter().enumerate() {
        guest_memory.write(&mut store, TARGET_PC as usize + i, &[b]).unwrap();
    }

    // Dispatch again: CheckCode inside the stale compiled function must
    // detect the mismatch, evict its own dynamic-table entry via
    // jit_invalidate, then tail-call back to the lookup stub for the same
    // PC -- which must NOT find the (now-evicted) entry again, and instead
    // fall through cleanly to the interpreter, without looping or trapping.
    let (r0, r1) = call_lookup(&mut store, SEED, RET_TARGET, TARGET_PC as i64);
    assert_eq!((r0, r1), (SEED + 0xBAD, RET_TARGET), "must fall through to the interpreter after eviction");

    // And the dynamic table entry is verifiably gone.
    let data = dyn_table_mem.data(&store);
    let table_slot = i32::from_le_bytes(data[8..12].try_into().unwrap());
    assert_eq!(table_slot, -1, "slot 0's entry must have been evicted");
}
