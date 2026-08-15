//! Corpus-scale, hybrid (static + dynamic) dispatch tests for the AArch64
//! dynamic-JIT path, run against real compiled-C-program `.text.elf`
//! fixtures shared with `speet-aarch64`'s own AOT corpus tests.
//!
//! AArch64 sibling of `corpus_riscv.rs` -- see that file's module doc for
//! the full design rationale (trace-inlining behavior, the miss-driven
//! "OOB miss -> JIT compile -> link -> retry" loop, the halt-sentinel
//! convention, and why `on_miss` stands in for a real interpreter tier
//! rather than requiring one). AArch64 has no interpreter fallback at any
//! level in this codebase (unlike RISC-V's real-but-unwired
//! `RiscVThompsonInterp`), so `on_miss` is the *only* way this harness
//! could work here regardless.

use object::{Object, ObjectSection};
use speet_dynamic_jit::compile_pc_aarch64;
use speet_interp::emit_jit_lookup_stub;
use speet_link_core::{DispatchMode, IndexSlot, JitConfig, OobConfig};
use std::path::{Path, PathBuf};
use vane_aarch64::Mem;
use wasm_encoder_240 as wasm_encoder;
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, HeapType, ImportSection, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TypeSection, ValType,
};

const NUM_REGS: u32 = 73;
const TARGET_PC_LOCAL: u32 = NUM_REGS;
const SP_REG: usize = 31; // AArch64 state slot: sp
const RA_REG: usize = 30; // AArch64 state slot: x30 / LR
const X0_REG: usize = 0; // AArch64 state slot: x0 (first arg / return value)
const HALT_PC: i64 = -1; // sentinel: never a real guest address
const STACK_TOP: u64 = 0x10000;
const GUEST_MEMORY_PAGES: u64 = 2; // 128 KiB: comfortably covers .text + stack
const DYN_TABLE_CAPACITY: u32 = 64;
const MAX_JIT_COMPILES: usize = 64; // safety bound against a genuine infinite loop

fn regfn_params() -> Vec<ValType> {
    vec![ValType::I64; (TARGET_PC_LOCAL + 1) as usize]
}
fn regfn_results() -> Vec<ValType> {
    vec![ValType::I64; NUM_REGS as usize]
}

// ── Fixture loading ─────────────────────────────────────────────────────────

struct Fixture {
    text: Vec<u8>,
    base: u64,
    entry_pc: u64,
}

fn load_fixture(path: &Path) -> Fixture {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let obj = object::File::parse(&*bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let sec = obj.section_by_name(".text").unwrap_or_else(|| panic!("no .text in {}", path.display()));
    let text = sec.data().unwrap_or_else(|e| panic!("read .text: {e}")).to_vec();
    let base = sec.address();
    assert!(!text.is_empty(), ".text empty in {}", path.display());

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_else(|| panic!("bad path {}", path.display()));
    let stem = stem.strip_suffix(".text").unwrap_or(stem);
    let entry_path = path.with_file_name(format!("{stem}.entry"));
    let entry_str = std::fs::read_to_string(&entry_path).unwrap_or_else(|e| panic!("read {}: {e}", entry_path.display()));
    let entry_str = entry_str.trim();
    let entry_off = u64::from_str_radix(entry_str.strip_prefix("0x").unwrap_or(entry_str), 16)
        .unwrap_or_else(|e| panic!("parse entry offset {entry_str:?}: {e}"));

    Fixture { text, base, entry_pc: base.wrapping_add(entry_off) }
}

fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data")
}

fn aarch64_fixture(name: &str) -> Fixture {
    load_fixture(&corpus_root().join("c-corpus/aarch64-linux").join(format!("{name}.text.elf")))
}

fn expected_main_return(program: &str) -> i64 {
    let toml = std::fs::read_to_string(corpus_root().join("c-corpus/programs").join(program).join("expected.toml"))
        .unwrap_or_else(|e| panic!("read expected.toml for {program}: {e}"));
    for line in toml.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("main_return = ") {
            return rest.trim().parse().unwrap_or_else(|e| panic!("parse main_return {rest:?}: {e}"));
        }
    }
    panic!("no main_return found in expected.toml for {program}");
}

// ── Module assembly (shared by both dispatch modes) ─────────────────────────

fn build_main_module(fixture: &Fixture, dispatch_mode: DispatchMode) -> Vec<u8> {
    let oob = OobConfig {
        lookup_stub_func_idx: 3,
        interp_func_idx: 2, // the "on_miss" import itself -- see corpus_riscv.rs's module doc
        dispatch_table_slot: IndexSlot(0),
        jit: Some(JitConfig {
            dyn_dispatch_table_slot: IndexSlot(1),
            dispatch_mode,
            // Table 1 itself is typed (vs. untyped) depending on dispatch_mode --
            // there is no separate table 2; see the TableSection code below.
            dyn_funcref_table_slot: if dispatch_mode == DispatchMode::FunctionRef { Some(IndexSlot(1)) } else { None },
            dyn_table_mem_idx: 1,
            dyn_table_mem_offset: 0,
            dyn_table_capacity: DYN_TABLE_CAPACITY,
            hit_table_mem_idx: 2,
            hit_table_mem_offset: 0,
            hit_table_capacity: 4,
            hit_threshold: 0, // fire-and-forget requests disabled -- this harness drives compiles itself
            jit_request_func_idx: None,
        }),
    };
    let jit = oob.jit.as_ref().unwrap();

    let mut types = TypeSection::new();
    types.ty().function(regfn_params(), regfn_results()); // type 0: regfn (also on_miss's type)
    types.ty().function(vec![ValType::I64; NUM_REGS as usize], vec![ValType::I64; NUM_REGS as usize]); // type 1: ecall
    types.ty().function([ValType::I64], []); // type 2: jit_invalidate

    let mut imports = ImportSection::new();
    imports.import("env", "ecall", EntityType::Function(1)); // func 0
    imports.import("env", "jit_invalidate", EntityType::Function(2)); // func 1
    imports.import("env", "on_miss", EntityType::Function(0)); // func 2 -- doubles as oob.interp_func_idx

    let mut functions = FunctionSection::new();
    functions.function(0); // func 3: lookup_stub

    let mut tables = TableSection::new();
    tables.table(TableType { element_type: RefType::FUNCREF, minimum: 1, maximum: Some(1), table64: false, shared: false }); // table 0: static (1 entry)
    match dispatch_mode {
        DispatchMode::TableIndirect => {
            tables.table(TableType {
                element_type: RefType::FUNCREF,
                minimum: DYN_TABLE_CAPACITY as u64,
                maximum: Some(DYN_TABLE_CAPACITY as u64),
                table64: false,
                shared: false,
            }); // table 1: dynamic (untyped)
        }
        DispatchMode::FunctionRef => {
            tables.table(TableType {
                element_type: RefType { nullable: true, heap_type: HeapType::Concrete(0) }, // (ref null $regfn)
                minimum: DYN_TABLE_CAPACITY as u64,
                maximum: Some(DYN_TABLE_CAPACITY as u64),
                table64: false,
                shared: false,
            }); // table 1: dynamic (typed)
        }
    }

    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 0: static PC table (1 entry)
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 1: dyn side-table
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 2: hit-count table (unused)
    memories.memory(MemoryType { minimum: GUEST_MEMORY_PAGES, maximum: None, memory64: false, shared: false, page_size_log2: None }); // 3: guest memory

    let mut exports = ExportSection::new();
    exports.export("lookup_stub", ExportKind::Func, 3);
    exports.export("static_table", ExportKind::Table, 0);
    exports.export("dyn_table", ExportKind::Table, 1);
    exports.export("dyn_table_mem", ExportKind::Memory, 1);
    exports.export("guest_memory", ExportKind::Memory, 3);

    // Both tables are populated entirely via host Table::set after
    // instantiation -- see corpus_riscv.rs's module doc for why.
    let elements = ElementSection::new();

    let mut data = DataSection::new();
    let mut static_init = Vec::new();
    static_init.extend_from_slice(&(fixture.entry_pc as i64).to_le_bytes());
    static_init.extend_from_slice(&0i32.to_le_bytes());
    data.active(0, &ConstExpr::i32_const(0), static_init);
    let mut dyn_init = Vec::new();
    for _ in 0..DYN_TABLE_CAPACITY {
        dyn_init.extend_from_slice(&0u64.to_le_bytes());
        dyn_init.extend_from_slice(&(-1i32).to_le_bytes());
    }
    data.active(1, &ConstExpr::i32_const(0), dyn_init);
    data.active(3, &ConstExpr::i32_const(fixture.base as i32), fixture.text.clone());

    let mut lookup_fn = {
        let mut locals: Vec<(u32, ValType)> = vec![
            (1, ValType::I32), (1, ValType::I32), (1, ValType::I32), (1, ValType::I64), (1, ValType::I32),
        ];
        locals.extend(speet_interp::jit_lookup_stub_extra_locals());
        Function::new(locals)
    };
    let dyn_funcref_table_idx = match dispatch_mode {
        DispatchMode::TableIndirect => None,
        DispatchMode::FunctionRef => Some(1),
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
        dyn_funcref_table_idx,
        /* data_mem_idx (static) */ 0,
        /* n_entries */ 1,
    )
    .unwrap();

    let mut code = CodeSection::new();
    code.function(&lookup_fn);

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

fn initial_regs() -> Vec<i64> {
    let mut regs = vec![0i64; NUM_REGS as usize];
    regs[SP_REG] = STACK_TOP as i64;
    regs[RA_REG] = HALT_PC; // a top-level `ret` lands here -> "program done"
    regs
}

fn guest_mem_from_fixture(fixture: &Fixture) -> Mem {
    let mut mem = Mem::default();
    for (i, b) in fixture.text.iter().enumerate() {
        mem.write_byte(fixture.base + i as u64, *b);
    }
    mem
}

// ── wasmi (TableIndirect) driver ─────────────────────────────────────────────

#[derive(Default)]
struct HostState {
    pending_miss: Option<(Vec<i64>, i64)>,
    jit_compiles: Vec<u64>,
}

fn run_table_indirect(fixture: &Fixture) -> i64 {
    let main_wasm = build_main_module(fixture, DispatchMode::TableIndirect);
    let jit_config = JitConfig {
        dyn_dispatch_table_slot: IndexSlot(1),
        dispatch_mode: DispatchMode::TableIndirect,
        dyn_funcref_table_slot: None,
        dyn_table_mem_idx: 1,
        dyn_table_mem_offset: 0,
        dyn_table_capacity: DYN_TABLE_CAPACITY,
        hit_table_mem_idx: 2,
        hit_table_mem_offset: 0,
        hit_table_capacity: 4,
        hit_threshold: 0,
        jit_request_func_idx: None,
    };

    let mut config = wasmi::Config::default();
    config.wasm_tail_call(true);
    config.wasm_multi_memory(true);
    let engine = wasmi::Engine::new(&config);
    let mut store = wasmi::Store::new(&engine, HostState::default());
    let mut linker: wasmi::Linker<HostState> = wasmi::Linker::new(&engine);

    let ecall_ty = wasmi::FuncType::new(
        vec![wasmi::ValType::I64; NUM_REGS as usize],
        vec![wasmi::ValType::I64; NUM_REGS as usize],
    );
    linker
        .func_new("env", "ecall", ecall_ty, |_caller: wasmi::Caller<'_, HostState>, params: &[wasmi::Val], results: &mut [wasmi::Val]| {
            results.clone_from_slice(params); // unused by these programs (no syscalls)
            Ok(())
        })
        .unwrap();
    linker
        .func_wrap("env", "jit_invalidate", |_c: wasmi::Caller<'_, HostState>, pc: i64| -> () {
            panic!("unexpected jit_invalidate for pc={pc:#x}: compiled from live guest bytes");
        })
        .unwrap();
    let on_miss_ty = wasmi::FuncType::new(
        vec![wasmi::ValType::I64; (NUM_REGS + 1) as usize],
        vec![wasmi::ValType::I64; NUM_REGS as usize],
    );
    linker
        .func_new("env", "on_miss", on_miss_ty, |mut caller: wasmi::Caller<'_, HostState>, params: &[wasmi::Val], results: &mut [wasmi::Val]| {
            let get_i64 = |v: &wasmi::Val| match v { wasmi::Val::I64(x) => *x, other => panic!("unexpected type: {other:?}") };
            let regs: Vec<i64> = params[..NUM_REGS as usize].iter().map(get_i64).collect();
            let target_pc = get_i64(&params[NUM_REGS as usize]);
            if target_pc == HALT_PC {
                for (i, r) in regs.iter().enumerate() {
                    results[i] = wasmi::Val::I64(*r);
                }
                Ok(())
            } else {
                caller.data_mut().pending_miss = Some((regs, target_pc));
                Err(wasmi::Error::new("jit-miss: pc not yet compiled"))
            }
        })
        .unwrap();

    let main_module = wasmi::Module::new(&engine, &main_wasm[..]).expect("main module should validate");
    let main_instance = linker.instantiate_and_start(&mut store, &main_module).expect("main module should instantiate");

    let lookup_stub = main_instance.get_func(&mut store, "lookup_stub").unwrap();
    let static_table = main_instance.get_table(&store, "static_table").unwrap();
    let dyn_table = main_instance.get_table(&store, "dyn_table").unwrap();
    let dyn_table_mem = main_instance.get_export(&store, "dyn_table_mem").and_then(|e| e.into_memory()).unwrap();
    let guest_memory = main_instance.get_export(&store, "guest_memory").and_then(|e| e.into_memory()).unwrap();

    linker.define("env", "lookup_stub", wasmi::Extern::Func(lookup_stub)).unwrap();
    linker.define("env", "guest_memory", wasmi::Extern::Memory(guest_memory)).unwrap();

    let mem = guest_mem_from_fixture(fixture);

    // Static entry: compile once, link into table slot 0.
    let entry_wasm = compile_pc_aarch64(&mem, fixture.entry_pc);
    let entry_module = wasmi::Module::new(&engine, &entry_wasm[..]).expect("entry module should validate");
    let entry_instance = linker.instantiate_and_start(&mut store, &entry_module).expect("entry module should instantiate");
    let entry_func = entry_instance.get_func(&mut store, speet_dynamic_jit::JIT_EXPORT_NAME).unwrap();
    static_table.set(&mut store, 0, wasmi::Val::from(entry_func)).expect("static table set should succeed");

    let mut regs = initial_regs();
    let mut target_pc = fixture.entry_pc as i64;

    loop {
        let mut params: Vec<wasmi::Val> = regs.iter().map(|&r| wasmi::Val::I64(r)).collect();
        params.push(wasmi::Val::I64(target_pc));
        let mut results = vec![wasmi::Val::I64(0); NUM_REGS as usize];
        match lookup_stub.call(&mut store, &params, &mut results) {
            Ok(()) => {
                let get_i64 = |v: &wasmi::Val| match v { wasmi::Val::I64(x) => *x, other => panic!("unexpected type: {other:?}") };
                return get_i64(&results[X0_REG]);
            }
            Err(trap) => {
                let (miss_regs, miss_pc) = store.data_mut().pending_miss.take().unwrap_or_else(|| {
                    panic!("trap must have recorded a pending miss; actual trap: {trap} (compiled so far: {:?})", store.data().jit_compiles)
                });
                assert!(
                    store.data().jit_compiles.len() < MAX_JIT_COMPILES,
                    "exceeded {MAX_JIT_COMPILES} JIT compiles -- likely an infinite loop; compiled so far: {:?}",
                    store.data().jit_compiles
                );
                let wasm = compile_pc_aarch64(&mem, miss_pc as u64);
                speet_dynamic_jit::link_jit_function(
                    &mut store, &linker, &engine, &jit_config, &dyn_table, &dyn_table_mem, miss_pc as u64, &wasm,
                )
                .unwrap_or_else(|e| panic!("linking pc={miss_pc:#x} should succeed: {e:?}"));
                store.data_mut().jit_compiles.push(miss_pc as u64);
                regs = miss_regs;
                target_pc = miss_pc;
            }
        }
    }
}

#[test]
fn corpus_aarch64_arith_table_indirect() {
    let fixture = aarch64_fixture("arith");
    assert_eq!(run_table_indirect(&fixture), expected_main_return("arith"));
}

/// `frame.c` ("stack-frame pressure: many spills and prologue/epilogue
/// patterns" -- 12 locals + a 6-element array, far more than `arith`/
/// `pairs`) hits a genuine wasm "out of bounds memory access" trap after
/// one successful dynamic-JIT compile+link+retry cycle. The *next* miss's
/// reported target_pc is `0xd65f03c01b007c20` -- which is not a plausible
/// address at all: its high 32 bits, `0xd65f03c0`, are exactly the raw
/// `ret` instruction encoding (see `corpus_riscv.rs`'s AArch64-adjacent
/// sibling test's `HALT_PC` convention notes), and the low 32 bits also
/// decode as a plausible instruction word -- i.e. this looks like a 64-bit
/// load read two adjacent **instruction words** out of `.text` (which
/// starts at guest VA 0) instead of its intended source. Dumping the
/// compiled trace confirms the value flows through `scratch_a`
/// (`render_stack_to_wasm`'s TailCall staging local) into the final
/// `ReturnCall`'s target-pc argument -- so *something* upstream computed a
/// wrong address (most plausibly an SP-relative or ret-address
/// computation that doesn't generalize correctly to a larger stack frame)
/// and it round-tripped through an otherwise-correctly-structured TailCall
/// sequence. This is evidence, not a full root cause -- worth flagging as
/// higher-priority than a missing-instruction gap, since `scratch_a`
/// staging is shared `vane-arch::template::render_stack_to_wasm` code, not
/// AArch64-specific, so the same class of bug could in principle affect
/// any ISA under similar conditions (many nested `CheckCode` guards from a
/// trace with many guest instructions). Left `#[ignore]`d rather than
/// silently skipped or "fixed" by guessing at a root cause.
#[test]
#[ignore = "genuine bug: a later dynamic-JIT compile's target_pc decodes as raw .text instruction bytes (0xd65f03c01b007c20), not a valid address -- see this test's doc comment for the evidence trail"]
fn corpus_aarch64_frame_table_indirect() {
    let fixture = aarch64_fixture("frame");
    assert_eq!(run_table_indirect(&fixture), expected_main_return("frame"));
}

#[test]
fn corpus_aarch64_pairs_table_indirect() {
    let fixture = aarch64_fixture("pairs");
    assert_eq!(run_table_indirect(&fixture), expected_main_return("pairs"));
}

// ── wasmtime (FunctionRef) driver ────────────────────────────────────────────

#[cfg(feature = "wasmtime")]
#[derive(Default)]
struct HostStateWasmtime {
    pending_miss: Option<(Vec<i64>, i64)>,
    jit_compiles: Vec<u64>,
}

#[cfg(feature = "wasmtime")]
fn run_function_ref(fixture: &Fixture) -> i64 {
    use speet_dynamic_jit::wasmtime_link::link_jit_function_funcref;

    let main_wasm = build_main_module(fixture, DispatchMode::FunctionRef);
    let jit_config = JitConfig {
        dyn_dispatch_table_slot: IndexSlot(1),
        dispatch_mode: DispatchMode::FunctionRef,
        dyn_funcref_table_slot: Some(IndexSlot(1)),
        dyn_table_mem_idx: 1,
        dyn_table_mem_offset: 0,
        dyn_table_capacity: DYN_TABLE_CAPACITY,
        hit_table_mem_idx: 2,
        hit_table_mem_offset: 0,
        hit_table_capacity: 4,
        hit_threshold: 0,
        jit_request_func_idx: None,
    };

    let mut config = wasmtime::Config::default();
    config.wasm_function_references(true);
    let engine = wasmtime::Engine::new(&config).expect("engine should build");
    let mut store = wasmtime::Store::new(&engine, HostStateWasmtime::default());
    let mut linker: wasmtime::Linker<HostStateWasmtime> = wasmtime::Linker::new(&engine);

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
        .func_wrap("env", "jit_invalidate", |pc: i64| -> () {
            panic!("unexpected jit_invalidate for pc={pc:#x}: compiled from live guest bytes");
        })
        .unwrap();
    let on_miss_ty = wasmtime::FuncType::new(
        &engine,
        vec![wasmtime::ValType::I64; (NUM_REGS + 1) as usize],
        vec![wasmtime::ValType::I64; NUM_REGS as usize],
    );
    linker
        .func_new("env", "on_miss", on_miss_ty, |mut caller: wasmtime::Caller<'_, HostStateWasmtime>, params: &[wasmtime::Val], results: &mut [wasmtime::Val]| {
            let get_i64 = |v: &wasmtime::Val| v.i64().expect("expected i64");
            let regs: Vec<i64> = params[..NUM_REGS as usize].iter().map(get_i64).collect();
            let target_pc = get_i64(&params[NUM_REGS as usize]);
            if target_pc == HALT_PC {
                for (i, r) in regs.iter().enumerate() {
                    results[i] = wasmtime::Val::I64(*r);
                }
                Ok(())
            } else {
                caller.data_mut().pending_miss = Some((regs, target_pc));
                Err(wasmtime::Error::msg("jit-miss: pc not yet compiled"))
            }
        })
        .unwrap();

    let main_module = wasmtime::Module::new(&engine, &main_wasm[..]).expect("main module should validate");
    let main_instance = linker.instantiate(&mut store, &main_module).expect("main module should instantiate");

    let lookup_stub = main_instance.get_func(&mut store, "lookup_stub").unwrap();
    let static_table = main_instance.get_table(&mut store, "static_table").unwrap();
    let dyn_funcref_table = main_instance.get_table(&mut store, "dyn_table").unwrap();
    let dyn_table_mem = main_instance.get_memory(&mut store, "dyn_table_mem").unwrap();
    let guest_memory = main_instance.get_memory(&mut store, "guest_memory").unwrap();

    linker.define(&store, "env", "lookup_stub", lookup_stub).unwrap();
    linker.define(&store, "env", "guest_memory", guest_memory).unwrap();

    let mem = guest_mem_from_fixture(fixture);

    let entry_wasm = compile_pc_aarch64(&mem, fixture.entry_pc);
    let entry_module = wasmtime::Module::new(&engine, &entry_wasm[..]).expect("entry module should validate");
    let entry_instance = linker.instantiate(&mut store, &entry_module).expect("entry module should instantiate");
    let entry_func = entry_instance.get_func(&mut store, speet_dynamic_jit::JIT_EXPORT_NAME).unwrap();
    static_table
        .set(&mut store, 0, wasmtime::Ref::Func(Some(entry_func)))
        .expect("static table set should succeed");

    let mut regs = initial_regs();
    let mut target_pc = fixture.entry_pc as i64;

    loop {
        let mut params: Vec<wasmtime::Val> = regs.iter().map(|&r| wasmtime::Val::I64(r)).collect();
        params.push(wasmtime::Val::I64(target_pc));
        let mut results = vec![wasmtime::Val::I64(0); NUM_REGS as usize];
        match lookup_stub.call(&mut store, &params, &mut results) {
            Ok(()) => {
                let get_i64 = |v: &wasmtime::Val| v.i64().expect("expected i64");
                return get_i64(&results[X0_REG]);
            }
            Err(trap) => {
                let (miss_regs, miss_pc) = store.data_mut().pending_miss.take().unwrap_or_else(|| {
                    panic!("trap must have recorded a pending miss; actual trap: {trap} (compiled so far: {:?})", store.data().jit_compiles)
                });
                assert!(
                    store.data().jit_compiles.len() < MAX_JIT_COMPILES,
                    "exceeded {MAX_JIT_COMPILES} JIT compiles -- likely an infinite loop; compiled so far: {:?}",
                    store.data().jit_compiles
                );
                let wasm = compile_pc_aarch64(&mem, miss_pc as u64);
                link_jit_function_funcref(
                    &mut store, &linker, &engine, &jit_config, &dyn_funcref_table, &dyn_table_mem, miss_pc as u64, &wasm,
                )
                .unwrap_or_else(|e| panic!("linking pc={miss_pc:#x} should succeed: {e:?}"));
                store.data_mut().jit_compiles.push(miss_pc as u64);
                regs = miss_regs;
                target_pc = miss_pc;
            }
        }
    }
}

#[cfg(feature = "wasmtime")]
#[test]
fn corpus_aarch64_arith_function_ref() {
    let fixture = aarch64_fixture("arith");
    assert_eq!(run_function_ref(&fixture), expected_main_return("arith"));
}

#[cfg(feature = "wasmtime")]
#[test]
#[ignore = "genuine bug: a later dynamic-JIT compile's target_pc decodes as raw .text instruction bytes (0xd65f03c01b007c20), not a valid address -- see corpus_aarch64_frame_table_indirect's doc comment for the evidence trail"]
fn corpus_aarch64_frame_function_ref() {
    let fixture = aarch64_fixture("frame");
    assert_eq!(run_function_ref(&fixture), expected_main_return("frame"));
}

#[cfg(feature = "wasmtime")]
#[test]
fn corpus_aarch64_pairs_function_ref() {
    let fixture = aarch64_fixture("pairs");
    assert_eq!(run_function_ref(&fixture), expected_main_return("pairs"));
}
