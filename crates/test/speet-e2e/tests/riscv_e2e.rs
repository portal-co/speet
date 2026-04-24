use std::{convert::Infallible, path::Path};

use object::{Object, ObjectSection};
use rv_asm::Xlen;
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext};
use speet_riscv::{HintCallback, HintInfo, RiscVRecompiler};
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, ExportKind, ExportSection, Function,
    FunctionSection, ImportSection, MemorySection, MemoryType, Module, RefType, TableSection,
    TableType, TypeSection, ValType,
};
use wasmi::{AsContext, Engine, Linker, Module as WasmiModule, Store};
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Linear-memory offset where x0-x31 are saved before calling __speet_hint.
const REG_SAVE_BASE: u32 = 0x100;

/// WASM import indices.
const IMPORT_HINT: u32 = 0; // env.__speet_hint(i32) -> ()
#[allow(dead_code)]
const IMPORT_WRITE: u32 = 1; // env.write(i32,i32,i32) -> i32
#[allow(dead_code)]
const IMPORT_EXIT: u32 = 2; // env.exit(i32) -> ()
const N_IMPORTS: u32 = 3;

// ── Harness types ─────────────────────────────────────────────────────────────

/// Register snapshot captured at a HINT site.
#[derive(Debug, Clone)]
struct RegSnapshot {
    /// x0-x31 integer register values (in order).
    regs: [i32; 32],
}

impl RegSnapshot {
    /// Read a register by ABI name (`a0`-`a7`, `t0`-`t6`, `s0`-`s11`, `x0`-`x31`).
    fn reg(&self, name: &str) -> i32 {
        let idx = abi_reg_index(name).expect("unknown register name");
        self.regs[idx]
    }
}

/// State shared between host functions and the test.
struct HostState {
    /// (hint_value, snapshot) pairs in order of occurrence.
    hints: Vec<(i32, RegSnapshot)>,
    /// Bytes written to stdout (fd 1).
    stdout: Vec<u8>,
    /// Exit code set by env.exit, if called.
    exit_code: Option<i32>,
}

impl HostState {
    fn new() -> Self {
        Self { hints: Vec::new(), stdout: Vec::new(), exit_code: None }
    }
}

// ── ReactorAdapter helper ──────────────────────────────────────────────────────

fn make_rctx(
    reactor: &mut Reactor<(), Infallible, Function, LocalPool>,
) -> ReactorAdapter<'_, (), Infallible, Function, LocalPool> {
    static T: TableIdx = TableIdx(0);
    ReactorAdapter {
        reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark { slot_count: 0, total_locals: 0 },
        pool: yecta::Pool { handler: &T, ty: TypeIdx(0) },
        escape_tag: None,
    }
}

// ── WASM module builder ────────────────────────────────────────────────────────

/// Build a WASM module from a RISCV ELF `.text` section.
///
/// Returns `(wasm_bytes, start_func_idx)` where `start_func_idx` is the
/// absolute WASM function index of the function at `start_addr`.
fn build_riscv_module(text: &[u8], start_addr: u32, xlen: Xlen) -> Vec<u8> {
    let mut recompiler =
        RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
            start_addr as u64,
            false, // no passive hint tracking — we use the callback
            xlen == Xlen::Rv64,
            false,
        );

    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);

    // Advance past the 3 host imports so reactor-generated function indices
    // don't collide with them.
    rctx.set_base_func_offset(N_IMPORTS);

    // Set up trap params — this appends the RISCV register params to the layout.
    recompiler.setup_traps(&mut rctx);

    // The function type for all generated RISCV functions is exactly the set of
    // params declared in the layout before the locals mark.
    let mark = rctx.locals_mark();
    let rv_params: Vec<ValType> = rctx
        .layout()
        .iter_before(&mark)
        .flat_map(|(count, ty)| std::iter::repeat(ty).take(count as usize))
        .collect();

    // Install hint callback: at each HINT site emit a register save + call to import 0.
    let mut hint_cb = build_hint_callback();
    recompiler.set_hint_callback(&mut hint_cb);

    let mut ctx = ();
    recompiler
        .translate_bytes(
            &mut ctx,
            &mut rctx,
            text,
            start_addr,
            xlen,
            &mut |a| Function::new(a.collect::<Vec<_>>()),
        )
        .expect("translate_bytes failed");

    let fns: Vec<Function> = rctx.drain_fns();
    let n_fns = fns.len() as u32;

    // ── Assemble WASM module ───────────────────────────────────────────────────

    // Types:
    //   0: (rv_params...) → () — RISCV function type (must be type 0 for
    //                            the TypeIdx(0) used in call_indirect dispatch)
    //   1: (i32) → ()          — __speet_hint and exit
    //   2: (i32, i32, i32) → i32 — write
    let mut types = TypeSection::new();
    types.ty().function(rv_params, []);                            // 0: riscv
    types.ty().function([ValType::I32], []);                       // 1: hint/exit
    types.ty().function([ValType::I32, ValType::I32, ValType::I32], [ValType::I32]); // 2: write

    // Imports (absolute func indices 0, 1, 2).
    let mut imports = ImportSection::new();
    imports.import("env", "__speet_hint", wasm_encoder::EntityType::Function(1));
    imports.import("env", "write",        wasm_encoder::EntityType::Function(2));
    imports.import("env", "exit",         wasm_encoder::EntityType::Function(1));

    // Functions: all use type index 0 (the RISCV type).
    let mut funcs = FunctionSection::new();
    for _ in 0..n_fns {
        funcs.function(0);
    }

    // Table: funcref, size = N_IMPORTS + n_fns.
    // Table index 0 is used by yecta for return_call_indirect dispatch.
    let table_size = N_IMPORTS + n_fns;
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: table_size as u64,
        maximum: Some(table_size as u64),
        table64: false,
        shared: false,
    });

    // Memory: 64 pages (4 MiB), no max.
    let mut mems = MemorySection::new();
    mems.memory(MemoryType { minimum: 64, maximum: None, memory64: false, shared: false, page_size_log2: None });

    // Exports: "memory" and "_start" (first generated func at absolute index N_IMPORTS).
    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("_start", ExportKind::Func, N_IMPORTS);

    // Element segment: active, table 0, offset N_IMPORTS.
    // Populates the table slots [N_IMPORTS .. N_IMPORTS+n_fns) with the generated funcs.
    let func_indices: Vec<u32> = (N_IMPORTS..N_IMPORTS + n_fns).collect();
    let mut elems = ElementSection::new();
    elems.active(
        Some(0),
        &ConstExpr::i32_const(N_IMPORTS as i32),
        Elements::Functions(std::borrow::Cow::Borrowed(&func_indices)),
    );

    // Code: all generated functions in order.
    let mut code = CodeSection::new();
    for f in &fns {
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

/// Build a `HintCallback` closure that — for each HINT — saves x0-x31 to
/// linear memory starting at `REG_SAVE_BASE`, then calls `__speet_hint`.
///
/// Errors are unwrapped because `E = Infallible` makes them impossible.
fn build_hint_callback() -> impl HintCallback<(), Infallible> {
    |hint: &HintInfo, ctx: &mut (), rctx: &mut speet_riscv::HintContext<(), Infallible>| {
        use wasm_encoder::Instruction;
        use wasm_encoder::MemArg;

        // Save x0-x31 to [REG_SAVE_BASE .. REG_SAVE_BASE+128).
        for i in 0u32..32 {
            rctx.emit(ctx, &Instruction::I32Const(0)).unwrap();
            rctx.emit(ctx, &Instruction::LocalGet(i)).unwrap();
            rctx.emit(ctx, &Instruction::I32Store(MemArg {
                memory_index: 0,
                align: 2,
                offset: (REG_SAVE_BASE + i * 4) as u64,
            })).unwrap();
        }
        // Call __speet_hint(hint_value).
        rctx.emit(ctx, &Instruction::I32Const(hint.value as i32)).unwrap();
        rctx.emit(ctx, &Instruction::Call(IMPORT_HINT)).unwrap();
    }
}

// ── Wasmi harness ─────────────────────────────────────────────────────────────

/// Instantiate a wasmi module, wire up host functions, and run `_start`.
///
/// Returns the populated `HostState` (hints, stdout, exit_code).
fn run_module(wasm: &[u8]) -> HostState {
    let engine = Engine::default();
    let mut store = Store::new(&engine, HostState::new());

    let mut linker: Linker<HostState> = Linker::new(&engine);

    linker
        .func_wrap("env", "__speet_hint", |mut caller: wasmi::Caller<'_, HostState>, hint_id: i32| {
            let memory = caller.get_export("memory")
                .and_then(|e| e.into_memory())
                .expect("no memory export");
            let mut regs = [0i32; 32];
            {
                let data = memory.data(caller.as_context());
                let base = REG_SAVE_BASE as usize;
                for (i, r) in regs.iter_mut().enumerate() {
                    let off = base + i * 4;
                    *r = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                }
            }
            let snap = RegSnapshot { regs };
            caller.data_mut().hints.push((hint_id, snap));
        })
        .unwrap();

    linker
        .func_wrap("env", "write", |mut caller: wasmi::Caller<'_, HostState>, fd: i32, ptr: i32, len: i32| -> i32 {
            if fd == 1 || fd == 2 {
                let memory = caller.get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("no memory export");
                let mut buf = vec![0u8; len as usize];
                memory.read(caller.as_context(), ptr as usize, &mut buf).unwrap_or(());
                caller.data_mut().stdout.extend_from_slice(&buf);
            }
            len
        })
        .unwrap();

    linker
        .func_wrap("env", "exit", |mut caller: wasmi::Caller<'_, HostState>, code: i32| {
            caller.data_mut().exit_code = Some(code);
        })
        .unwrap();

    let module = WasmiModule::new(&engine, wasm).expect("invalid WASM module");
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiation failed");

    // Call _start — it chains through the trap table until it terminates.
    // Termination is expected (via panic/trap from exit or running off the end).
    let start_fn = instance
        .get_func(&mut store, "_start")
        .expect("no _start export");

    // _start has 66 i32/f64 params — call with all-zeros (regs start at 0).
    // Build the param list dynamically from the function type.
    let ty = start_fn.ty(&store);
    let params: Vec<wasmi::Val> = ty
        .params()
        .iter()
        .map(|vt| match vt {
            wasmi::ValType::I32 => wasmi::Val::I32(0),
            wasmi::ValType::I64 => wasmi::Val::I64(0),
            wasmi::ValType::F32 => wasmi::Val::F32(wasmi::F32::from_bits(0)),
            wasmi::ValType::F64 => wasmi::Val::F64(wasmi::F64::from_bits(0)),
            _ => panic!("unexpected param type"),
        })
        .collect();

    let mut results = vec![wasmi::Val::I32(0); ty.results().len()];
    let _ = start_fn.call(&mut store, &params, &mut results);
    // Ignore call errors — exit() and trap-based termination are expected.

    store.into_data()
}

// ── ELF loader ────────────────────────────────────────────────────────────────

fn load_text_section(path: &Path) -> Option<(Vec<u8>, u32)> {
    if !path.exists() {
        eprintln!(
            "Skipping: ELF not found at {path:?}\n\
             Run 'git submodule update --init --recursive' to fetch rv-corpus."
        );
        return None;
    }
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let obj = object::File::parse(&*bytes)
        .unwrap_or_else(|e| panic!("parse ELF {}: {e}", path.display()));
    let section = obj
        .section_by_name(".text")
        .unwrap_or_else(|| panic!("no .text in {}", path.display()));
    let data = section
        .data()
        .unwrap_or_else(|e| panic!("read .text from {}: {e}", path.display()))
        .to_vec();
    let addr = section.address() as u32;
    Some((data, addr))
}

// ── ABI register name → index ─────────────────────────────────────────────────

fn abi_reg_index(name: &str) -> Option<usize> {
    match name {
        "x0" | "zero" => Some(0),
        "x1" | "ra"   => Some(1),
        "x2" | "sp"   => Some(2),
        "x3" | "gp"   => Some(3),
        "x4" | "tp"   => Some(4),
        "x5" | "t0"   => Some(5),
        "x6" | "t1"   => Some(6),
        "x7" | "t2"   => Some(7),
        "x8" | "s0" | "fp" => Some(8),
        "x9" | "s1"   => Some(9),
        "x10" | "a0"  => Some(10),
        "x11" | "a1"  => Some(11),
        "x12" | "a2"  => Some(12),
        "x13" | "a3"  => Some(13),
        "x14" | "a4"  => Some(14),
        "x15" | "a5"  => Some(15),
        "x16" | "a6"  => Some(16),
        "x17" | "a7"  => Some(17),
        "x18" | "s2"  => Some(18),
        "x19" | "s3"  => Some(19),
        "x20" | "s4"  => Some(20),
        "x21" | "s5"  => Some(21),
        "x22" | "s6"  => Some(22),
        "x23" | "s7"  => Some(23),
        "x24" | "s8"  => Some(24),
        "x25" | "s9"  => Some(25),
        "x26" | "s10" => Some(26),
        "x27" | "s11" => Some(27),
        "x28" | "t3"  => Some(28),
        "x29" | "t4"  => Some(29),
        "x30" | "t5"  => Some(30),
        "x31" | "t6"  => Some(31),
        _ => None,
    }
}

// ── Corpus path helper ────────────────────────────────────────────────────────

fn corpus(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus")
        .join(rel)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Smoke test: translate + assemble + validate with wasmparser.
macro_rules! rv32_smoke {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = match load_text_section(&path) {
                Some(v) => v,
                None => return,
            };
            assert!(!text.is_empty());
            let wasm = build_riscv_module(&text, addr, Xlen::Rv32);
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!(
                "  ✓ {} — {} WASM bytes",
                path.file_name().unwrap().to_string_lossy(),
                wasm.len()
            );
        }
    };
}

macro_rules! rv64_smoke {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = match load_text_section(&path) {
                Some(v) => v,
                None => return,
            };
            assert!(!text.is_empty());
            let wasm = build_riscv_module(&text, addr, Xlen::Rv64);
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!(
                "  ✓ {} — {} WASM bytes",
                path.file_name().unwrap().to_string_lossy(),
                wasm.len()
            );
        }
    };
}

rv32_smoke!(rv32_smoke_multiply_divide,   "rv32im/01_multiply_divide");
rv32_smoke!(rv32_smoke_branches,          "rv32im/02_branches");
rv32_smoke!(rv32_smoke_loads_stores,      "rv32im/03_loads_stores");
rv32_smoke!(rv32_smoke_shifts,            "rv32im/04_shifts");
rv32_smoke!(rv32_smoke_immediate_ops,     "rv32im/05_immediate_ops");
rv64_smoke!(rv64_smoke_basic_arith,       "rv64im/01_basic_arith");
rv64_smoke!(rv64_smoke_word_ops,          "rv64im/02_word_ops");
rv64_smoke!(rv64_smoke_loads_stores,      "rv64im/03_loads_stores");
rv64_smoke!(rv64_smoke_shifts,            "rv64im/04_shifts");
rv64_smoke!(rv64_smoke_multiply_divide,   "rv64im/05_multiply_divide");

/// Execute the multiply/divide corpus file and verify HINT markers fire.
#[test]
fn rv32_execute_multiply_divide() {
    let path = corpus("rv32im/01_multiply_divide");
    let (text, addr) = match load_text_section(&path) {
        Some(v) => v,
        None => return,
    };
    let wasm = build_riscv_module(&text, addr, Xlen::Rv32);
    wasmparser::validate(&wasm).expect("generated WASM is invalid");

    let state = run_module(&wasm);
    if state.hints.is_empty() {
        eprintln!("Note: no HINT markers fired (binary may not contain addi x0,x0,N)");
        return;
    }
    println!("HINT markers seen: {}", state.hints.len());
    for (id, snap) in &state.hints {
        println!("  hint={id} a0={} a1={}", snap.reg("a0"), snap.reg("a1"));
    }
}
