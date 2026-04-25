//! End-to-end RISC-V recompiler tests.
//!
//! Three test kinds, each with two exception-handling variants:
//!
//! - **smoke** — translate → assemble → `wasmparser::validate` only
//! - **run**   — translate → assemble → execute in wasmi
//! - **link**  — two binaries merged via `MegabinaryBuilder` → assemble → execute

use std::{borrow::Cow, convert::Infallible, path::Path};

use object::{Object, ObjectSection};
use rv_asm::Xlen;
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext};
use speet_module_builder::MegabinaryBuilder;
use speet_link_core::{linker::LinkerPlugin, unit::{BinaryUnit, FuncType}};
use speet_riscv::{HintCallback, HintInfo, RiscVRecompiler};
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, ExportKind, ExportSection, Function,
    FunctionSection, ImportSection, MemorySection, MemoryType, Module, RefType, TableSection,
    TableType, TagSection, TagType, TypeSection, ValType,
};
use wasmi::{AsContext, Engine, Linker, Module as WasmiModule, Store};
use yecta::{EscapeTag, LocalPool, Reactor, TableIdx, TagIdx, TypeIdx};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Linear-memory offset where x0–x31 are saved before calling `__speet_hint`.
const REG_SAVE_BASE: u32 = 0x100;
/// Absolute WASM index of the `env.__speet_hint` import (always first).
const IMPORT_HINT: u32 = 0;
/// Total number of host-function imports at the front of the module.
const N_IMPORTS: u32 = 3; // hint, write, exit

// ── Exception-handling variant ────────────────────────────────────────────────

/// Whether to build the module with WASM exception-handling support.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Eh {
    /// No EH — `escape_tag = None`, no tag section.
    None,
    /// Full EH — `EscapeTag` set, tag section present.
    With,
}

// ── Harness state ──────────────────────────────────────────────────────────────

/// Register snapshot captured at a HINT site.
#[derive(Debug, Clone)]
struct RegSnapshot {
    regs: [i32; 32],
}
impl RegSnapshot {
    fn reg(&self, name: &str) -> i32 {
        self.regs[abi_reg_index(name).expect("unknown register")]
    }
}

struct HostState {
    hints: Vec<(i32, RegSnapshot)>,
    stdout: Vec<u8>,
    exit_code: Option<i32>,
}
impl HostState {
    fn new() -> Self {
        Self { hints: Vec::new(), stdout: Vec::new(), exit_code: None }
    }
}

// ── Reactor/adapter helpers ───────────────────────────────────────────────────

fn make_rctx<'r>(
    reactor: &'r mut Reactor<(), Infallible, Function, LocalPool>,
    base_func_offset: u32,
    type_idx: TypeIdx,   // pool / escape-tag type index in the assembled module
    eh: Eh,
) -> ReactorAdapter<'r, (), Infallible, Function, LocalPool> {
    static T: TableIdx = TableIdx(0);
    let escape_tag = match eh {
        Eh::None => None,
        Eh::With => Some(EscapeTag { tag: TagIdx(0), ty: type_idx }),
    };
    let mut rctx = ReactorAdapter {
        reactor,
        layout:       yecta::LocalLayout::empty(),
        locals_mark:  yecta::Mark { slot_count: 0, total_locals: 0 },
        pool:         yecta::Pool { handler: &T, ty: type_idx },
        escape_tag,
    };
    rctx.set_base_func_offset(base_func_offset);
    rctx
}

/// Collect the WASM parameter types for RISCV functions after `setup_traps`.
fn collect_rv_params(rctx: &ReactorAdapter<'_, (), Infallible, Function, LocalPool>) -> Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout()
        .iter_before(&mark)
        .flat_map(|(count, ty)| std::iter::repeat(ty).take(count as usize))
        .collect()
}

// ── Hint callback ─────────────────────────────────────────────────────────────

/// Build a `HintCallback` that — at each HINT — saves x0–x31 to
/// `[REG_SAVE_BASE .. REG_SAVE_BASE+128)` then calls the hint import.
fn build_hint_callback() -> impl HintCallback<(), Infallible> {
    |hint: &HintInfo, ctx: &mut (), rctx: &mut speet_riscv::HintContext<(), Infallible>| {
        use wasm_encoder::{Instruction, MemArg};
        for i in 0u32..32 {
            rctx.emit(ctx, &Instruction::I32Const(0)).unwrap();
            rctx.emit(ctx, &Instruction::LocalGet(i)).unwrap();
            rctx.emit(ctx, &Instruction::I32Store(MemArg {
                memory_index: 0,
                align: 2,
                offset: (REG_SAVE_BASE + i * 4) as u64,
            })).unwrap();
        }
        rctx.emit(ctx, &Instruction::I32Const(hint.value as i32)).unwrap();
        rctx.emit(ctx, &Instruction::Call(IMPORT_HINT)).unwrap();
    }
}

// ── Single-binary translation ─────────────────────────────────────────────────

/// Translate one binary; returns (functions, rv_param_types).
///
/// `type_idx` is the WASM type index the caller has assigned to these
/// functions in the module's type section (used for `call_indirect` dispatch
/// and, when `eh == Eh::With`, the escape tag).
fn translate_binary(
    text: &[u8],
    start_addr: u32,
    xlen: Xlen,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> (Vec<Function>, Vec<ValType>) {
    let mut recompiler =
        RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
            start_addr as u64,
            false, // no passive hint tracking — use callback
            xlen == Xlen::Rv64,
            false,
        );
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor, base_func_offset, type_idx, eh);
    recompiler.setup_traps(&mut rctx);
    let rv_params = collect_rv_params(&rctx);

    let mut hint_cb = build_hint_callback();
    recompiler.set_hint_callback(&mut hint_cb);

    let mut ctx = ();
    recompiler
        .translate_bytes(&mut ctx, &mut rctx, text, start_addr, xlen,
            &mut |a| Function::new(a.collect::<Vec<_>>()))
        .expect("translate_bytes failed");

    (rctx.drain_fns(), rv_params)
}

// ── Module assembly ───────────────────────────────────────────────────────────

/// Description of one binary slice for a multi-binary module.
struct BinarySlice {
    fns: Vec<Function>,
    rv_params: Vec<ValType>,
    /// Absolute WASM index of the entry function (first in this slice).
    start_func_idx: u32,
    /// Export name for this binary's entry function.
    entry_name: String,
}

/// Assemble a complete, standalone WASM module from one or more binary slices.
///
/// The module layout is:
/// ```text
/// Type    0 ..N-1  : one type per unique rv_params signature, then import types
/// Import  0 ..2    : env.__speet_hint, env.write, env.exit
/// Function        : all RISCV functions (deduplicated types)
/// Table   0       : funcref, size = N_IMPORTS + total_fns
/// Memory  0       : 64 pages
/// [Tag    0 …]    : one tag per unique rv type (only when eh == Eh::With)
/// Export          : "memory", entry function(s)
/// Element 0       : active, table 0, offset N_IMPORTS, all func refs
/// Code            : all RISCV functions
/// ```
fn assemble_module(slices: &[BinarySlice], eh: Eh) -> Vec<u8> {
    // ── Collect unique rv_params types ────────────────────────────────────────
    let mut unique_rv: Vec<Vec<ValType>> = Vec::new();
    for s in slices {
        if !unique_rv.iter().any(|t| t == &s.rv_params) {
            unique_rv.push(s.rv_params.clone());
        }
    }
    let n_rv_types = unique_rv.len() as u32;
    let type_idx_of = |rv_params: &Vec<ValType>| -> u32 {
        unique_rv.iter().position(|t| t == rv_params).unwrap() as u32
    };

    // ── Type section ─────────────────────────────────────────────────────────
    // rv types first (so TypeIdx(i) matches type i), then import types.
    let hint_ty_idx = n_rv_types;       // (i32) → ()
    let write_ty_idx = n_rv_types + 1;  // (i32, i32, i32) → i32
    // exit reuses hint_ty_idx (same signature)

    let mut types = TypeSection::new();
    for rv_params in &unique_rv {
        types.ty().function(rv_params.clone(), []);
    }
    types.ty().function([ValType::I32], []);                                           // hint/exit
    types.ty().function([ValType::I32, ValType::I32, ValType::I32], [ValType::I32]);   // write

    // ── Import section ────────────────────────────────────────────────────────
    let mut imports = ImportSection::new();
    imports.import("env", "__speet_hint", wasm_encoder::EntityType::Function(hint_ty_idx));
    imports.import("env", "write",        wasm_encoder::EntityType::Function(write_ty_idx));
    imports.import("env", "exit",         wasm_encoder::EntityType::Function(hint_ty_idx));

    // ── Function section ──────────────────────────────────────────────────────
    let mut funcs = FunctionSection::new();
    for s in slices {
        let ty_idx = type_idx_of(&s.rv_params);
        for _ in &s.fns {
            funcs.function(ty_idx);
        }
    }

    // ── Table section ─────────────────────────────────────────────────────────
    let total_fns: u32 = slices.iter().map(|s| s.fns.len() as u32).sum();
    let table_size = N_IMPORTS + total_fns;
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum:  table_size as u64,
        maximum:  Some(table_size as u64),
        table64:  false,
        shared:   false,
    });

    // ── Memory section ────────────────────────────────────────────────────────
    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 64, maximum: None, memory64: false, shared: false, page_size_log2: None,
    });

    // ── Tag section (EH only) ─────────────────────────────────────────────────
    let mut tags = TagSection::new();
    if eh == Eh::With {
        for i in 0..n_rv_types {
            tags.tag(TagType { kind: wasm_encoder::TagKind::Exception, func_type_idx: i });
        }
    }

    // ── Export section ────────────────────────────────────────────────────────
    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    for s in slices {
        exports.export(&s.entry_name, ExportKind::Func, s.start_func_idx);
    }

    // ── Element section ───────────────────────────────────────────────────────
    // Populate table[0][N_IMPORTS .. N_IMPORTS+total_fns] = all generated funcs.
    let all_func_indices: Vec<u32> = (N_IMPORTS..N_IMPORTS + total_fns).collect();
    let mut elems = ElementSection::new();
    elems.active(
        Some(0),
        &ConstExpr::i32_const(N_IMPORTS as i32),
        Elements::Functions(Cow::Borrowed(&all_func_indices)),
    );

    // ── Code section ──────────────────────────────────────────────────────────
    let mut code = CodeSection::new();
    for s in slices {
        for f in &s.fns {
            code.function(f);
        }
    }

    // ── Final module ──────────────────────────────────────────────────────────
    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    module.section(&tables);
    module.section(&mems);
    if eh == Eh::With && n_rv_types > 0 {
        module.section(&tags);
    }
    module.section(&exports);
    module.section(&elems);
    module.section(&code);
    module.finish()
}

// ── High-level module builders ────────────────────────────────────────────────

/// Build a single-binary WASM module.
fn build_single(text: &[u8], start_addr: u32, xlen: Xlen, eh: Eh) -> Vec<u8> {
    // rv type is always at index 0 for single-binary modules.
    let type_idx = TypeIdx(0);
    let (fns, rv_params) = translate_binary(text, start_addr, xlen, N_IMPORTS, type_idx, eh);
    let slice = BinarySlice {
        fns,
        rv_params,
        start_func_idx: N_IMPORTS,
        entry_name: "_start".into(),
    };
    assemble_module(&[slice], eh)
}

/// Build a linked (multi-binary) WASM module via `MegabinaryBuilder`.
///
/// `specs` is `[(text, start_addr, xlen, entry_name)]`.  The binaries can
/// have different XLEN values, exercising different types, pools, and cell
/// layouts in the same module.
fn build_linked(
    specs: &[(&[u8], u32, Xlen, &str)],
    eh: Eh,
) -> Vec<u8> {
    // ── Pre-compute unique rv types to assign stable TypeIdx values ────────
    // We must know the TypeIdx before translating so the Pool and EscapeTag
    // reference the right index.  We run a dry-run setup_traps to get params.
    let mut unique_rv: Vec<Vec<ValType>> = Vec::new();
    for &(_, start_addr, xlen, _) in specs {
        let params = dry_run_rv_params(start_addr, xlen);
        if !unique_rv.iter().any(|p| p == &params) {
            unique_rv.push(params);
        }
    }

    let type_idx_for = |xlen: Xlen, start_addr: u32, unique: &[Vec<ValType>]| -> TypeIdx {
        let params = dry_run_rv_params(start_addr, xlen);
        let idx = unique.iter().position(|p| p == &params).unwrap();
        TypeIdx(idx as u32)
    };

    // ── Translate each binary ─────────────────────────────────────────────
    // Use `MegabinaryBuilder` to exercise the full linking pipeline.
    let mut builder = MegabinaryBuilder::<Function>::new();
    let mut slices: Vec<BinarySlice> = Vec::new();
    let mut running_offset = N_IMPORTS;

    for &(text, start_addr, xlen, entry_name) in specs {
        let type_idx = type_idx_for(xlen, start_addr, &unique_rv);
        let (fns, rv_params) = translate_binary(text, start_addr, xlen, running_offset, type_idx, eh);
        let n_fns = fns.len() as u32;
        let start_func_idx = running_offset;

        // Construct a BinaryUnit and feed it to the builder.
        let func_type = FuncType::from_val_types(&rv_params, &[]);
        let unit = BinaryUnit {
            base_func_offset: running_offset,
            entry_points: vec![(entry_name.to_string(), start_func_idx)],
            func_types: vec![func_type; n_fns as usize],
            fns,
            data_segments: vec![],
            data_init_fn: None,
        };
        builder.on_unit(unit);

        // Build the slice for assembly (we re-drain from the builder's output).
        slices.push(BinarySlice {
            fns: vec![], // filled from builder output below
            rv_params,
            start_func_idx,
            entry_name: entry_name.to_string(),
        });
        running_offset += n_fns;
    }

    // ── Extract accumulated functions from the builder ─────────────────────
    let output = builder.finish();
    // Re-partition output.fns into per-binary slices.
    let mut fn_iter = output.fns.into_iter();
    for (i, &(_, _, _, _)) in specs.iter().enumerate() {
        // Count: how many fns in this slice?
        let start = slices[i].start_func_idx - N_IMPORTS;
        let end = if i + 1 < specs.len() {
            slices[i + 1].start_func_idx - N_IMPORTS
        } else {
            running_offset - N_IMPORTS
        };
        slices[i].fns = fn_iter.by_ref().take((end - start) as usize).collect();
    }

    assemble_module(&slices, eh)
}

/// Quick "dry-run" to get the WASM parameter types for a given (addr, xlen)
/// without performing actual translation.
fn dry_run_rv_params(start_addr: u32, xlen: Xlen) -> Vec<ValType> {
    let recompiler =
        RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
            start_addr as u64, false, xlen == Xlen::Rv64, false,
        );
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor, 0, TypeIdx(0), Eh::None);
    recompiler.setup_traps(&mut rctx);
    collect_rv_params(&rctx)
}

// ── Wasmi execution ───────────────────────────────────────────────────────────

fn run_module(wasm: &[u8], entry: &str) -> Result<HostState, String> {
    let engine = Engine::default();
    let mut store = Store::new(&engine, HostState::new());
    let mut linker: Linker<HostState> = Linker::new(&engine);

    linker.func_wrap("env", "__speet_hint", |mut caller: wasmi::Caller<'_, HostState>, id: i32| {
        let memory = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
        let mut regs = [0i32; 32];
        {
            let data = memory.data(caller.as_context());
            let base = REG_SAVE_BASE as usize;
            for (i, r) in regs.iter_mut().enumerate() {
                let off = base + i * 4;
                *r = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            }
        }
        caller.data_mut().hints.push((id, RegSnapshot { regs }));
    }).map_err(|e| e.to_string())?;

    linker.func_wrap("env", "write", |mut caller: wasmi::Caller<'_, HostState>, fd: i32, ptr: i32, len: i32| -> i32 {
        if fd == 1 || fd == 2 {
            let memory = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let mut buf = vec![0u8; len as usize];
            let _ = memory.read(caller.as_context(), ptr as usize, &mut buf);
            caller.data_mut().stdout.extend_from_slice(&buf);
        }
        len
    }).map_err(|e| e.to_string())?;

    linker.func_wrap("env", "exit", |mut caller: wasmi::Caller<'_, HostState>, code: i32| {
        caller.data_mut().exit_code = Some(code);
    }).map_err(|e| e.to_string())?;

    let module = WasmiModule::new(&engine, wasm).map_err(|e| e.to_string())?;
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(|e| e.to_string())?;

    let start_fn = instance.get_func(&mut store, entry).ok_or("no entry export")?;
    let ty = start_fn.ty(&store);
    let params: Vec<wasmi::Val> = ty.params().iter().map(|vt| match vt {
        wasmi::ValType::I32 => wasmi::Val::I32(0),
        wasmi::ValType::I64 => wasmi::Val::I64(0),
        wasmi::ValType::F32 => wasmi::Val::F32(wasmi::F32::from_bits(0)),
        wasmi::ValType::F64 => wasmi::Val::F64(wasmi::F64::from_bits(0)),
        _ => panic!("unexpected param type"),
    }).collect();

    let mut results = vec![wasmi::Val::I32(0); ty.results().len()];
    let _ = start_fn.call(&mut store, &params, &mut results);

    Ok(store.into_data())
}

// ── ELF helpers ───────────────────────────────────────────────────────────────

fn load_text(path: &Path) -> Option<(Vec<u8>, u32)> {
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
    let sec = obj.section_by_name(".text")
        .unwrap_or_else(|| panic!("no .text in {}", path.display()));
    let data = sec.data()
        .unwrap_or_else(|e| panic!("read .text from {}: {e}", path.display()))
        .to_vec();
    Some((data, sec.address() as u32))
}

fn corpus(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus")
        .join(rel)
}

fn abi_reg_index(name: &str) -> Option<usize> {
    match name {
        "x0"|"zero"=>Some(0), "x1"|"ra"=>Some(1), "x2"|"sp"=>Some(2),
        "x3"|"gp"=>Some(3),   "x4"|"tp"=>Some(4), "x5"|"t0"=>Some(5),
        "x6"|"t1"=>Some(6),   "x7"|"t2"=>Some(7), "x8"|"s0"|"fp"=>Some(8),
        "x9"|"s1"=>Some(9),   "x10"|"a0"=>Some(10), "x11"|"a1"=>Some(11),
        "x12"|"a2"=>Some(12), "x13"|"a3"=>Some(13), "x14"|"a4"=>Some(14),
        "x15"|"a5"=>Some(15), "x16"|"a6"=>Some(16), "x17"|"a7"=>Some(17),
        "x18"|"s2"=>Some(18), "x19"|"s3"=>Some(19), "x20"|"s4"=>Some(20),
        "x21"|"s5"=>Some(21), "x22"|"s6"=>Some(22), "x23"|"s7"=>Some(23),
        "x24"|"s8"=>Some(24), "x25"|"s9"=>Some(25), "x26"|"s10"=>Some(26),
        "x27"|"s11"=>Some(27),"x28"|"t3"=>Some(28), "x29"|"t4"=>Some(29),
        "x30"|"t5"=>Some(30), "x31"|"t6"=>Some(31), _ => None,
    }
}

// ── Test macros ───────────────────────────────────────────────────────────────

/// Smoke test: translate + assemble + validate.
macro_rules! smoke {
    ($name:ident, $rel:expr, $xlen:expr, $eh:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = match load_text(&path) { Some(v) => v, None => return };
            let wasm = build_single(&text, addr, $xlen, $eh);
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!("  ✓ {} ({:?}) — {} bytes",
                path.file_name().unwrap().to_string_lossy(), $eh as u8, wasm.len());
        }
    };
}

/// Run test: translate + assemble + execute, optionally skipping EH if unsupported.
macro_rules! run {
    ($name:ident, $rel:expr, $xlen:expr, $eh:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = match load_text(&path) { Some(v) => v, None => return };
            let wasm = build_single(&text, addr, $xlen, $eh);
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => {
                    println!("  ✓ {} hints seen", state.hints.len());
                    for (id, snap) in &state.hints {
                        println!("    hint={id} a0={}", snap.reg("a0"));
                    }
                }
                Err(e) => {
                    // EH features may not be supported by this wasmi build.
                    if $eh == Eh::With {
                        eprintln!("  ! EH run skipped (wasmi error): {e}");
                    } else {
                        panic!("run failed: {e}");
                    }
                }
            }
        }
    };
}

/// Linkage test: two binaries merged via `MegabinaryBuilder` + run.
macro_rules! link {
    ($name:ident, $rel1:expr, $xlen1:expr, $rel2:expr, $xlen2:expr, $eh:expr) => {
        #[test]
        fn $name() {
            let p1 = corpus($rel1);
            let p2 = corpus($rel2);
            let (t1, a1) = match load_text(&p1) { Some(v) => v, None => return };
            let (t2, a2) = match load_text(&p2) { Some(v) => v, None => return };

            let wasm = build_linked(
                &[
                    (&t1, a1, $xlen1, "start_0"),
                    (&t2, a2, $xlen2, "start_1"),
                ],
                $eh,
            );
            wasmparser::validate(&wasm).expect("linked WASM is invalid");
            println!("  linked module: {} bytes", wasm.len());

            // Run both entry points.
            for entry in ["start_0", "start_1"] {
                match run_module(&wasm, entry) {
                    Ok(state) => {
                        println!("  ✓ {entry}: {} hints", state.hints.len());
                    }
                    Err(e) => {
                        if $eh == Eh::With {
                            eprintln!("  ! EH run skipped ({entry}): {e}");
                        } else {
                            panic!("run {entry} failed: {e}");
                        }
                    }
                }
            }
        }
    };
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

smoke!(smoke_rv32_mul_no_eh,   "rv32im/01_multiply_divide", Xlen::Rv32, Eh::None);
smoke!(smoke_rv32_mul_eh,      "rv32im/01_multiply_divide", Xlen::Rv32, Eh::With);
smoke!(smoke_rv32_br_no_eh,    "rv32im/02_branches",        Xlen::Rv32, Eh::None);
smoke!(smoke_rv32_br_eh,       "rv32im/02_branches",        Xlen::Rv32, Eh::With);
smoke!(smoke_rv32_ls_no_eh,    "rv32im/03_loads_stores",    Xlen::Rv32, Eh::None);
smoke!(smoke_rv32_ls_eh,       "rv32im/03_loads_stores",    Xlen::Rv32, Eh::With);
smoke!(smoke_rv32_sh_no_eh,    "rv32im/04_shifts",          Xlen::Rv32, Eh::None);
smoke!(smoke_rv32_sh_eh,       "rv32im/04_shifts",          Xlen::Rv32, Eh::With);
smoke!(smoke_rv32_imm_no_eh,   "rv32im/05_immediate_ops",   Xlen::Rv32, Eh::None);
smoke!(smoke_rv32_imm_eh,      "rv32im/05_immediate_ops",   Xlen::Rv32, Eh::With);
smoke!(smoke_rv64_arith_no_eh, "rv64im/01_basic_arith",     Xlen::Rv64, Eh::None);
smoke!(smoke_rv64_arith_eh,    "rv64im/01_basic_arith",     Xlen::Rv64, Eh::With);
smoke!(smoke_rv64_word_no_eh,  "rv64im/02_word_ops",        Xlen::Rv64, Eh::None);
smoke!(smoke_rv64_word_eh,     "rv64im/02_word_ops",        Xlen::Rv64, Eh::With);
smoke!(smoke_rv64_ls_no_eh,    "rv64im/03_loads_stores",    Xlen::Rv64, Eh::None);
smoke!(smoke_rv64_ls_eh,       "rv64im/03_loads_stores",    Xlen::Rv64, Eh::With);
smoke!(smoke_rv64_sh_no_eh,    "rv64im/04_shifts",          Xlen::Rv64, Eh::None);
smoke!(smoke_rv64_sh_eh,       "rv64im/04_shifts",          Xlen::Rv64, Eh::With);
smoke!(smoke_rv64_mul_no_eh,   "rv64im/05_multiply_divide", Xlen::Rv64, Eh::None);
smoke!(smoke_rv64_mul_eh,      "rv64im/05_multiply_divide", Xlen::Rv64, Eh::With);

// ── Run tests ─────────────────────────────────────────────────────────────────

run!(run_rv32_mul_no_eh,   "rv32im/01_multiply_divide", Xlen::Rv32, Eh::None);
run!(run_rv32_mul_eh,      "rv32im/01_multiply_divide", Xlen::Rv32, Eh::With);
run!(run_rv32_br_no_eh,    "rv32im/02_branches",        Xlen::Rv32, Eh::None);
run!(run_rv32_br_eh,       "rv32im/02_branches",        Xlen::Rv32, Eh::With);
run!(run_rv64_arith_no_eh, "rv64im/01_basic_arith",     Xlen::Rv64, Eh::None);
run!(run_rv64_arith_eh,    "rv64im/01_basic_arith",     Xlen::Rv64, Eh::With);

// ── Linkage tests ─────────────────────────────────────────────────────────────

// Same type: two RV32 binaries in one module (same LocalPool, same cell layout).
link!(link_rv32_rv32_no_eh, "rv32im/01_multiply_divide", Xlen::Rv32,
                             "rv32im/02_branches",        Xlen::Rv32, Eh::None);
link!(link_rv32_rv32_eh,    "rv32im/01_multiply_divide", Xlen::Rv32,
                             "rv32im/02_branches",        Xlen::Rv32, Eh::With);

// Different types: RV32 + RV64 — different param types, different pools,
// different cells in the assembled type section.
link!(link_rv32_rv64_no_eh, "rv32im/01_multiply_divide", Xlen::Rv32,
                             "rv64im/01_basic_arith",     Xlen::Rv64, Eh::None);
link!(link_rv32_rv64_eh,    "rv32im/01_multiply_divide", Xlen::Rv32,
                             "rv64im/01_basic_arith",     Xlen::Rv64, Eh::With);
