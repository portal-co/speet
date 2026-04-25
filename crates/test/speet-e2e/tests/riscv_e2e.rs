//! End-to-end recompiler tests for RISC-V and x86-64.
//!
//! Three test kinds, each with two exception-handling variants:
//!
//! - **smoke** — translate → assemble → `wasmparser::validate` only
//! - **run**   — translate → assemble → execute in wasmi
//! - **link**  — two binaries (possibly different arches) merged via
//!               `MegabinaryBuilder` → assemble → execute
//!
//! C programs are compiled by `build.rs` (requires a suitable clang).
//! Tests that need compiled C objects are skipped when the object is absent.

use std::{borrow::Cow, convert::Infallible, path::Path};

use object::{Object, ObjectSection};
use rv_asm::Xlen;
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext};
use speet_link_core::{linker::LinkerPlugin, unit::{BinaryUnit, FuncType}};
use speet_module_builder::MegabinaryBuilder;
use speet_riscv::{HintCallback, HintInfo, RiscVRecompiler};
use speet_x86_64::X86Recompiler;
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, ExportKind, ExportSection, Function,
    FunctionSection, ImportSection, MemorySection, MemoryType, Module, RefType, TableSection,
    TableType, TagSection, TagType, TypeSection, ValType,
};
use wasmi::{AsContext, Engine, Linker, Module as WasmiModule, Store};
use yecta::{EscapeTag, LocalPool, Reactor, TableIdx, TagIdx, TypeIdx};

// ── Constants ─────────────────────────────────────────────────────────────────

const REG_SAVE_BASE: u32 = 0x100;
const IMPORT_HINT: u32 = 0;
const N_IMPORTS: u32 = 3; // hint, write, exit

// ── EH variant ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Eh { None, With }

// ── Architecture ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arch { Rv32, Rv64, X86_64 }

// ── Harness state ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RegSnapshot { regs: [i32; 32] }
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
    fn new() -> Self { Self { hints: Vec::new(), stdout: Vec::new(), exit_code: None } }
}

// ── ReactorAdapter helpers ────────────────────────────────────────────────────

fn make_rctx<'r>(
    reactor: &'r mut Reactor<(), Infallible, Function, LocalPool>,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> ReactorAdapter<'r, (), Infallible, Function, LocalPool> {
    static T: TableIdx = TableIdx(0);
    let escape_tag = match eh {
        Eh::None => None,
        Eh::With => Some(EscapeTag { tag: TagIdx(0), ty: type_idx }),
    };
    let mut rctx = ReactorAdapter {
        reactor,
        layout:      yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark { slot_count: 0, total_locals: 0 },
        pool:        yecta::Pool { handler: &T, ty: type_idx },
        escape_tag,
    };
    rctx.set_base_func_offset(base_func_offset);
    rctx
}

fn collect_rv_params(rctx: &ReactorAdapter<'_, (), Infallible, Function, LocalPool>) -> Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout().iter_before(&mark)
        .flat_map(|(count, ty)| std::iter::repeat(ty).take(count as usize))
        .collect()
}

// ── RISCV hint callback ───────────────────────────────────────────────────────

/// Build a hint callback that saves x0–x31 to `REG_SAVE_BASE` then calls the
/// hint import.  With memory64 all addresses are i64.  When `regs_are_i64` is
/// true (RV64) each register value is truncated via `i32.wrap_i64` before the
/// 32-bit store so the `RegSnapshot.regs` slice always holds 32-bit values.
fn build_hint_callback(regs_are_i64: bool) -> impl HintCallback<(), Infallible> {
    move |hint: &HintInfo, ctx: &mut (), rctx: &mut speet_riscv::HintContext<(), Infallible>| {
        use wasm_encoder::{Instruction, MemArg};
        let mem = MemArg { memory_index: 0, align: 2, offset: 0 };
        for i in 0u32..32 {
            // memory64: addresses are i64
            rctx.emit(ctx, &Instruction::I64Const((REG_SAVE_BASE + i * 4) as i64)).unwrap();
            rctx.emit(ctx, &Instruction::LocalGet(i)).unwrap();
            if regs_are_i64 {
                rctx.emit(ctx, &Instruction::I32WrapI64).unwrap();
            }
            rctx.emit(ctx, &Instruction::I32Store(mem)).unwrap();
        }
        rctx.emit(ctx, &Instruction::I32Const(hint.value as i32)).unwrap();
        rctx.emit(ctx, &Instruction::Call(IMPORT_HINT)).unwrap();
    }
}

// ── Translation backends ──────────────────────────────────────────────────────

/// Result of translating one binary.
struct Translated {
    fns: Vec<Function>,
    params: Vec<ValType>,
    /// Unsupported mnemonics (x86_64 only; empty for RISCV).
    unsupported: Vec<String>,
}

fn translate_rv(
    text: &[u8],
    start_addr: u32,
    xlen: Xlen,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> Translated {
    let rv64 = xlen == Xlen::Rv64;
    let mut recompiler = RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
        start_addr as u64, false, rv64, true, // use_memory64 = true
    );
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor, base_func_offset, type_idx, eh);
    recompiler.setup_traps(&mut rctx);
    let params = collect_rv_params(&rctx);

    let mut hint_cb = build_hint_callback(rv64);
    recompiler.set_hint_callback(&mut hint_cb);

    let mut ctx = ();
    recompiler.translate_bytes(&mut ctx, &mut rctx, text, start_addr, xlen,
        &mut |a| Function::new(a.collect::<Vec<_>>()))
        .expect("translate_bytes failed");

    Translated { fns: rctx.drain_fns(), params, unsupported: vec![] }
}

fn translate_x86(
    text: &[u8],
    rip: u64,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> Translated {
    let mut recompiler = X86Recompiler::new_with_base_rip(rip);
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor, base_func_offset, type_idx, eh);
    recompiler.setup_traps(&mut rctx);
    let params = collect_rv_params(&rctx);

    let mut ctx = ();
    recompiler.translate_bytes(&mut ctx, &mut rctx, text, rip,
        &mut |a| Function::new(a.collect::<Vec<_>>()))
        .expect("translate_bytes failed");

    let unsupported: Vec<String> = recompiler.unsupported_insns().iter().cloned().collect();
    Translated { fns: rctx.drain_fns(), params, unsupported }
}

fn translate(
    text: &[u8],
    start_addr: u64,
    arch: Arch,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> Translated {
    match arch {
        Arch::Rv32  => translate_rv(text, start_addr as u32, Xlen::Rv32, base_func_offset, type_idx, eh),
        Arch::Rv64  => translate_rv(text, start_addr as u32, Xlen::Rv64, base_func_offset, type_idx, eh),
        Arch::X86_64 => translate_x86(text, start_addr, base_func_offset, type_idx, eh),
    }
}

// ── Module assembly ───────────────────────────────────────────────────────────

struct BinarySlice {
    fns:   Vec<Function>,
    params: Vec<ValType>,
    start_func_idx: u32,
    entry_name: String,
}

fn assemble_module(slices: &[BinarySlice], eh: Eh) -> Vec<u8> {
    // Collect unique param-type signatures.
    let mut unique_params: Vec<Vec<ValType>> = Vec::new();
    for s in slices {
        if !unique_params.iter().any(|p| p == &s.params) {
            unique_params.push(s.params.clone());
        }
    }
    let type_idx_of = |params: &Vec<ValType>| -> u32 {
        unique_params.iter().position(|p| p == params).unwrap() as u32
    };
    let n_rv_types = unique_params.len() as u32;
    let hint_ty_idx  = n_rv_types;
    let write_ty_idx = n_rv_types + 1;

    let mut types = TypeSection::new();
    for p in &unique_params {
        types.ty().function(p.clone(), []);
    }
    types.ty().function([ValType::I32], []);
    types.ty().function([ValType::I32, ValType::I32, ValType::I32], [ValType::I32]);

    let mut imports = ImportSection::new();
    imports.import("env", "__speet_hint", wasm_encoder::EntityType::Function(hint_ty_idx));
    imports.import("env", "write",        wasm_encoder::EntityType::Function(write_ty_idx));
    imports.import("env", "exit",         wasm_encoder::EntityType::Function(hint_ty_idx));

    let mut funcs = FunctionSection::new();
    for s in slices {
        let ti = type_idx_of(&s.params);
        for _ in &s.fns { funcs.function(ti); }
    }

    let total_fns: u32 = slices.iter().map(|s| s.fns.len() as u32).sum();
    let table_size = N_IMPORTS + total_fns;
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: table_size as u64, maximum: Some(table_size as u64),
        table64: true, shared: false,
    });

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 64, maximum: None, memory64: true, shared: false, page_size_log2: None,
    });

    let mut tags = TagSection::new();
    if eh == Eh::With {
        for i in 0..n_rv_types {
            tags.tag(TagType { kind: wasm_encoder::TagKind::Exception, func_type_idx: i });
        }
    }

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    for s in slices {
        exports.export(&s.entry_name, ExportKind::Func, s.start_func_idx);
    }

    let all_indices: Vec<u32> = (N_IMPORTS..N_IMPORTS + total_fns).collect();
    let mut elems = ElementSection::new();
    elems.active(Some(0), &ConstExpr::i64_const(N_IMPORTS as i64),
        Elements::Functions(Cow::Borrowed(&all_indices)));

    let mut code = CodeSection::new();
    for s in slices { for f in &s.fns { code.function(f); } }

    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    module.section(&tables);
    module.section(&mems);
    if eh == Eh::With && n_rv_types > 0 { module.section(&tags); }
    module.section(&exports);
    module.section(&elems);
    module.section(&code);
    module.finish()
}

// ── High-level module builders ────────────────────────────────────────────────

fn dry_run_params(arch: Arch, addr: u64) -> Vec<ValType> {
    match arch {
        Arch::Rv32 | Arch::Rv64 => {
            let xlen = if arch == Arch::Rv64 { Xlen::Rv64 } else { Xlen::Rv32 };
            let recompiler = RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
                addr, false, xlen == Xlen::Rv64, false,
            );
            let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
            let mut rctx = make_rctx(&mut reactor, 0, TypeIdx(0), Eh::None);
            recompiler.setup_traps(&mut rctx);
            collect_rv_params(&rctx)
        }
        Arch::X86_64 => {
            let recompiler = X86Recompiler::new_with_base_rip(addr);
            let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
            let mut rctx = make_rctx(&mut reactor, 0, TypeIdx(0), Eh::None);
            recompiler.setup_traps(&mut rctx);
            collect_rv_params(&rctx)
        }
    }
}

fn build_single(text: &[u8], start_addr: u64, arch: Arch, eh: Eh) -> (Vec<u8>, Vec<String>) {
    let t = translate(text, start_addr, arch, N_IMPORTS, TypeIdx(0), eh);
    let unsupported = t.unsupported.clone();
    let slice = BinarySlice {
        params: t.params, fns: t.fns,
        start_func_idx: N_IMPORTS, entry_name: "_start".into(),
    };
    (assemble_module(&[slice], eh), unsupported)
}

/// Spec for one binary inside a linked module.
struct LinkSpec<'a> {
    text:       &'a [u8],
    start_addr: u64,
    arch:       Arch,
    entry:      &'static str,
}

fn build_linked(specs: &[LinkSpec<'_>], eh: Eh) -> (Vec<u8>, Vec<String>) {
    // Pre-compute unique param signatures for stable TypeIdx assignment.
    let mut unique_params: Vec<Vec<ValType>> = Vec::new();
    for s in specs {
        let p = dry_run_params(s.arch, s.start_addr);
        if !unique_params.iter().any(|q| q == &p) { unique_params.push(p); }
    }
    let type_idx_of_params = |p: &Vec<ValType>| -> TypeIdx {
        TypeIdx(unique_params.iter().position(|q| q == p).unwrap() as u32)
    };

    let mut builder = MegabinaryBuilder::<Function>::new();
    let mut slices: Vec<BinarySlice> = Vec::new();
    let mut running_offset = N_IMPORTS;
    let mut all_unsupported: Vec<String> = Vec::new();

    for s in specs {
        let p = dry_run_params(s.arch, s.start_addr);
        let type_idx = type_idx_of_params(&p);
        let t = translate(s.text, s.start_addr, s.arch, running_offset, type_idx, eh);

        if !t.unsupported.is_empty() {
            all_unsupported.extend(t.unsupported.iter().cloned());
        }

        let n = t.fns.len() as u32;
        let start_func_idx = running_offset;
        let func_type = FuncType::from_val_types(&t.params, &[]);
        let unit = BinaryUnit {
            base_func_offset: running_offset,
            entry_points: vec![(s.entry.to_string(), start_func_idx)],
            func_types: vec![func_type; n as usize],
            fns: t.fns,
            data_segments: vec![],
            data_init_fn: None,
        };
        builder.on_unit(unit);
        slices.push(BinarySlice {
            fns: vec![], params: t.params,
            start_func_idx, entry_name: s.entry.to_string(),
        });
        running_offset += n;
    }

    // Redistribute fns from builder output back into slices.
    let output = builder.finish();
    let mut fn_iter = output.fns.into_iter();
    for (i, s) in specs.iter().enumerate() {
        let start = slices[i].start_func_idx - N_IMPORTS;
        let end = if i + 1 < specs.len() {
            slices[i + 1].start_func_idx - N_IMPORTS
        } else {
            running_offset - N_IMPORTS
        };
        slices[i].fns = fn_iter.by_ref().take((end - start) as usize).collect();
    }

    (assemble_module(&slices, eh), all_unsupported)
}

// ── Wasmi execution ───────────────────────────────────────────────────────────

fn run_module(wasm: &[u8], entry: &str) -> Result<HostState, String> {
    let engine = Engine::default();
    let mut store = Store::new(&engine, HostState::new());
    let mut linker: Linker<HostState> = Linker::new(&engine);

    linker.func_wrap("env", "__speet_hint",
        |mut caller: wasmi::Caller<'_, HostState>, id: i32| {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let mut regs = [0i32; 32];
            {
                let data = mem.data(caller.as_context());
                let base = REG_SAVE_BASE as usize;
                for (i, r) in regs.iter_mut().enumerate() {
                    let off = base + i * 4;
                    *r = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                }
            }
            caller.data_mut().hints.push((id, RegSnapshot { regs }));
        }).map_err(|e| e.to_string())?;

    linker.func_wrap("env", "write",
        |mut caller: wasmi::Caller<'_, HostState>, fd: i32, ptr: i32, len: i32| -> i32 {
            if fd == 1 || fd == 2 {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let mut buf = vec![0u8; len as usize];
                let _ = mem.read(caller.as_context(), ptr as usize, &mut buf);
                caller.data_mut().stdout.extend_from_slice(&buf);
            }
            len
        }).map_err(|e| e.to_string())?;

    linker.func_wrap("env", "exit",
        |mut caller: wasmi::Caller<'_, HostState>, code: i32| {
            caller.data_mut().exit_code = Some(code);
        }).map_err(|e| e.to_string())?;

    let module = WasmiModule::new(&engine, wasm).map_err(|e| e.to_string())?;
    let instance = linker.instantiate_and_start(&mut store, &module)
        .map_err(|e| e.to_string())?;

    let func = instance.get_func(&mut store, entry).ok_or("no entry export")?;
    let ty = func.ty(&store);
    let params: Vec<wasmi::Val> = ty.params().iter().map(|vt| match vt {
        wasmi::ValType::I32 => wasmi::Val::I32(0),
        wasmi::ValType::I64 => wasmi::Val::I64(0),
        wasmi::ValType::F32 => wasmi::Val::F32(wasmi::F32::from_bits(0)),
        wasmi::ValType::F64 => wasmi::Val::F64(wasmi::F64::from_bits(0)),
        _ => panic!("unexpected param type"),
    }).collect();
    let mut results = vec![wasmi::Val::I32(0); ty.results().len()];
    let _ = func.call(&mut store, &params, &mut results);
    Ok(store.into_data())
}

// ── ELF / object helpers ──────────────────────────────────────────────────────

fn load_text(path: &Path) -> Option<(Vec<u8>, u64)> {
    if !path.exists() {
        eprintln!("Skipping: not found at {path:?}");
        return None;
    }
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let obj = object::File::parse(&*bytes)
        .unwrap_or_else(|e| panic!("parse ELF {}: {e}", path.display()));
    let sec = obj.section_by_name(".text")
        .unwrap_or_else(|| panic!("no .text in {}", path.display()));
    let data = sec.data()
        .unwrap_or_else(|e| panic!("read .text from {}: {e}", path.display()))
        .to_vec();
    if data.is_empty() { return None; }
    Some((data, sec.address()))
}

fn corpus(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus").join(rel)
}

fn c_obj(path_env: &str) -> Option<std::path::PathBuf> {
    let p = std::env::var(path_env).unwrap_or_default();
    if p.is_empty() { return None; }
    let p = std::path::PathBuf::from(p);
    if p.exists() { Some(p) } else { None }
}

fn abi_reg_index(name: &str) -> Option<usize> {
    match name {
        "x0"|"zero"=>Some(0),"x1"|"ra"=>Some(1),"x2"|"sp"=>Some(2),
        "x3"|"gp"=>Some(3),"x4"|"tp"=>Some(4),"x5"|"t0"=>Some(5),
        "x6"|"t1"=>Some(6),"x7"|"t2"=>Some(7),"x8"|"s0"|"fp"=>Some(8),
        "x9"|"s1"=>Some(9),"x10"|"a0"=>Some(10),"x11"|"a1"=>Some(11),
        "x12"|"a2"=>Some(12),"x13"|"a3"=>Some(13),"x14"|"a4"=>Some(14),
        "x15"|"a5"=>Some(15),"x16"|"a6"=>Some(16),"x17"|"a7"=>Some(17),
        "x18"|"s2"=>Some(18),"x19"|"s3"=>Some(19),"x20"|"s4"=>Some(20),
        "x21"|"s5"=>Some(21),"x22"|"s6"=>Some(22),"x23"|"s7"=>Some(23),
        "x24"|"s8"=>Some(24),"x25"|"s9"=>Some(25),"x26"|"s10"=>Some(26),
        "x27"|"s11"=>Some(27),"x28"|"t3"=>Some(28),"x29"|"t4"=>Some(29),
        "x30"|"t5"=>Some(30),"x31"|"t6"=>Some(31),_=>None,
    }
}

// ── Report unsupported instructions ──────────────────────────────────────────

fn report_unsupported(unsupported: &[String], context: &str) {
    if !unsupported.is_empty() {
        eprintln!(
            "  [x86_64 unsupported in {context}]: {}",
            unsupported.join(", ")
        );
    }
}

// ── Test macros ───────────────────────────────────────────────────────────────

macro_rules! smoke {
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = match load_text(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single(&text, addr, $arch, $eh);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!("  ✓ {} ({:?}) — {} bytes",
                path.file_name().unwrap().to_string_lossy(), $eh, wasm.len());
        }
    };
}

macro_rules! smoke_c {
    ($name:ident, env = $env:expr, arch = $arch:expr, $eh:expr) => {
        #[test]
        fn $name() {
            let path = match c_obj($env) { Some(p) => p, None => {
                eprintln!("  skipping {}: C object not built", stringify!($name));
                return;
            }};
            let (text, addr) = match load_text(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single(&text, addr, $arch, $eh);
            report_unsupported(&unsupported, stringify!($name));
            assert!(!wasm.is_empty());
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            println!("  ✓ {} ({:?}) — {} bytes", path.file_name().unwrap().to_string_lossy(), $eh, wasm.len());
        }
    };
}

macro_rules! run {
    ($name:ident, $rel:expr, arch = $arch:expr, $eh:expr) => {
        #[test]
        fn $name() {
            let path = corpus($rel);
            let (text, addr) = match load_text(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single(&text, addr, $arch, $eh);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => {
                    println!("  ✓ {} hints, {:?}", state.hints.len(), $eh);
                    for (id, snap) in &state.hints {
                        println!("    hint={id} a0={}", snap.reg("a0"));
                    }
                }
                Err(e) if $eh == Eh::With => eprintln!("  ! EH run skipped: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
}

macro_rules! run_c {
    ($name:ident, env = $env:expr, arch = $arch:expr, $eh:expr) => {
        #[test]
        fn $name() {
            let path = match c_obj($env) { Some(p) => p, None => {
                eprintln!("  skipping {}: C object not built", stringify!($name));
                return;
            }};
            let (text, addr) = match load_text(&path) { Some(v) => v, None => return };
            let (wasm, unsupported) = build_single(&text, addr, $arch, $eh);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("generated WASM is invalid");
            match run_module(&wasm, "_start") {
                Ok(state) => println!("  ✓ {} hints, {:?}", state.hints.len(), $eh),
                Err(e) if $eh == Eh::With => eprintln!("  ! EH run skipped: {e}"),
                Err(e) => panic!("run failed: {e}"),
            }
        }
    };
}

macro_rules! link {
    ($name:ident, [ $( ($rel:expr, arch = $arch:expr, entry = $entry:expr) ),+ ], $eh:expr) => {
        #[test]
        fn $name() {
            let mut specs_data: Vec<(Vec<u8>, u64, Arch, &'static str)> = Vec::new();
            $(
                {
                    let path = corpus($rel);
                    let (text, addr) = match load_text(&path) { Some(v) => v, None => return };
                    specs_data.push((text, addr, $arch, $entry));
                }
            )+
            let specs: Vec<LinkSpec<'_>> = specs_data.iter()
                .map(|(t, a, arch, e)| LinkSpec { text: t, start_addr: *a, arch: *arch, entry: e })
                .collect();
            let (wasm, unsupported) = build_linked(&specs, $eh);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("linked WASM is invalid");
            println!("  linked: {} bytes, {:?}", wasm.len(), $eh);
            for spec in &specs {
                match run_module(&wasm, spec.entry) {
                    Ok(state) => println!("  ✓ {}: {} hints", spec.entry, state.hints.len()),
                    Err(e) if $eh == Eh::With => eprintln!("  ! EH skipped ({}): {e}", spec.entry),
                    Err(e) => panic!("run {} failed: {e}", spec.entry),
                }
            }
        }
    };
}

macro_rules! link_c {
    ($name:ident, [ $( ($env_or_corpus:expr, is_corpus = $is_corpus:expr, arch = $arch:expr, entry = $entry:expr) ),+ ], $eh:expr) => {
        #[test]
        fn $name() {
            let mut specs_data: Vec<(Vec<u8>, u64, Arch, &'static str)> = Vec::new();
            $(
                {
                    let path: Option<std::path::PathBuf> = if $is_corpus {
                        Some(corpus($env_or_corpus))
                    } else {
                        c_obj($env_or_corpus)
                    };
                    let path = match path { Some(p) => p, None => {
                        eprintln!("  skipping {}: missing input", stringify!($name));
                        return;
                    }};
                    let (text, addr) = match load_text(&path) { Some(v) => v, None => return };
                    specs_data.push((text, addr, $arch, $entry));
                }
            )+
            let specs: Vec<LinkSpec<'_>> = specs_data.iter()
                .map(|(t, a, arch, e)| LinkSpec { text: t, start_addr: *a, arch: *arch, entry: e })
                .collect();
            let (wasm, unsupported) = build_linked(&specs, $eh);
            report_unsupported(&unsupported, stringify!($name));
            wasmparser::validate(&wasm).expect("linked WASM is invalid");
            println!("  linked: {} bytes, {:?}", wasm.len(), $eh);
            for spec in &specs {
                match run_module(&wasm, spec.entry) {
                    Ok(state) => println!("  ✓ {}: {} hints", spec.entry, state.hints.len()),
                    Err(e) if $eh == Eh::With => eprintln!("  ! EH skipped ({}): {e}", spec.entry),
                    Err(e) => panic!("run {} failed: {e}", spec.entry),
                }
            }
        }
    };
}

// ── RISCV corpus smoke tests ──────────────────────────────────────────────────

smoke!(smoke_rv32_mul_no_eh,   "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::None);
smoke!(smoke_rv32_mul_eh,      "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With);
smoke!(smoke_rv32_br_no_eh,    "rv32im/02_branches",        arch=Arch::Rv32, Eh::None);
smoke!(smoke_rv32_br_eh,       "rv32im/02_branches",        arch=Arch::Rv32, Eh::With);
smoke!(smoke_rv32_ls_no_eh,    "rv32im/03_loads_stores",    arch=Arch::Rv32, Eh::None);
smoke!(smoke_rv32_ls_eh,       "rv32im/03_loads_stores",    arch=Arch::Rv32, Eh::With);
smoke!(smoke_rv32_sh_no_eh,    "rv32im/04_shifts",          arch=Arch::Rv32, Eh::None);
smoke!(smoke_rv32_sh_eh,       "rv32im/04_shifts",          arch=Arch::Rv32, Eh::With);
smoke!(smoke_rv32_imm_no_eh,   "rv32im/05_immediate_ops",   arch=Arch::Rv32, Eh::None);
smoke!(smoke_rv32_imm_eh,      "rv32im/05_immediate_ops",   arch=Arch::Rv32, Eh::With);
smoke!(smoke_rv64_arith_no_eh, "rv64im/01_basic_arith",     arch=Arch::Rv64, Eh::None);
smoke!(smoke_rv64_arith_eh,    "rv64im/01_basic_arith",     arch=Arch::Rv64, Eh::With);
smoke!(smoke_rv64_word_no_eh,  "rv64im/02_word_ops",        arch=Arch::Rv64, Eh::None);
smoke!(smoke_rv64_word_eh,     "rv64im/02_word_ops",        arch=Arch::Rv64, Eh::With);
smoke!(smoke_rv64_ls_no_eh,    "rv64im/03_loads_stores",    arch=Arch::Rv64, Eh::None);
smoke!(smoke_rv64_ls_eh,       "rv64im/03_loads_stores",    arch=Arch::Rv64, Eh::With);
smoke!(smoke_rv64_sh_no_eh,    "rv64im/04_shifts",          arch=Arch::Rv64, Eh::None);
smoke!(smoke_rv64_sh_eh,       "rv64im/04_shifts",          arch=Arch::Rv64, Eh::With);
smoke!(smoke_rv64_mul_no_eh,   "rv64im/05_multiply_divide", arch=Arch::Rv64, Eh::None);
smoke!(smoke_rv64_mul_eh,      "rv64im/05_multiply_divide", arch=Arch::Rv64, Eh::With);

// ── RISCV C smoke tests ───────────────────────────────────────────────────────

smoke_c!(smoke_rv32_c_arith_no_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::None);
smoke_c!(smoke_rv32_c_arith_eh,    env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::With);
smoke_c!(smoke_rv64_c_arith_no_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::None);
smoke_c!(smoke_rv64_c_arith_eh,    env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::With);

// ── x86_64 C smoke tests (corpus-equivalent: compile C programs) ──────────────

smoke_c!(smoke_x86_c_arith_no_eh, env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::None);
smoke_c!(smoke_x86_c_arith_eh,    env="E2E_X86_ARITH", arch=Arch::X86_64, Eh::With);

// ── RISCV run tests ───────────────────────────────────────────────────────────

run!(run_rv32_mul_no_eh,   "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::None);
run!(run_rv32_mul_eh,      "rv32im/01_multiply_divide", arch=Arch::Rv32, Eh::With);
run!(run_rv32_br_no_eh,    "rv32im/02_branches",        arch=Arch::Rv32, Eh::None);
run!(run_rv32_br_eh,       "rv32im/02_branches",        arch=Arch::Rv32, Eh::With);
run!(run_rv64_arith_no_eh, "rv64im/01_basic_arith",     arch=Arch::Rv64, Eh::None);
run!(run_rv64_arith_eh,    "rv64im/01_basic_arith",     arch=Arch::Rv64, Eh::With);

// ── C run tests ───────────────────────────────────────────────────────────────

run_c!(run_rv32_c_arith_no_eh, env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::None);
run_c!(run_rv32_c_arith_eh,    env="E2E_RV32_ARITH", arch=Arch::Rv32, Eh::With);
run_c!(run_rv64_c_arith_no_eh, env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::None);
run_c!(run_rv64_c_arith_eh,    env="E2E_RV64_ARITH", arch=Arch::Rv64, Eh::With);
run_c!(run_x86_c_arith_no_eh,  env="E2E_X86_ARITH",  arch=Arch::X86_64, Eh::None);
run_c!(run_x86_c_arith_eh,     env="E2E_X86_ARITH",  arch=Arch::X86_64, Eh::With);

// ── Same-arch linkage tests ───────────────────────────────────────────────────

link!(link_rv32_rv32_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="start_0"),
     ("rv32im/02_branches",        arch=Arch::Rv32, entry="start_1")],
    Eh::None);
link!(link_rv32_rv32_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="start_0"),
     ("rv32im/02_branches",        arch=Arch::Rv32, entry="start_1")],
    Eh::With);

// ── Mixed-arch linkage: RV32 + RV64 ──────────────────────────────────────────

link!(link_rv32_rv64_no_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="rv32_start"),
     ("rv64im/01_basic_arith",     arch=Arch::Rv64, entry="rv64_start")],
    Eh::None);
link!(link_rv32_rv64_eh,
    [("rv32im/01_multiply_divide", arch=Arch::Rv32, entry="rv32_start"),
     ("rv64im/01_basic_arith",     arch=Arch::Rv64, entry="rv64_start")],
    Eh::With);

// ── Mixed-arch linkage: RISCV + x86_64 (C programs) ─────────────────────────

link_c!(link_rv32_x86_no_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32,   entry="rv32_start"),
     ("E2E_X86_ARITH",             is_corpus=false, arch=Arch::X86_64, entry="x86_start")],
    Eh::None);
link_c!(link_rv32_x86_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32,   entry="rv32_start"),
     ("E2E_X86_ARITH",             is_corpus=false, arch=Arch::X86_64, entry="x86_start")],
    Eh::With);

// ── Three-arch linkage: RV32 + RV64 + x86_64 ─────────────────────────────────

link_c!(link_rv32_rv64_x86_no_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32,   entry="rv32_fn"),
     ("rv64im/01_basic_arith",     is_corpus=true,  arch=Arch::Rv64,   entry="rv64_fn"),
     ("E2E_X86_ARITH",             is_corpus=false, arch=Arch::X86_64, entry="x86_fn")],
    Eh::None);
link_c!(link_rv32_rv64_x86_eh,
    [("rv32im/01_multiply_divide", is_corpus=true,  arch=Arch::Rv32,   entry="rv32_fn"),
     ("rv64im/01_basic_arith",     is_corpus=true,  arch=Arch::Rv64,   entry="rv64_fn"),
     ("E2E_X86_ARITH",             is_corpus=false, arch=Arch::X86_64, entry="x86_fn")],
    Eh::With);

#[test]
fn debug_rv32_c_arith() {
    let path = match c_obj("E2E_RV32_ARITH") { Some(p) => p, None => return };
    let (text, addr) = match load_text(&path) { Some(v) => v, None => return };

    // Disassemble the RISCV .text section — linear and conservative.
    eprintln!("── RV32 .text linear disassembly ({} bytes @ {addr:#x}) ──", text.len());
    disasm_rv(&text, addr as u64, Xlen::Rv32);
    disasm_rv_conservative(&text, addr as u64, Xlen::Rv32);

    let (wasm, _) = build_single(&text, addr, Arch::Rv32, Eh::None);

    // Decode operators near the error using wasmparser.
    if let Err(e) = wasmparser::validate(&wasm) {
        let err_offset = e.offset();
        let window = 40;
        let start = err_offset.saturating_sub(window);
        eprintln!("Validation error at offset {err_offset:#x}: {e}");
        eprintln!("Decoding operators in range [{start:#x}..{:#x}]:", err_offset + window);
        decode_operators_near(&wasm, start, err_offset + window);
        panic!("invalid WASM");
    }
}

#[test]
fn debug_rv64_c_arith() {
    let path = match c_obj("E2E_RV64_ARITH") { Some(p) => p, None => return };
    let (text, addr) = match load_text(&path) { Some(v) => v, None => return };

    eprintln!("── RV64 .text disassembly ({} bytes @ {addr:#x}) ──", text.len());
    disasm_rv(&text, addr as u64, Xlen::Rv64);
    disasm_rv_conservative(&text, addr as u64, Xlen::Rv64);

    let (wasm, _) = build_single(&text, addr, Arch::Rv64, Eh::None);
    if let Err(e) = wasmparser::validate(&wasm) {
        let err_offset = e.offset();
        let window = 40;
        let start = err_offset.saturating_sub(window);
        eprintln!("Validation error at offset {err_offset:#x}: {e}");
        eprintln!("Decoding operators in range [{start:#x}..{:#x}]:", err_offset + window);
        decode_operators_near(&wasm, start, err_offset + window);
        panic!("invalid WASM");
    }
}

/// Disassemble RISC-V bytes at base `pc`.
///
/// When `conservative = true`, attempts to decode at every 2-byte boundary
/// (matching the recompiler's conservative CFG reconstruction).  Otherwise,
/// advances by the actual instruction width to produce a clean listing.
fn disasm_rv(text: &[u8], pc: u64, xlen: Xlen) {
    disasm_rv_inner(text, pc, xlen, false);
}

fn disasm_rv_conservative(text: &[u8], pc: u64, xlen: Xlen) {
    eprintln!("── conservative 2-byte-boundary decode ──");
    disasm_rv_inner(text, pc, xlen, true);
}

fn disasm_rv_inner(text: &[u8], pc: u64, xlen: Xlen, conservative: bool) {
    use rv_asm::Inst;
    let mut i = 0usize;
    while i + 2 <= text.len() {
        let lo = u16::from_le_bytes([text[i], text[i + 1]]);
        if lo & 0x3 != 0x3 {
            // 16-bit compressed
            match Inst::decode_compressed(lo, xlen) {
                Ok(inst) => eprintln!("  {:#010x}  {:04x}          {inst}", pc + i as u64, lo),
                Err(_)   => eprintln!("  {:#010x}  {:04x}          <bad-c>", pc + i as u64, lo),
            }
            i += 2;
        } else {
            // 32-bit
            if i + 4 > text.len() { break; }
            let word = u32::from_le_bytes([text[i], text[i+1], text[i+2], text[i+3]]);
            match Inst::decode(word, xlen) {
                Ok((inst, _)) => eprintln!("  {:#010x}  {:08x}  {inst}", pc + i as u64, word),
                Err(_)         => eprintln!("  {:#010x}  {:08x}  <bad>",  pc + i as u64, word),
            }
            i += if conservative { 2 } else { 4 };
        }
    }
}

fn decode_operators_near(wasm: &[u8], from: usize, to: usize) {
    use wasmparser::{Parser, Payload};
    for payload in Parser::new(0).parse_all(wasm) {
        let Ok(payload) = payload else { continue };
        if let Payload::CodeSectionEntry(body) = payload {
            // Print local declarations for any function that overlaps [from, to].
            let body_range = body.range();
            if body_range.end < from || body_range.start > to { continue; }

            eprintln!("  -- function body [{:#x}..{:#x}]", body_range.start, body_range.end);
            if let Ok(locals_reader) = body.get_locals_reader() {
                let mut local_idx = 0u32;
                for local in locals_reader {
                    let Ok((count, ty)) = local else { break };
                    eprintln!("    locals {local_idx}..{}: {ty:?}", local_idx + count);
                    local_idx += count;
                }
            }

            let Ok(reader) = body.get_operators_reader() else { continue };
            let mut ops = reader;
            loop {
                let pos = ops.original_position();
                match ops.read() {
                    Ok(op) => {
                        if pos >= from && pos <= to {
                            eprintln!("  [{pos:#06x}] {op:?}");
                        }
                        if pos > to { break; }
                    }
                    Err(_) => break,
                }
            }
        }
    }
}
