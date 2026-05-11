//! Shared test harness for speet-e2e integration tests.
//!
//! Contains all runtime helpers (wasmi execution, ELF loading, module assembly),
//! native-recompiler translation helpers, WasmFrontend translation helpers, and
//! programmatic WASM fixture builders.

use std::{borrow::Cow, convert::Infallible, path::Path};

use object::{Object, ObjectSection};
use rv_asm::Xlen;
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext, TrapReactorAdapter};
use speet_link_core::{linker::LinkerPlugin, unit::{BinaryUnit, FuncType}};
use speet_memory::{AddressWidth, DirectMemory, IntWidth, MemoryAccess};
use speet_module_builder::MegabinaryBuilder;
use speet_riscv::{HintCallback, HintInfo, RiscVRecompiler};
use speet_traps::{JumpInfo, JumpKind, JumpTrap, LocalDeclarator, LocalLayout, LocalSlot, TrapAction, TrapContext};
use speet_traps::cond::{ConditionInfo, ConditionTrap};
use speet_wasm::{GuestMemoryConfig, IndexOffsets, WasmFrontend};
use speet_x86_64::X86Recompiler;
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, ElementSection, Elements, ExportKind, ExportSection,
    Function, FunctionSection, ImportSection, MemorySection, MemoryType, Module, RefType,
    TableSection, TableType, TagSection, TagType, TypeSection, ValType,
};
use wasmi::{AsContext, Engine, Linker, Module as WasmiModule, Store};
use wasmparser::BinaryReaderError;
use yecta::{EscapeTag, LocalPool, Reactor, TableIdx, TagIdx, TypeIdx};
use yecta::layout::CellIdx;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const REG_SAVE_BASE: u32 = 0x100;
pub const IMPORT_HINT: u32 = 0;
pub const N_IMPORTS: u32 = 3; // hint, write, exit

pub const HINT_RETURN: i32 = 0xCA11_u32 as i32;
pub const HINT_CALL:   i32 = 0xCA12_u32 as i32;

// ── EH / Arch enums ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Eh { None, With }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch { Rv32, Rv64, X86_64 }

// ── Harness state ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RegSnapshot { pub regs: [i32; 32] }
impl RegSnapshot {
    pub fn reg(&self, name: &str) -> i32 {
        self.regs[abi_reg_index(name).expect("unknown register")]
    }
}

pub struct HostState {
    pub hints:     Vec<(i32, RegSnapshot)>,
    pub stdout:    Vec<u8>,
    pub exit_code: Option<i32>,
}
impl HostState {
    pub fn new() -> Self { Self { hints: Vec::new(), stdout: Vec::new(), exit_code: None } }
}

// ── ABI register index ────────────────────────────────────────────────────────

pub fn abi_reg_index(name: &str) -> Option<usize> {
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
        "x30"|"t5"=>Some(30),"x31"|"t6"=>Some(31),_=>Option::None,
    }
}

// ── Corpus / C-object path helpers ───────────────────────────────────────────

pub fn corpus(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus").join(rel)
}

pub fn c_obj(path_env: &str) -> Option<std::path::PathBuf> {
    let p = std::env::var(path_env).unwrap_or_default();
    if p.is_empty() { return Option::None; }
    let p = std::path::PathBuf::from(p);
    if p.exists() { Some(p) } else { Option::None }
}

// ── ELF loader ────────────────────────────────────────────────────────────────

pub fn load_text(path: &Path) -> Option<(Vec<u8>, u64)> {
    if !path.exists() {
        eprintln!("Skipping: not found at {path:?}");
        return Option::None;
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
    if data.is_empty() { return Option::None; }
    Some((data, sec.address()))
}

pub fn report_unsupported(unsupported: &[String], context: &str) {
    if !unsupported.is_empty() {
        eprintln!("  [x86_64 unsupported in {context}]: {}", unsupported.join(", "));
    }
}

// ── ReactorAdapter helpers ────────────────────────────────────────────────────

pub fn make_rctx<'r>(
    reactor: &'r mut Reactor<(), Infallible, Function, LocalPool>,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> ReactorAdapter<'r, (), Infallible, Function, LocalPool> {
    static T: TableIdx = TableIdx(0);
    let escape_tag = match eh {
        Eh::None => Option::None,
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

pub fn collect_rv_params(rctx: &ReactorAdapter<'_, (), Infallible, Function, LocalPool>) -> Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout().iter_before(&mark)
        .flat_map(|(count, ty)| std::iter::repeat(ty).take(count as usize))
        .collect()
}

pub fn build_hint_callback(regs_are_i64: bool) -> impl HintCallback<(), Infallible> {
    move |hint: &HintInfo, ctx: &mut (), rctx: &mut speet_riscv::HintContext<(), Infallible>| {
        use wasm_encoder::{Instruction, MemArg};
        let mem = MemArg { memory_index: 0, align: 2, offset: 0 };
        for i in 0u32..32 {
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

// ── JumpEventTrap ─────────────────────────────────────────────────────────────

pub struct JumpEventTrap {
    pub scratch_slot: LocalSlot,
}

impl JumpEventTrap {
    pub fn new() -> Self { Self { scratch_slot: LocalSlot::default() } }
}

impl LocalDeclarator for JumpEventTrap {
    fn declare_params(&mut self, _cell: CellIdx, params: &mut LocalLayout) {
        self.scratch_slot = params.append(1, ValType::I32);
    }
}

impl JumpTrap<(), Infallible> for JumpEventTrap {
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        _ctx: &mut (),
        _trap_ctx: &mut TrapContext<(), Infallible>,
    ) -> Result<TrapAction, Infallible> {
        match info.kind {
            JumpKind::Return | JumpKind::Call | JumpKind::IndirectCall => {}
            _ => return Ok(TrapAction::Continue),
        }
        Ok(TrapAction::Continue)
    }
}

// ── Trap-aware ReactorAdapter helpers ─────────────────────────────────────────

pub fn make_trap_rctx<'r>(
    reactor: &'r mut Reactor<(), Infallible, Function, LocalPool>,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
    jump_trap: &'r mut JumpEventTrap,
) -> TrapReactorAdapter<'r, 'r, 'r, (), Infallible, Function, LocalPool> {
    static T: TableIdx = TableIdx(0);
    let escape_tag = match eh {
        Eh::None => Option::None,
        Eh::With => Some(EscapeTag { tag: TagIdx(0), ty: type_idx }),
    };
    let mut rctx = TrapReactorAdapter::new(
        reactor,
        yecta::Pool { handler: &T, ty: type_idx },
        escape_tag,
    );
    rctx.traps.set_jump_trap(jump_trap);
    rctx.set_base_func_offset(base_func_offset);
    rctx
}

pub fn collect_trap_params(
    rctx: &TrapReactorAdapter<'_, '_, '_, (), Infallible, Function, LocalPool>,
) -> Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout().iter_before(&mark)
        .flat_map(|(count, ty)| std::iter::repeat(ty).take(count as usize))
        .collect()
}

// ── Translation helpers ───────────────────────────────────────────────────────

pub struct Translated {
    pub fns:         Vec<Function>,
    pub params:      Vec<ValType>,
    pub unsupported: Vec<String>,
}

pub fn translate_rv(
    text: &[u8],
    start_addr: u32,
    xlen: Xlen,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> Translated {
    let rv64 = xlen == Xlen::Rv64;
    let mut recompiler = RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
        start_addr as u64, false, rv64, true,
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

pub fn translate_rv_with_trap(
    text: &[u8],
    start_addr: u32,
    xlen: Xlen,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> Translated {
    let rv64 = xlen == Xlen::Rv64;
    let mut recompiler = RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
        start_addr as u64, false, rv64, true,
    );
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut jump_trap = JumpEventTrap::new();
    let mut rctx = make_trap_rctx(&mut reactor, base_func_offset, type_idx, eh, &mut jump_trap);
    recompiler.setup_traps(&mut rctx);
    let params = collect_trap_params(&rctx);

    let mut hint_cb = build_hint_callback(rv64);
    recompiler.set_hint_callback(&mut hint_cb);

    let mut ctx = ();
    recompiler.translate_bytes(&mut ctx, &mut rctx, text, start_addr, xlen,
        &mut |a| Function::new(a.collect::<Vec<_>>()))
        .expect("translate_bytes failed");

    Translated { fns: rctx.drain_fns(), params, unsupported: vec![] }
}

pub fn translate_x86(
    text: &[u8],
    rip: u64,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> Translated {
    let mut recompiler = X86Recompiler::new_with_base_rip(rip);
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor, base_func_offset, type_idx, eh);
    recompiler.setup_traps(&mut rctx, &mut ());
    let params = collect_rv_params(&rctx);

    let mut ctx = ();
    recompiler.translate_bytes(&mut ctx, &mut rctx, text, rip,
        &mut |a| Function::new(a.collect::<Vec<_>>()))
        .expect("translate_bytes failed");

    let unsupported: Vec<String> = recompiler.unsupported_insns().iter().cloned().collect();
    Translated { fns: rctx.drain_fns(), params, unsupported }
}

pub fn translate(
    text: &[u8],
    start_addr: u64,
    arch: Arch,
    base_func_offset: u32,
    type_idx: TypeIdx,
    eh: Eh,
) -> Translated {
    match arch {
        Arch::Rv32   => translate_rv(text, start_addr as u32, Xlen::Rv32, base_func_offset, type_idx, eh),
        Arch::Rv64   => translate_rv(text, start_addr as u32, Xlen::Rv64, base_func_offset, type_idx, eh),
        Arch::X86_64 => translate_x86(text, start_addr, base_func_offset, type_idx, eh),
    }
}

// ── Module assembly ───────────────────────────────────────────────────────────

pub struct BinarySlice {
    pub fns:            Vec<Function>,
    pub params:         Vec<ValType>,
    pub start_func_idx: u32,
    pub entry_name:     String,
}

pub fn assemble_module(slices: &[BinarySlice], eh: Eh) -> Vec<u8> {
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
    for p in &unique_params { types.ty().function(p.clone(), []); }
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
        minimum: 64, maximum: Option::None, memory64: true, shared: false, page_size_log2: Option::None,
    });

    let mut tags = TagSection::new();
    if eh == Eh::With {
        for i in 0..n_rv_types {
            tags.tag(TagType { kind: wasm_encoder::TagKind::Exception, func_type_idx: i });
        }
    }

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    for s in slices { exports.export(&s.entry_name, ExportKind::Func, s.start_func_idx); }

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

// ── High-level native module builders ────────────────────────────────────────

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
            recompiler.setup_traps(&mut rctx, &mut ());
            collect_rv_params(&rctx)
        }
    }
}

pub fn build_single(text: &[u8], start_addr: u64, arch: Arch, eh: Eh) -> (Vec<u8>, Vec<String>) {
    let t = translate(text, start_addr, arch, N_IMPORTS, TypeIdx(0), eh);
    let unsupported = t.unsupported.clone();
    let slice = BinarySlice {
        params: t.params, fns: t.fns,
        start_func_idx: N_IMPORTS, entry_name: "_start".into(),
    };
    (assemble_module(&[slice], eh), unsupported)
}

pub fn build_single_with_trap(text: &[u8], start_addr: u64, arch: Arch, eh: Eh) -> (Vec<u8>, Vec<String>) {
    let t = match arch {
        Arch::Rv32 => translate_rv_with_trap(text, start_addr as u32, Xlen::Rv32, N_IMPORTS, TypeIdx(0), eh),
        Arch::Rv64 => translate_rv_with_trap(text, start_addr as u32, Xlen::Rv64, N_IMPORTS, TypeIdx(0), eh),
        Arch::X86_64 => {
            let t = translate(text, start_addr, arch, N_IMPORTS, TypeIdx(0), eh);
            let unsupported = t.unsupported.clone();
            let slice = BinarySlice { params: t.params, fns: t.fns, start_func_idx: N_IMPORTS, entry_name: "_start".into() };
            return (assemble_module(&[slice], eh), unsupported);
        }
    };
    let unsupported = t.unsupported.clone();
    let slice = BinarySlice { params: t.params, fns: t.fns, start_func_idx: N_IMPORTS, entry_name: "_start".into() };
    (assemble_module(&[slice], eh), unsupported)
}

pub struct LinkSpec<'a> {
    pub text:       &'a [u8],
    pub start_addr: u64,
    pub arch:       Arch,
    pub entry:      &'static str,
}

pub fn build_linked(specs: &[LinkSpec<'_>], eh: Eh) -> (Vec<u8>, Vec<String>) {
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
            data_init_fn: Option::None,
        };
        builder.on_unit(unit);
        slices.push(BinarySlice {
            fns: vec![], params: t.params,
            start_func_idx, entry_name: s.entry.to_string(),
        });
        running_offset += n;
    }

    let output = builder.finish();
    let mut fn_iter = output.fns.into_iter();
    for (i, s) in specs.iter().enumerate() {
        let start = slices[i].start_func_idx - N_IMPORTS;
        let end = if i + 1 < specs.len() {
            slices[i + 1].start_func_idx - N_IMPORTS
        } else {
            running_offset - N_IMPORTS
        };
        let _ = s;
        slices[i].fns = fn_iter.by_ref().take((end - start) as usize).collect();
    }

    (assemble_module(&slices, eh), all_unsupported)
}

// ── Wasmi execution ───────────────────────────────────────────────────────────

pub fn run_module(wasm: &[u8], entry: &str) -> Result<HostState, String> {
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

// ── WasmFrontend helpers ──────────────────────────────────────────────────────

pub struct WasmTranslateConfig {
    pub mapper:    Option<Box<dyn MemoryAccess<(), BinaryReaderError>>>,
    pub cond_trap: Option<Box<dyn ConditionTrap<(), BinaryReaderError>>>,
}

impl WasmTranslateConfig {
    pub fn plain() -> Self { Self { mapper: Option::None, cond_trap: Option::None } }
}

/// Translate a WASM module through `WasmFrontend` and assemble a runnable
/// output module.  All translated functions are exported by index
/// (`"fn0"`, `"fn1"`, …); the caller supplies `entry_exports` to add named
/// exports pointing at specific function indices.
pub fn translate_wasm(
    input: &[u8],
    cfg: WasmTranslateConfig,
    entry_exports: &[(&str, u32)], // (export_name, fn_index_in_output)
) -> Vec<u8> {
    let per_memory = vec![GuestMemoryConfig {
        addr_width: AddressWidth::W32,
        memory_access: cfg.mapper,
    }];

    let mut frontend: WasmFrontend<(), BinaryReaderError> =
        WasmFrontend::with_wasm_encoder_fn(per_memory, 0, IndexOffsets::default());

    if let Some(trap) = cfg.cond_trap {
        frontend.set_condition_trap(trap);
    }

    let mut linker: speet_linker::Linker<'_, '_, (), BinaryReaderError> =
        speet_linker::Linker::new();
    frontend.translate_module(&mut (), &mut linker, input)
        .expect("WasmFrontend::translate_module failed");

    let compiled = frontend.take_compiled();

    // Collect unique function types.
    let mut unique_types: Vec<(Vec<ValType>, Vec<ValType>)> = Vec::new();
    for (_, ft) in &compiled {
        let params: Vec<ValType>  = ft.params_val_types().collect();
        let results: Vec<ValType> = ft.results_val_types().collect();
        if !unique_types.iter().any(|(p, r)| p == &params && r == &results) {
            unique_types.push((params, results));
        }
    }
    let type_idx_of = |ft: &FuncType| -> u32 {
        let params:  Vec<ValType> = ft.params_val_types().collect();
        let results: Vec<ValType> = ft.results_val_types().collect();
        unique_types.iter().position(|(p, r)| p == &params && r == &results).unwrap() as u32
    };

    let mut types = TypeSection::new();
    for (p, r) in &unique_types { types.ty().function(p.clone(), r.clone()); }

    let mut funcs = FunctionSection::new();
    for (_, ft) in &compiled { funcs.function(type_idx_of(ft)); }

    // Include a memory section only when the input module declared a memory.
    let needs_memory = !frontend.memory_infos().is_empty();
    let mut mems = MemorySection::new();
    if needs_memory {
        mems.memory(MemoryType {
            minimum: 1, maximum: Option::None, memory64: false, shared: false, page_size_log2: Option::None,
        });
    }

    let mut exports = ExportSection::new();
    for (name, idx) in entry_exports {
        exports.export(name, ExportKind::Func, *idx);
    }
    if needs_memory {
        exports.export("memory", ExportKind::Memory, 0);
    }

    let mut code = CodeSection::new();
    for (f, _) in &compiled { code.function(f); }

    let mut module = Module::new();
    module.section(&types);
    module.section(&funcs);
    if needs_memory { module.section(&mems); }
    module.section(&exports);
    module.section(&code);
    module.finish()
}

/// Translate a WASM module through `WasmFrontend` with a `"env" "decide"`
/// import at index 0 and a `HookConditionTrap` that calls it on every branch
/// condition.  The output module exports the (single) translated function as
/// `entry_name` at function index 1.
///
/// Use `run_wasm_with_decide` to execute the result.
pub fn translate_wasm_with_decide_import(input: &[u8], entry_name: &str) -> Vec<u8> {
    let per_memory = vec![GuestMemoryConfig {
        addr_width: AddressWidth::W32,
        memory_access: Option::None,
    }];
    let mut frontend: WasmFrontend<(), BinaryReaderError> =
        WasmFrontend::with_wasm_encoder_fn(per_memory, 0, IndexOffsets::default());
    frontend.set_condition_trap(Box::new(HookConditionTrap { decide_fn_idx: 0 }));

    let mut linker: speet_linker::Linker<'_, '_, (), BinaryReaderError> =
        speet_linker::Linker::new();
    frontend.translate_module(&mut (), &mut linker, input)
        .expect("WasmFrontend::translate_module failed");

    let compiled = frontend.take_compiled();
    let (translated_fn, func_type) = compiled.into_iter().next().expect("one function");
    let fn_params:  Vec<ValType> = func_type.params_val_types().collect();
    let fn_results: Vec<ValType> = func_type.results_val_types().collect();

    // Type 0: (i32) -> i32 for the decide import (and the test fn if same sig).
    let decide_ty: (Vec<ValType>, Vec<ValType>) =
        (vec![ValType::I32], vec![ValType::I32]);
    let fn_ty_same_as_decide = fn_params == decide_ty.0 && fn_results == decide_ty.1;

    let mut types = TypeSection::new();
    types.ty().function(decide_ty.0.clone(), decide_ty.1.clone()); // type 0 = decide sig
    let fn_type_idx = if fn_ty_same_as_decide {
        0u32
    } else {
        types.ty().function(fn_params, fn_results);
        1u32
    };

    let mut imports = ImportSection::new();
    imports.import("env", "decide", wasm_encoder::EntityType::Function(0));
    // decide is at fn index 0; translated fn is at fn index 1.

    let mut funcs = FunctionSection::new();
    funcs.function(fn_type_idx);

    let mut exports = ExportSection::new();
    exports.export(entry_name, ExportKind::Func, 1);

    let mut codes = CodeSection::new();
    codes.function(&translated_fn);

    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    module.section(&exports);
    module.section(&codes);
    module.finish()
}

/// Run a translated WASM module that returns a single `i32`.
/// The module must export `entry` as `() -> i32`.
pub fn run_wasm_returning_i32(wasm: &[u8], entry: &str) -> i32 {
    let engine = Engine::default();
    type State = ();
    let mut store: Store<State> = Store::new(&engine, ());
    let linker: Linker<State> = Linker::new(&engine);
    let module = WasmiModule::new(&engine, wasm).expect("valid wasm");
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    let func = instance.get_typed_func::<i32, i32>(&mut store, entry).unwrap();
    func.call(&mut store, 0).unwrap()
}

/// Run a translated WASM module calling `entry(input) -> i32` with a
/// `"env" "decide"` host import that maps condition values.
pub fn run_wasm_with_decide(
    wasm: &[u8],
    entry: &str,
    input: i32,
    decide_fn: impl Fn(i32) -> i32 + Send + Sync + 'static,
) -> i32 {
    let engine = Engine::default();
    type State = ();
    let mut store: Store<State> = Store::new(&engine, ());
    let mut linker: Linker<State> = Linker::new(&engine);
    linker.func_wrap("env", "decide", move |v: i32| -> i32 { decide_fn(v) }).unwrap();
    let module = WasmiModule::new(&engine, wasm).expect("valid wasm");
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    let func = instance.get_typed_func::<i32, i32>(&mut store, entry).unwrap();
    func.call(&mut store, input).unwrap()
}

/// Return a `standard_page_table_mapper` suitable for smoke tests (validates
/// structural correctness of mapper-annotated WASM; page table not initialised
/// at runtime).
pub fn make_test_mapper() -> Box<dyn MemoryAccess<(), BinaryReaderError>> {
    use speet_memory::{standard_page_table_mapper, PageTableBase};
    Box::new(DirectMemory::new(
        standard_page_table_mapper(
            PageTableBase::Constant(0x0),
            PageTableBase::Constant(0x0),
            0,
            false,
        ),
        0,
        AddressWidth::W32,
        IntWidth::I32,
    ))
}

// ── Condition trap implementations ────────────────────────────────────────────

pub struct FlipConditionTrap;
impl LocalDeclarator for FlipConditionTrap {}
impl ConditionTrap<(), BinaryReaderError> for FlipConditionTrap {
    fn on_condition(
        &self,
        _info: &ConditionInfo,
        _ctx: &mut (),
        go: &mut (dyn FnMut(&mut (), &wasm_encoder::Instruction<'_>) -> Result<(), BinaryReaderError> + '_),
    ) -> Result<(), BinaryReaderError> {
        go(&mut (), &wasm_encoder::Instruction::I32Eqz)
    }
}

/// Calls `call $decide_fn_idx` where the host decides the new condition value.
pub struct HookConditionTrap { pub decide_fn_idx: u32 }
impl LocalDeclarator for HookConditionTrap {}
impl ConditionTrap<(), BinaryReaderError> for HookConditionTrap {
    fn on_condition(
        &self,
        _info: &ConditionInfo,
        _ctx: &mut (),
        go: &mut (dyn FnMut(&mut (), &wasm_encoder::Instruction<'_>) -> Result<(), BinaryReaderError> + '_),
    ) -> Result<(), BinaryReaderError> {
        go(&mut (), &wasm_encoder::Instruction::Call(self.decide_fn_idx))
    }
}

// ── WASM fixture builders ─────────────────────────────────────────────────────

/// ```wat
/// (module
///   (func (export "compute") (param i32 i32) (result i32)
///     local.get 0  local.get 1  i32.add))
/// ```
pub fn wasm_arith() -> Vec<u8> {
    use wasm_encoder::Instruction as I;
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32, ValType::I32], [ValType::I32]);

    let mut funcs = FunctionSection::new();
    funcs.function(0);

    let mut exports = ExportSection::new();
    exports.export("compute", ExportKind::Func, 0);

    let mut codes = CodeSection::new();
    let mut f = Function::new([]);
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::LocalGet(1));
    f.instruction(&I::I32Add);
    f.instruction(&I::End);
    codes.function(&f);

    let mut m = Module::new();
    m.section(&types);
    m.section(&funcs);
    m.section(&exports);
    m.section(&codes);
    m.finish()
}

/// ```wat
/// (module
///   (func (export "test") (param i32) (result i32)
///     local.get 0
///     if (result i32)  i32.const 1  else  i32.const 0  end))
/// ```
/// Exercises condition trap hook points (`if` and `br_if`).
pub fn wasm_branches() -> Vec<u8> {
    use wasm_encoder::Instruction as I;
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I32]);

    let mut funcs = FunctionSection::new();
    funcs.function(0);

    let mut exports = ExportSection::new();
    exports.export("test", ExportKind::Func, 0);

    let mut codes = CodeSection::new();
    let mut f = Function::new([]);
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::If(BlockType::Result(ValType::I32)));
    f.instruction(&I::I32Const(1));
    f.instruction(&I::Else);
    f.instruction(&I::I32Const(0));
    f.instruction(&I::End);
    f.instruction(&I::End);
    codes.function(&f);

    let mut m = Module::new();
    m.section(&types);
    m.section(&funcs);
    m.section(&exports);
    m.section(&codes);
    m.finish()
}

/// ```wat
/// (module
///   (memory 1)
///   (func (export "roundtrip") (param i32) (result i32)
///     i32.const 8  local.get 0  i32.store
///     i32.const 8  i32.load))
/// ```
/// Exercises mapper code paths (every load/store has its address translated).
pub fn wasm_memory_rw() -> Vec<u8> {
    use wasm_encoder::{Instruction as I, MemArg};
    let mem_arg = MemArg { memory_index: 0, align: 2, offset: 0 };

    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I32]);

    let mut funcs = FunctionSection::new();
    funcs.function(0);

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 1, maximum: Option::None, memory64: false, shared: false, page_size_log2: Option::None,
    });

    let mut exports = ExportSection::new();
    exports.export("roundtrip", ExportKind::Func, 0);
    exports.export("memory",    ExportKind::Memory, 0);

    let mut codes = CodeSection::new();
    let mut f = Function::new([]);
    f.instruction(&I::I32Const(8));
    f.instruction(&I::LocalGet(0));
    f.instruction(&I::I32Store(mem_arg));
    f.instruction(&I::I32Const(8));
    f.instruction(&I::I32Load(mem_arg));
    f.instruction(&I::End);
    codes.function(&f);

    let mut m = Module::new();
    m.section(&types);
    m.section(&funcs);
    m.section(&mems);
    m.section(&exports);
    m.section(&codes);
    m.finish()
}
