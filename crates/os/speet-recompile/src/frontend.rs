//! Frontend stage: load a host binary and drive speet to a WASM megabinary.
//!
//! Responsibilities (M1 wires the minimal path; M3 adds PLT tunneling):
//! 1. Load via [`binary_io::load_auto`]; assert OS+arch == host (v1 same-platform).
//! 2. Build an external-target table from `imports` (undefined dyn syms + PLT
//!    addrs) and undefined-symbol relocations, mapping `plt_addr -> guest_name`.
//!    During speet translation, a call/jmp whose target is in this table is
//!    lowered via `AmbientSink::call_ambient(name, sig)` instead of an internal
//!    tail-call.
//! 3. Feed non-text sections as passive WASM data segments + a `data_init_fn`.
//! 4. Register the page-table memory model (`speet_memory::VirtualMemory`).
//! 5. Resolve the entry point to a megabinary export (`__guest_entry`).

use binary_io::{BinArch, BinOs, ImportSym, LoadedBinary};
use speet_plugin_api::external_target::{ExternalTargetTable, LibraryId};

/// Build an [`ExternalTargetTable`] from a loaded binary's import list (PLT
/// addresses in [`LibraryId::MAIN_IMAGE`]).
pub fn external_targets_from_imports(imports: &[ImportSym]) -> ExternalTargetTable {
    let mut table = ExternalTargetTable::new();
    for imp in imports {
        if let Some(addr) = imp.plt_addr {
            table.insert(LibraryId::MAIN_IMAGE, addr, imp.name.clone());
        }
    }
    table
}

/// Alias kept for call-site brevity — prefer [`ExternalTargetTable`] in new code.
pub type ExternalTargets = ExternalTargetTable;

/// Returns the current host (OS, arch) so the driver can enforce same-platform.
pub fn host_platform() -> (BinOs, BinArch) {
    let os = if cfg!(target_os = "macos") {
        BinOs::MacOs
    } else {
        BinOs::Linux
    };
    let arch = if cfg!(target_arch = "aarch64") {
        BinArch::AArch64
    } else {
        BinArch::X86_64
    };
    (os, arch)
}

/// Assert the loaded binary matches the host platform (v1 constraint).
pub fn assert_same_platform(bin: &LoadedBinary) -> Result<(), String> {
    let (os, arch) = host_platform();
    if !matches!((&bin.os, os), (BinOs::Linux, BinOs::Linux) | (BinOs::MacOs, BinOs::MacOs)) {
        return Err(format!("input OS {:?} != host {:?} (v1 is same-platform)", bin.os, os));
    }
    if !matches!((&bin.arch, arch), (BinArch::X86_64, BinArch::X86_64) | (BinArch::AArch64, BinArch::AArch64)) {
        return Err(format!("input arch {:?} != host {:?} (v1 is same-platform)", bin.arch, arch));
    }
    Ok(())
}

// ── speet translate: guest machine code -> WASM module ───────────────────────

use crate::plt::PltCallPlan;
use core::convert::Infallible;
use speet_host_api::{ImportManifest, WasmValType};
use speet_link_core::layout::{EntityIndexSpace, IndexSlot};
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext, RuntimeLayoutParams, TextBaseSource};
use speet_link_core::image_layout::MemoryModel;
use speet_memory::memory_access_for_model;
use speet_reach::{CfgDecoder, CfgEdges, PcSlotMap};
use wasm_encoder::{
    CodeSection, ConstExpr, DataCountSection, DataSection, ElementSection, Elements, EntityType,
    ExportKind, ExportSection, Function, FunctionSection, ImportSection, Instruction, MemorySection,
    MemoryType, Module, RefType, TableSection, TableType, TypeSection, ValType,
};
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

static REACTOR_TABLE: TableIdx = TableIdx(0);

fn wasm_val_type(t: WasmValType) -> ValType {
    match t {
        WasmValType::I32 => ValType::I32,
        WasmValType::I64 => ValType::I64,
        WasmValType::F32 => ValType::F32,
        WasmValType::F64 => ValType::F64,
    }
}

/// Total host-capability slots declared for `manifest`. Read this back from
/// [`EntityIndexSpace`] rather than assuming it equals
/// `manifest.func_imports.len()` — see `docs/guides/thin-runtime-genericity.md`
/// principle 1.
pub fn host_capability_total(manifest: &ImportManifest) -> u32 {
    let mut space = EntityIndexSpace::empty();
    for _ in &manifest.func_imports {
        space.host_capabilities.append(1);
    }
    space.host_capabilities.total()
}

/// Assemble pre-translated function bodies into a runnable WASM module.
///
/// `entry_func_idx` is 0-based among translated functions only (before
/// `n_imports`), matching [`finish_module`].
pub fn assemble_translated_module(
    manifest: &ImportManifest,
    fns: &[Function],
    register_file_params: Vec<ValType>,
    entry_func_idx: u32,
    memory64: bool,
) -> Vec<u8> {
    let total = fns.len() as u32;
    let n_params = register_file_params.len() as u32;
    let (types, imports, space, func_slot) =
        build_import_section(manifest, register_file_params, total);
    finish_module(
        types,
        imports,
        &space,
        func_slot,
        fns,
        &[],
        memory64,
        entry_func_idx,
        n_params,
        &[],
    )
}

/// Build the register-file type (type 0) plus one function type per
/// `manifest` import, the import section itself, and the
/// [`EntityIndexSpace`] that allocates host-capability slots (one per
/// import, in manifest order) followed by `n_fns` translated function
/// bodies. This is the *only* place a WASM module's import-derived function
/// indices should be computed — never hand-count a
/// `N_IMPORTS`/`N_INTEGRATED_IMPORTS`/`N_SYSCALL_IMPORTS`-style constant
/// instead. See `docs/guides/thin-runtime-genericity.md` principle 1.
///
/// The WASM function index for import `i` is `space.host_capabilities.base(..)`
/// (i.e. simply `i`, since imports are always contiguous from 0 in the WASM
/// binary format); the base index for the `n_fns` translated bodies is
/// `space.host_capabilities.total()` — read that back rather than assuming
/// it equals `manifest.func_imports.len()` textually, so a future
/// non-WASM-import host-capability realization can change this rule in one
/// place.
fn build_import_section(
    manifest: &ImportManifest,
    register_file_params: Vec<ValType>,
    n_fns: u32,
) -> (TypeSection, ImportSection, EntityIndexSpace, IndexSlot) {
    let mut types = TypeSection::new();
    // type 0: (registers) -> (registers) — symmetric, not `-> ()`. Every
    // speet-emitted function shares this type; see
    // `docs/guides/thin-runtime-genericity.md` principle 4 for why (the
    // halt stub `finish_module` appends needs to surface the live register
    // file as real WASM results when a guest `ret`/indirect return lands
    // past the end of the translated set — e.g. `main` returning with no
    // crt0 chain — and every other function trivially satisfies this same
    // type via `return_call`/`return_call_indirect`, never a bare `Return`).
    types.ty().function(register_file_params.clone(), register_file_params);
    let mut imports = ImportSection::new();
    let mut space = EntityIndexSpace::empty();
    for (i, imp) in manifest.func_imports.iter().enumerate() {
        let type_idx = (i + 1) as u32; // type 0 is the register file
        types.ty().function(
            imp.params.iter().copied().map(wasm_val_type),
            imp.results.iter().copied().map(wasm_val_type),
        );
        imports.import(&imp.module, &imp.name, EntityType::Function(type_idx));
        space.host_capabilities.append(1);
    }
    let func_slot = space.functions.append(n_fns);
    (types, imports, space, func_slot)
}

/// Build the **halt stub**: the one extra function every module reserves
/// one slot past the last translated function, at guest address
/// `start_addr + text.len()` (see [`halt_addr`]) — the landing spot for a
/// guest `ret`/indirect return past the end of the translated set (most
/// commonly `main` returning with no crt0 chain to call `exit`). Its body
/// is exactly `local.get 0 .. local.get n-1; return`: it surfaces the live
/// register file at the point of return as this call's WASM results,
/// mirroring a real `crt0` calling `exit(main())`. See
/// `docs/guides/thin-runtime-genericity.md` principle 4 — copy this shape
/// for any new terminal control-flow path, don't invent a shorter one.
pub fn build_halt_stub(n_params: u32) -> Function {
    let mut f = Function::new([]);
    // Push in normal (ascending) param order, per the WASM multi-value
    // spec: result 0 pushed first/deepest, result n-1 pushed last/on top.
    // `wasm-blitz`'s SysV/AAPCS64 backend (`sysv_emit_epilogue` in
    // `blitz-aarch64`/`blitz-x86-64`'s `sysv.rs`) reads *this* spec order
    // to marshal result 0 into X0/RAX (the guest's return-value register —
    // always register-file param index 0), so a guest program's real
    // return value ends up as `__guest_entry`'s C return value when a
    // guest legitimately returns instead of calling `exit`. See
    // `docs/guides/thin-runtime-genericity.md` principle 4.
    for p in 0..n_params {
        f.instruction(&Instruction::LocalGet(p));
    }
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);
    f
}

/// Build one redirect-shim WASM function: marshal register file → import call → return register file.
pub fn build_redirect_shim_fn(
    convention: &speet_plugin_api::external_target::CallingConvention,
    import_idx: u32,
    n_register_params: u32,
) -> Function {
    let mut f = Function::new([]);
    for (i, &local) in convention.arg_locals.iter().enumerate() {
        f.instruction(&Instruction::LocalGet(local));
        if convention.wraps_i32(i) {
            f.instruction(&Instruction::I32WrapI64);
        }
    }
    f.instruction(&Instruction::Call(import_idx));
    if let Some(result_local) = convention.result_local {
        if convention.result_extend_i32 {
            f.instruction(&Instruction::I64ExtendI32U);
        }
        f.instruction(&Instruction::LocalSet(result_local));
    }
    for p in 0..n_register_params {
        f.instruction(&Instruction::LocalGet(p));
    }
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);
    f
}

/// Build redirect shim bodies and a symbol → shim-index map from a [`PltCallPlan`].
pub fn build_redirect_shims_from_plan(
    plt_plan: &PltCallPlan,
    arch: BinArch,
    manifest: &ImportManifest,
    n_register_params: u32,
) -> (Vec<Function>, std::collections::BTreeMap<String, u32>) {
    let mut shims = Vec::new();
    let mut by_symbol = std::collections::BTreeMap::new();
    let table = plt_plan.to_hook_table(arch, manifest);
    let mut ordered: std::collections::BTreeSet<(speet_plugin_api::external_target::LibraryId, u64)> =
        plt_plan.wasm_import_by_addr.keys().copied().collect();
    ordered.extend(plt_plan.native_shim_by_addr.keys().copied());
    for (shim_i, &(library, addr)) in ordered.iter().enumerate() {
        let label = plt_plan
            .targets
            .lookup(library, addr)
            .unwrap_or("?")
            .to_string();
        let import_idx = if let Some(&idx) = plt_plan.wasm_import_by_addr.get(&(library, addr)) {
            idx
        } else if let Some(core) = plt_plan.native_shim_by_addr.get(&(library, addr)) {
            plt_plan
                .import_idx_for_core_symbol(manifest, core)
                .unwrap_or_else(|| panic!("no manifest import for native shim {core}"))
        } else {
            continue;
        };
        let convention = if let Some(hook) = table.lookup(library, addr) {
            hook.convention.clone()
        } else {
            speet_abi_stubs::plt_calling_convention(manifest, arch, &label)
        };
        shims.push(build_redirect_shim_fn(
            &convention,
            import_idx,
            n_register_params,
        ));
        by_symbol.insert(label, shim_i as u32);
    }
    (shims, by_symbol)
}

/// Table/elements/exports scaffolding shared by every `assemble_*` variant.
/// Appends optional redirect shim functions, then the halt stub.
fn finish_module(
    mut types: TypeSection,
    imports: ImportSection,
    space: &EntityIndexSpace,
    func_slot: IndexSlot,
    fns: &[Function],
    redirect_shims: &[Function],
    memory64: bool,
    entry_func_idx: u32,
    n_register_params: u32,
    data_segments: &[DataSegment],
) -> Vec<u8> {
    let n_imports = space.host_capabilities.total();
    let total = space.functions.count(func_slot);
    let n_shims = redirect_shims.len() as u32;
    let halt_stub = build_halt_stub(n_register_params);

    // The data-init function (if any) gets its own `() -> ()` type, appended
    // after every import's type — `types.len()` before the append is exactly
    // that new type's index regardless of how many import types precede it.
    let data_init_type_idx = (!data_segments.is_empty()).then(|| {
        let idx = types.len();
        types.ty().function([], []);
        idx
    });

    let mut funcs = FunctionSection::new();
    for _ in fns {
        funcs.function(0);
    }
    for _ in redirect_shims {
        funcs.function(0);
    }
    funcs.function(0); // halt stub: also type 0, (registers) -> (registers)
    if let Some(ty) = data_init_type_idx {
        funcs.function(ty);
    }

    let table_size = n_imports + total + n_shims + 1;
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: table_size as u64,
        maximum: Some(table_size as u64),
        table64: true,
        shared: false,
    });

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 64,
        maximum: None,
        memory64,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("_start", ExportKind::Func, n_imports + entry_func_idx);
    // The data-init function sits right after the halt stub in both the
    // function and code sections (neither is part of the guest-function
    // dispatch table, so this doesn't disturb the elem/table indices below)
    // — exported by name since the C entry shim calls it directly, not
    // through the table.
    if data_init_type_idx.is_some() {
        exports.export(
            DATA_INIT_EXPORT_NAME,
            ExportKind::Func,
            n_imports + total + n_shims + 1,
        );
    }

    let indices: Vec<u32> = (n_imports..n_imports + total + n_shims + 1).collect();
    let mut elems = ElementSection::new();
    elems.active(
        Some(0),
        &ConstExpr::i64_const(n_imports as i64),
        Elements::Functions(std::borrow::Cow::Borrowed(&indices)),
    );

    let mut code = CodeSection::new();
    for f in fns {
        code.function(f);
    }
    for f in redirect_shims {
        code.function(f);
    }
    code.function(&halt_stub);
    let data_init_fn = data_init_type_idx.map(|_| build_data_init_fn(data_segments));
    if let Some(f) = &data_init_fn {
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
    // Bulk-memory-operations spec: a module using `memory.init`/`data.drop`
    // must carry a `DataCount` section (between elements and code) so
    // decoders can validate those instructions' data-index operands without
    // a forward reference to the later `Data` section.
    if !data_segments.is_empty() {
        module.section(&DataCountSection {
            count: data_segments.len() as u32,
        });
    }
    module.section(&code);
    if !data_segments.is_empty() {
        let mut data = DataSection::new();
        for seg in data_segments {
            data.passive(seg.bytes.iter().copied());
        }
        module.section(&data);
    }
    module.finish()
}

/// Guest address of the halt stub (see [`build_halt_stub`]) for a
/// translation whose `.text` blob starts at `start_addr` and is
/// `text_len` bytes long: one slot past the last translated function, at
/// exactly the address the granularity-based addr-to-slot formula already
/// maps to `total` (the halt stub's table index) with no special-casing.
/// Callers seed this as the guest's initial return-address register/slot
/// before invoking the entry function, so a guest `ret` that's never
/// overwritten that register (i.e. really is the final return) lands here
/// instead of on whatever garbage the register held — see
/// `docs/guides/thin-runtime-genericity.md` principle 4.
pub fn halt_addr(start_addr: u64, text_len: usize) -> u64 {
    start_addr + text_len as u64
}

/// One guest data section (`.rodata`/`.data`) to populate `__wasm_mem` with
/// at load time, via the generated data-init function (see
/// [`DATA_INIT_EXPORT_NAME`], [`build_data_init_fn`]). `addr` is the
/// section's own guest virtual address; `bytes` its literal content. Never
/// pass `.bss` here — `__wasm_mem` already starts zeroed, so a zero-fill
/// segment would be redundant work, not a correctness fix.
#[derive(Clone, Debug)]
pub struct DataSegment {
    pub addr: u64,
    pub bytes: Vec<u8>,
}

/// Export name of the generated data-init function (see
/// [`build_data_init_fn`]) — the C entry shim (`speet_rt::entry_bridge`)
/// calls this once, before seeding registers/invoking the guest entry,
/// whenever the guest has any non-empty [`DataSegment`]s.
pub const DATA_INIT_EXPORT_NAME: &str = "__speet_data_init";

/// Build the data-init function: for each segment, `memory.init` copies its
/// bytes to its own guest address, then `data.drop` releases the (already
/// fully consumed, by construction — see `Instruction::DataDrop`'s doc
/// comment in `wasm-blitz`) segment. `dest` is pushed as `i64.const` since
/// this module's memory is always memory64 (see `finish_module`'s
/// `memory64` parameter) — `src_offset`/`len` stay `i32` regardless, per the
/// WASM bulk-memory-operations spec (they index into the segment itself,
/// not the address space).
pub fn build_data_init_fn(segments: &[DataSegment]) -> Function {
    let mut f = Function::new([]);
    for (i, seg) in segments.iter().enumerate() {
        let data_index = i as u32;
        f.instruction(&Instruction::I64Const(seg.addr as i64));
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::I32Const(seg.bytes.len() as i32));
        f.instruction(&Instruction::MemoryInit { mem: 0, data_index });
        f.instruction(&Instruction::DataDrop(data_index));
    }
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);
    f
}

/// Build a fresh [`ReactorAdapter`] (mirrors the speet-e2e harness `make_rctx`,
/// with exception handling disabled). Generic over `E` so the
/// `ArchPlugin`-backed path (which needs `E: From<PluginError>`, never
/// satisfiable by the native path's `Infallible`) can reuse this builder
/// with its own error type — see `RecompilerChoice::Plugin` below.
fn make_rctx<'r, E>(
    reactor: &'r mut Reactor<(), E, Function, LocalPool>,
    base_func_offset: u32,
    text_base: TextBaseSource,
) -> ReactorAdapter<'r, (), E, Function, LocalPool> {
    let mut rctx = ReactorAdapter {
        reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark { slot_count: 0, total_locals: 0 },
        injected_start: yecta::Mark { slot_count: 0, total_locals: 0 },
        layout_params: RuntimeLayoutParams::with_text_base_source(text_base),
        pool: yecta::Pool { handler: &REACTOR_TABLE, ty: TypeIdx(0) },
        escape_tag: None,
    };
    rctx.set_base_func_offset(base_func_offset);
    rctx
}

fn collect_params<E>(rctx: &ReactorAdapter<'_, (), E, Function, LocalPool>) -> Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout()
        .iter_before(&mark)
        .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
        .collect()
}

/// Recompiled output of one guest binary: function bodies + their (shared) param
/// signature (the guest register file) + any unsupported instructions seen.
pub struct Translated {
    pub fns: Vec<Function>,
    pub params: Vec<ValType>,
    pub unsupported: Vec<String>,
    /// Index into `fns` (0-based, *before* the WASM host-capability/import
    /// offset is added) of the function to export as `_start` — the guest's
    /// real entry point, not necessarily `fns[0]`. A linked binary's `.text`
    /// commonly holds several functions before the one at the actual entry
    /// address (see the disassembly note in `docs/guides/thin-runtime-genericity.md`
    /// principle 2); exporting `fns[0]` unconditionally silently runs the
    /// wrong function whenever that's not also the entry. Defaults to `0`
    /// for every caller that doesn't know a distinct entry address (single-
    /// snippet translations, the RV64 syscall path, the WASM-plugin path).
    pub entry_func_idx: u32,
}

/// Which recompiler drives [`translate`]: one of speet's own built-in native
/// arch frontends, or an external [`ArchPlugin`](speet_plugin_api::ArchPlugin)
/// (see `docs/guides/plugin-api.md`). `BinArch` is owned by the external
/// `binary_io` crate and can't grow a plugin variant itself, hence this
/// wrapper local to `speet-recompile`.
pub enum RecompilerChoice {
    Native(BinArch),
    Plugin(Box<dyn speet_plugin_api::ArchPlugin>),
}

/// Translate a `.text` blob of guest machine code to speet WASM functions.
/// `manifest` must be the same [`ImportManifest`] the caller will later
/// assemble the module with — its import count fixes `base_func_offset`
/// (the WASM index the first translated function lands at), so a mismatch
/// here silently makes every internal `call`/`return_call` target the wrong
/// function. See `docs/guides/thin-runtime-genericity.md` principle 1.
pub fn translate(text: &[u8], start_addr: u64, choice: RecompilerChoice, manifest: &ImportManifest) -> Translated {
    translate_with_plt(text, start_addr, choice, None, None, manifest, MemoryModel::OwnedLinear)
}

/// Guest-address-offset → relative function-slot-index granularity (bytes
/// per decode slot) for a native arch — the same mapping each recompiler's
/// own `pc_to_func_idx` uses internally (`speet-x86_64`: 1, every byte
/// offset is a candidate slot; `speet-aarch64`: 4, fixed-width encoding).
/// Kept in sync manually today; see `docs/guides/thin-runtime-genericity.md`
/// principle 1 for why a mismatch here (vs. what the arch's own recompiler
/// uses) would silently point `_start` at the wrong function.
fn slot_granularity(arch: BinArch) -> u64 {
    match arch {
        BinArch::X86_64 => 1,
        BinArch::AArch64 => 4,
    }
}

fn redirect_shim_count(plt_plan: Option<&PltCallPlan>) -> u32 {
    let Some(plan) = plt_plan else {
        return 0;
    };
    let mut keys: std::collections::BTreeSet<_> =
        plan.wasm_import_by_addr.keys().copied().collect();
    keys.extend(plan.native_shim_by_addr.keys().copied());
    keys.len() as u32
}

struct Fixed4CfgDecoder;

impl CfgDecoder for Fixed4CfgDecoder {
    fn decode_edges(&self, _pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
        if bytes.len() < 4 {
            return None;
        }
        Some(CfgEdges {
            static_successors: vec![],
            fallthrough: true,
            insn_len: 4,
            has_indirect: false,
        })
    }
}

fn bind_memory_after_traps<E>(
    rc: &mut speet_x86_64::X86Recompiler<(), E>,
    rctx: &ReactorAdapter<'_, (), E, Function, LocalPool>,
) {
    rc.bind_memory_layout(rctx);
}

fn bind_memory_after_traps_aarch64<E>(
    rc: &mut speet_aarch64::AArch64Recompiler<(), E>,
    rctx: &ReactorAdapter<'_, (), E, Function, LocalPool>,
) {
    rc.bind_memory_layout(rctx);
}


fn build_pc_slot_map(
    arch: BinArch,
    text: &[u8],
    start_addr: u64,
    plt_plan: Option<&PltCallPlan>,
) -> PcSlotMap {
    let n_shims = redirect_shim_count(plt_plan);
    let halt = halt_addr(start_addr, text.len());
    let gran = slot_granularity(arch);
    let map = match arch {
        BinArch::X86_64 => {
            let dec = speet_x86_64::cfg::X86CfgDecoder;
            PcSlotMap::all_slots(text, start_addr, &dec)
        }
        BinArch::AArch64 => PcSlotMap::all_slots(text, start_addr, &Fixed4CfgDecoder),
    };
    let n_text = map.len() as u32;
    let mut map = map.with_redirect_shims(halt, n_shims, gran);
    if let Some(plan) = plt_plan {
        // Same address order as `build_redirect_shims_from_plan`.
        let mut stub_addrs: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        stub_addrs.extend(plan.wasm_import_by_addr.keys().map(|(_, a)| *a));
        stub_addrs.extend(plan.native_shim_by_addr.keys().map(|(_, a)| *a));
        map = map.with_plt_stub_aliases(stub_addrs, n_text);
    }
    map
}


/// Like [`translate`], but redirects PLT/external calls per `plt_plan` and,
/// when `entry_addr` is given, exports the function at that guest address
/// (rather than `fns[0]`) as `_start` — see [`Translated::entry_func_idx`].
pub fn translate_with_plt(
    text: &[u8],
    start_addr: u64,
    choice: RecompilerChoice,
    plt_plan: Option<&PltCallPlan>,
    entry_addr: Option<u64>,
    manifest: &ImportManifest,
    memory_model: MemoryModel,
) -> Translated {
    let n_imports = manifest.func_imports.len() as u32;
    let entry_func_idx = match (&choice, entry_addr) {
        (RecompilerChoice::Native(arch), Some(addr)) => {
            let granularity = slot_granularity(*arch);
            addr.saturating_sub(start_addr) / granularity
        }
        _ => 0,
    } as u32;
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(
        &mut reactor,
        n_imports,
        TextBaseSource::Constant(start_addr),
    );
    let mut ctx = ();

    let (params, unsupported) = match choice {
        RecompilerChoice::Native(arch) => match arch {
            BinArch::X86_64 => {
                let mut rc = speet_x86_64::X86Recompiler::new_with_base_rip(start_addr);
                rc.set_slot_assigner(build_pc_slot_map(
                    BinArch::X86_64,
                    text,
                    start_addr,
                    plt_plan,
                ));
                if let Some(idx) = manifest.index_of("env", "__speet_stub_for_pc") {
                    rc.set_stub_for_pc_import_idx(idx);
                }
                rc.set_memory_access(memory_access_for_model::<(), Infallible>(memory_model));
                rc.setup_traps(&mut rctx, &mut ctx);
                bind_memory_after_traps(&mut rc, &rctx);
                let params = collect_params(&rctx);
                rc.translate_bytes(&mut ctx, &mut rctx, text, start_addr, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .expect("translate_bytes");
                (params, rc.unsupported_insns().iter().cloned().collect())
            }
            BinArch::AArch64 => {
                let mut rc =
                    speet_aarch64::AArch64Recompiler::<(), Infallible>::new_with_base_pc(start_addr);
                rc.set_slot_assigner(build_pc_slot_map(
                    BinArch::AArch64,
                    text,
                    start_addr,
                    plt_plan,
                ));
                if let Some(idx) = manifest.index_of("env", "__speet_stub_for_pc") {
                    rc.set_stub_for_pc_import_idx(idx);
                }
                rc.set_memory_access(memory_access_for_model::<(), Infallible>(memory_model));
                rc.setup_traps(&mut rctx, &mut ctx);
                bind_memory_after_traps_aarch64(&mut rc, &rctx);
                let params = collect_params(&rctx);
                rc.translate_bytes(&mut ctx, &mut rctx, text, start_addr, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .expect("translate_bytes");
                (params, rc.unsupported_insns().iter().cloned().collect())
            }
        },
        RecompilerChoice::Plugin(plugin) => {
            // `ArchPluginRecompiler` requires `E: From<PluginError>`, which
            // `Infallible` (this function's native-path error type) can
            // never satisfy — so the plugin path gets its own, separate
            // `Reactor`/`ReactorAdapter` instance typed at `E = PluginError`
            // (trivially `From<PluginError>` via the reflexive identity
            // impl) instead of reusing `reactor`/`rctx` above. `Function`
            // itself doesn't depend on `E`, so the two paths' outputs still
            // unify into the same `Translated`.
            let mut plugin_reactor: Reactor<(), speet_plugin_api::error::PluginError, Function, LocalPool> =
                Reactor::default();
            let mut plugin_rctx = make_rctx(
                &mut plugin_reactor,
                n_imports,
                TextBaseSource::Constant(start_addr),
            );
            let mut plugin_ctx = ();
            let mut rc = speet_plugin_adapter::ArchPluginRecompiler::<
                (),
                speet_plugin_api::error::PluginError,
                Function,
            >::new(std::sync::Arc::from(plugin));
            rc.setup_traps(&mut plugin_rctx, &mut plugin_ctx);
            let params = collect_params(&plugin_rctx);
            rc.translate_bytes(&mut plugin_ctx, &mut plugin_rctx, text, start_addr, &mut |a| {
                Function::new(a.collect::<Vec<_>>())
            })
            .expect("ArchPlugin::step");
            return Translated { fns: plugin_rctx.drain_fns(), params, unsupported: Vec::new(), entry_func_idx: 0 };
        }
    };

    Translated { fns: rctx.drain_fns(), params, unsupported, entry_func_idx }
}

/// Assemble with runtime unreachable logging (integrated thin runtime).
/// `manifest` must be the same [`ImportManifest`] instance (or an equal one)
/// used to generate the C link shim (`speet_rt::generate_shim_integrated`)
/// and to resolve indices in [`crate::plt::PltCallPlan`] — see
/// `docs/guides/thin-runtime-genericity.md` principle 1.
pub fn assemble_module_instrumented(t: &Translated, arch: BinArch, manifest: &ImportManifest) -> Vec<u8> {
    assemble_module_instrumented_with_data(t, arch, manifest, &[], None)
}

/// Like [`assemble_module_instrumented`], but also emits `data_segments` as
/// passive WASM data segments plus a generated init function the C entry
/// shim calls once at startup — see [`DataSegment`]/[`DATA_INIT_EXPORT_NAME`].
pub fn assemble_module_instrumented_with_data(
    t: &Translated,
    arch: BinArch,
    manifest: &ImportManifest,
    data_segments: &[DataSegment],
    plt_plan: Option<&PltCallPlan>,
) -> Vec<u8> {
    use crate::instrument::{instrument_unreachable_logging, GuestPcRef};

    let mut fns = t.fns.clone();
    let log_unreachable_idx = manifest.index_of("env", "__speet_log_unreachable");
    instrument_unreachable_logging(&mut fns, GuestPcRef::for_arch(arch), log_unreachable_idx);

    let total = fns.len() as u32;
    let n_params = t.params.len() as u32;
    let (redirect_shims, _) = plt_plan
        .map(|p| build_redirect_shims_from_plan(p, arch, manifest, n_params))
        .unwrap_or_default();
    let (types, imports, space, func_slot) = build_import_section(manifest, t.params.clone(), total);
    finish_module(
        types,
        imports,
        &space,
        func_slot,
        &fns,
        &redirect_shims,
        true,
        t.entry_func_idx,
        n_params,
        data_segments,
    )
}

/// Recompile with integrated unreachable instrumentation and optional PLT hooks.
pub fn recompile_to_wasm_instrumented(
    text: &[u8],
    start_addr: u64,
    arch: BinArch,
) -> (Vec<u8>, Vec<String>) {
    recompile_to_wasm_instrumented_plt(text, start_addr, arch, None, None, &ImportManifest::integrated_native())
}

/// Like [`recompile_to_wasm_instrumented`], but the caller supplies the
/// exact [`ImportManifest`] to translate and assemble against (so a host
/// backend with a different capability set than `integrated_native()` still
/// gets consistent indices end-to-end), an optional PLT hook plan, and an
/// optional real guest entry address (see [`Translated::entry_func_idx`] —
/// `None` keeps the old "function 0 is `_start`" behavior, correct only
/// when `text` starts exactly at the entry point).
pub fn recompile_to_wasm_instrumented_plt(
    text: &[u8],
    start_addr: u64,
    arch: BinArch,
    plt_plan: Option<&PltCallPlan>,
    entry_addr: Option<u64>,
    manifest: &ImportManifest,
) -> (Vec<u8>, Vec<String>) {
    recompile_to_wasm_instrumented_plt_with_data(text, start_addr, arch, plt_plan, entry_addr, manifest, &[])
}

/// Like [`recompile_to_wasm_instrumented_plt`], but also emits
/// `data_segments` as passive WASM data segments plus a generated init
/// function the C entry shim calls once at startup — see
/// [`DataSegment`]/[`DATA_INIT_EXPORT_NAME`].
#[allow(clippy::too_many_arguments)]
pub fn recompile_to_wasm_instrumented_plt_with_data(
    text: &[u8],
    start_addr: u64,
    arch: BinArch,
    plt_plan: Option<&PltCallPlan>,
    entry_addr: Option<u64>,
    manifest: &ImportManifest,
    data_segments: &[DataSegment],
) -> (Vec<u8>, Vec<String>) {
    let layout = speet_link_core::GuestImageLayout {
        text_base: start_addr,
        text_len: text.len(),
        slot_granularity: slot_granularity(arch),
        data_sections: data_segments
            .iter()
            .map(|s| speet_link_core::image_layout::DataSectionSpec {
                name: String::new(),
                addr: s.addr,
                bytes: s.bytes.clone(),
            })
            .collect(),
        relocs: vec![],
        libraries: vec![],
        memory_model: speet_link_core::image_layout::MemoryModel::OwnedLinear,
    };
    recompile_to_wasm_instrumented_plt_with_layout(
        text, &layout, arch, plt_plan, entry_addr, manifest,
    )
}

/// Like [`recompile_to_wasm_instrumented_plt_with_data`], but accepts a full
/// [`GuestImageLayout`] (relocs, memory model, libraries) from the loaded binary.
pub fn recompile_to_wasm_instrumented_plt_with_layout(
    text: &[u8],
    layout: &speet_link_core::GuestImageLayout,
    arch: BinArch,
    plt_plan: Option<&PltCallPlan>,
    entry_addr: Option<u64>,
    manifest: &ImportManifest,
) -> (Vec<u8>, Vec<String>) {
    let start_addr = layout.text_base;
    let t = translate_with_plt(
        text,
        start_addr,
        RecompilerChoice::Native(arch),
        plt_plan,
        entry_addr,
        manifest,
        layout.memory_model,
    );
    let unsupported = t.unsupported.clone();
    let n_params = t.params.len() as u32;
    let (_, shim_by_sym) = plt_plan
        .map(|p| build_redirect_shims_from_plan(p, arch, manifest, n_params))
        .unwrap_or_default();
    let linked_data = crate::data_link::link_data_segments(layout, plt_plan, &shim_by_sym);
    (
        assemble_module_instrumented_with_data(&t, arch, manifest, &linked_data, plt_plan),
        unsupported,
    )
}

/// Assemble a single translated binary into a complete WASM module (no exception
/// handling). The entry (first function) is exported as `_start`. Mirrors the
/// speet-e2e harness `assemble_module` for one slice.
///
/// This builder's `make_rctx` always sets `escape_tag: None` — no
/// exceptions/speculative calls ever run through this path. Unlike the
/// separate speet-e2e test harness's own `assemble_module` (which still
/// deliberately keeps `Eh::None` archs at `-> ()` — see the warning
/// preserved in that file), this production path's shared register-file
/// type is unconditionally `(registers) -> (registers)`: every internal
/// exit is a `return_call`/`return_call_indirect` (trivially type-correct,
/// same type on both ends) or the halt stub `finish_module` appends
/// (which actually pushes the full register file) — never a bare,
/// zero-arity `Return`. See `docs/guides/thin-runtime-genericity.md`
/// principle 4.
pub fn assemble_module(t: &Translated, manifest: &ImportManifest) -> Vec<u8> {
    let total = t.fns.len() as u32;
    let n_params = t.params.len() as u32;
    let (types, imports, space, func_slot) = build_import_section(manifest, t.params.clone(), total);
    finish_module(types, imports, &space, func_slot, &t.fns, &[], true, t.entry_func_idx, n_params, &[])
}

/// Recompile a `.text` blob of guest machine code to a complete WASM module.
pub fn recompile_to_wasm(text: &[u8], start_addr: u64, arch: BinArch) -> (Vec<u8>, Vec<String>) {
    let manifest = ImportManifest::native_syscall();
    let t = translate(text, start_addr, RecompilerChoice::Native(arch), &manifest);
    let unsupported = t.unsupported.clone();
    (assemble_module(&t, &manifest), unsupported)
}

// ── RISC-V guest with host syscall lowering ──────────────────────────────────

/// Import layout for the native syscall shim, read back from
/// [`ImportManifest::rv64_syscall`] rather than a hand-counted constant —
/// see `docs/guides/thin-runtime-genericity.md` principle 1.
pub fn native_syscall_imports() -> speet_host_syscall::NativeSyscallImports {
    let m = ImportManifest::rv64_syscall();
    speet_host_syscall::NativeSyscallImports {
        exit: m.index_of("env", "exit").expect("rv64_syscall manifest declares env.exit"),
        write: m.index_of("env", "write").expect("rv64_syscall manifest declares env.write"),
    }
}

/// Translate an RV64 Linux `.text` blob, lowering `ecall` to the native host
/// syscall imports (`env.exit`/`env.write`) via [`speet_syscall::WasmSyscallDispatcher`].
pub fn translate_rv64(text: &[u8], start_addr: u64) -> Translated {
    use rv_asm::Xlen;
    let manifest = ImportManifest::rv64_syscall();
    let n_imports = manifest.func_imports.len() as u32;
    let imports = native_syscall_imports();
    let table = speet_host_syscall::linux_rv64_table(&imports);

    let mut recompiler =
        speet_riscv::RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
            start_addr, false, true, false,
        );
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(
        &mut reactor,
        n_imports,
        TextBaseSource::Constant(start_addr),
    );
    let mut ctx = ();
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let params = collect_params(&rctx);

    let mut dispatcher = speet_syscall::WasmSyscallDispatcher {
        table: &table,
        syscall_num_local: speet_host_syscall::A7,
        syscall_num_is_i64: true,
        num_params: params.len() as u32,
        next_pc_func: n_imports, // dummy: exit terminates, no continuation
    };
    recompiler.set_ecall_callback(&mut dispatcher);

    recompiler
        .translate_bytes(&mut ctx, &mut rctx, text, start_addr as u32, Xlen::Rv64, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        })
        .expect("translate_bytes");

    Translated { fns: rctx.drain_fns(), params, unsupported: vec![], entry_func_idx: 0 }
}

/// Assemble an RV64 translation with the native syscall imports
/// (`env.exit`/`env.write`), entry exported as `_start`. 32-bit linear memory.
pub fn assemble_syscall_module(t: &Translated, manifest: &ImportManifest) -> Vec<u8> {
    let total = t.fns.len() as u32;
    let n_params = t.params.len() as u32;
    let (types, imports, space, func_slot) = build_import_section(manifest, t.params.clone(), total);
    finish_module(types, imports, &space, func_slot, &t.fns, &[], false, t.entry_func_idx, n_params, &[])
}

/// Recompile an RV64 Linux `.text` blob (with `ecall` → host syscalls) to WASM.
pub fn recompile_rv64_to_wasm(text: &[u8], start_addr: u64) -> Vec<u8> {
    let t = translate_rv64(text, start_addr);
    assemble_syscall_module(&t, &ImportManifest::rv64_syscall())
}

/// Recompile an RV64 Linux `.text` blob with Linux→WASI preview1 lowering.
pub fn recompile_rv64_wasi_to_wasm(text: &[u8], start_addr: u64) -> Vec<u8> {
    speet_linux_wasi::recompile_rv64_wasi_to_wasm(text, start_addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_plugin_api::arch::{ArchOp, ArchPlugin};
    use speet_plugin_api::snippet::PluginValType;
    use speet_plugin_api::PResult;
    use std::sync::Mutex;

    /// One register param, one instruction (`reg0` read then discarded),
    /// sealed with `Unreachable` — exercises `RecompilerChoice::Plugin`
    /// end-to-end through the real `translate()` entry point, not just
    /// `ArchPluginRecompiler` directly (see `speet-plugin-adapter::arch`'s
    /// own unit tests for that).
    struct ToyArch {
        step: Mutex<u32>,
    }
    impl ArchPlugin for ToyArch {
        fn reset_for_next_binary(&self, _args: &[u8]) {}
        fn count_fns(&self, _bytes: &[u8]) -> u32 {
            1
        }
        fn declare_params(&self) -> Vec<PluginValType> {
            vec![PluginValType::I64]
        }
        fn step(&self, _feedback: Option<&[u8]>) -> PResult<ArchOp> {
            let mut step = self.step.lock().unwrap();
            let op = match *step {
                0 => ArchOp::OpenFn { len: 1 },
                1 => ArchOp::Feed {
                    snippet: speet_plugin_api::CodeSnippet::from_instructions(&[
                        Instruction::LocalGet(0),
                        Instruction::Drop,
                    ]),
                },
                2 => ArchOp::Seal {
                    snippet: speet_plugin_api::CodeSnippet::from_instructions(&[
                        Instruction::Unreachable,
                    ]),
                },
                _ => ArchOp::Done,
            };
            *step += 1;
            Ok(op)
        }
    }

    #[test]
    fn recompiler_choice_plugin_produces_one_function() {
        let plugin: Box<dyn ArchPlugin> = Box::new(ToyArch { step: Mutex::new(0) });
        let manifest = ImportManifest::native_syscall();
        let t = translate(&[0u8; 1], 0x1000, RecompilerChoice::Plugin(plugin), &manifest);
        assert_eq!(t.fns.len(), 1);
        // ToyArch emits one i64 param; layout params (text_base, host_mem_base) append after traps setup.
        assert!(t.params.len() >= 1);
        assert_eq!(t.params[0], ValType::I64);
        assert!(t.unsupported.is_empty());

        let mut expected = Function::new([]);
        expected.instruction(&Instruction::LocalGet(0));
        expected.instruction(&Instruction::Drop);
        expected.instruction(&Instruction::Unreachable);
        expected.instruction(&Instruction::End);
        assert_eq!(t.fns[0], expected);
    }
}
