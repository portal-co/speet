//! Link [`CANONICAL_GUEST_WASM`] as the first megabinary slot via [`WasmFrontend`].
//!
//! Megabinary imports come from [`guest_module::guest_func_imports`]. The guest
//! module is multi-memory: memory 0 is private/unmapped, memory 1 is shared with
//! the host (identity-mapped via [`DirectMemory`]).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use speet_aarch64::cfg::AArch64CfgDecoder;
use binary_io::BinArch;
use speet_host_api::{FuncImport, ImportManifest, WasmValType};
use speet_link_core::{
    unit::FuncType, BinaryUnit, ReactorContext, Recompile,
};
use speet_linker::Linker;
use speet_memory::{AddressWidth, DirectMemory, IntWidth};
use speet_module_builder::{assemble, ElementsOwned, MegabinaryBuilder};
use speet_reach::PcSlotMap;
use speet_schedule::FuncSchedule;
use speet_wasm::{GuestMemoryConfig, IndexOffsets, WasmFrontend};
use speet_plugin_api::external_target::{CallingConvention, ExternalTargetTable};
use wasm_encoder::{
    ConstExpr, Function, Instruction, MemoryType, RefType, TableType, ValType,
};
use wasmparser::{BinaryReaderError, Parser, Payload};
use yecta::SlotAssigner;

use crate::guest_module::{self, guest_defined_fn_count, handler_indices, CANONICAL_GUEST_WASM};
use crate::svc::{build_syscall_dispatch, DarwinWasiSvc, HandlerIndices, X0_LOCAL};

pub(crate) const N_SYNTHETIC_DISPATCH: u32 = 1;
use crate::{WasiImports, WasiImportsExt};

/// Host scratch memory index (guest linear memory stays at 0).
pub const HOST_MEMORY_INDEX: u32 = 1;

type LinkErr = BinaryReaderError;

/// Link the embedded guest module alone through the WASM frontend pipeline.
pub fn link_canonical_guest_wasm() -> Vec<u8> {
    let mut schedule: FuncSchedule<(), LinkErr, Function> = FuncSchedule::new();
    let wasi = WasiImports::register(schedule.entity_space_mut());
    let n_guest = guest_defined_fn_count(CANONICAL_GUEST_WASM) + N_SYNTHETIC_DISPATCH;
    let handlers = handler_indices(CANONICAL_GUEST_WASM);
    let _guest_slot = schedule.push(n_guest, move |rctx, ctx| {
        emit_guest_unit(rctx, ctx, handlers)
    });

    let mut builder = MegabinaryBuilder::<Function>::new();
    let mut ctx = ();
    wasi.declare(&mut builder, &mut ctx).expect("declare wasi");
    declare_guest_memories(&mut builder);

    let mut linker = Linker::with_plugin(builder);
    linker.execute_schedule(schedule, &mut ());

    assemble(linker.plugin.finish()).finish()
}

/// Link guest (slot 0) + translated aarch64 (slot 1) into one WASI megabinary.
pub fn link_wasi_megabinary(text: &[u8], start_addr: u64) -> Vec<u8> {
    link_wasi_megabinary_with_escape(text, start_addr, yecta::SpeculativeEscape::JUMP)
}

/// Like [`link_wasi_megabinary`], with an explicit speculative-call escape policy.
pub fn link_wasi_megabinary_with_escape(
    text: &[u8],
    start_addr: u64,
    speculative: yecta::SpeculativeEscape,
) -> Vec<u8> {
    link_wasi_megabinary_inner(text, start_addr, &[], speculative)
}

/// Link guest handlers plus aarch64 text and virtual PLT redirect shims.
///
/// `targets` must name the virtual shim PCs placed directly after the halt
/// slot. This is intentionally address-driven: an indirect `blr`, a function
/// pointer, and a GOT/lazy pointer all reach the same slot-map entry.
pub fn link_wasi_megabinary_with_targets(
    text: &[u8],
    start_addr: u64,
    targets: &ExternalTargetTable,
) -> Vec<u8> {
    let redirects = resolve_redirects(targets);
    let halt_pc = start_addr + text.len() as u64;
    for (i, redirect) in redirects.iter().enumerate() {
        let expected = halt_pc + (i as u64 + 1) * 4;
        assert_eq!(
            redirect.address, expected,
            "Darwin redirect target {} must use virtual shim PC {expected:#x}, got {:#x}",
            redirect.symbol, redirect.address,
        );
    }
    link_wasi_megabinary_inner(text, start_addr, &redirects, yecta::SpeculativeEscape::JUMP)
}

fn link_wasi_megabinary_inner(
    text: &[u8],
    start_addr: u64,
    redirects: &[DarwinRedirect],
    speculative: yecta::SpeculativeEscape,
) -> Vec<u8> {
    let text_slots = PcSlotMap::all_slots(text, start_addr, &AArch64CfgDecoder);
    let n_aarch64 = text_slots.total_slots();
    // Redirect PCs are reached by indirect-table arithmetic, so they do not
    // need decoded instruction slots. Keeping the assigner text-only avoids
    // materializing synthetic "instructions" outside the input blob.
    let slots = text_slots;
    let n_halt = 1u32;

    let mut schedule: FuncSchedule<(), LinkErr, Function> = FuncSchedule::new();
    let wasi = WasiImports::register(schedule.entity_space_mut());

    let n_handlers = guest_defined_fn_count(CANONICAL_GUEST_WASM);
    let n_guest = n_handlers + N_SYNTHETIC_DISPATCH;
    let handlers = handler_indices(CANONICAL_GUEST_WASM);
    let guest_slot = schedule.push(n_guest, move |rctx, ctx| {
        emit_guest_unit(rctx, ctx, handlers)
    });
    let guest_base = schedule.entity_space().functions.base(guest_slot);
    let syscall_dispatch_idx = guest_base + n_handlers;

    let n_shims = redirects.len() as u32;
    let redirects = redirects.to_vec();
    let aarch64_slot = schedule.push(n_aarch64 + n_shims + n_halt, move |rctx, ctx| {
        rctx.set_escape(speculative.escape);
        emit_aarch64_unit(
            rctx,
            ctx,
            text,
            start_addr,
            &slots,
            syscall_dispatch_idx,
            &redirects,
            speculative,
        )
    });
    let aarch64_base = schedule.entity_space().functions.base(aarch64_slot);
    let aarch64_total = schedule.entity_space().functions.count(aarch64_slot);

    let mut builder = MegabinaryBuilder::<Function>::new();
    // Pool `TypeIdx(0)` must be the aarch64 register-file signature before any
    // WASI import types are interned — otherwise `blr`/`return_call_indirect`
    // validates against an i32-returning import type.
    let reg_types = aarch64_register_val_types();
    let reg_results = register_file_results(&reg_types, speculative.escape);
    let reg_ty = FuncType::from_val_types(&reg_types, &reg_results);
    assert_eq!(builder.intern_type(reg_ty), 0);
    let mut declare_ctx = ();
    wasi.declare(&mut builder, &mut declare_ctx).expect("declare wasi");
    declare_guest_memories(&mut builder);

    let table_size = aarch64_base + aarch64_total;
    builder.declare_table(
        TableType {
            element_type: RefType::FUNCREF,
            minimum: table_size as u64,
            maximum: Some(table_size as u64),
            table64: true,
            shared: false,
        },
        None,
    );
    // The address-to-table formula reserves the PC at `halt_pc` for the halt
    // function. Redirect PCs start one slot after it, while their function
    // bodies stay physically before halt in the code section. The element
    // segment is therefore the indirection that keeps virtual PCs and bodies
    // in their respective contractual orders.
    let mut elem_indices: Vec<u32> = (aarch64_base..aarch64_base + n_aarch64).collect();
    elem_indices.push(aarch64_base + n_aarch64 + n_shims);
    elem_indices.extend(aarch64_base + n_aarch64..aarch64_base + n_aarch64 + n_shims);
    builder.add_element_segment(
        0,
        ConstExpr::i64_const(aarch64_base as i64),
        ElementsOwned::Functions(elem_indices),
    );

    let mut linker = Linker::with_plugin(builder);
    linker.execute_schedule(schedule, &mut ());

    assemble(linker.plugin.finish()).finish()
}

#[derive(Clone)]
struct DarwinRedirect {
    address: u64,
    symbol: String,
    handler: HandlerKind,
}

#[derive(Clone, Copy)]
enum HandlerKind {
    Read,
    Write,
    Close,
    Exit,
}

impl HandlerKind {
    fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol {
            "read" | "_read" => Some(Self::Read),
            "write" | "_write" => Some(Self::Write),
            "close" | "_close" => Some(Self::Close),
            "exit" | "_exit" => Some(Self::Exit),
            _ => None,
        }
    }

    fn arity(self) -> usize {
        match self {
            Self::Read | Self::Write => 3,
            Self::Close | Self::Exit => 1,
        }
    }

    fn syscall_number(self) -> u64 {
        use os_darwin_wasi::sysno;
        match self {
            Self::Read => sysno::READ,
            Self::Write => sysno::WRITE,
            Self::Close => sysno::CLOSE,
            Self::Exit => sysno::EXIT,
        }
    }
}

fn resolve_redirects(targets: &ExternalTargetTable) -> Vec<DarwinRedirect> {
    targets
        .iter()
        .filter_map(|entry| {
            HandlerKind::from_symbol(&entry.label).map(|handler| DarwinRedirect {
                address: entry.address,
                symbol: entry.label,
                handler,
            })
        })
        .collect()
}

/// Same as [`link_wasi_megabinary`] but also returns layout metadata.
pub fn link_wasi_megabinary_with_plan(text: &[u8], start_addr: u64) -> (Vec<u8>, WasiLinkPlan) {
    let n_imports = guest_module::guest_import_count(CANONICAL_GUEST_WASM);
    let slots = PcSlotMap::all_slots(text, start_addr, &AArch64CfgDecoder);
    let n_aarch64 = slots.total_slots();
    let n_handlers = guest_defined_fn_count(CANONICAL_GUEST_WASM);
    let n_guest = n_handlers + N_SYNTHETIC_DISPATCH;
    let handlers = handler_indices(CANONICAL_GUEST_WASM);
    let guest_base = n_imports;
    let aarch64_base = guest_base + n_guest;
    let wasm = link_wasi_megabinary(text, start_addr);
    let wasi = {
        let mut schedule = FuncSchedule::<(), LinkErr, Function>::new();
        WasiImports::register(schedule.entity_space_mut())
    };
    let plan = WasiLinkPlan {
        wasi,
        n_imports,
        guest_base,
        guest_count: n_guest,
        handlers,
        syscall_dispatch_idx: guest_base + n_handlers,
        aarch64_base,
        aarch64_count: n_aarch64 + 1,
    };
    (wasm, plan)
}

/// Resolved indices after [`link_wasi_megabinary`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasiLinkPlan {
    pub wasi: WasiImports,
    pub n_imports: u32,
    pub guest_base: u32,
    pub guest_count: u32,
    pub handlers: HandlerIndices,
    pub syscall_dispatch_idx: u32,
    pub aarch64_base: u32,
    pub aarch64_count: u32,
}

fn declare_guest_memories(builder: &mut MegabinaryBuilder<Function>) {
    builder.declare_memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    builder.declare_memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    // WASI hosts look up export `"memory"`; guest scratch is mem0, host buffer is mem1.
    builder.export_memory("memory", HOST_MEMORY_INDEX);
}

fn guest_memory_configs() -> Vec<GuestMemoryConfig<(), LinkErr>> {
    vec![
        GuestMemoryConfig {
            addr_width: AddressWidth::W32,
            memory_access: None,
        },
        GuestMemoryConfig {
            addr_width: AddressWidth::W32,
            memory_access: Some(Box::new(DirectMemory::new(
                (),
                HOST_MEMORY_INDEX,
                AddressWidth::W32,
                IntWidth::I32,
            ))),
        },
    ]
}

fn guest_export_entry_points() -> Vec<(String, u32)> {
    let mut points = Vec::new();
    for payload in Parser::new(0).parse_all(CANONICAL_GUEST_WASM) {
        if let Payload::ExportSection(reader) = payload.expect("guest parse") {
            for export in reader {
                let export = export.expect("guest export");
                if export.kind == wasmparser::ExternalKind::Func {
                    points.push((export.name.to_string(), export.index));
                }
            }
        }
    }
    points
}

/// Absolute WASM function index of an exported guest function.
pub fn func_export_index(wasm: &[u8], name: &str) -> Option<u32> {
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::ExportSection(reader) = payload.ok()? {
            for export in reader {
                let export = export.ok()?;
                if export.kind == wasmparser::ExternalKind::Func && export.name == name {
                    return Some(export.index);
                }
            }
        }
    }
    None
}

fn emit_guest_unit(
    rctx: &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
    ctx: &mut (),
    handlers: HandlerIndices,
) -> BinaryUnit<Function> {
    let mut frontend = WasmFrontend::with_wasm_encoder_fn(
        guest_memory_configs(),
        HOST_MEMORY_INDEX,
        IndexOffsets::default(),
    );
    frontend
        .translate_module(ctx, rctx, CANONICAL_GUEST_WASM)
        .expect("guest translate_module");
    let mut unit = frontend.drain_unit(rctx, guest_export_entry_points());
    let dispatch_ty = FuncType::from_val_types(
        &[ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        &[ValType::I64],
    );
    unit.fns.push(build_syscall_dispatch(handlers));
    unit.func_types.push(dispatch_ty);
    unit
}

fn emit_aarch64_unit(
    rctx: &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
    ctx: &mut (),
    text: &[u8],
    start_addr: u64,
    slots: &PcSlotMap,
    syscall_dispatch_idx: u32,
    redirects: &[DarwinRedirect],
    speculative: yecta::SpeculativeEscape,
) -> BinaryUnit<Function> {
    let mut recompiler =
        speet_aarch64::AArch64Recompiler::<(), LinkErr>::new_with_base_pc(start_addr);
    recompiler.set_slot_assigner(slots.clone());
    recompiler.set_speculative_calls(speculative.enable);
    recompiler.setup_traps(rctx, ctx);
    let params = collect_params(rctx);

    let mut svc_cb = DarwinWasiSvc {
        syscall_dispatch_idx,
        slots: slots.clone(),
        base_func_offset: rctx.base_func_offset(),
        halt_stub_idx: rctx.base_func_offset() + slots.total_slots(),
        num_params: params.len() as u32,
    };
    recompiler.set_svc_callback(&mut svc_cb);

    recompiler
        .translate_bytes(
            ctx,
            rctx,
            text,
            start_addr,
            &mut |a| Function::new(a.collect::<Vec<_>>()),
        )
        .expect("translate_bytes");

    let mut fns = rctx.drain_fns();
    let results = register_file_results(&params, speculative.escape);
    let register_type = FuncType::from_val_types(&params, &results);
    let mut func_types = vec![register_type.clone(); fns.len()];
    // Redirect PCs are table-only synthetic addresses, not decoded slots;
    // append their bodies after the translated text functions.
    for redirect in redirects {
        fns.push(build_redirect_shim(
            redirect,
            syscall_dispatch_idx,
            params.len() as u32,
            start_addr,
            rctx.base_func_offset(),
            speculative.escape,
        ));
        func_types.push(register_type.clone());
    }
    fns.push(build_halt_stub(&params, speculative.escape));
    func_types.push(register_type);

    BinaryUnit {
        fns,
        base_func_offset: rctx.base_func_offset(),
        entry_points: vec![(String::from("_start"), rctx.base_func_offset())],
        func_types,
        data_segments: vec![],
        data_init_fn: None,
    }
}

fn register_file_results(params: &[ValType], escape: yecta::CallEscape) -> Vec<ValType> {
    match escape {
        yecta::CallEscape::Flag => {
            let mut r = params.to_vec();
            r.push(ValType::I32);
            r
        }
        yecta::CallEscape::Jump | yecta::CallEscape::Exception(_) => params.to_vec(),
    }
}

fn collect_params(rctx: &dyn ReactorContext<(), LinkErr, FnType = Function>) -> Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout()
        .iter_before(&mark)
        .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
        .collect()
}

/// Fixed aarch64 register-file param types matching
/// [`speet_aarch64::AArch64Recompiler::setup_traps`] plus the two
/// [`RuntimeLayoutParams`](speet_link_core::RuntimeLayoutParams) i64s the
/// linker injects (`text_base`, `host_mem_base`).
fn aarch64_register_val_types() -> Vec<ValType> {
    let mut v = Vec::with_capacity(
        speet_aarch64::AArch64Recompiler::<(), LinkErr>::BASE_PARAMS as usize + 2,
    );
    v.extend(core::iter::repeat(ValType::I64).take(31)); // x0–x30
    v.push(ValType::I64); // PC
    v.extend(core::iter::repeat(ValType::I32).take(4)); // NZCV
    v.extend(core::iter::repeat(ValType::I64).take(3)); // scratch
    v.extend(core::iter::repeat(ValType::F64).take(32)); // V0–V31
    v.push(ValType::I64); // expected_ra
    v.push(ValType::I64); // SP
    v.push(ValType::I64); // layout: text_base
    v.push(ValType::I64); // layout: host_mem_base
    v
}

fn build_halt_stub(params: &[ValType], escape: yecta::CallEscape) -> Function {
    let mut f = Function::new([]);
    for p in 0..params.len() as u32 {
        f.instruction(&wasm_encoder::Instruction::LocalGet(p));
    }
    if matches!(escape, yecta::CallEscape::Flag) {
        f.instruction(&wasm_encoder::Instruction::I32Const(0));
    }
    f.instruction(&wasm_encoder::Instruction::Return);
    f.instruction(&wasm_encoder::Instruction::End);
    f
}

/// Build a redirect slot that preserves the aarch64 register-file ABI while
/// calling one of the shared Unix/WASI handlers.
///
/// Guest `blr` is lowered as `return_call_indirect`, so the shim must
/// `return_call_indirect` through LR (`x30`) to resume after the call site —
/// a bare `return` would unwind past `_start` and skip the rest of the guest.
fn build_redirect_shim(
    redirect: &DarwinRedirect,
    syscall_dispatch_idx: u32,
    n_register_params: u32,
    text_base: u64,
    base_func_offset: u32,
    _escape: yecta::CallEscape,
) -> Function {
    let convention = handler_calling_convention(&redirect.symbol, redirect.handler.arity());
    let mut f = Function::new([]);
    f.instruction(&Instruction::I64Const(redirect.handler.syscall_number() as i64));
    for &local in convention.arg_locals.iter().take(redirect.handler.arity()) {
        f.instruction(&Instruction::LocalGet(local));
    }
    for _ in redirect.handler.arity()..3 {
        f.instruction(&Instruction::I64Const(0));
    }
    f.instruction(&Instruction::Call(syscall_dispatch_idx));
    if matches!(redirect.handler, HandlerKind::Exit) {
        // `proc_exit` may return under wasmi stubs; never resume the guest.
        f.instruction(&Instruction::Unreachable);
        f.instruction(&Instruction::End);
        return f;
    }
    f.instruction(&Instruction::LocalSet(X0_LOCAL));
    for p in 0..n_register_params {
        f.instruction(&Instruction::LocalGet(p));
    }
    // table_idx = (x30 - text_base) / 4 + base_func_offset
    f.instruction(&Instruction::LocalGet(30)); // LR
    f.instruction(&Instruction::I64Const(text_base as i64));
    f.instruction(&Instruction::I64Sub);
    f.instruction(&Instruction::I64Const(2));
    f.instruction(&Instruction::I64ShrU);
    f.instruction(&Instruction::I64Const(base_func_offset as i64));
    f.instruction(&Instruction::I64Add);
    f.instruction(&Instruction::ReturnCallIndirect {
        type_index: 0,
        table_index: 0,
    });
    f.instruction(&Instruction::End);
    f
}

fn handler_calling_convention(symbol: &str, arity: usize) -> CallingConvention {
    let manifest = ImportManifest {
        func_imports: vec![FuncImport {
            module: String::from("speet_darwin_wasi"),
            name: String::from("handler"),
            params: core::iter::repeat_n(WasmValType::I64, arity).collect(),
            results: vec![WasmValType::I64],
            intercepts: vec![String::from(symbol)],
        }],
    };
    let mut convention =
        speet_abi_stubs::plt_calling_convention(&manifest, BinArch::AArch64, symbol);
    // The shared handler functions accept/return i64 even where the source
    // libSystem declaration is i32; preserve ABI-selected registers but not
    // the source-import conversion operations.
    convention.arg_wrap_i32 = vec![false; convention.arg_locals.len()];
    convention.result_local = Some(X0_LOCAL);
    convention.result_extend_i32 = false;
    convention
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EXPORT_HANDLER_EXIT, EXPORT_HANDLER_WRITE};

    /// Hand-written aarch64: write(1, 520, 6) then exit(0).
    const WRITE_EXIT: &[u8] = &[
        0x20, 0x00, 0x80, 0xD2, // mov x0, #1
        0x01, 0x41, 0x80, 0xD2, // mov x1, #520
        0xC2, 0x00, 0x80, 0xD2, // mov x2, #6
        0x90, 0x00, 0x80, 0xD2, // mov x16, #4
        0x01, 0x10, 0x00, 0xD4, // svc #0x80
        0x00, 0x00, 0x80, 0xD2, // mov x0, #0
        0x30, 0x00, 0x80, 0xD2, // mov x16, #1
        0x01, 0x10, 0x00, 0xD4, // svc #0x80
    ];

    #[test]
    fn link_guest_via_wasm_frontend_validates() {
        let wasm = link_canonical_guest_wasm();
        wasmparser::validate(&wasm).expect("linked guest module");
        assert!(func_export_index(&wasm, EXPORT_HANDLER_WRITE).is_some());
        assert!(func_export_index(&wasm, EXPORT_HANDLER_EXIT).is_some());
    }

    #[test]
    fn guest_first_slot_preserves_handler_indices() {
        let (_wasm, plan) = link_wasi_megabinary_with_plan(WRITE_EXIT, 0x1000);
        assert_eq!(plan.guest_base, plan.n_imports);
        assert_eq!(plan.guest_count, 5);
        assert_eq!(plan.handlers, handler_indices(CANONICAL_GUEST_WASM));
        assert_eq!(
            plan.syscall_dispatch_idx,
            plan.guest_base + guest_defined_fn_count(CANONICAL_GUEST_WASM)
        );
        assert_eq!(plan.aarch64_base, plan.guest_base + plan.guest_count);
    }

    #[test]
    fn linked_wasi_megabinary_validates() {
        let wasm = link_wasi_megabinary(WRITE_EXIT, 0x1000);
        wasmparser::validate(&wasm).expect("linked megabinary");
    }
}
