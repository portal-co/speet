//! Link [`CANONICAL_GUEST_WASM`] as the first megabinary slot via [`WasmFrontend`].
//!
//! Megabinary imports come from [`guest_module::guest_func_imports`]. The guest
//! module is multi-memory: memory 0 is private/unmapped, memory 1 is shared with
//! the host (identity-mapped via [`DirectMemory`]).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use rv_asm::Xlen;
use speet_link_core::{
    unit::FuncType, BinaryUnit, ReactorContext, Recompile,
};
use speet_linker::Linker;
use speet_memory::{AddressWidth, DirectMemory, IntWidth};
use speet_module_builder::{assemble, ElementsOwned, MegabinaryBuilder};
use speet_reach::PcSlotMap;
use speet_riscv::cfg::RiscVCfgDecoder;
use speet_aarch64::cfg::AArch64CfgDecoder;
use speet_mips::cfg::MipsCfgDecoder;
use speet_x86_64::cfg::X86CfgDecoder;
use speet_schedule::FuncSchedule;
use speet_wasm::{GuestMemoryConfig, IndexOffsets, WasmFrontend};
use wasm_encoder::{
    ConstExpr, Function, MemoryType, RefType, TableType, ValType,
};
use wasmparser::{BinaryReaderError, Parser, Payload};
use yecta::SlotAssigner;

use crate::ecall::{
    build_syscall_dispatch, HandlerIndices, LinuxSyscallNums, LinuxWasiEcall,
    LinuxWasiMipsSyscall, LinuxWasiSvc, LinuxWasiSyscall,
};
use crate::guest_module::{self, guest_defined_fn_count, handler_indices, CANONICAL_GUEST_WASM};
use crate::{WasiImports, WasiImportsExt};

/// Handlers from the unix guest plus one synthetic Linux number dispatch.
pub(crate) const N_SYNTHETIC_DISPATCH: u32 = 1;

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
        emit_guest_unit(rctx, ctx, handlers, LinuxSyscallNums::default())
    });

    let mut builder = MegabinaryBuilder::<Function>::new();
    let mut ctx = ();
    wasi.declare(&mut builder, &mut ctx).expect("declare wasi");
    declare_guest_memories(&mut builder);

    let mut linker = Linker::with_plugin(builder);
    linker.execute_schedule(schedule, &mut ());

    assemble(linker.plugin.finish()).finish()
}

/// Link guest (slot 0) + translated RV64 (slot 1) into one WASI megabinary.
pub fn link_wasi_megabinary(text: &[u8], start_addr: u64) -> Vec<u8> {
    link_wasi_megabinary_with_escape(text, start_addr, yecta::SpeculativeEscape::JUMP)
}

/// Like [`link_wasi_megabinary`], with an explicit speculative-call escape policy.
pub fn link_wasi_megabinary_with_escape(
    text: &[u8],
    start_addr: u64,
    speculative: yecta::SpeculativeEscape,
) -> Vec<u8> {
    let slots = PcSlotMap::all_slots(
        text,
        start_addr,
        &RiscVCfgDecoder {
            xlen: Xlen::Rv64,
        },
    );
    let n_rv64 = slots.total_slots();
    let n_halt = 1u32;

    let mut schedule: FuncSchedule<(), LinkErr, Function> = FuncSchedule::new();
    let wasi = WasiImports::register(schedule.entity_space_mut());

    let n_handlers = guest_defined_fn_count(CANONICAL_GUEST_WASM);
    let n_guest = n_handlers + N_SYNTHETIC_DISPATCH;
    let handlers = handler_indices(CANONICAL_GUEST_WASM);
    let guest_slot = schedule.push(n_guest, move |rctx, ctx| {
        emit_guest_unit(rctx, ctx, handlers, LinuxSyscallNums::default())
    });
    let guest_base = schedule.entity_space().functions.base(guest_slot);
    let syscall_dispatch_idx = guest_base + n_handlers;

    let rv64_slot = schedule.push(n_rv64 + n_halt, move |rctx, ctx| {
        rctx.set_escape(speculative.escape);
        emit_rv64_unit(
            rctx,
            ctx,
            text,
            start_addr,
            &slots,
            syscall_dispatch_idx,
            guest_base + n_guest,
            speculative,
        )
    });
    let rv64_base = schedule.entity_space().functions.base(rv64_slot);
    let rv64_total = schedule.entity_space().functions.count(rv64_slot);

    let mut builder = MegabinaryBuilder::<Function>::new();
    let mut declare_ctx = ();
    wasi.declare(&mut builder, &mut declare_ctx).expect("declare wasi");
    declare_guest_memories(&mut builder);

    let table_size = rv64_base + rv64_total;
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
    let elem_indices: Vec<u32> = (rv64_base..table_size).collect();
    builder.add_element_segment(
        0,
        ConstExpr::i64_const(rv64_base as i64),
        ElementsOwned::Functions(elem_indices),
    );

    let mut linker = Linker::with_plugin(builder);
    linker.execute_schedule(schedule, &mut ());

    assemble(linker.plugin.finish()).finish()
}

/// Link Linux aarch64 text (`svc #0`, x8/x0–x2) into a WASI megabinary.
pub fn link_aarch64_linux_wasi_megabinary(text: &[u8], start_addr: u64) -> Vec<u8> {
    let slots = PcSlotMap::all_slots(text, start_addr, &AArch64CfgDecoder);
    link_wasi_megabinary_arch(text, start_addr, slots.clone(), move |rctx, ctx, dispatch, halt| {
        emit_aarch64_unit(rctx, ctx, text, start_addr, &slots, dispatch, halt)
    })
}

/// Link Linux x86-64 text (`syscall`, rax/rdi/rsi/rdx) into a WASI megabinary.
pub fn link_x86_64_linux_wasi_megabinary(text: &[u8], start_addr: u64) -> Vec<u8> {
    let slots = PcSlotMap::all_slots(text, start_addr, &X86CfgDecoder);
    link_wasi_megabinary_arch(text, start_addr, slots.clone(), move |rctx, ctx, dispatch, halt| {
        emit_x86_64_unit(rctx, ctx, text, start_addr, &slots, dispatch, halt)
    })
}

/// Link big-endian MIPS N64 text (`syscall`, v0/a0–a2) into a WASI megabinary.
pub fn link_mips_linux_wasi_megabinary(text: &[u8], start_addr: u64) -> Vec<u8> {
    let slots = PcSlotMap::all_slots(text, start_addr, &MipsCfgDecoder);
    link_wasi_megabinary_arch(text, start_addr, slots.clone(), move |rctx, ctx, dispatch, halt| {
        emit_mips_unit(rctx, ctx, text, start_addr as u32, &slots, dispatch, halt)
    })
}

fn link_wasi_megabinary_arch(
    _text: &[u8],
    _start_addr: u64,
    slots: PcSlotMap,
    emit_arch: impl FnOnce(
        &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
        &mut (),
        u32,
        u32,
    ) -> BinaryUnit<Function>,
) -> Vec<u8> {
    let n_arch = slots.total_slots();
    let mut schedule: FuncSchedule<(), LinkErr, Function> = FuncSchedule::new();
    let wasi = WasiImports::register(schedule.entity_space_mut());
    let n_handlers = guest_defined_fn_count(CANONICAL_GUEST_WASM);
    let n_guest = n_handlers + N_SYNTHETIC_DISPATCH;
    let handlers = handler_indices(CANONICAL_GUEST_WASM);
    let guest_slot = schedule.push(n_guest, move |rctx, ctx| {
        emit_guest_unit(rctx, ctx, handlers, LinuxSyscallNums::default())
    });
    let guest_base = schedule.entity_space().functions.base(guest_slot);
    let dispatch = guest_base + n_handlers;
    let arch_slot = schedule.push(n_arch + 1, move |rctx, ctx| {
        emit_arch(rctx, ctx, dispatch, guest_base + n_guest)
    });
    let arch_base = schedule.entity_space().functions.base(arch_slot);
    let arch_total = schedule.entity_space().functions.count(arch_slot);

    let mut builder = MegabinaryBuilder::<Function>::new();
    let mut ctx = ();
    wasi.declare(&mut builder, &mut ctx).expect("declare wasi");
    declare_guest_memories(&mut builder);
    let table_size = arch_base + arch_total;
    builder.declare_table(TableType {
        element_type: RefType::FUNCREF,
        minimum: table_size as u64,
        maximum: Some(table_size as u64),
        table64: true,
        shared: false,
    }, None);
    builder.add_element_segment(0, ConstExpr::i64_const(arch_base as i64),
        ElementsOwned::Functions((arch_base..table_size).collect()));
    let mut linker = Linker::with_plugin(builder);
    linker.execute_schedule(schedule, &mut ());
    assemble(linker.plugin.finish()).finish()
}

/// Same as [`link_wasi_megabinary`] but also returns layout metadata.
pub fn link_wasi_megabinary_with_plan(text: &[u8], start_addr: u64) -> (Vec<u8>, WasiLinkPlan) {
    let n_imports = guest_module::guest_import_count(CANONICAL_GUEST_WASM);
    let slots = PcSlotMap::all_slots(
        text,
        start_addr,
        &RiscVCfgDecoder {
            xlen: Xlen::Rv64,
        },
    );
    let n_rv64 = slots.total_slots();
    let n_handlers = guest_defined_fn_count(CANONICAL_GUEST_WASM);
    let n_guest = n_handlers + N_SYNTHETIC_DISPATCH;
    let handlers = handler_indices(CANONICAL_GUEST_WASM);
    let guest_base = n_imports;
    let rv64_base = guest_base + n_guest;
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
        rv64_base,
        rv64_count: n_rv64 + 1,
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
    pub rv64_base: u32,
    pub rv64_count: u32,
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
        // Private guest linear memory — keep index 0, no address rewrite.
        GuestMemoryConfig {
            addr_width: AddressWidth::W32,
            memory_access: None,
        },
        // Host-shared memory — already lowered to memory 1; identity map.
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
    nums: LinuxSyscallNums,
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
    unit.fns.push(build_syscall_dispatch(handlers, nums));
    unit.func_types.push(dispatch_ty);
    unit
}

fn emit_rv64_unit(
    rctx: &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
    ctx: &mut (),
    text: &[u8],
    start_addr: u64,
    slots: &PcSlotMap,
    syscall_dispatch_idx: u32,
    halt_stub_idx: u32,
    speculative: yecta::SpeculativeEscape,
) -> BinaryUnit<Function> {
    let mut recompiler =
        speet_riscv::RiscVRecompiler::<(), LinkErr, Function>::new_with_full_config(
            start_addr, false, true, false,
        );
    recompiler.set_slot_assigner(slots.clone());
    recompiler.set_speculative_calls(speculative.enable);
    recompiler.setup_traps(rctx, ctx);
    let params = collect_params(rctx);

    let mut ecall_cb = LinuxWasiEcall::rv64(
        syscall_dispatch_idx,
        slots.clone(),
        rctx.base_func_offset(),
        halt_stub_idx,
        params.len() as u32,
    );
    recompiler.set_ecall_callback(&mut ecall_cb);

    recompiler
        .translate_bytes(
            ctx,
            rctx,
            text,
            start_addr as u32,
            Xlen::Rv64,
            &mut |a| Function::new(a.collect::<Vec<_>>()),
        )
        .expect("translate_bytes");

    let mut fns = rctx.drain_fns();
    let results = register_file_results(&params, speculative.escape);
    let register_type = FuncType::from_val_types(&params, &results);
    let mut func_types = vec![register_type.clone(); fns.len()];
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

fn emit_aarch64_unit(
    rctx: &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
    ctx: &mut (),
    text: &[u8],
    start_addr: u64,
    slots: &PcSlotMap,
    syscall_dispatch_idx: u32,
    halt_stub_idx: u32,
) -> BinaryUnit<Function> {
    let mut recompiler =
        speet_aarch64::AArch64Recompiler::<(), LinkErr>::new_with_base_pc(start_addr);
    recompiler.set_slot_assigner(slots.clone());
    recompiler.setup_traps(rctx, ctx);
    let params = collect_params(rctx);
    let mut svc = LinuxWasiSvc {
        syscall_dispatch_idx, slots: slots.clone(), base_func_offset: rctx.base_func_offset(),
        halt_stub_idx, num_params: params.len() as u32,
    };
    recompiler.set_svc_callback(&mut svc);
    recompiler.translate_bytes(ctx, rctx, text, start_addr,
        &mut |a| Function::new(a.collect::<Vec<_>>())).expect("translate aarch64");
    finish_arch_unit(rctx, params)
}

fn emit_x86_64_unit(
    rctx: &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
    ctx: &mut (),
    text: &[u8],
    start_addr: u64,
    slots: &PcSlotMap,
    syscall_dispatch_idx: u32,
    halt_stub_idx: u32,
) -> BinaryUnit<Function> {
    let mut recompiler = speet_x86_64::X86Recompiler::<(), LinkErr>::new_with_base_rip(start_addr);
    recompiler.set_slot_assigner(slots.clone());
    recompiler.setup_traps(rctx, ctx);
    let params = collect_params(rctx);
    recompiler.set_syscall_callback(LinuxWasiSyscall {
        syscall_dispatch_idx, slots: slots.clone(), base_func_offset: rctx.base_func_offset(),
        halt_stub_idx, num_params: params.len() as u32,
    });
    recompiler.translate_bytes(ctx, rctx, text, start_addr,
        &mut |a| Function::new(a.collect::<Vec<_>>())).expect("translate x86-64");
    finish_arch_unit(rctx, params)
}

fn emit_mips_unit(
    rctx: &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
    ctx: &mut (),
    text: &[u8],
    start_addr: u32,
    slots: &PcSlotMap,
    syscall_dispatch_idx: u32,
    halt_stub_idx: u32,
) -> BinaryUnit<Function> {
    use rabbitizer::{InstrCategory, Instruction as MipsInstruction};
    // `syscall` must be declared before `recompiler`: the callback borrow is
    // held in the recompiler's 'cb slot, and values drop in reverse
    // declaration order.
    let mut syscall = LinuxWasiMipsSyscall {
        syscall_dispatch_idx,
        slots: slots.clone(),
        base_func_offset: 0,
        halt_stub_idx,
        num_params: 0,
    };
    let mut recompiler =
        speet_mips::MipsRecompiler::<'_, '_, (), LinkErr, Function>::new_with_full_config(start_addr, true);
    recompiler.set_slot_assigner(slots.clone());
    // `MipsRecompiler::setup_traps` retains the caller layout; the preceding
    // WASI guest slot may contain unrelated locals.
    *rctx.layout_mut() = yecta::LocalLayout::empty();
    recompiler.setup_traps(rctx, ctx);
    let params = collect_params(rctx);
    syscall.base_func_offset = rctx.base_func_offset();
    syscall.num_params = params.len() as u32;
    recompiler.set_syscall_callback(&mut syscall);
    // Branch/jump delay slots: fetching is available via
    // set_delay_slot_fetcher, but inline delay execution is disabled until
    // the yecta Else-arm retarget lands (see comparison-fuzzing-plan.md
    // finding #6) — enabling it here double-executes the delay word on the
    // not-taken path.
    for (offset, bytes) in text.chunks_exact(4).enumerate() {
        let pc = start_addr + (offset * 4) as u32;
        // Delay-slot words are executed inline by their branch/jump.
        if recompiler.is_absorbed_delay_pc(pc) {
            continue;
        }
        let insn = MipsInstruction::new(
            u32::from_be_bytes(bytes.try_into().expect("exact chunk")),
            pc,
            InstrCategory::CPU,
        );
        recompiler.translate_instruction(ctx, rctx, &insn,
            &mut |a| Function::new(a.collect::<Vec<_>>())).expect("translate mips");
    }
    let mut unit = finish_arch_unit(rctx, params);
    // MIPS' single-instruction translation path leaves the wasm function body
    // open for its caller to finish (unlike the byte translators above).
    let translated_count = unit.fns.len() - 1;
    for f in &mut unit.fns[..translated_count] {
        f.instruction(&wasm_encoder::Instruction::End);
    }
    unit
}

fn finish_arch_unit(
    rctx: &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
    params: Vec<ValType>,
) -> BinaryUnit<Function> {
    let mut fns = rctx.drain_fns();
    let escape = rctx.escape();
    let results = register_file_results(&params, escape);
    let register_type = FuncType::from_val_types(&params, &results);
    let mut func_types = vec![register_type.clone(); fns.len()];
    fns.push(build_halt_stub(&params, escape));
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

fn collect_params(rctx: &dyn ReactorContext<(), LinkErr, FnType = Function>) -> Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout()
        .iter_before(&mark)
        .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EXPORT_HANDLER_EXIT, EXPORT_HANDLER_WRITE};

    #[test]
    fn link_guest_via_wasm_frontend_validates() {
        let wasm = link_canonical_guest_wasm();
        wasmparser::validate(&wasm).expect("linked guest module");
        assert!(func_export_index(&wasm, EXPORT_HANDLER_WRITE).is_some());
        assert!(func_export_index(&wasm, EXPORT_HANDLER_EXIT).is_some());
    }

    #[test]
    fn guest_first_slot_preserves_handler_indices() {
        const WRITE_EXIT: &[u8] = &[
            0x13, 0x05, 0x10, 0x00,
            0x93, 0x05, 0x80, 0x20,
            0x13, 0x06, 0x60, 0x00,
            0x93, 0x08, 0x00, 0x04,
            0x73, 0x00, 0x00, 0x00,
            0x13, 0x05, 0x00, 0x00,
            0x93, 0x08, 0xd0, 0x05,
            0x73, 0x00, 0x00, 0x00,
        ];
        let (_wasm, plan) = link_wasi_megabinary_with_plan(WRITE_EXIT, 0x1000);
        assert_eq!(plan.guest_base, plan.n_imports);
        assert_eq!(plan.guest_count, 5); // 4 handlers + synthetic dispatch
        assert_eq!(plan.handlers, handler_indices(CANONICAL_GUEST_WASM));
        assert_eq!(
            plan.syscall_dispatch_idx,
            plan.guest_base + guest_defined_fn_count(CANONICAL_GUEST_WASM)
        );
        assert_eq!(plan.rv64_base, plan.guest_base + plan.guest_count);
    }

    #[test]
    fn linked_wasi_megabinary_validates() {
        const WRITE_EXIT: &[u8] = &[
            0x13, 0x05, 0x10, 0x00,
            0x93, 0x05, 0x80, 0x20,
            0x13, 0x06, 0x60, 0x00,
            0x93, 0x08, 0x00, 0x04,
            0x73, 0x00, 0x00, 0x00,
            0x13, 0x05, 0x00, 0x00,
            0x93, 0x08, 0xd0, 0x05,
            0x73, 0x00, 0x00, 0x00,
        ];
        let wasm = link_wasi_megabinary(WRITE_EXIT, 0x1000);
        wasmparser::validate(&wasm).expect("linked megabinary");
    }

    #[test]
    fn aarch64_linux_exit_megabinary_validates() {
        // mov x0,#42; mov x8,#93; svc #0
        let text = [0x40, 0x05, 0x80, 0xd2, 0xa8, 0x0b, 0x80, 0xd2, 0x01, 0x00, 0x00, 0xd4];
        wasmparser::validate(&link_aarch64_linux_wasi_megabinary(&text, 0x1000))
            .expect("aarch64 Linux megabinary");
    }

    #[test]
    fn x86_64_linux_exit_megabinary_validates() {
        // mov rax,60; mov rdi,42; syscall
        let text = [
            0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00,
            0x48, 0xc7, 0xc7, 0x2a, 0x00, 0x00, 0x00, 0x0f, 0x05,
        ];
        wasmparser::validate(&link_x86_64_linux_wasi_megabinary(&text, 0x1000))
            .expect("x86-64 Linux megabinary");
    }

    #[test]
    fn mips_n64_linux_exit_megabinary_validates() {
        // The synthetic dispatcher receives N64 v0/a0–a2 from this syscall.
        // Register setup is intentionally left to the embedder's entry state:
        // MIPS64 immediate materialization is not required to validate routing.
        let text = [0x00, 0x00, 0x00, 0x0c];
        wasmparser::validate(&link_mips_linux_wasi_megabinary(&text, 0x1000))
            .expect("MIPS N64 Linux megabinary");
    }
}
