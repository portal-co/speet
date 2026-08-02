//! Link [`CANONICAL_GUEST_WASM`] as the first megabinary slot via [`WasmFrontend`].
//!
//! Megabinary imports come from [`guest_module::guest_func_imports`]. The guest slot uses
//! [`WasmFrontend::set_preserve_guest_module`] for the pre-lowered multi-memory module.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use rv_asm::Xlen;
use speet_link_core::{
    unit::FuncType, BinaryUnit, ReactorContext, Recompile,
};
use speet_linker::Linker;
use speet_memory::AddressWidth;
use speet_module_builder::{assemble, ElementsOwned, MegabinaryBuilder};
use speet_reach::PcSlotMap;
use speet_riscv::cfg::RiscVCfgDecoder;
use speet_schedule::FuncSchedule;
use speet_wasm::{GuestMemoryConfig, IndexOffsets, WasmFrontend};
use wasm_encoder::{
    ConstExpr, Function, MemoryType, RefType, TableType, ValType,
};
use wasmparser::{BinaryReaderError, Parser, Payload};
use yecta::SlotAssigner;

use crate::ecall::LinuxWasiEcall;
use crate::guest_module::{self, guest_defined_fn_count, CANONICAL_GUEST_WASM, EXPORT_SYSCALL_DISPATCH};
use crate::{WasiImports, WasiImportsExt};

/// Host scratch memory index (guest linear memory stays at 0).
pub const HOST_MEMORY_INDEX: u32 = 1;

type LinkErr = BinaryReaderError;

/// Link the embedded guest module alone through the WASM frontend pipeline.
pub fn link_canonical_guest_wasm() -> Vec<u8> {
    let mut schedule: FuncSchedule<(), LinkErr, Function> = FuncSchedule::new();
    let wasi = WasiImports::register(schedule.entity_space_mut());
    let n_guest = guest_defined_fn_count(CANONICAL_GUEST_WASM);
    let wasi_for_guest = wasi.clone();
    let _guest_slot = schedule.push(n_guest, move |rctx, ctx| {
        emit_guest_unit(rctx, ctx)
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

    let n_guest = guest_defined_fn_count(CANONICAL_GUEST_WASM);
    let guest_slot = schedule.push(n_guest, move |rctx, ctx| emit_guest_unit(rctx, ctx));
    let guest_base = schedule.entity_space().functions.base(guest_slot);
    let syscall_dispatch_idx =
        func_export_index(CANONICAL_GUEST_WASM, EXPORT_SYSCALL_DISPATCH)
            .expect("syscall_dispatch export");

    let rv64_slot = schedule.push(n_rv64 + n_halt, move |rctx, ctx| {
        emit_rv64_unit(
            rctx,
            ctx,
            text,
            start_addr,
            &slots,
            syscall_dispatch_idx,
            guest_base + n_guest,
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

/// Same as [`link_wasi_megabinary`] but also returns layout metadata.
pub fn link_wasi_megabinary_with_plan(text: &[u8], start_addr: u64) -> (Vec<u8>, WasiLinkPlan) {
    let n_imports = guest_module::guest_import_count(CANONICAL_GUEST_WASM);
    let n_guest = guest_defined_fn_count(CANONICAL_GUEST_WASM);
    let slots = PcSlotMap::all_slots(
        text,
        start_addr,
        &RiscVCfgDecoder {
            xlen: Xlen::Rv64,
        },
    );
    let n_rv64 = slots.total_slots();
    let syscall_dispatch_idx =
        func_export_index(CANONICAL_GUEST_WASM, EXPORT_SYSCALL_DISPATCH).unwrap();
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
        syscall_dispatch_idx,
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
}

fn guest_memory_configs() -> Vec<GuestMemoryConfig<(), LinkErr>> {
    vec![
        GuestMemoryConfig {
            addr_width: AddressWidth::W32,
            memory_access: None,
        },
        GuestMemoryConfig {
            addr_width: AddressWidth::W32,
            memory_access: None,
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
) -> BinaryUnit<Function> {
    let mut frontend = WasmFrontend::with_wasm_encoder_fn(
        guest_memory_configs(),
        HOST_MEMORY_INDEX,
        IndexOffsets::default(),
    );
    frontend.set_preserve_guest_module(true);
    frontend
        .translate_module(ctx, rctx, CANONICAL_GUEST_WASM)
        .expect("guest translate_module");
    frontend.drain_unit(rctx, guest_export_entry_points())
}

fn emit_rv64_unit(
    rctx: &mut dyn ReactorContext<(), LinkErr, FnType = Function>,
    ctx: &mut (),
    text: &[u8],
    start_addr: u64,
    slots: &PcSlotMap,
    syscall_dispatch_idx: u32,
    halt_stub_idx: u32,
) -> BinaryUnit<Function> {
    let mut recompiler =
        speet_riscv::RiscVRecompiler::<(), LinkErr, Function>::new_with_full_config(
            start_addr, false, true, false,
        );
    recompiler.set_slot_assigner(slots.clone());
    recompiler.setup_traps(rctx, ctx);
    let params = collect_params(rctx);

    let mut ecall_cb = LinuxWasiEcall {
        syscall_dispatch_idx,
        slots: slots.clone(),
        base_func_offset: rctx.base_func_offset(),
        halt_stub_idx,
        num_params: params.len() as u32,
    };
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
    let register_type = FuncType::from_val_types(&params, &params);
    let mut func_types = vec![register_type.clone(); fns.len()];
    fns.push(build_halt_stub(&params));
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

fn build_halt_stub(params: &[ValType]) -> Function {
    let mut f = Function::new([]);
    for p in 0..params.len() as u32 {
        f.instruction(&wasm_encoder::Instruction::LocalGet(p));
    }
    f.instruction(&wasm_encoder::Instruction::Return);
    f.instruction(&wasm_encoder::Instruction::End);
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EXPORT_HANDLER_WRITE;

    #[test]
    fn link_guest_via_wasm_frontend_validates() {
        let wasm = link_canonical_guest_wasm();
        wasmparser::validate(&wasm).expect("linked guest module");
        assert!(func_export_index(&wasm, EXPORT_SYSCALL_DISPATCH).is_some());
        assert!(func_export_index(&wasm, EXPORT_HANDLER_WRITE).is_some());
    }

    #[test]
    fn guest_first_slot_preserves_syscall_dispatch_index() {
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
        assert_eq!(plan.guest_count, 5);
        assert_eq!(
            plan.syscall_dispatch_idx,
            func_export_index(CANONICAL_GUEST_WASM, EXPORT_SYSCALL_DISPATCH).unwrap()
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
}
