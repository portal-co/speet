//! Translate aarch64 Darwin `.text` with `svc` → merged guest [`syscall_dispatch`].

use core::convert::Infallible;

use speet_aarch64::cfg::AArch64CfgDecoder;
use speet_host_api::ImportManifest;
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext, TextBaseSource};
use speet_reach::PcSlotMap;
use wasm_encoder::{Function, ValType};
use yecta::{LocalPool, Reactor, SlotAssigner, TableIdx, TypeIdx};

use crate::manifest::wasi_preview1_manifest;
use crate::svc::DarwinWasiSvc;
use crate::WasiImports;

static REACTOR_TABLE: TableIdx = TableIdx(0);

/// Output of [`translate_aarch64_darwin_wasi`]: translated bodies plus merge metadata.
pub struct WasiTranslation {
    pub fns: alloc::vec::Vec<Function>,
    pub params: alloc::vec::Vec<ValType>,
    pub entry_func_idx: u32,
    pub start_addr: u64,
    pub wasi: WasiImports,
    pub syscall_dispatch_idx: u32,
}

fn make_rctx<'r>(
    reactor: &'r mut Reactor<(), Infallible, Function, LocalPool>,
    base_func_offset: u32,
    start_addr: u64,
) -> ReactorAdapter<'r, (), Infallible, Function, LocalPool> {
    let mut rctx = ReactorAdapter {
        reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark {
            slot_count: 0,
            total_locals: 0,
        },
        injected_start: yecta::Mark {
            slot_count: 0,
            total_locals: 0,
        },
        layout_params: speet_link_core::RuntimeLayoutParams::with_text_base_source(
            TextBaseSource::Constant(start_addr),
        ),
        pool: yecta::Pool {
            handler: &REACTOR_TABLE,
            ty: TypeIdx(0),
        },
        escape_tag: None,
    };
    rctx.set_base_func_offset(base_func_offset);
    rctx
}

fn collect_params(rctx: &ReactorAdapter<'_, (), Infallible, Function, LocalPool>) -> alloc::vec::Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout()
        .iter_before(&mark)
        .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
        .collect()
}

fn wasi_imports_from_manifest(manifest: &ImportManifest) -> WasiImports {
    WasiImports {
        fd_write: manifest
            .index_of("wasi_snapshot_preview1", "fd_write")
            .expect("WASI manifest declares fd_write"),
        fd_read: manifest
            .index_of("wasi_snapshot_preview1", "fd_read")
            .expect("WASI manifest declares fd_read"),
        fd_close: manifest
            .index_of("wasi_snapshot_preview1", "fd_close")
            .expect("WASI manifest declares fd_close"),
        proc_exit: manifest
            .index_of("wasi_snapshot_preview1", "proc_exit")
            .expect("WASI manifest declares proc_exit"),
    }
}

/// Translate an aarch64 Darwin `.text` blob, lowering `svc` to the embedded guest module.
pub fn translate_aarch64_darwin_wasi(text: &[u8], start_addr: u64) -> WasiTranslation {
    let manifest = wasi_preview1_manifest();
    let wasi = wasi_imports_from_manifest(&manifest);
    let n_imports = manifest.func_imports.len() as u32;

    let slots = PcSlotMap::all_slots(text, start_addr, &AArch64CfgDecoder);
    let n_translated = slots.total_slots();
    let n_guest = crate::guest_module::guest_defined_fn_count(crate::CANONICAL_GUEST_WASM);
    let aarch64_base = n_imports + n_guest;
    let syscall_dispatch_idx = crate::link::func_export_index(
        crate::CANONICAL_GUEST_WASM,
        crate::EXPORT_SYSCALL_DISPATCH,
    )
    .expect("syscall_dispatch export");
    let halt_stub_idx = aarch64_base + n_translated;

    let mut recompiler = speet_aarch64::AArch64Recompiler::<(), Infallible>::new_with_base_pc(start_addr);
    recompiler.set_slot_assigner(slots.clone());

    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor, aarch64_base, start_addr);
    let mut ctx = ();
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let params = collect_params(&rctx);

    let mut svc_cb = DarwinWasiSvc {
        syscall_dispatch_idx,
        slots,
        base_func_offset: aarch64_base,
        halt_stub_idx,
        num_params: params.len() as u32,
    };
    recompiler.set_svc_callback(&mut svc_cb);

    recompiler
        .translate_bytes(
            &mut ctx,
            &mut rctx,
            text,
            start_addr,
            &mut |a| Function::new(a.collect::<alloc::vec::Vec<_>>()),
        )
        .expect("translate_bytes");

    WasiTranslation {
        fns: rctx.drain_fns(),
        params,
        entry_func_idx: 0,
        start_addr,
        wasi,
        syscall_dispatch_idx,
    }
}
