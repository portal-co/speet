//! GOT/PLT shim foundation integration tests.

use binary_io::BinArch;
use speet_host_api::{HostApi, ImportManifest, PltRedirect, RedirectingHostApi, TunneledHostApi};
use speet_link_core::{GuestImageLayout, MemoryModel, RuntimeLayoutParams, TextBaseSource};
use speet_plugin_api::external_target::{ExternalTargetTable, LibraryId};
use speet_recompile::frontend::{build_redirect_shims_from_plan, halt_addr};
use speet_recompile::guest_func_catalog::GuestFuncCatalog;
use speet_recompile::plt::PltCallPlan;
use yecta::{LocalDeclarator, LocalLayout};

struct NativeWriteHost;

impl HostApi for NativeWriteHost {
    fn import_manifest(&self) -> ImportManifest {
        ImportManifest::native_syscall()
    }
    fn resolve_plt_redirect(&self, guest_symbol: &str) -> Option<PltRedirect> {
        let bare = guest_symbol.strip_prefix('_').unwrap_or(guest_symbol);
        if bare == "write" {
            Some(PltRedirect::NativeShim {
                core_symbol: "write".into(),
            })
        } else {
            None
        }
    }
    fn resolve_ambient(&self, _: &str) -> Option<tunnel::TunnelResolution> {
        None
    }
    fn link_recipe(&self) -> speet_host_api::LinkRecipe {
        speet_host_api::LinkRecipe {
            arch: BinArch::AArch64,
            os: binary_io::BinOs::Linux,
            dylib_flags: vec![],
            ambient_aliases: vec![],
        }
    }
}

#[test]
fn guest_image_layout_shim_pc_after_halt() {
    let layout = GuestImageLayout {
        text_base: 0x1000,
        text_len: 0x20,
        slot_granularity: 4,
        data_sections: vec![],
        relocs: vec![],
        libraries: vec![],
        memory_model: MemoryModel::OwnedLinear,
    };
    assert_eq!(layout.halt_guest_pc(), 0x1020);
    assert_eq!(layout.shim_guest_pc(0), 0x1024);
}

#[test]
fn runtime_layout_params_declare_local_slots() {
    let mut layout = LocalLayout::empty();
    let mut params = RuntimeLayoutParams::with_text_base_source(TextBaseSource::Constant(0x1000));
    params.declare_params(yecta::CellIdx(0), &mut layout);
    params.bind_snippets(&layout);
    assert!(layout.total_locals() >= 2);
}

#[test]
fn catalog_includes_redirect_shim_entries() {
    let catalog = GuestFuncCatalog::from_layout(
        0x1000,
        BinArch::AArch64,
        3,
        2,
        0x20,
        1,
    );
    assert_eq!(catalog.entries.len(), 4); // 2 translated + 1 shim + halt
    assert_eq!(catalog.n_redirect_shims, 1);
    assert_eq!(catalog.halt_entry().guest_pc, halt_addr(0x1000, 0x20));
}

#[test]
fn plt_plan_native_shim_addr_map() {
    let mut targets = ExternalTargetTable::new();
    targets.insert(LibraryId::MAIN_IMAGE, 0x5000, "write");
    let plan = PltCallPlan::from_targets(&targets, &NativeWriteHost);
    assert!(plan.wasm_import_by_addr.is_empty());
    assert_eq!(
        plan.native_shim_by_addr.get(&(LibraryId::MAIN_IMAGE, 0x5000)),
        Some(&"write".to_string())
    );
}

#[test]
fn redirect_shims_emitted_for_wasm_import_hooks() {
    let manifest = ImportManifest::integrated_native();
    let mut targets = ExternalTargetTable::new();
    targets.insert(LibraryId::MAIN_IMAGE, 0x3000, "execve");
    let host = RedirectingHostApi::integrated(
        TunneledHostApi::for_host().with_manifest(manifest.clone()),
    );
    let plan = PltCallPlan::from_targets(&targets, &host);
    let (shims, by_sym) = build_redirect_shims_from_plan(&plan, BinArch::AArch64, &manifest, 34);
    assert_eq!(shims.len(), 1);
    assert!(by_sym.contains_key("execve"));
}

#[test]
fn redirect_shims_emitted_for_native_shim_hooks() {
    let manifest = ImportManifest::native_syscall();
    let mut targets = ExternalTargetTable::new();
    targets.insert(LibraryId::MAIN_IMAGE, 0x5000, "write");
    let plan = PltCallPlan::from_targets(&targets, &NativeWriteHost);
    let (shims, by_sym) =
        build_redirect_shims_from_plan(&plan, BinArch::AArch64, &manifest, 34);
    assert_eq!(shims.len(), 1);
    assert!(by_sym.contains_key("write"));
}
