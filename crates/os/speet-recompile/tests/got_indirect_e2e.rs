//! GOT → redirect shim pipeline (virtual GOT, no PC-check in translated `.text`).

use binary_io::BinArch;
use speet_host_api::{HostApi, ImportManifest, PltRedirect, RedirectingHostApi, TunneledHostApi};
use speet_link_core::image_layout::{DataSectionSpec, GuestImageLayout, MemoryModel};
use speet_plugin_api::external_target::{ExternalTargetTable, LibraryId};
use speet_recompile::data_link::link_data_segments;
use speet_recompile::frontend::recompile_to_wasm_instrumented_plt_with_layout;
use speet_recompile::plt::PltCallPlan;

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
            arch: BinArch::X86_64,
            os: binary_io::BinOs::Linux,
            dylib_flags: vec![],
            ambient_aliases: vec![],
        }
    }
}

/// Minimal guest: `nop` + `ret` — PLT resolution goes through patched GOT, not PC-check hooks.
const GUEST_TEXT: &[u8] = &[0x90, 0xC3];

const GOT_ADDR: u64 = 0x2000;

#[test]
fn got_patch_recompile_validates_without_pc_check_imports_in_text() {
    let mut targets = ExternalTargetTable::new();
    targets.insert(LibraryId::MAIN_IMAGE, GOT_ADDR, "write");
    let plan = PltCallPlan::from_targets(&targets, &NativeWriteHost);
    let manifest = ImportManifest::native_syscall();
    let write_import = manifest.index_of("env", "write").expect("write import");

    let layout = GuestImageLayout {
        text_base: 0x1000,
        text_len: GUEST_TEXT.len(),
        slot_granularity: 1,
        data_sections: vec![DataSectionSpec {
            name: ".got".into(),
            addr: GOT_ADDR,
            bytes: vec![0u8; 8],
        }],
        relocs: vec![],
        libraries: vec![],
        memory_model: MemoryModel::OwnedLinear,
    };

    let (_, shim_by_sym) = speet_recompile::frontend::build_redirect_shims_from_plan(
        &plan,
        BinArch::X86_64,
        &manifest,
        42,
    );
    let linked = link_data_segments(&layout, Some(&plan), &shim_by_sym);
    let shim_pc = layout.shim_guest_pc(0);
    assert_eq!(&linked[0].bytes, &shim_pc.to_le_bytes());

    let (wasm, _unsupported) = recompile_to_wasm_instrumented_plt_with_layout(
        GUEST_TEXT,
        &layout,
        BinArch::X86_64,
        Some(&plan),
        Some(0x1000),
        &manifest,
    );

    wasmparser::validate(&wasm).expect("recompiled module must validate");

    // Translated `.text` functions must not embed direct host-import calls for PLT
    // hooks (retired PC-check path); redirect shims live in separate WASM funcs.
    let mut in_text_region = true;
    let mut text_func_count = 0usize;
    for payload in wasmparser::Parser::new(0).parse_all(&wasm) {
        let payload = payload.expect("parse");
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            if in_text_region {
                text_func_count += 1;
                for op in body.get_operators_reader().expect("ops").into_iter() {
                    let op = op.expect("op");
                    if let wasmparser::Operator::Call { function_index } = op {
                        assert_ne!(
                            function_index, write_import,
                            "text-region func must not call write import directly (PC-check retired)"
                        );
                    }
                }
            }
            // After all byte-aligned text slots (2 bytes → 2 funcs for this snippet).
            if text_func_count >= GUEST_TEXT.len() {
                in_text_region = false;
            }
        }
    }
    assert!(text_func_count >= GUEST_TEXT.len());
}
