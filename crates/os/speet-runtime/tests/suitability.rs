//! Import-based suitability tests.

use speet_host_api::integrated_host_api;
use speet_runtime::{IntegratedNativeRuntime, NativeRuntime};
use std::path::Path;
use std::sync::Arc;

fn corpus_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus")
}

fn linked_exit42() -> Option<std::path::PathBuf> {
    let root = corpus_root();
    if cfg!(target_arch = "aarch64") {
        let p = root.join("aarch64-macos/exit42.linked.macho");
        if p.is_file() {
            return Some(p);
        }
    }
    if cfg!(target_arch = "x86_64") {
        let p = root.join("x86_64-macos/exit42.linked.macho");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[test]
fn exit42_linked_is_suitable() {
    let Some(path) = linked_exit42() else {
        eprintln!("SKIP: no exit42 linked artifact for this host");
        return;
    };
    let rt = IntegratedNativeRuntime::new(Arc::new(integrated_host_api()));
    let report = rt.analyze(&path).expect("analyze");
    assert!(
        report.suitable,
        "exit42 should be suitable: unresolved={:?} fn_ptr={:?}",
        report.unresolved_deps, report.fn_ptr_deps
    );
}

#[test]
fn extra_wired_clears_unknown_import_name() {
    use binary_io::{BinArch, BinFormat, BinOs, ImportSym, LoadedBinary};
    use speet_link_core::MemoryModel;
    use speet_runtime::analyze_imports_with_model_and_wired;

    let host = integrated_host_api();
    let bin = LoadedBinary {
        format: BinFormat::MachO,
        arch: if cfg!(target_arch = "aarch64") {
            BinArch::AArch64
        } else {
            BinArch::X86_64
        },
        os: BinOs::MacOs,
        entry: 0,
        sections: vec![],
        symbols: vec![],
        relocs: vec![],
        dyn_deps: vec![],
        imports: vec![ImportSym {
            name: "not_a_real_symbol".into(),
            plt_addr: None,
        }],
    };
    let (unresolved, _) = analyze_imports_with_model_and_wired(
        &host,
        &bin,
        MemoryModel::OwnedLinear,
        &["not_a_real_symbol".into()],
    );
    assert!(
        unresolved.is_empty(),
        "hot-wired name must not remain unresolved: {unresolved:?}"
    );
}

#[test]
fn cache_key_includes_plugin_hashes_when_hot() {
    let rt = IntegratedNativeRuntime::new(Arc::new(integrated_host_api()));
    let Some(path) = linked_exit42() else {
        eprintln!("SKIP: no exit42 linked artifact for this host");
        return;
    };
    let native = rt.artifact_cache_key(&path).expect("key");
    assert!(
        !native.contains(":rec="),
        "production key must omit rec=/stubs= segments: {native}"
    );
    rt.set_hot_plugins(
        speet_runtime::PluginHashes {
            recompiler: "aaa".into(),
            stubs: "bbb".into(),
        },
        vec![],
        None,
    );
    let hot = rt.artifact_cache_key(&path).expect("key");
    assert!(hot.contains(":rec=aaa:stubs=bbb"), "{hot}");
    assert_ne!(native, hot);
}

#[test]
fn printf_import_is_wired_via_abi_stub() {
    use binary_io::{BinArch, BinFormat, BinOs, ImportSym, LoadedBinary};
    use speet_runtime::analyze_imports;

    let host = integrated_host_api();
    let bin = LoadedBinary {
        format: BinFormat::MachO,
        arch: if cfg!(target_arch = "aarch64") {
            BinArch::AArch64
        } else {
            BinArch::X86_64
        },
        os: BinOs::MacOs,
        entry: 0,
        sections: vec![],
        symbols: vec![],
        relocs: vec![],
        dyn_deps: vec![],
        imports: vec![ImportSym {
            name: "printf".into(),
            plt_addr: Some(0x1000),
        }],
    };
    let (unresolved, fn_ptr) = analyze_imports(&host, &bin);
    assert!(unresolved.is_empty());
    assert!(
        fn_ptr.is_empty(),
        "printf is wired via speet-abi-stubs, not a fn-ptr dep: {fn_ptr:?}"
    );
}
