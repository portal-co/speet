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
        report.unresolved_deps,
        report.fn_ptr_deps
    );
}

#[test]
fn printf_import_is_fn_ptr_dep() {
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
    assert_eq!(fn_ptr, vec!["printf"]);
}
