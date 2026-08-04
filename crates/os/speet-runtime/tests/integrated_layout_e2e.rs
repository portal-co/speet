//! Regression: integrated obtain with full [`GuestImageLayout`] data segments.
//!
//! `memory.init` must emit ADRP+ADD for `__wasm_data_seg_*` / `__wasm_memory_init_copy`.
//! Plain ADR was historically mapped to Mach-O `ARM64_RELOC_PAGE21`, which `ld`
//! rejects on non-ADRP instructions.

use speet_host_api::integrated_host_api;
use speet_runtime::{IntegratedNativeRuntime, NativeRuntime};
use std::sync::Arc;

#[test]
fn integrated_layout_exit42_links_and_runs() {
    let mut rt = IntegratedNativeRuntime::new(Arc::new(integrated_host_api()));
    if !rt.llvm_available() {
        eprintln!("SKIP: no LLVM");
        return;
    }
    let guest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/c-corpus/aarch64-macos/exit42.linked.macho");
    let exe = rt.obtain_executable(&guest).unwrap_or_else(|e| panic!("{e}"));
    let st = rt.spawn(&exe, &[], None).unwrap();
    assert_eq!(st.code(), Some(42), "status={st:?}");
}
