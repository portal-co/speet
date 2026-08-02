//! Test host contract for `__speet_host_mem_*` imports (pre-shimming phase).

use speet_host_api::ImportManifest;

/// Manifest slice shared by test mock hosts and parity harness lane A.
pub fn test_mock_manifest() -> ImportManifest {
    ImportManifest::native_syscall().with_host_mem_imports()
}

#[test]
fn test_mock_manifest_includes_host_mem_imports() {
    let m = test_mock_manifest();
    assert!(m.index_of("__speet_host_mem", "load_i64").is_some());
    assert!(m.index_of("__speet_host_mem", "store_i64").is_some());
}
