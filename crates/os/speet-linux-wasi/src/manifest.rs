//! [`ImportManifest`] for WASI preview1 megabinary assembly.

use alloc::vec;

use speet_host_api::{FuncImport, ImportManifest, WasmValType::I32};

/// [`ImportManifest`] for host-side WASI preview1 wiring (lookup by name).
/// Megabinary import **order and signatures** come from [`guest_module::guest_func_imports`].
pub fn wasi_preview1_manifest() -> ImportManifest {
    ImportManifest {
        func_imports: vec![
            FuncImport {
                module: "wasi_snapshot_preview1".into(),
                name: "fd_write".into(),
                params: vec![I32, I32, I32, I32],
                results: vec![I32],
                intercepts: vec![],
            },
            FuncImport {
                module: "wasi_snapshot_preview1".into(),
                name: "fd_read".into(),
                params: vec![I32, I32, I32, I32],
                results: vec![I32],
                intercepts: vec![],
            },
            FuncImport {
                module: "wasi_snapshot_preview1".into(),
                name: "fd_close".into(),
                params: vec![I32],
                results: vec![I32],
                intercepts: vec![],
            },
            FuncImport {
                module: "wasi_snapshot_preview1".into(),
                name: "proc_exit".into(),
                params: vec![I32],
                results: vec![],
                intercepts: vec![],
            },
        ],
    }
}
