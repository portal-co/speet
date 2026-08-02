//! Host-mem-lowered Linux→WASI guest module (compiled from Rust, not raw bytecode).

use alloc::vec::Vec;
use os_linux_wasi::LinuxToWasi;
use wasmparser::{Parser, Payload};

/// Canonical multi-memory guest module bytes (post [`lower_host_mem_imports`]).
pub const CANONICAL_GUEST_WASM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/canonical_guest.wasm"));

/// Stable export names inside the guest module.
pub const EXPORT_SYSCALL_DISPATCH: &str = "syscall_dispatch";
pub const EXPORT_HANDLER_READ: &str = "handler_read";
pub const EXPORT_HANDLER_WRITE: &str = "handler_write";
pub const EXPORT_HANDLER_CLOSE: &str = "handler_close";
pub const EXPORT_HANDLER_EXIT: &str = "handler_exit";

/// Map Linux syscall numbers to guest handler export names.
pub fn handler_export_name(syscall_num: u64) -> Option<&'static str> {
    match syscall_num {
        63 => Some(EXPORT_HANDLER_READ),
        64 => Some(EXPORT_HANDLER_WRITE),
        57 => Some(EXPORT_HANDLER_CLOSE),
        93 | 94 => Some(EXPORT_HANDLER_EXIT),
        _ => None,
    }
}

/// Plan mapping Linux syscall numbers to handler function indices inside a linked module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandlerModulePlan {
    pub handlers: Vec<(u64, u32)>,
}

impl HandlerModulePlan {
    /// Build a plan from a [`LinuxToWasi`] table and the base index where guest handlers land.
    pub fn from_linux_to_wasi(table: &LinuxToWasi, base_func_idx: u32) -> Self {
        Self::from_syscall_table(&table.build_table(|n| n as u32), base_func_idx)
    }

    pub fn from_syscall_table(
        table: &os_syscall_emit::SyscallTable,
        base_func_idx: u32,
    ) -> Self {
        let handlers = table
            .entries()
            .iter()
            .enumerate()
            .map(|(i, (num, _))| (*num, base_func_idx + i as u32))
            .collect();
        Self { handlers }
    }
}

/// Collect exported function names from the canonical guest module.
pub fn guest_export_names(wasm: &[u8]) -> Result<Vec<alloc::string::String>, alloc::string::String> {
    let mut names = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::ExportSection(reader) = payload.map_err(|e| alloc::format!("{e:?}"))? {
            for export in reader {
                let export = export.map_err(|e| alloc::format!("{e:?}"))?;
                if export.kind == wasmparser::ExternalKind::Func {
                    names.push(alloc::string::String::from(export.name));
                }
            }
        }
    }
    Ok(names)
}

/// Validate the embedded guest module and confirm host-mem imports were lowered away.
pub fn validate_canonical_guest() -> Result<(), alloc::string::String> {
    wasmparser::validate(CANONICAL_GUEST_WASM).map_err(|e| alloc::format!("{e:?}"))?;
    let host_imports = host_mem_import_map(CANONICAL_GUEST_WASM)?;
    if !host_imports.is_empty() {
        return Err(alloc::format!(
            "canonical guest still has {} host-mem import(s)",
            host_imports.len()
        ));
    }
    Ok(())
}

fn host_mem_import_map(wasm: &[u8]) -> Result<alloc::collections::BTreeMap<u32, alloc::string::String>, alloc::string::String> {
    use alloc::collections::BTreeMap;
    use wasmparser::TypeRef;

    let mut map = BTreeMap::new();
    let mut import_func_idx = 0u32;
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::ImportSection(reader) = payload.map_err(|e| alloc::format!("{e:?}"))? {
            for imp in reader {
                let imp = imp.map_err(|e| alloc::format!("{e:?}"))?;
                if let TypeRef::Func(_) = imp.ty {
                    if imp.module == "__speet_host_mem" {
                        map.insert(import_func_idx, alloc::string::String::from(imp.name));
                    }
                    import_func_idx += 1;
                }
            }
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use os_linux_wasi::WasiImports;

    #[test]
    fn canonical_guest_validates_and_exports_handlers() {
        validate_canonical_guest().expect("embedded guest module");
        let exports = guest_export_names(CANONICAL_GUEST_WASM).unwrap();
        for name in [
            EXPORT_SYSCALL_DISPATCH,
            EXPORT_HANDLER_READ,
            EXPORT_HANDLER_WRITE,
            EXPORT_HANDLER_CLOSE,
            EXPORT_HANDLER_EXIT,
        ] {
            assert!(exports.iter().any(|e| e == name), "missing export {name}");
        }
    }

    #[test]
    fn handler_plan_covers_linux_to_wasi_table() {
        let imports = WasiImports {
            fd_write: 0,
            fd_read: 1,
            fd_close: 2,
            proc_exit: 3,
        };
        let table = LinuxToWasi::new(imports).build_table(|n| n as u32);
        let plan = HandlerModulePlan::from_syscall_table(&table, 10);
        assert_eq!(plan.handlers.len(), table.entries().len());
        assert!(plan.handlers.iter().any(|(n, _)| *n == 64));
    }

    #[test]
    fn syscall_numbers_map_to_guest_exports() {
        assert_eq!(handler_export_name(64), Some(EXPORT_HANDLER_WRITE));
        assert_eq!(handler_export_name(57), Some(EXPORT_HANDLER_CLOSE));
        assert_eq!(handler_export_name(999), None);
    }
}
