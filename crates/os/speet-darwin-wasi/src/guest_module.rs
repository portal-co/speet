//! Host-mem-lowered Darwin/BSD→WASI guest module (compiled from Rust, not raw bytecode).

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use os_darwin_wasi::DarwinToWasi;
use wasmparser::{CompositeInnerType, Parser, Payload, TypeRef};
use wasm_encoder::ValType;

/// One function import declared in a guest WASM module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestFuncImport {
    pub module: String,
    pub name: String,
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

/// Canonical multi-memory guest module bytes (post [`lower_host_mem_imports`]).
pub const CANONICAL_GUEST_WASM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/canonical_guest.wasm"));

/// Stable export names inside the guest module.
pub const EXPORT_SYSCALL_DISPATCH: &str = "syscall_dispatch";
pub const EXPORT_HANDLER_READ: &str = "handler_read";
pub const EXPORT_HANDLER_WRITE: &str = "handler_write";
pub const EXPORT_HANDLER_CLOSE: &str = "handler_close";
pub const EXPORT_HANDLER_EXIT: &str = "handler_exit";

/// Map Darwin/BSD syscall numbers to guest handler export names.
pub fn handler_export_name(syscall_num: u64) -> Option<&'static str> {
    match syscall_num {
        3 => Some(EXPORT_HANDLER_READ),
        4 => Some(EXPORT_HANDLER_WRITE),
        6 => Some(EXPORT_HANDLER_CLOSE),
        1 => Some(EXPORT_HANDLER_EXIT),
        _ => None,
    }
}

/// Plan mapping Darwin/BSD syscall numbers to handler function indices inside a linked module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandlerModulePlan {
    pub handlers: Vec<(u64, u32)>,
}

impl HandlerModulePlan {
    /// Build a plan from a [`DarwinToWasi`] table and the base index where guest handlers land.
    pub fn from_darwin_to_wasi(table: &DarwinToWasi, base_func_idx: u32) -> Self {
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

/// Function imports from `wasm` in link order (type signatures taken from the guest type section).
pub fn guest_func_imports(wasm: &[u8]) -> Vec<GuestFuncImport> {
    let types = guest_func_types(wasm);
    let mut imports = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::ImportSection(reader) = payload.expect("guest parse") {
            for imp in reader {
                let imp = imp.expect("guest import");
                if let TypeRef::Func(type_idx) = imp.ty {
                    let (params, results) = types
                        .get(type_idx as usize)
                        .cloned()
                        .unwrap_or_default();
                    imports.push(GuestFuncImport {
                        module: String::from(imp.module),
                        name: String::from(imp.name),
                        params,
                        results,
                    });
                }
            }
        }
    }
    imports
}

/// Number of function imports in `wasm`.
pub fn guest_import_count(wasm: &[u8]) -> u32 {
    guest_func_imports(wasm).len() as u32
}

/// [`WasiImports`] indices matching guest-module call targets at `base`.
pub fn wasi_imports_for_guest_layout(base: u32) -> os_darwin_wasi::WasiImports {
    wasi_imports_from_guest_imports(base, &guest_func_imports(CANONICAL_GUEST_WASM))
}

/// Map guest import list to [`WasiImports`] field indices at `base`.
pub fn wasi_imports_from_guest_imports(
    base: u32,
    imports: &[GuestFuncImport],
) -> os_darwin_wasi::WasiImports {
    use os_darwin_wasi::WasiImports;

    let mut wasi = WasiImports {
        fd_write: 0,
        fd_read: 0,
        fd_close: 0,
        proc_exit: 0,
    };
    for (i, imp) in imports.iter().enumerate() {
        let idx = base + i as u32;
        match imp.name.as_str() {
            "fd_write" => wasi.fd_write = idx,
            "fd_read" => wasi.fd_read = idx,
            "fd_close" => wasi.fd_close = idx,
            "proc_exit" => wasi.proc_exit = idx,
            other => panic!("unexpected WASI import in guest module: {other}"),
        }
    }
    wasi
}

/// Resolve a guest-module type index to [`FuncType`].
pub fn guest_func_type_at(wasm: &[u8], type_idx: u32) -> speet_link_core::unit::FuncType {
    let (params, results) = guest_func_types(wasm)
        .into_iter()
        .nth(type_idx as usize)
        .unwrap_or_default();
    speet_link_core::unit::FuncType::from_val_types(&params, &results)
}

fn guest_func_types(wasm: &[u8]) -> Vec<(Vec<ValType>, Vec<ValType>)> {
    let mut types = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::TypeSection(reader) = payload.expect("guest parse") {
            for rec in reader {
                let rec = rec.expect("guest type");
                for sub in rec.types() {
                    match &sub.composite_type.inner {
                        CompositeInnerType::Func(ft) => {
                            types.push((
                                ft.params().iter().copied().map(val_type).collect(),
                                ft.results().iter().copied().map(val_type).collect(),
                            ));
                        }
                        _ => types.push((Vec::new(), Vec::new())),
                    }
                }
            }
        }
    }
    types
}

fn val_type(ty: wasmparser::ValType) -> ValType {
    match ty {
        wasmparser::ValType::I32 => ValType::I32,
        wasmparser::ValType::I64 => ValType::I64,
        wasmparser::ValType::F32 => ValType::F32,
        wasmparser::ValType::F64 => ValType::F64,
        wasmparser::ValType::V128 => ValType::V128,
        wasmparser::ValType::Ref(rt) => {
            use wasm_encoder::{AbstractHeapType, HeapType, RefType};
            ValType::Ref(RefType {
                nullable: rt.is_nullable(),
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Func,
                },
            })
        }
    }
}

/// Count defined functions in a WASM module (Function section size).
pub fn guest_defined_fn_count(wasm: &[u8]) -> u32 {
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::FunctionSection(reader) = payload.expect("guest parse") {
            return reader.count();
        }
    }
    0
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
    use os_darwin_wasi::WasiImports;

    #[test]
    fn guest_func_imports_match_canonical_module() {
        let imports = guest_func_imports(CANONICAL_GUEST_WASM);
        assert_eq!(imports.len(), 4);
        assert!(imports.iter().all(|i| i.module == "wasi_snapshot_preview1"));
        for name in ["fd_close", "fd_read", "fd_write", "proc_exit"] {
            assert!(
                imports.iter().any(|i| i.name == name),
                "missing guest import {name}"
            );
        }
        let write = imports.iter().find(|i| i.name == "fd_write").unwrap();
        assert_eq!(
            write.params,
            vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32]
        );
        assert_eq!(write.results, vec![ValType::I32]);
    }

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
    fn handler_plan_covers_darwin_to_wasi_table() {
        let imports = WasiImports {
            fd_write: 0,
            fd_read: 1,
            fd_close: 2,
            proc_exit: 3,
        };
        let table = DarwinToWasi::new(imports).build_table(|n| n as u32);
        let plan = HandlerModulePlan::from_syscall_table(&table, 10);
        assert_eq!(plan.handlers.len(), table.entries().len());
        assert!(plan.handlers.iter().any(|(n, _)| *n == 4));
    }

    #[test]
    fn syscall_numbers_map_to_guest_exports() {
        assert_eq!(handler_export_name(4), Some(EXPORT_HANDLER_WRITE));
        assert_eq!(handler_export_name(6), Some(EXPORT_HANDLER_CLOSE));
        assert_eq!(handler_export_name(999), None);
    }
}
