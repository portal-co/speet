//! Reloadable ABI-stub table for `speet-rtd`'s `hot-recompiler` feature.
//!
//! Implements [`TargetPlugin`]: `module_manifest().func_imports` lists every
//! wired symbol as `field`. The daemon treats those names as `has_wired_impl`
//! before the statically linked registry. Keep [`WIRED_SYMBOLS`] in sync with
//! `os-abi-stubs` `STUB_SYMBOLS` (the agent edits that source, then rebuilds
//! this crate to wasm32).

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use speet_plugin_api::snippet::PluginValType;
use speet_plugin_api::target::{FuncImportDecl, ModuleManifest, PluginSyscallTable, TargetPlugin};
use speet_plugin_guest::{install, wasm_alloc, wasm_call, GuestPlugin};

/// Wired symbols — mirror of `os_abi_stubs::STUB_SYMBOLS`. Extra names the
/// agent adds here (and in the generated registry) become hot-reloadable.
pub const WIRED_SYMBOLS: &[&str] = &[
    "write", "exit", "printf", "strcpy", "strcat", "strncpy", "strncat", "malloc", "free",
];

pub struct StubsTarget;

impl TargetPlugin for StubsTarget {
    fn module_manifest(&self) -> ModuleManifest {
        ModuleManifest {
            func_imports: WIRED_SYMBOLS
                .iter()
                .map(|s| FuncImportDecl {
                    module: String::from("stub"),
                    field: String::from(*s),
                    params: Vec::new(),
                    results: Vec::<PluginValType>::new(),
                })
                .collect(),
            ..Default::default()
        }
    }
    fn syscall_table(&self) -> PluginSyscallTable {
        PluginSyscallTable::default()
    }
}

fn ensure_installed() {
    let _ = install(GuestPlugin::Target(Arc::new(StubsTarget)));
}

/// Names the daemon should treat as wired after loading this guest.
pub fn wired_symbol_list() -> Vec<String> {
    WIRED_SYMBOLS.iter().map(|s| String::from(*s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_plugin_api::target::TargetPlugin;

    #[test]
    fn wired_symbols_are_manifest_fields() {
        let m = StubsTarget.module_manifest();
        let fields: Vec<_> = m.func_imports.iter().map(|f| f.field.as_str()).collect();
        assert_eq!(fields, WIRED_SYMBOLS);
    }
}

#[no_mangle]
pub extern "C" fn speet_plugin_alloc(len: i32) -> i32 {
    ensure_installed();
    wasm_alloc(len)
}

#[no_mangle]
pub extern "C" fn speet_plugin_call(req_ptr: i32, req_len: i32) -> i64 {
    ensure_installed();
    wasm_call(req_ptr, req_len)
}

#[no_mangle]
pub extern "C" fn speet_plugin_dealloc(_ptr: i32, _len: i32) {}
