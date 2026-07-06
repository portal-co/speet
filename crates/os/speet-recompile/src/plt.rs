//! PLT call redirection during guest → WASM translation.

use binary_io::BinArch;
use speet_host_api::{HostApi, PltRedirect};
use std::collections::BTreeMap;

use crate::frontend::ExternalTargets;

/// WASM import index for `env.__speet_execve` in [`super::frontend::assemble_module_instrumented`].
pub const INTEGRATED_IMPORT_EXECVE: u32 = 4;

/// Per-guest-address PLT table plus symbol → WASM import index map.
#[derive(Debug, Clone, Default)]
pub struct PltCallPlan {
    pub by_addr: BTreeMap<u64, String>,
    pub import_by_symbol: BTreeMap<String, u32>,
}

impl PltCallPlan {
    pub fn from_targets(targets: &ExternalTargets, host: &dyn HostApi) -> Self {
        let mut import_by_symbol = BTreeMap::new();
        for sym in targets.by_plt_addr.values() {
            if import_by_symbol.contains_key(sym) {
                continue;
            }
            let Some(PltRedirect::WasmImport { module, name }) = host.resolve_plt_redirect(sym)
            else {
                continue;
            };
            if module != "env" {
                continue;
            }
            let idx = import_index_for_env_name(&name);
            if let Some(idx) = idx {
                import_by_symbol.insert(sym.clone(), idx);
            }
        }
        Self {
            by_addr: targets.by_plt_addr.clone(),
            import_by_symbol,
        }
    }

    pub fn lookup_import(&self, target: u64) -> Option<(u32, &str)> {
        let sym = self.by_addr.get(&target)?;
        let idx = self.import_by_symbol.get(sym)?;
        Some((*idx, sym.as_str()))
    }
}

fn import_index_for_env_name(name: &str) -> Option<u32> {
    match name {
        "__speet_execve" => Some(INTEGRATED_IMPORT_EXECVE),
        _ => None,
    }
}

/// SysV x86-64 argument local indices for a hooked PLT symbol.
pub fn x86_sysv_arg_locals(symbol: &str) -> Option<&'static [u32]> {
    match symbol.strip_prefix('_').unwrap_or(symbol) {
        "execve" => Some(&[7, 6, 2]), // RDI, RSI, RDX
        _ => None,
    }
}

/// AAPCS64 argument local indices for a hooked PLT symbol.
pub fn aarch64_arg_locals(symbol: &str) -> Option<&'static [u32]> {
    match symbol.strip_prefix('_').unwrap_or(symbol) {
        "execve" => Some(&[0, 1, 2]), // x0, x1, x2
        _ => None,
    }
}

/// Result GPR local after a hooked import call.
pub fn result_local(arch: BinArch) -> u32 {
    match arch {
        BinArch::X86_64 => 0,   // RAX
        BinArch::AArch64 => 0,  // x0
    }
}
