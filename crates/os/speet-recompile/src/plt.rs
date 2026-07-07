//! PLT call redirection during guest → WASM translation.

use binary_io::BinArch;
use speet_host_api::{HostApi, PltRedirect};
use speet_plugin_api::external_target::{CallingConvention, PltHook, PltHookTable};
use std::collections::BTreeMap;

use crate::frontend::ExternalTargets;

/// Per-guest-address PLT table plus symbol → WASM import index map.
#[derive(Debug, Clone, Default)]
pub struct PltCallPlan {
    pub by_addr: BTreeMap<u64, String>,
    pub import_by_symbol: BTreeMap<String, u32>,
}

impl PltCallPlan {
    /// Resolve every hooked guest symbol's WASM import index against
    /// `host.import_manifest()`'s own `index_of` — never a separately
    /// hand-maintained `match name { "x" => 4, ... }` table (see
    /// `docs/guides/thin-runtime-genericity.md` principle 1). This also
    /// generalizes past the old "module must be `env`" restriction: any
    /// module name the manifest actually declares works.
    pub fn from_targets(targets: &ExternalTargets, host: &dyn HostApi) -> Self {
        let manifest = host.import_manifest();
        let mut import_by_symbol = BTreeMap::new();
        for sym in targets.by_plt_addr.values() {
            if import_by_symbol.contains_key(sym) {
                continue;
            }
            let Some(PltRedirect::WasmImport { module, name }) = host.resolve_plt_redirect(sym)
            else {
                continue;
            };
            if let Some(idx) = manifest.index_of(&module, &name) {
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

    /// Realize this plan as a [`PltHookTable`] for `arch`'s calling
    /// convention — the single shared type `speet-x86_64`/`speet-aarch64`
    /// hold by composition (`Option<PltHookTable>`) instead of each
    /// independently declaring its own `plt_by_addr` + `plt_imports` pair
    /// and a duplicated `lookup_plt_import`/`emit_plt_import_call` pair.
    /// See `docs/guides/thin-runtime-genericity.md` principle 1.
    pub fn to_hook_table(&self, arch: BinArch) -> PltHookTable {
        let mut table = PltHookTable::new();
        for (&addr, sym) in &self.by_addr {
            let Some(&import_idx) = self.import_by_symbol.get(sym) else {
                continue;
            };
            table.insert(
                addr,
                PltHook {
                    label: sym.clone(),
                    import_idx,
                    convention: calling_convention_for(arch, sym),
                },
            );
        }
        table
    }
}

/// Per-arch calling convention for a hooked guest symbol's redirected call.
///
/// Delegates to checked-in ABI-spec stubs when available (`speet-abi-stubs`),
/// then falls back to hand-maintained entries for symbols not yet generated.
/// See `docs/future/abi-spec-redirects.md`.
fn calling_convention_for(arch: BinArch, symbol: &str) -> CallingConvention {
    if let Some(cc) = speet_abi_stubs::calling_convention(arch, symbol) {
        return cc;
    }
    let bare = symbol.strip_prefix('_').unwrap_or(symbol);
    match (arch, bare) {
        (BinArch::X86_64, "execve") => CallingConvention {
            arg_locals: vec![7, 6, 2], // RDI, RSI, RDX
            arg_wrap_i32: vec![false, false, false],
            result_local: Some(0), // RAX
            result_extend_i32: true,
        },
        (BinArch::AArch64, "execve") => CallingConvention {
            arg_locals: vec![0, 1, 2], // x0, x1, x2
            arg_wrap_i32: vec![false, false, false],
            result_local: Some(0), // x0
            result_extend_i32: true,
        },
        _ => CallingConvention::default(),
    }
}
