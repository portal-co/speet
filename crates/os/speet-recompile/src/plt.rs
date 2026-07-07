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
/// Only `execve`/`exit` are wired today (matching
/// `ImportManifest::native_syscall`'s `env.exit` import, `(i32) -> ()`, and
/// `integrated_native`'s extra `env.__speet_execve`, `(i64,i64,i64) -> i32`).
/// This is deliberately a small, explicit table rather than a general ABI
/// decoder — see `docs/future/abi-spec-redirects.md` for the generalized
/// path (ABI-spec ingestion + generated stubs) that replaces hand-written
/// entries like these, without changing the
/// [`PltHookTable`]/[`CallingConvention`] contract those generated stubs
/// also target. Adding a new manifest `intercepts` entry (principle 1) is
/// necessary but not sufficient for a new symbol to actually redirect
/// correctly — a matching arm here is also required until abi-spec codegen
/// replaces this table.
///
/// `arg_locals`/`result_local` are interpreted per-arch by the consuming
/// recompiler: `speet-x86_64` treats them as literal WASM local indices
/// (its GPRs are fixed WASM locals 0–15); `speet-aarch64` treats them as
/// architectural register numbers, resolved to the actual WASM local via
/// its own dynamic layout at emission time.
fn calling_convention_for(arch: BinArch, symbol: &str) -> CallingConvention {
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
        (BinArch::X86_64, "exit" | "Exit") => CallingConvention {
            arg_locals: vec![7], // RDI: status
            arg_wrap_i32: vec![true], // env.exit wants i32, RDI holds i64
            result_local: None, // env.exit never returns
            result_extend_i32: false,
        },
        (BinArch::AArch64, "exit" | "Exit") => CallingConvention {
            arg_locals: vec![0], // x0: status
            arg_wrap_i32: vec![true],
            result_local: None,
            result_extend_i32: false,
        },
        _ => CallingConvention::default(),
    }
}
