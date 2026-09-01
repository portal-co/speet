//! PLT call redirection during guest → WASM translation.

use binary_io::BinArch;
use speet_host_api::{HostApi, ImportManifest, PltRedirect};
use speet_plugin_api::external_target::{
    ExternalTargetTable, LibraryId, PltHook, PltHookTable, PltHookTarget,
};
use std::collections::BTreeMap;

/// Resolved redirect plan: guest `(library, address)` hooks plus per-symbol
/// metadata. Only [`PltHookTarget::WasmImport`] entries become PC-check hooks
/// in the interim path; [`PltHookTarget::Ambient`] is link-time only until
/// virtual-GOT shims land — see `docs/future/redirect-shim-got.md`.
#[derive(Debug, Clone, Default)]
pub struct PltCallPlan {
    pub targets: ExternalTargetTable,
    /// Guest addresses that resolve to a WASM-import redirect (PC-check).
    pub wasm_import_by_addr: BTreeMap<(LibraryId, u64), u32>,
    /// Guest addresses resolved to native `os_shim_*` shims (redirect shim slots).
    pub native_shim_by_addr: BTreeMap<(LibraryId, u64), String>,
    /// Symbols resolved to ambient link aliases (not PC-check emitted).
    pub ambient_labels: BTreeMap<String, ()>,
}

impl PltCallPlan {
    /// Resolve every hooked guest symbol against `host.import_manifest()`'s
    /// `index_of` — never a hand-maintained match table (principle 1).
    pub fn from_targets(targets: &ExternalTargetTable, host: &dyn HostApi) -> Self {
        let manifest = host.import_manifest();
        let mut wasm_import_by_addr = BTreeMap::new();
        let mut native_shim_by_addr = BTreeMap::new();
        let mut ambient_labels = BTreeMap::new();

        for entry in targets.iter() {
            let Some(redirect) = host.resolve_plt_redirect(&entry.label) else {
                continue;
            };
            match redirect {
                PltRedirect::WasmImport { module, name } => {
                    if let Some(idx) = manifest.index_of(&module, &name) {
                        wasm_import_by_addr.insert((entry.library, entry.address), idx);
                    }
                }
                PltRedirect::NativeShim { core_symbol } => {
                    native_shim_by_addr.insert((entry.library, entry.address), core_symbol);
                }
                PltRedirect::Ambient => {
                    ambient_labels.insert(entry.label.clone(), ());
                }
            }
        }

        Self {
            targets: targets.clone(),
            wasm_import_by_addr,
            native_shim_by_addr,
            ambient_labels,
        }
    }

    pub fn lookup_wasm_import(&self, library: LibraryId, target: u64) -> Option<u32> {
        self.wasm_import_by_addr.get(&(library, target)).copied()
    }

    /// Resolve a native `os_shim_*` core symbol to a WASM import index via manifest intercepts.
    pub fn import_idx_for_core_symbol(
        &self,
        manifest: &ImportManifest,
        core_symbol: &str,
    ) -> Option<u32> {
        let bare = core_symbol.strip_prefix('_').unwrap_or(core_symbol);
        for imp in &manifest.func_imports {
            if imp
                .intercepts
                .iter()
                .any(|s| s.strip_prefix('_').unwrap_or(s.as_str()) == bare)
            {
                return manifest.index_of(&imp.module, &imp.name);
            }
        }
        None
    }

    /// Realize WASM-import hooks as a [`PltHookTable`] for `arch`.
    ///
    /// `manifest` supplies WASM import param/result types for marshalling
    /// (checked-in ABI stubs take precedence — see `speet_abi_stubs`).
    pub fn to_hook_table(&self, arch: BinArch, manifest: &ImportManifest) -> PltHookTable {
        let mut table = PltHookTable::new();
        for (&(library, addr), &import_idx) in &self.wasm_import_by_addr {
            let label = self
                .targets
                .lookup(library, addr)
                .unwrap_or("?")
                .to_string();
            table.insert(
                library,
                addr,
                PltHook {
                    label: label.clone(),
                    target: PltHookTarget::WasmImport { import_idx },
                    convention: speet_abi_stubs::plt_calling_convention(manifest, arch, &label),
                },
            );
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_host_api::{
        HostApi, ImportManifest, PltRedirect, RedirectingHostApi, TunneledHostApi,
    };
    use tunnel::{DylibRef, HostAbi, TunnelResolution};

    #[test]
    fn execve_becomes_wasm_import_hook() {
        let manifest = ImportManifest::integrated_native();
        let execve_idx = manifest
            .index_of("env", "__speet_execve")
            .expect("execve import slot");
        let mut targets = ExternalTargetTable::new();
        targets.insert(LibraryId::MAIN_IMAGE, 0x3000, "execve");
        let host = RedirectingHostApi::integrated(
            TunneledHostApi::for_host().with_manifest(manifest.clone()),
        );
        let plan = PltCallPlan::from_targets(&targets, &host);
        assert_eq!(
            plan.wasm_import_by_addr
                .get(&(LibraryId::MAIN_IMAGE, 0x3000)),
            Some(&execve_idx)
        );
        let table = plan.to_hook_table(BinArch::AArch64, &manifest);
        let hook = table.lookup_main(0x3000).expect("hook");
        assert_eq!(hook.import_idx(), Some(execve_idx));
    }

    #[test]
    fn ambient_redirect_not_in_hook_table() {
        struct AmbientWriteHost;
        impl HostApi for AmbientWriteHost {
            fn import_manifest(&self) -> ImportManifest {
                ImportManifest::native_syscall()
            }
            fn resolve_ambient(&self, guest_name: &str) -> Option<TunnelResolution> {
                if guest_name == "write" || guest_name == "_write" {
                    Some(TunnelResolution {
                        host_symbol: "write".into(),
                        dylib: DylibRef::Lc,
                        abi: HostAbi::SysV,
                        variadic: false,
                        arg_count: Some(3),
                    })
                } else {
                    None
                }
            }
            fn link_recipe(&self) -> speet_host_api::LinkRecipe {
                speet_host_api::LinkRecipe {
                    arch: binary_io::BinArch::X86_64,
                    os: binary_io::BinOs::Linux,
                    dylib_flags: vec![],
                    ambient_aliases: vec![("write".into(), "write".into())],
                }
            }
            fn resolve_plt_redirect(&self, guest_symbol: &str) -> Option<PltRedirect> {
                let bare = guest_symbol.strip_prefix('_').unwrap_or(guest_symbol);
                if bare == "write" {
                    Some(PltRedirect::Ambient)
                } else {
                    None
                }
            }
            fn supports_ambient_linking(&self) -> bool {
                true
            }
        }

        let manifest = ImportManifest::native_syscall();
        let mut targets = ExternalTargetTable::new();
        targets.insert(LibraryId::MAIN_IMAGE, 0x4000, "write");
        let host = AmbientWriteHost;
        let plan = PltCallPlan::from_targets(&targets, &host);
        assert!(plan.wasm_import_by_addr.is_empty());
        assert!(plan.ambient_labels.contains_key("write"));
        assert!(plan.to_hook_table(BinArch::X86_64, &manifest).is_empty());
    }
}
