//! Compile-time PLT redirection for integrated runtime hooks.

use crate::{HostApi, ImportManifest, LinkRecipe};
use std::collections::BTreeMap;
use tunnel::TunnelResolution;

/// Where a guest PLT/external call is redirected at compile time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PltRedirect {
    /// Lower to a WASM import call; the link shim implements the symbol.
    WasmImport { module: String, name: String },
}

/// Wraps an inner [`HostApi`] and redirects selected guest externals to
/// integrated-runtime hooks instead of tunnel ambient aliases.
pub struct RedirectingHostApi<H: HostApi> {
    inner: H,
    hooks: BTreeMap<String, PltRedirect>,
}

impl<H: HostApi> RedirectingHostApi<H> {
    pub fn new(inner: H, hooks: BTreeMap<String, PltRedirect>) -> Self {
        Self { inner, hooks }
    }

    /// Integrated thin-runtime hooks (`execve` → `env.__speet_execve`).
    pub fn integrated(inner: H) -> Self {
        let mut hooks = BTreeMap::new();
        for sym in ["execve", "_execve"] {
            hooks.insert(
                sym.into(),
                PltRedirect::WasmImport {
                    module: "env".into(),
                    name: "__speet_execve".into(),
                },
            );
        }
        Self::new(inner, hooks)
    }

    fn is_hooked(&self, guest_symbol: &str) -> bool {
        let bare = guest_symbol.strip_prefix('_').unwrap_or(guest_symbol);
        self.hooks.contains_key(guest_symbol) || self.hooks.contains_key(bare)
    }
}

impl<H: HostApi> HostApi for RedirectingHostApi<H> {
    fn import_manifest(&self) -> ImportManifest {
        self.inner.import_manifest()
    }

    fn resolve_ambient(&self, guest_name: &str) -> Option<TunnelResolution> {
        self.inner.resolve_ambient(guest_name)
    }

    fn resolve_plt_redirect(&self, guest_symbol: &str) -> Option<PltRedirect> {
        if let Some(r) = self.hooks.get(guest_symbol) {
            return Some(r.clone());
        }
        let bare = guest_symbol.strip_prefix('_').unwrap_or(guest_symbol);
        self.hooks.get(bare).cloned().or_else(|| self.inner.resolve_plt_redirect(guest_symbol))
    }

    fn link_recipe(&self) -> LinkRecipe {
        let mut recipe = self.inner.link_recipe();
        recipe
            .ambient_aliases
            .retain(|(alias, _)| !self.is_hooked(alias));
        recipe
    }

    fn syscall(&mut self, nr: u64, args: &[u64]) -> i64 {
        self.inner.syscall(nr, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TunneledHostApi;

    #[test]
    fn execve_redirects_to_wasm_import() {
        let api = RedirectingHostApi::integrated(
            TunneledHostApi::for_host().with_manifest(ImportManifest::integrated_native()),
        );
        let r = api
            .resolve_plt_redirect("execve")
            .expect("execve hook");
        assert_eq!(
            r,
            PltRedirect::WasmImport {
                module: "env".into(),
                name: "__speet_execve".into(),
            }
        );
    }

    #[test]
    fn hooked_symbols_drop_ambient_aliases() {
        let inner = TunneledHostApi::for_host();
        let inner_aliases = inner.link_recipe().ambient_aliases.len();
        let api = RedirectingHostApi::integrated(inner);
        let recipe = api.link_recipe();
        if inner_aliases > 0 {
            assert!(
                recipe.ambient_aliases.len() <= inner_aliases,
                "hooked aliases should be removed"
            );
        }
        assert!(
            !recipe
                .ambient_aliases
                .iter()
                .any(|(a, _)| a == "execve" || a == "_execve")
        );
    }
}
