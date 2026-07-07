//! Policy-wrapped host API with syscall and ambient whitelists.

use crate::{HostApi, ImportManifest, LinkRecipe};
use std::collections::BTreeSet;
use tunnel::TunnelResolution;

/// Per-binary host policy (subset of container `manifest.json` syscall lists).
#[derive(Debug, Clone, Default)]
pub struct HostPolicy {
    pub allowed_ambient: BTreeSet<String>,
    pub allowed_syscalls: BTreeSet<u64>,
}

impl HostPolicy {
    pub fn exit_write_only() -> Self {
        let mut syscalls = BTreeSet::new();
        syscalls.insert(93); // RV64 exit
        syscalls.insert(64); // RV64 write
        let mut ambient = BTreeSet::new();
        ambient.insert("write".into());
        ambient.insert("exit".into());
        ambient.insert("_exit".into());
        Self {
            allowed_ambient: ambient,
            allowed_syscalls: syscalls,
        }
    }

    /// Build from manifest-style string lists (syscall names are numeric strings).
    pub fn from_lists(syscall_numbers: &[u64], ambient_symbols: &[&str]) -> Self {
        Self {
            allowed_syscalls: syscall_numbers.iter().copied().collect(),
            allowed_ambient: ambient_symbols.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Wraps an inner [`HostApi`] and denies ambient symbols / syscalls not on the allowlist.
pub struct FilteredHostApi<H: HostApi> {
    inner: H,
    allowed_ambient: BTreeSet<String>,
    allowed_syscalls: BTreeSet<u64>,
}

impl<H: HostApi> FilteredHostApi<H> {
    pub fn new(inner: H, allowed_ambient: BTreeSet<String>, allowed_syscalls: BTreeSet<u64>) -> Self {
        Self {
            inner,
            allowed_ambient,
            allowed_syscalls,
        }
    }

    /// Minimal whitelist for `exit` + `write` on Linux RV64.
    pub fn exit_write_only(inner: H) -> Self {
        let p = HostPolicy::exit_write_only();
        Self::new(inner, p.allowed_ambient, p.allowed_syscalls)
    }

    pub fn from_policy(inner: H, policy: HostPolicy) -> Self {
        Self::new(inner, policy.allowed_ambient, policy.allowed_syscalls)
    }
}

impl<H: HostApi> HostApi for FilteredHostApi<H> {
    fn import_manifest(&self) -> ImportManifest {
        self.inner.import_manifest()
    }

    fn resolve_ambient(&self, guest_name: &str) -> Option<TunnelResolution> {
        let bare = guest_name
            .strip_prefix(tunnel::AMBIENT_PREFIX)
            .unwrap_or(guest_name);
        if !self.allowed_ambient.contains(bare) {
            return None;
        }
        self.inner.resolve_ambient(guest_name)
    }

    fn link_recipe(&self) -> LinkRecipe {
        let mut recipe = self.inner.link_recipe();
        recipe.ambient_aliases.retain(|(alias, real)| {
            let bare = alias
                .strip_prefix(tunnel::AMBIENT_PREFIX)
                .unwrap_or(alias.as_str());
            self.allowed_ambient.contains(bare) || self.allowed_ambient.contains(real)
        });
        recipe
    }

    fn resolve_plt_redirect(&self, guest_symbol: &str) -> Option<crate::PltRedirect> {
        self.inner.resolve_plt_redirect(guest_symbol)
    }

    fn syscall(&mut self, nr: u64, args: &[u64]) -> i64 {
        if !self.allowed_syscalls.contains(&nr) {
            return -1; // ENOSYS
        }
        self.inner.syscall(nr, args)
    }

    fn supports_ambient_linking(&self) -> bool {
        self.inner.supports_ambient_linking()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TunneledHostApi;

    #[test]
    fn denied_ambient_returns_none() {
        let api = FilteredHostApi::new(
            TunneledHostApi::for_host(),
            BTreeSet::from(["exit".into()]),
            BTreeSet::new(),
        );
        assert!(api.resolve_ambient("printf").is_none());
        assert!(api.resolve_ambient("exit").is_some());
    }

    #[test]
    fn denied_syscall_returns_enosys() {
        let mut api = FilteredHostApi::new(
            TunneledHostApi::for_host(),
            BTreeSet::new(),
            BTreeSet::from([93]),
        );
        assert_eq!(api.syscall(64, &[]), -1);
        assert_eq!(api.syscall(93, &[42]), -1); // inner default also ENOSYS
    }
}
