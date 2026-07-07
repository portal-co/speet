//! Tunnelled-by-default host API.

use crate::{HostApi, ImportManifest, LinkRecipe};
use binary_io::{BinArch, BinOs};
use tunnel::{dylib_link_flag, Tunnel, TunnelResolution};

/// Resolves guest externals through a [`Tunnel`] allowlist.
pub struct TunneledHostApi {
    tunnel: Box<dyn Tunnel + Send + Sync>,
    arch: BinArch,
    os: BinOs,
    manifest: ImportManifest,
}

impl TunneledHostApi {
    pub fn new(tunnel: Box<dyn Tunnel + Send + Sync>, arch: BinArch, os: BinOs) -> Self {
        Self {
            tunnel,
            arch,
            os,
            manifest: ImportManifest::native_syscall(),
        }
    }

    pub fn for_host() -> Self {
        let (arch, os) = crate::host_link_target();
        Self::new(crate::default_tunnel(), arch, os)
    }

    pub fn with_manifest(mut self, manifest: ImportManifest) -> Self {
        self.manifest = manifest;
        self
    }
}

impl HostApi for TunneledHostApi {
    fn import_manifest(&self) -> ImportManifest {
        self.manifest.clone()
    }

    fn resolve_ambient(&self, guest_name: &str) -> Option<TunnelResolution> {
        self.tunnel.resolve(guest_name)
    }

    fn link_recipe(&self) -> LinkRecipe {
        LinkRecipe {
            arch: self.arch,
            os: self.os,
            dylib_flags: self
                .tunnel
                .required_dylibs()
                .iter()
                .map(dylib_link_flag)
                .collect(),
            ambient_aliases: self.tunnel.aliases(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_resolves_write() {
        let api = TunneledHostApi::for_host();
        let r = api.resolve_ambient("write").expect("write in allowlist");
        assert_eq!(r.host_symbol, "write");
        assert!(!api.link_recipe().dylib_flags.is_empty());
    }
}
