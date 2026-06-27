//! Host entity imports — plugins calling back into host-resolved (built-in or
//! other-plugin) resources. See `docs/guides/plugin-api.md` §7.
//!
//! An import resolves to the same `Arc<dyn XPlugin>` handle already defined
//! for host→plugin calls — no new call shape, just a binding step. The
//! in-process host calls [`HostImports`] methods directly; the WASM host
//! realizes the same lookup as `Linker` imports; the subprocess host realizes
//! it as nested `plugin→host` frames. All three are restricted to an
//! embedder-chosen, manifest-declared allowlist — see
//! `PluginRegistry::restricted_view` in `speet-plugin-host`.

use alloc::sync::Arc;

use crate::arch::ArchPlugin;
use crate::memory::{AddressMapperPlugin, MemoryAccessPlugin};
use crate::object_model::ObjectModelPlugin;
use crate::table::TablePlugin;
use crate::target::TargetPlugin;

/// Named lookup of host-callable resources, handed to a plugin's
/// `bind_imports` so it can call back into a host-resolved (built-in or
/// other-plugin) entity. `PluginRegistry` is the canonical implementation.
///
/// `Send + Sync`: a remote host (WASM, subprocess) stores its granted view
/// behind an `Arc<dyn HostImports>` shared into engine/store state that
/// itself must be `Send + Sync` (every plugin trait in this crate is), so
/// this bound has to hold here too — every real implementation
/// (`PluginRegistry`, `RestrictedHostImports`, `NoImports`) already
/// satisfies it trivially.
pub trait HostImports: Send + Sync {
    fn address_mapper(&self, _name: &str) -> Option<Arc<dyn AddressMapperPlugin>> {
        None
    }
    fn memory_access(&self, _name: &str) -> Option<Arc<dyn MemoryAccessPlugin>> {
        None
    }
    fn table(&self, _name: &str) -> Option<Arc<dyn TablePlugin>> {
        None
    }
    fn object_model(&self, _name: &str) -> Option<Arc<dyn ObjectModelPlugin>> {
        None
    }
    fn target(&self, _name: &str) -> Option<Arc<dyn TargetPlugin>> {
        None
    }
    fn arch(&self, _name: &str) -> Option<Arc<dyn ArchPlugin>> {
        None
    }
}

/// A [`HostImports`] that grants nothing — the default for any plugin whose
/// manifest declares no imports.
pub struct NoImports;
impl HostImports for NoImports {}
