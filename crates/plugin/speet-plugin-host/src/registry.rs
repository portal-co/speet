//! [`PluginRegistry`] — the single by-name lookup table for every loaded
//! plugin, regardless of which transport produced it. Doubles as the
//! resolution table for host-entity imports (restricted via
//! [`RestrictedHostImports`]).

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use core::any::Any;

use speet_plugin_api::{
    AddressMapperPlugin, ArchPlugin, HostImports, MemoryAccessPlugin, ObjectModelPlugin,
    PluginKind, TablePlugin, TargetPlugin,
};

use crate::transport::{PluginHandle, PluginLoadError, PluginTransport};

/// The single by-name lookup table for every loaded plugin. An embedder
/// registers the transports it wants, loads named instances, and reads them
/// back out via the typed accessors below — or hands `&registry` (or a
/// [`RestrictedHostImports`] view of it) to another plugin as its
/// [`HostImports`].
#[derive(Default)]
pub struct PluginRegistry {
    plugins: BTreeMap<String, PluginHandle>,
    transports: BTreeMap<String, Box<dyn PluginTransport>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_transport(&mut self, scheme: &str, transport: Box<dyn PluginTransport>) {
        self.transports.insert(scheme.to_string(), transport);
    }

    /// Load a plugin named `name` via the transport registered under
    /// `scheme`, using `descriptor` (a transport-specific value, e.g. a
    /// `PathBuf` for the WASM/dylib hosts or a command line for the
    /// subprocess host).
    pub fn load(
        &mut self,
        name: &str,
        scheme: &str,
        descriptor: &dyn Any,
    ) -> Result<(), PluginLoadError> {
        let transport = self
            .transports
            .get(scheme)
            .ok_or_else(|| PluginLoadError::new(alloc::format!("unknown host scheme: {scheme}")))?;
        let handle = transport.load(descriptor)?;
        self.plugins.insert(name.to_string(), handle);
        Ok(())
    }

    /// Register an already-constructed handle directly, bypassing
    /// `PluginTransport::load` entirely — the in-process static-mode bypass
    /// (no descriptor-based indirection needed for code already linked into
    /// the same binary).
    pub fn register_static(&mut self, name: &str, handle: PluginHandle) {
        self.plugins.insert(name.to_string(), handle);
    }

    pub fn kind_of(&self, name: &str) -> Option<PluginKind> {
        self.plugins.get(name).map(|h| h.kind)
    }

    pub fn arch(&self, name: &str) -> Option<Arc<dyn ArchPlugin>> {
        self.plugins.get(name)?.as_arch()
    }
    pub fn address_mapper(&self, name: &str) -> Option<Arc<dyn AddressMapperPlugin>> {
        self.plugins.get(name)?.as_address_mapper()
    }
    pub fn memory_access(&self, name: &str) -> Option<Arc<dyn MemoryAccessPlugin>> {
        self.plugins.get(name)?.as_memory_access()
    }
    pub fn table(&self, name: &str) -> Option<Arc<dyn TablePlugin>> {
        self.plugins.get(name)?.as_table()
    }
    pub fn object_model(&self, name: &str) -> Option<Arc<dyn ObjectModelPlugin>> {
        self.plugins.get(name)?.as_object_model()
    }
    pub fn target(&self, name: &str) -> Option<Arc<dyn TargetPlugin>> {
        self.plugins.get(name)?.as_target()
    }

    /// A [`HostImports`] view restricted to exactly the `(kind, name)` pairs
    /// in `granted` — the manifest-declared, embedder-approved allowlist a
    /// loaded plugin's imports are bound against. Never hand a plugin the
    /// unrestricted registry directly; this is the load-bearing trust
    /// boundary for the WASM (and subprocess) hosts. See
    /// `docs/guides/plugin-api.md` §7.
    pub fn restricted_view<'a>(
        &'a self,
        granted: &'a [(PluginKind, String)],
    ) -> RestrictedHostImports<'a> {
        RestrictedHostImports {
            registry: self,
            granted,
        }
    }
}

impl HostImports for PluginRegistry {
    fn address_mapper(&self, name: &str) -> Option<Arc<dyn AddressMapperPlugin>> {
        self.address_mapper(name)
    }
    fn memory_access(&self, name: &str) -> Option<Arc<dyn MemoryAccessPlugin>> {
        self.memory_access(name)
    }
    fn table(&self, name: &str) -> Option<Arc<dyn TablePlugin>> {
        self.table(name)
    }
    fn object_model(&self, name: &str) -> Option<Arc<dyn ObjectModelPlugin>> {
        self.object_model(name)
    }
    fn target(&self, name: &str) -> Option<Arc<dyn TargetPlugin>> {
        self.target(name)
    }
    fn arch(&self, name: &str) -> Option<Arc<dyn ArchPlugin>> {
        self.arch(name)
    }
}

/// A [`HostImports`] view restricted to an explicit, embedder-chosen
/// allowlist of `(kind, name)` pairs — see
/// [`PluginRegistry::restricted_view`].
pub struct RestrictedHostImports<'a> {
    registry: &'a PluginRegistry,
    granted: &'a [(PluginKind, String)],
}

impl<'a> RestrictedHostImports<'a> {
    fn is_granted(&self, kind: PluginKind, name: &str) -> bool {
        self.granted
            .iter()
            .any(|(k, n)| *k == kind && n == name)
    }
}

impl<'a> HostImports for RestrictedHostImports<'a> {
    fn address_mapper(&self, name: &str) -> Option<Arc<dyn AddressMapperPlugin>> {
        self.is_granted(PluginKind::Memory, name)
            .then(|| self.registry.address_mapper(name))
            .flatten()
    }
    fn memory_access(&self, name: &str) -> Option<Arc<dyn MemoryAccessPlugin>> {
        self.is_granted(PluginKind::Memory, name)
            .then(|| self.registry.memory_access(name))
            .flatten()
    }
    fn table(&self, name: &str) -> Option<Arc<dyn TablePlugin>> {
        self.is_granted(PluginKind::Table, name)
            .then(|| self.registry.table(name))
            .flatten()
    }
    fn object_model(&self, name: &str) -> Option<Arc<dyn ObjectModelPlugin>> {
        self.is_granted(PluginKind::ObjectModel, name)
            .then(|| self.registry.object_model(name))
            .flatten()
    }
    fn target(&self, name: &str) -> Option<Arc<dyn TargetPlugin>> {
        self.is_granted(PluginKind::Target, name)
            .then(|| self.registry.target(name))
            .flatten()
    }
    fn arch(&self, name: &str) -> Option<Arc<dyn ArchPlugin>> {
        self.is_granted(PluginKind::Arch, name)
            .then(|| self.registry.arch(name))
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_plugin_api::{CodeSnippet, PResult, TargetPlugin};
    use speet_plugin_api::target::{ModuleManifest, PluginSyscallTable};

    struct ToyTarget;
    impl TargetPlugin for ToyTarget {
        fn module_manifest(&self) -> ModuleManifest {
            ModuleManifest::default()
        }
        fn syscall_table(&self) -> PluginSyscallTable {
            PluginSyscallTable::default()
        }
    }

    struct DenyAllMemory;
    impl AddressMapperPlugin for DenyAllMemory {
        fn translate(&self, _addr_local: u32) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::empty())
        }
    }

    #[test]
    fn register_static_and_lookup() {
        let mut registry = PluginRegistry::new();
        registry.register_static("toy", PluginHandle::target(Arc::new(ToyTarget)));
        assert!(registry.target("toy").is_some());
        assert!(registry.arch("toy").is_none());
        assert_eq!(registry.kind_of("toy"), Some(PluginKind::Target));
        assert!(registry.target("missing").is_none());
    }

    #[test]
    fn restricted_view_enforces_allowlist() {
        let mut registry = PluginRegistry::new();
        registry.register_static(
            "host-mmu",
            PluginHandle::address_mapper(Arc::new(DenyAllMemory)),
        );
        let granted = [(PluginKind::Memory, "host-mmu".to_string())];
        let view = registry.restricted_view(&granted);
        assert!(view.address_mapper("host-mmu").is_some());
        // Not in the allowlist under this name:
        assert!(view.address_mapper("other").is_none());

        let empty: [(PluginKind, String); 0] = [];
        let locked_down = registry.restricted_view(&empty);
        assert!(locked_down.address_mapper("host-mmu").is_none());
    }
}
