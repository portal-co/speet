//! [`PluginTransport`] — the extensibility seam for new host kinds, and
//! [`PluginHandle`] — the type-erased, resource-kind-tagged value it
//! produces.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use core::any::Any;

use speet_plugin_api::{
    AddressMapperPlugin, ArchPlugin, MemoryAccessPlugin, ObjectModelPlugin, PluginKind,
    TablePlugin, TargetPlugin,
};

/// A host-side failure to load/spawn/connect a plugin — distinct from
/// [`speet_plugin_api::PluginError`], which is a failure *reported by* an
/// already-loaded plugin during a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginLoadError(pub String);

impl PluginLoadError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl core::fmt::Display for PluginLoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "plugin load error: {}", self.0)
    }
}

/// A type-erased, resource-kind-tagged handle a [`PluginTransport`] produces.
///
/// Internally wraps an `Arc<dyn XPlugin>` for whichever kind this is — the
/// `Arc` itself (not the trait object) is the concrete, `Any`-compatible
/// type stored, so it can be downcast back out by [`PluginRegistry`].
pub struct PluginHandle {
    pub kind: PluginKind,
    inner: Box<dyn Any + Send + Sync>,
}

impl PluginHandle {
    pub fn arch(plugin: Arc<dyn ArchPlugin>) -> Self {
        Self {
            kind: PluginKind::Arch,
            inner: Box::new(plugin),
        }
    }

    pub fn address_mapper(plugin: Arc<dyn AddressMapperPlugin>) -> Self {
        Self {
            kind: PluginKind::Memory,
            inner: Box::new(plugin),
        }
    }

    pub fn memory_access(plugin: Arc<dyn MemoryAccessPlugin>) -> Self {
        Self {
            kind: PluginKind::Memory,
            inner: Box::new(plugin),
        }
    }

    pub fn table(plugin: Arc<dyn TablePlugin>) -> Self {
        Self {
            kind: PluginKind::Table,
            inner: Box::new(plugin),
        }
    }

    pub fn object_model(plugin: Arc<dyn ObjectModelPlugin>) -> Self {
        Self {
            kind: PluginKind::ObjectModel,
            inner: Box::new(plugin),
        }
    }

    pub fn target(plugin: Arc<dyn TargetPlugin>) -> Self {
        Self {
            kind: PluginKind::Target,
            inner: Box::new(plugin),
        }
    }

    pub fn as_arch(&self) -> Option<Arc<dyn ArchPlugin>> {
        self.inner.downcast_ref::<Arc<dyn ArchPlugin>>().cloned()
    }
    pub fn as_address_mapper(&self) -> Option<Arc<dyn AddressMapperPlugin>> {
        self.inner
            .downcast_ref::<Arc<dyn AddressMapperPlugin>>()
            .cloned()
    }
    pub fn as_memory_access(&self) -> Option<Arc<dyn MemoryAccessPlugin>> {
        self.inner
            .downcast_ref::<Arc<dyn MemoryAccessPlugin>>()
            .cloned()
    }
    pub fn as_table(&self) -> Option<Arc<dyn TablePlugin>> {
        self.inner.downcast_ref::<Arc<dyn TablePlugin>>().cloned()
    }
    pub fn as_object_model(&self) -> Option<Arc<dyn ObjectModelPlugin>> {
        self.inner
            .downcast_ref::<Arc<dyn ObjectModelPlugin>>()
            .cloned()
    }
    pub fn as_target(&self) -> Option<Arc<dyn TargetPlugin>> {
        self.inner.downcast_ref::<Arc<dyn TargetPlugin>>().cloned()
    }
}

/// A host backend. Implementors load, spawn, or connect to a plugin given a
/// host-specific descriptor (erased as `&dyn Any` here so this trait stays
/// object-safe — e.g. a `.wasm` path, a subprocess command+args, an
/// in-process type name) and produce a [`PluginHandle`].
///
/// A 4th host kind implements this trait in its own new crate and registers
/// via `PluginRegistry::register_transport` — no change to
/// `speet-plugin-api`, this trait, or `speet-plugin-adapter`.
pub trait PluginTransport: Send + Sync {
    fn load(&self, descriptor: &dyn Any) -> Result<PluginHandle, PluginLoadError>;
}
