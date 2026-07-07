//! [`DylibTransport`] — the `PluginTransport` impl for dylib guests.

use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;

use speet_plugin_api::remote::ImportRole;
use speet_plugin_api::HostImports;
use speet_plugin_host::{PluginHandle, PluginLoadError, PluginTransport};

use super::adapters::{
    DylibAddressMapperPlugin, DylibArchPlugin, DylibMemoryAccessPlugin, DylibObjectModelPlugin,
    DylibTablePlugin, DylibTargetPlugin,
};
use super::engine::DylibPlugin;

/// Descriptor for [`PluginTransport::load`]: which of the six roles this
/// dylib implements, the path to the `.so`/`.dylib`/`.dll`, and the
/// (already-restricted) view of host entities it may import (§2.7) — pass
/// `Arc::new(speet_plugin_api::imports::NoImports)` for a plugin with no
/// imports.
pub struct DylibDescriptor {
    pub role: ImportRole,
    pub path: PathBuf,
    pub imports: Arc<dyn HostImports>,
}

pub struct DylibTransport;

impl PluginTransport for DylibTransport {
    fn load(&self, descriptor: &dyn Any) -> Result<PluginHandle, PluginLoadError> {
        let desc = descriptor
            .downcast_ref::<DylibDescriptor>()
            .ok_or_else(|| PluginLoadError::new("speet-plugin-host-inproc dylib mode expected a DylibDescriptor"))?;
        let engine = DylibPlugin::load(&desc.path, desc.role, desc.imports.clone())
            .map_err(|e| PluginLoadError::new(e.to_string()))?;
        Ok(match desc.role {
            ImportRole::Arch => PluginHandle::arch(Arc::new(DylibArchPlugin::new(engine))),
            ImportRole::AddressMapper => {
                PluginHandle::address_mapper(Arc::new(DylibAddressMapperPlugin::new(engine)))
            }
            ImportRole::MemoryAccess => {
                PluginHandle::memory_access(Arc::new(DylibMemoryAccessPlugin::new(engine)))
            }
            ImportRole::Table => PluginHandle::table(Arc::new(DylibTablePlugin::new(engine))),
            ImportRole::ObjectModel => {
                PluginHandle::object_model(Arc::new(DylibObjectModelPlugin::new(engine)))
            }
            ImportRole::Target => PluginHandle::target(Arc::new(DylibTargetPlugin::new(engine))),
        })
    }
}
