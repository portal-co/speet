//! [`SubprocessTransport`] — the `PluginTransport` impl for plugins
//! implemented as a separate OS process.

use std::any::Any;
use std::sync::Arc;

use speet_plugin_api::remote::ImportRole;
use speet_plugin_api::HostImports;
use speet_plugin_host::{PluginHandle, PluginLoadError, PluginTransport};

use crate::adapters::{
    SubprocessAddressMapperPlugin, SubprocessArchPlugin, SubprocessMemoryAccessPlugin,
    SubprocessObjectModelPlugin, SubprocessTablePlugin, SubprocessTargetPlugin,
};
use crate::process::SubprocessPlugin;

/// Descriptor for [`PluginTransport::load`]: which of the six remote-callable
/// roles this plugin implements (see `speet_plugin_api::remote::ImportRole`),
/// the command + args to spawn it, and the (already-restricted) view of host
/// entities it may import (§2.7). Pass
/// `Arc::new(speet_plugin_api::imports::NoImports)` for a plugin with no
/// imports.
pub struct SubprocessDescriptor {
    pub role: ImportRole,
    pub command: String,
    pub args: Vec<String>,
    pub imports: Arc<dyn HostImports>,
}

pub struct SubprocessTransport;

impl PluginTransport for SubprocessTransport {
    fn load(&self, descriptor: &dyn Any) -> Result<PluginHandle, PluginLoadError> {
        let desc = descriptor.downcast_ref::<SubprocessDescriptor>().ok_or_else(|| {
            PluginLoadError::new("speet-plugin-host-subprocess expected a SubprocessDescriptor")
        })?;
        let process = SubprocessPlugin::spawn(&desc.command, &desc.args, desc.imports.clone())
            .map_err(|e| PluginLoadError::new(e.to_string()))?;
        Ok(match desc.role {
            ImportRole::Arch => PluginHandle::arch(Arc::new(SubprocessArchPlugin::new(process))),
            ImportRole::AddressMapper => {
                PluginHandle::address_mapper(Arc::new(SubprocessAddressMapperPlugin::new(process)))
            }
            ImportRole::MemoryAccess => {
                PluginHandle::memory_access(Arc::new(SubprocessMemoryAccessPlugin::new(process)))
            }
            ImportRole::Table => PluginHandle::table(Arc::new(SubprocessTablePlugin::new(process))),
            ImportRole::ObjectModel => {
                PluginHandle::object_model(Arc::new(SubprocessObjectModelPlugin::new(process)))
            }
            ImportRole::Target => PluginHandle::target(Arc::new(SubprocessTargetPlugin::new(process))),
        })
    }
}
