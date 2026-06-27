//! [`WasmTransport`] — the `PluginTransport` impl for `.wasm` guests.

use std::any::Any;
use std::sync::Arc;

use speet_plugin_api::HostImports;
use speet_plugin_host::{PluginHandle, PluginLoadError, PluginTransport};

use crate::abi::ImportRole;
use crate::adapters::{
    WasmAddressMapperPlugin, WasmArchPlugin, WasmMemoryAccessPlugin, WasmObjectModelPlugin,
    WasmTablePlugin, WasmTargetPlugin,
};
use crate::engine::WasmPlugin;

/// Descriptor for [`PluginTransport::load`]: which of the six remote-callable
/// roles this guest implements ([`ImportRole`], finer-grained than
/// `speet_plugin_api::PluginKind` — see `crate::abi` docs), the guest bytes,
/// and the (already-restricted) view of host entities this guest may import
/// (§2.7) — typically a `speet_plugin_host::RestrictedHostImports` built from
/// the plugin's manifest-declared `imports` list. Pass
/// `Arc::new(speet_plugin_api::imports::NoImports)` for a guest with no
/// imports.
pub struct WasmDescriptor<'a> {
    pub role: ImportRole,
    pub wasm_bytes: &'a [u8],
    pub imports: Arc<dyn HostImports>,
}

pub struct WasmTransport;

impl PluginTransport for WasmTransport {
    fn load(&self, descriptor: &dyn Any) -> Result<PluginHandle, PluginLoadError> {
        let desc = descriptor
            .downcast_ref::<WasmDescriptor>()
            .ok_or_else(|| PluginLoadError::new("speet-plugin-host-wasm expected a WasmDescriptor"))?;
        let engine = WasmPlugin::load(desc.wasm_bytes, desc.imports.clone())
            .map_err(|e| PluginLoadError::new(e.to_string()))?;
        Ok(match desc.role {
            ImportRole::Arch => PluginHandle::arch(Arc::new(WasmArchPlugin::new(engine))),
            ImportRole::AddressMapper => {
                PluginHandle::address_mapper(Arc::new(WasmAddressMapperPlugin::new(engine)))
            }
            ImportRole::MemoryAccess => {
                PluginHandle::memory_access(Arc::new(WasmMemoryAccessPlugin::new(engine)))
            }
            ImportRole::Table => PluginHandle::table(Arc::new(WasmTablePlugin::new(engine))),
            ImportRole::ObjectModel => {
                PluginHandle::object_model(Arc::new(WasmObjectModelPlugin::new(engine)))
            }
            ImportRole::Target => PluginHandle::target(Arc::new(WasmTargetPlugin::new(engine))),
        })
    }
}
