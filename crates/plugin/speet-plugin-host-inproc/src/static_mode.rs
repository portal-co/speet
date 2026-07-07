//! Static in-process registration: construct a plugin, optionally bind its
//! imports, then hand it to the registry. No transport, no encoding — a
//! real Rust trait object the whole way.

use std::sync::Arc;

use speet_plugin_api::{
    AddressMapperPlugin, ArchPlugin, MemoryAccessPlugin, ObjectModelPlugin, TablePlugin,
    TargetPlugin,
};
use speet_plugin_host::{PluginHandle, PluginRegistry};

pub fn register_arch(registry: &mut PluginRegistry, name: &str, plugin: impl ArchPlugin + 'static) {
    registry.register_static(name, PluginHandle::arch(Arc::new(plugin)));
}

pub fn register_address_mapper(
    registry: &mut PluginRegistry,
    name: &str,
    plugin: impl AddressMapperPlugin + 'static,
) {
    registry.register_static(name, PluginHandle::address_mapper(Arc::new(plugin)));
}

pub fn register_memory_access(
    registry: &mut PluginRegistry,
    name: &str,
    plugin: impl MemoryAccessPlugin + 'static,
) {
    registry.register_static(name, PluginHandle::memory_access(Arc::new(plugin)));
}

pub fn register_table(registry: &mut PluginRegistry, name: &str, plugin: impl TablePlugin + 'static) {
    registry.register_static(name, PluginHandle::table(Arc::new(plugin)));
}

pub fn register_object_model(
    registry: &mut PluginRegistry,
    name: &str,
    plugin: impl ObjectModelPlugin + 'static,
) {
    registry.register_static(name, PluginHandle::object_model(Arc::new(plugin)));
}

pub fn register_target(
    registry: &mut PluginRegistry,
    name: &str,
    plugin: impl TargetPlugin + 'static,
) {
    registry.register_static(name, PluginHandle::target(Arc::new(plugin)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_plugin_api::{
        snippet::CodeSnippet, target::ModuleManifest, target::PluginSyscallTable, HostImports,
        PResult,
    };
    use std::sync::Mutex;

    struct ToyTarget;
    impl TargetPlugin for ToyTarget {
        fn module_manifest(&self) -> ModuleManifest {
            ModuleManifest::default()
        }
        fn syscall_table(&self) -> PluginSyscallTable {
            PluginSyscallTable::default()
        }
    }

    /// An `ArchPlugin` that imports a host-granted memory entity and uses it
    /// inside its own decode step — the in-process host-entity-import case.
    /// `Mutex` provides the interior mutability `bind_imports(&self)` and
    /// `step(&self)` require while keeping the plugin `Send + Sync`.
    struct ImportingArch {
        memory: Mutex<Option<Arc<dyn AddressMapperPlugin>>>,
    }
    impl ArchPlugin for ImportingArch {
        fn reset_for_next_binary(&self, _args: &[u8]) {}
        fn count_fns(&self, _bytes: &[u8]) -> u32 {
            1
        }
        fn step(&self, _feedback: Option<&[u8]>) -> PResult<speet_plugin_api::arch::ArchOp> {
            let guard = self.memory.lock().unwrap();
            let mem = guard.as_ref().expect("import bound");
            let mut snippet = CodeSnippet::empty();
            let translated = mem.translate(0).expect("translate");
            snippet.extend(&translated);
            Ok(speet_plugin_api::arch::ArchOp::Feed { snippet })
        }
        fn declare_params(&self) -> Vec<speet_plugin_api::snippet::PluginValType> {
            Vec::new()
        }
        fn bind_imports(&self, imports: &dyn HostImports) {
            *self.memory.lock().unwrap() = imports.address_mapper("host-mmu");
        }
    }

    struct ToyMemory;
    impl AddressMapperPlugin for ToyMemory {
        fn translate(&self, addr_local: u32) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::from_instructions(&[
                wasm_encoder::Instruction::LocalGet(addr_local),
            ]))
        }
    }

    #[test]
    fn register_and_lookup_target() {
        let mut registry = PluginRegistry::new();
        register_target(&mut registry, "toy", ToyTarget);
        assert!(registry.target("toy").is_some());
    }

    #[test]
    fn arch_plugin_imports_host_memory() {
        let mut registry = PluginRegistry::new();
        register_address_mapper(&mut registry, "host-mmu", ToyMemory);

        // Bind imports while only an immutable borrow of `registry` is live,
        // via the restricted view; `plugin` itself is still exclusively owned
        // here, so this doesn't conflict with registering it afterward.
        let plugin = ImportingArch {
            memory: Mutex::new(None),
        };
        let granted = [(
            speet_plugin_api::PluginKind::Memory,
            "host-mmu".to_string(),
        )];
        plugin.bind_imports(&registry.restricted_view(&granted));

        register_arch(&mut registry, "importing-arch", plugin);

        // Exercise the registered plugin directly through the shared handle —
        // `step` only needs `&self`, so the `Arc<dyn ArchPlugin>` the registry
        // hands back is directly callable without any clone-free workaround.
        let arch = registry.arch("importing-arch").unwrap();
        let op = arch.step(None).unwrap();
        match op {
            speet_plugin_api::arch::ArchOp::Feed { snippet } => {
                assert!(!snippet.wasm.is_empty());
            }
            _ => panic!("expected Feed"),
        }
    }
}
