//! End-to-end tests against real (non-Rust, Python) subprocess plugins —
//! demonstrating the subprocess host's language-neutrality, per the
//! verification plan's headline three-host cross-check. Skips gracefully if
//! `python3` isn't on `PATH` rather than failing the whole suite.

use std::sync::Arc;

use speet_plugin_api::memory::AddressMapperPlugin;
use speet_plugin_api::snippet::{CodeSnippet, PluginValType};
use speet_plugin_api::target::{FuncImportDecl, ModuleManifest, PluginSyscallTable, TargetPlugin};
use speet_plugin_api::{HostImports, PResult, PluginError, PluginKind};
use speet_plugin_host::PluginRegistry;
use speet_plugin_host_subprocess::{SubprocessAddressMapperPlugin, SubprocessPlugin, SubprocessTargetPlugin};
use wasm_encoder::Instruction;

fn python3_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_ok()
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn sample_manifest() -> ModuleManifest {
    ModuleManifest {
        func_imports: vec![
            FuncImportDecl {
                module: "wasi_snapshot_preview1".into(),
                field: "proc_exit".into(),
                params: vec![PluginValType::I32],
                results: vec![],
            },
            FuncImportDecl {
                module: "wasi_snapshot_preview1".into(),
                field: "fd_write".into(),
                params: vec![
                    PluginValType::I32,
                    PluginValType::I32,
                    PluginValType::I32,
                    PluginValType::I32,
                ],
                results: vec![PluginValType::I32],
            },
        ],
        ..Default::default()
    }
}

#[test]
fn target_plugin_constant_data_roundtrip() {
    if !python3_available() {
        eprintln!("skipping: python3 not found on PATH");
        return;
    }
    let process = SubprocessPlugin::spawn(
        "python3",
        &[fixture("target_plugin.py")],
        Arc::new(speet_plugin_api::imports::NoImports),
    )
    .expect("spawn");
    let plugin = SubprocessTargetPlugin::new(process);

    assert_eq!(plugin.module_manifest(), sample_manifest());
    assert_eq!(plugin.syscall_table(), PluginSyscallTable::default());
}

#[test]
fn address_mapper_translate_uses_dynamic_argument() {
    if !python3_available() {
        eprintln!("skipping: python3 not found on PATH");
        return;
    }
    let process = SubprocessPlugin::spawn(
        "python3",
        &[fixture("address_mapper_translate.py")],
        Arc::new(speet_plugin_api::imports::NoImports),
    )
    .expect("spawn");
    let plugin = SubprocessAddressMapperPlugin::new(process);

    let snippet = plugin.translate(7).expect("translate");
    assert_eq!(
        snippet,
        CodeSnippet::from_instructions(&[Instruction::LocalGet(7)])
    );
}

struct ToyMemory;
impl AddressMapperPlugin for ToyMemory {
    fn translate(&self, addr_local: u32) -> PResult<CodeSnippet> {
        Ok(CodeSnippet::from_instructions(&[Instruction::LocalGet(addr_local)]))
    }
}

struct OwnedRestrictedView {
    registry: PluginRegistry,
    granted: Vec<(PluginKind, String)>,
}
impl HostImports for OwnedRestrictedView {
    fn address_mapper(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::memory::AddressMapperPlugin>> {
        self.registry.restricted_view(&self.granted).address_mapper(name)
    }
    fn memory_access(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::memory::MemoryAccessPlugin>> {
        self.registry.restricted_view(&self.granted).memory_access(name)
    }
    fn table(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::table::TablePlugin>> {
        self.registry.restricted_view(&self.granted).table(name)
    }
    fn object_model(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::object_model::ObjectModelPlugin>> {
        self.registry.restricted_view(&self.granted).object_model(name)
    }
    fn target(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::target::TargetPlugin>> {
        self.registry.restricted_view(&self.granted).target(name)
    }
    fn arch(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::arch::ArchPlugin>> {
        self.registry.restricted_view(&self.granted).arch(name)
    }
}

fn owned_restricted_view(registry: PluginRegistry, granted: &[(PluginKind, String)]) -> OwnedRestrictedView {
    OwnedRestrictedView {
        registry,
        granted: granted.to_vec(),
    }
}

fn registry_with_host_mmu() -> PluginRegistry {
    let mut registry = PluginRegistry::new();
    registry.register_static(
        "host-mmu",
        speet_plugin_host::PluginHandle::address_mapper(Arc::new(ToyMemory)),
    );
    registry
}

#[test]
fn host_entity_import_granted_forwards_to_host_plugin() {
    if !python3_available() {
        eprintln!("skipping: python3 not found on PATH");
        return;
    }
    let granted = [(PluginKind::Memory, "host-mmu".to_string())];
    let imports: Arc<dyn HostImports> =
        Arc::new(owned_restricted_view(registry_with_host_mmu(), &granted));

    let process = SubprocessPlugin::spawn("python3", &[fixture("address_mapper_forward.py")], imports)
        .expect("spawn");
    let plugin = SubprocessAddressMapperPlugin::new(process);

    let snippet = plugin.translate(9).expect("translate via import");
    assert_eq!(
        snippet,
        CodeSnippet::from_instructions(&[Instruction::LocalGet(9)])
    );
}

#[test]
fn host_entity_import_denied_when_not_granted() {
    if !python3_available() {
        eprintln!("skipping: python3 not found on PATH");
        return;
    }
    let granted: [(PluginKind, String); 0] = [];
    let imports: Arc<dyn HostImports> =
        Arc::new(owned_restricted_view(registry_with_host_mmu(), &granted));

    let process = SubprocessPlugin::spawn("python3", &[fixture("address_mapper_forward.py")], imports)
        .expect("spawn");
    let plugin = SubprocessAddressMapperPlugin::new(process);

    let err = plugin.translate(9).expect_err("import must be denied");
    assert_eq!(err, PluginError::new(1, "import denied"));
}
