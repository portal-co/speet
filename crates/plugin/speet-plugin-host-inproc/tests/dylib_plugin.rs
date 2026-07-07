//! End-to-end tests against real, independently-compiled `cdylib` fixtures
//! (built on the fly with a bare `rustc` invocation — no shared crate
//! dependency between the test and the fixture, proving the `extern "C"`
//! ABI is genuinely self-describing, not an accidental Rust-ABI match).
//! Mirrors `speet-plugin-host-wasm`'s `wasm_plugin.rs` test shape.

#![cfg(feature = "dylib")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use speet_plugin_api::imports::NoImports;
use speet_plugin_api::memory::AddressMapperPlugin;
use speet_plugin_api::remote::ImportRole;
use speet_plugin_api::snippet::CodeSnippet;
use speet_plugin_api::target::TargetPlugin;
use speet_plugin_api::{HostImports, PResult, PluginError, PluginKind};
use speet_plugin_host::PluginRegistry;
use speet_plugin_host_inproc::dylib::{DylibAddressMapperPlugin, DylibPlugin, DylibTargetPlugin};
use wasm_encoder::Instruction;

/// Boilerplate every fixture source shares: the ABI types, independently
/// redefined (not `include!`d from the host crate) — that duplication *is*
/// the point, it's what a real third-party plugin author would write.
const FFI_PRELUDE: &str = r#"
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PluginBuffer { pub ptr: *mut u8, pub len: usize }

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HostImportsFfi {
    pub ctx: *mut c_void,
    pub call: extern "C" fn(*mut c_void, u8, *const u8, usize, *const u8, usize) -> PluginBuffer,
    pub free: extern "C" fn(PluginBuffer),
}

fn leak(bytes: Vec<u8>) -> PluginBuffer {
    let mut b = bytes.into_boxed_slice();
    let ptr = b.as_mut_ptr();
    let len = b.len();
    std::mem::forget(b);
    PluginBuffer { ptr, len }
}

unsafe fn free_leaked(buf: PluginBuffer) {
    if !buf.ptr.is_null() {
        drop(Box::from_raw(std::slice::from_raw_parts_mut(buf.ptr, buf.len)));
    }
}
"#;

/// `TargetPlugin` fixture: constant data, matching
/// `ModuleManifest::default()` / `PluginSyscallTable::default()` exactly
/// (both all-empty `Vec` fields, hence all-zero wire bytes).
fn fixture_target_source() -> String {
    format!(
        r#"{FFI_PRELUDE}
const MANIFEST_RESP: [u8; 37] = [0u8; 37];
const SYSCALL_RESP: [u8; 5] = [1, 0, 0, 0, 0];

#[no_mangle]
pub extern "C" fn speet_plugin_create_target(_imports: HostImportsFfi) -> *mut c_void {{
    std::ptr::null_mut()
}}

#[no_mangle]
pub extern "C" fn speet_plugin_call_target(_h: *mut c_void, p: *const u8, n: usize) -> PluginBuffer {{
    let payload = unsafe {{ std::slice::from_raw_parts(p, n) }};
    if payload.first().copied() == Some(0) {{
        leak(MANIFEST_RESP.to_vec())
    }} else {{
        leak(SYSCALL_RESP.to_vec())
    }}
}}

#[no_mangle]
pub extern "C" fn speet_plugin_free_buffer_target(buf: PluginBuffer) {{
    unsafe {{ free_leaked(buf) }}
}}

#[no_mangle]
pub extern "C" fn speet_plugin_destroy_target(_h: *mut c_void) {{}}
"#
    )
}

/// `AddressMapperPlugin` fixture computing `translate` itself — no host
/// imports. Only correct for `addr_local < 128` (single-byte LEB128), fine
/// for a test fixture.
fn fixture_address_mapper_direct_source() -> String {
    format!(
        r#"{FFI_PRELUDE}
#[no_mangle]
pub extern "C" fn speet_plugin_create_address_mapper(_imports: HostImportsFfi) -> *mut c_void {{
    std::ptr::null_mut()
}}

#[no_mangle]
pub extern "C" fn speet_plugin_call_address_mapper(_h: *mut c_void, p: *const u8, n: usize) -> PluginBuffer {{
    let payload = unsafe {{ std::slice::from_raw_parts(p, n) }};
    let addr_local = u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]);
    let leb = (addr_local & 0x7F) as u8;
    leak(vec![0u8, 0u8, 2, 0, 0, 0, 0x20, leb])
}}

#[no_mangle]
pub extern "C" fn speet_plugin_free_buffer_address_mapper(buf: PluginBuffer) {{
    unsafe {{ free_leaked(buf) }}
}}

#[no_mangle]
pub extern "C" fn speet_plugin_destroy_address_mapper(_h: *mut c_void) {{}}
"#
    )
}

/// `AddressMapperPlugin` fixture that forwards every `translate` call to a
/// host-granted import named `"host-mmu"` (§2.7's dylib realization). If
/// the import resolves to nothing (the `(null, 0)` denied sentinel — see
/// `speet-plugin-host-inproc::dylib::ffi` module docs), encodes its own
/// `Err(PluginError)` response instead of forwarding garbage.
fn fixture_address_mapper_forward_source() -> String {
    format!(
        r#"{FFI_PRELUDE}
#[no_mangle]
pub extern "C" fn speet_plugin_create_address_mapper(imports: HostImportsFfi) -> *mut c_void {{
    Box::into_raw(Box::new(imports)) as *mut c_void
}}

#[no_mangle]
pub extern "C" fn speet_plugin_call_address_mapper(h: *mut c_void, p: *const u8, n: usize) -> PluginBuffer {{
    let imports = unsafe {{ &*(h as *const HostImportsFfi) }};
    let name = b"host-mmu";
    let resp = (imports.call)(imports.ctx, 1u8, name.as_ptr(), name.len(), p, n);
    if resp.ptr.is_null() || resp.len == 0 {{
        let msg = b"import denied";
        let mut out = vec![0u8, 1u8];
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(msg.len() as u32).to_le_bytes());
        out.extend_from_slice(msg);
        return leak(out);
    }}
    let bytes = unsafe {{ std::slice::from_raw_parts(resp.ptr, resp.len) }}.to_vec();
    (imports.free)(resp);
    leak(bytes)
}}

#[no_mangle]
pub extern "C" fn speet_plugin_free_buffer_address_mapper(buf: PluginBuffer) {{
    unsafe {{ free_leaked(buf) }}
}}

#[no_mangle]
pub extern "C" fn speet_plugin_destroy_address_mapper(h: *mut c_void) {{
    if !h.is_null() {{
        unsafe {{ drop(Box::from_raw(h as *mut HostImportsFfi)) }};
    }}
}}
"#
    )
}

fn compile_cdylib(source: &str, name: &str) -> PathBuf {
    let out_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let src_path = out_dir.join(format!("{name}.rs"));
    std::fs::write(&src_path, source).unwrap();
    let dylib_path = out_dir.join(format!("lib{name}{}", std::env::consts::DLL_SUFFIX));

    let output = Command::new("rustc")
        .arg("--edition=2021")
        .arg("--crate-type=cdylib")
        .arg("-o")
        .arg(&dylib_path)
        .arg(&src_path)
        .output()
        .expect("failed to invoke rustc");
    assert!(
        output.status.success(),
        "rustc failed for fixture {name}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    dylib_path
}

fn load_path(path: &Path, role: ImportRole, imports: Arc<dyn HostImports>) -> DylibPlugin {
    DylibPlugin::load(path, role, imports).expect("dylib load")
}

#[test]
fn target_plugin_constant_data_roundtrip() {
    let path = compile_cdylib(&fixture_target_source(), "fixture_target");
    let engine = load_path(&path, ImportRole::Target, Arc::new(NoImports));
    let plugin = DylibTargetPlugin::new(engine);

    assert_eq!(
        plugin.module_manifest(),
        speet_plugin_api::target::ModuleManifest::default()
    );
    assert_eq!(
        plugin.syscall_table(),
        speet_plugin_api::target::PluginSyscallTable::default()
    );
}

#[test]
fn address_mapper_translate_uses_dynamic_argument() {
    let path = compile_cdylib(
        &fixture_address_mapper_direct_source(),
        "fixture_address_mapper_direct",
    );
    let engine = load_path(&path, ImportRole::AddressMapper, Arc::new(NoImports));
    let plugin = DylibAddressMapperPlugin::new(engine);

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

#[test]
fn host_entity_import_granted_forwards_to_host_plugin() {
    let mut registry = PluginRegistry::new();
    registry.register_static(
        "host-mmu",
        speet_plugin_host::PluginHandle::address_mapper(Arc::new(ToyMemory)),
    );
    let granted = vec![(PluginKind::Memory, "host-mmu".to_string())];
    let imports: Arc<dyn HostImports> = Arc::new(OwnedRestrictedView { registry, granted });

    let path = compile_cdylib(
        &fixture_address_mapper_forward_source(),
        "fixture_address_mapper_forward_granted",
    );
    let engine = load_path(&path, ImportRole::AddressMapper, imports);
    let plugin = DylibAddressMapperPlugin::new(engine);

    let snippet = plugin.translate(9).expect("translate via import");
    assert_eq!(
        snippet,
        CodeSnippet::from_instructions(&[Instruction::LocalGet(9)])
    );
}

#[test]
fn host_entity_import_denied_when_not_granted() {
    let mut registry = PluginRegistry::new();
    registry.register_static(
        "host-mmu",
        speet_plugin_host::PluginHandle::address_mapper(Arc::new(ToyMemory)),
    );
    let granted: Vec<(PluginKind, String)> = vec![];
    let imports: Arc<dyn HostImports> = Arc::new(OwnedRestrictedView { registry, granted });

    let path = compile_cdylib(
        &fixture_address_mapper_forward_source(),
        "fixture_address_mapper_forward_denied",
    );
    let engine = load_path(&path, ImportRole::AddressMapper, imports);
    let plugin = DylibAddressMapperPlugin::new(engine);

    let err = plugin.translate(9).expect_err("import must be denied");
    assert_eq!(err, PluginError::new(1, "import denied"));
}
