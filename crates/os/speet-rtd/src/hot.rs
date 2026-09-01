//! Auto-reload of recompiler.wasm + stubs.wasm on content-hash change.
//!
//! The daemon is the embedder: recompiler bytes load via wasmi
//! (`"recompiler-wasm"` conceptually — a bulk Translate ABI, not `ArchOp`);
//! stubs load through the existing `"wasm"` transport as `ImportRole::Target`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use binary_io::BinArch;
use speet_host_api::ImportManifest;
use speet_link_core::GuestImageLayout;
use speet_plugin_api::imports::NoImports;
use speet_plugin_api::target::TargetPlugin;
use speet_plugin_api::wire::{WireDecode, WireEncode};
use speet_plugin_host::{PluginHandle, PluginRegistry};
use speet_plugin_host_wasm::{WasmPlugin, WasmTargetPlugin, WasmTransport};
use speet_recompile::plt::PltCallPlan;
use speet_recompiler_guest::{TranslateRequest, TranslateResponse};
use speet_runtime::{HotFrontend, PluginHashes};

/// Transport scheme the daemon uses for the bulk-translate frontend blob.
pub const RECOMPILER_WASM_SCHEME: &str = "recompiler-wasm";

pub fn default_recompiler_wasm_path() -> PathBuf {
    if let Ok(p) = std::env::var("SPEET_RECOMPILER_WASM") {
        return PathBuf::from(p);
    }
    default_target_wasm("speet_recompiler_guest.wasm")
}

pub fn default_stubs_wasm_path() -> PathBuf {
    if let Ok(p) = std::env::var("SPEET_STUBS_WASM") {
        return PathBuf::from(p);
    }
    default_target_wasm("speet_stubs_guest.wasm")
}

fn default_target_wasm(name: &str) -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cwd.join("target/wasm32-unknown-unknown/release").join(name)
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    speet_runtime::ArtifactCache::hash_input(bytes)
}

pub fn hash_file(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(hash_bytes(&bytes))
}

struct WasmFront {
    plugin: WasmPlugin,
}

impl HotFrontend for WasmFront {
    fn recompile(
        &self,
        text: &[u8],
        layout: &GuestImageLayout,
        arch: BinArch,
        _plt_plan: Option<&PltCallPlan>,
        entry_addr: Option<u64>,
        _manifest: &ImportManifest,
    ) -> Result<(Vec<u8>, Vec<String>), String> {
        let req = TranslateRequest {
            arch,
            start_addr: layout.text_base,
            entry: entry_addr.unwrap_or(layout.text_base),
            text: text.to_vec(),
        };
        let mut buf = Vec::new();
        req.encode(&mut buf);
        let resp_bytes = self.plugin.call_raw(&buf).map_err(|e| e.to_string())?;
        let (resp, _) = TranslateResponse::decode(&resp_bytes)
            .map_err(|_| "malformed TranslateResponse from recompiler.wasm".to_string())?;
        if !resp.error.is_empty() {
            return Err(resp.error);
        }
        Ok((resp.wasm, resp.unsupported))
    }
}

pub struct HotLoader {
    rec_path: PathBuf,
    stubs_path: PathBuf,
    rec_hash: String,
    stubs_hash: String,
    frontend: Option<Arc<dyn HotFrontend>>,
    extra_wired: Vec<String>,
    registry: PluginRegistry,
}

impl HotLoader {
    pub fn new() -> Self {
        let mut registry = PluginRegistry::new();
        registry.register_transport("wasm", Box::new(WasmTransport));
        Self {
            rec_path: default_recompiler_wasm_path(),
            stubs_path: default_stubs_wasm_path(),
            rec_hash: PluginHashes::NATIVE.into(),
            stubs_hash: PluginHashes::LINKED_STUBS.into(),
            frontend: None,
            extra_wired: Vec::new(),
            registry,
        }
    }

    pub fn with_paths(rec: PathBuf, stubs: PathBuf) -> Self {
        let mut s = Self::new();
        s.rec_path = rec;
        s.stubs_path = stubs;
        s
    }

    pub fn hashes(&self) -> PluginHashes {
        PluginHashes {
            recompiler: self.rec_hash.clone(),
            stubs: self.stubs_hash.clone(),
        }
    }

    pub fn extra_wired(&self) -> Vec<String> {
        self.extra_wired.clone()
    }

    pub fn frontend(&self) -> Option<Arc<dyn HotFrontend>> {
        self.frontend.clone()
    }

    /// Reload either watched file whose content hash changed. Missing files
    /// keep the last good instance (or static-link sentinels if never loaded).
    pub fn ensure_fresh(&mut self) -> Result<(), String> {
        if let Some(h) = hash_file(&self.rec_path) {
            if h != self.rec_hash {
                let bytes = std::fs::read(&self.rec_path).map_err(|e| e.to_string())?;
                let plugin = WasmPlugin::load(&bytes, Arc::new(NoImports))
                    .map_err(|e| format!("{RECOMPILER_WASM_SCHEME} load: {e}"))?;
                self.frontend = Some(Arc::new(WasmFront { plugin }));
                self.rec_hash = h;
            }
        }
        if let Some(h) = hash_file(&self.stubs_path) {
            if h != self.stubs_hash {
                let bytes = std::fs::read(&self.stubs_path).map_err(|e| e.to_string())?;
                let plugin = WasmPlugin::load(&bytes, Arc::new(NoImports))
                    .map_err(|e| format!("load stubs.wasm: {e}"))?;
                let target = WasmTargetPlugin::new(plugin);
                self.extra_wired = target
                    .module_manifest()
                    .func_imports
                    .into_iter()
                    .map(|f| f.field)
                    .collect();
                self.registry
                    .register_static("stubs", PluginHandle::target(Arc::new(target)));
                self.stubs_hash = h;
            }
        }
        Ok(())
    }
}

impl Default for HotLoader {
    fn default() -> Self {
        Self::new()
    }
}
