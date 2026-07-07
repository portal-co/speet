//! [`WasmPlugin`] — one loaded `.wasm` guest instance, behind the
//! `speet_plugin_call` marshalling convention documented in [`crate::abi`].

use std::sync::{Arc, Mutex};

use speet_plugin_api::HostImports;
use wasmi::{Engine, Instance, Linker, Memory, Module, Store};

use crate::abi;
use crate::host_calls::{self, HostState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmLoadError(pub String);

impl core::fmt::Display for WasmLoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "wasm plugin error: {}", self.0)
    }
}
impl std::error::Error for WasmLoadError {}

struct Inner {
    store: Store<HostState>,
    instance: Instance,
    memory: Memory,
}

/// One loaded `.wasm` plugin instance: engine, store, and instance bound
/// together behind a lock so the `&self`-based plugin-api traits (shared via
/// `Arc` — see `speet_plugin_api::memory` module docs for why every plugin
/// trait here is `&self`) can drive wasmi's inherently `&mut`-based calling
/// convention.
pub struct WasmPlugin {
    inner: Mutex<Inner>,
}

impl WasmPlugin {
    /// Instantiate `wasm_bytes`, granting it exactly `imports` (already a
    /// restricted view — see `speet_plugin_host::PluginRegistry::restricted_view`
    /// and §2.7's trust note) as its `host.speet_host_call` resolution table.
    pub fn load(wasm_bytes: &[u8], imports: Arc<dyn HostImports>) -> Result<Self, WasmLoadError> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| WasmLoadError(format!("module parse: {e}")))?;
        let mut store = Store::new(&engine, HostState::new(imports));
        let mut linker = Linker::new(&engine);
        host_calls::register(&mut linker)
            .map_err(|e| WasmLoadError(format!("linker setup: {e}")))?;
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(|e| WasmLoadError(format!("instantiate: {e}")))?;
        let memory = instance
            .get_export(&store, abi::EXPORT_MEMORY)
            .and_then(|e| e.into_memory())
            .ok_or_else(|| WasmLoadError(format!("missing export: {}", abi::EXPORT_MEMORY)))?;
        Ok(Self {
            inner: Mutex::new(Inner {
                store,
                instance,
                memory,
            }),
        })
    }

    /// Marshal `payload` (a `speet_plugin_api::remote::XRequest` encoding)
    /// into the guest's own linear memory via its `speet_plugin_alloc`
    /// export, invoke `speet_plugin_call(ptr, len)`, and read the response
    /// bytes back out. The method tag is the request's own leading byte
    /// (every `XRequest::encode` writes it) — no separate tag parameter
    /// crosses this boundary. See `docs/guides/plugin-api.md` §5.2.
    pub fn call_raw(&self, payload: &[u8]) -> Result<Vec<u8>, WasmLoadError> {
        let mut guard = self.inner.lock().unwrap();
        let Inner {
            store,
            instance,
            memory,
        } = &mut *guard;

        let req_ptr = call_alloc(store, instance, payload.len() as u32)?;
        memory
            .write(&mut *store, req_ptr as usize, payload)
            .map_err(|e| WasmLoadError(format!("guest memory write: {e}")))?;

        let call_func = instance
            .get_export(&*store, abi::EXPORT_CALL)
            .and_then(|e| e.into_func())
            .ok_or_else(|| WasmLoadError(format!("missing export: {}", abi::EXPORT_CALL)))?;
        let typed = call_func
            .typed::<(i32, i32), i64>(&*store)
            .map_err(|e| WasmLoadError(format!("{}: {e}", abi::EXPORT_CALL)))?;
        let packed = typed
            .call(&mut *store, (req_ptr as i32, payload.len() as i32))
            .map_err(|e| WasmLoadError(format!("{} trapped: {e}", abi::EXPORT_CALL)))?;
        let (resp_ptr, resp_len) = abi::unpack(packed);

        let mut resp = vec![0u8; resp_len as usize];
        memory
            .read(&*store, resp_ptr as usize, &mut resp)
            .map_err(|e| WasmLoadError(format!("guest memory read: {e}")))?;

        if let Some(dealloc) = instance
            .get_export(&*store, abi::EXPORT_DEALLOC)
            .and_then(|e| e.into_func())
        {
            if let Ok(typed) = dealloc.typed::<(i32, i32), ()>(&*store) {
                let _ = typed.call(&mut *store, (resp_ptr as i32, resp_len as i32));
            }
        }
        Ok(resp)
    }
}

fn call_alloc(
    store: &mut Store<HostState>,
    instance: &Instance,
    len: u32,
) -> Result<u32, WasmLoadError> {
    let alloc = instance
        .get_export(&*store, abi::EXPORT_ALLOC)
        .and_then(|e| e.into_func())
        .ok_or_else(|| WasmLoadError(format!("missing export: {}", abi::EXPORT_ALLOC)))?;
    let typed = alloc
        .typed::<i32, i32>(&*store)
        .map_err(|e| WasmLoadError(format!("{}: {e}", abi::EXPORT_ALLOC)))?;
    let ptr = typed
        .call(&mut *store, len as i32)
        .map_err(|e| WasmLoadError(format!("{} trapped: {e}", abi::EXPORT_ALLOC)))?;
    Ok(ptr as u32)
}
