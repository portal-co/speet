//! [`DylibPlugin`] — one loaded shared library, behind the `extern "C"`
//! marshalling convention documented in [`crate::dylib::ffi`].

use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};

use libloading::Library;
use speet_plugin_api::remote::{dispatch_import, ImportRole};
use speet_plugin_api::HostImports;

use super::ffi::{
    role_symbol_name, CallFn, CreateFn, DestroyFn, FreeBufferFn, HostImportsFfi, PluginBuffer,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DylibLoadError(pub String);

impl core::fmt::Display for DylibLoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dylib plugin error: {}", self.0)
    }
}
impl std::error::Error for DylibLoadError {}

struct Symbols {
    call: CallFn,
    free_buffer: FreeBufferFn,
    destroy: DestroyFn,
}

struct Inner {
    // Order matters for `Drop`: `handle` must be destroyed before `_lib` is
    // unloaded (the destroy call needs the symbol, which borrows from
    // `_lib`), and `_imports_ctx` must outlive every call the dylib could
    // make back into it, including ones made during `destroy`. Rust drops
    // fields in declaration order, so this struct's field order is load-bearing.
    handle: *mut c_void,
    symbols: Symbols,
    _lib: Library,
    _imports_ctx: Box<Arc<dyn HostImports>>,
}

// SAFETY: `handle` and the function pointers in `Symbols` are only ever
// touched while holding `DylibPlugin`'s `Mutex`, which serializes access;
// the raw pointer itself carries no thread-affinity. The dylib's own
// implementation is responsible for being safe to call from whichever
// thread the host happens to call it on, exactly as for any other
// `extern "C"` library.
unsafe impl Send for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe { (self.symbols.destroy)(self.handle) };
    }
}

extern "C" fn host_imports_call(
    ctx: *mut c_void,
    role: u8,
    name_ptr: *const u8,
    name_len: usize,
    payload_ptr: *const u8,
    payload_len: usize,
) -> PluginBuffer {
    // SAFETY: `ctx` was produced by `DylibPlugin::load` as
    // `&*imports_ctx as *const Arc<dyn HostImports>` and stays valid for
    // the lifetime of the owning `Inner` (see its field-order comment); the
    // dylib is contractually required to pass it back unmodified.
    let imports = unsafe { &*(ctx as *const Arc<dyn HostImports>) };
    let name_bytes = if name_ptr.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(name_ptr, name_len) }
    };
    let Ok(name) = std::str::from_utf8(name_bytes) else {
        return PluginBuffer::empty();
    };
    let payload = if payload_ptr.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(payload_ptr, payload_len) }
    };
    let Some(role) = ImportRole::from_u8(role) else {
        return PluginBuffer::empty();
    };
    match dispatch_import(imports.as_ref(), role, name, payload) {
        Some(bytes) => PluginBuffer::from_vec(bytes),
        None => PluginBuffer::empty(),
    }
}

/// The host's half of [`HostImportsFfi`] — frees a buffer `host_imports_call`
/// produced (host-allocated, so the host frees it).
extern "C" fn host_imports_free(buf: PluginBuffer) {
    unsafe { buf.free_as_owned() }
}

/// One loaded dylib plugin instance.
pub struct DylibPlugin {
    inner: Mutex<Inner>,
}

impl DylibPlugin {
    /// Load `path`, resolve the four `speet_plugin_*_<role>` symbols, and
    /// call `speet_plugin_create_<role>`, granting it exactly `imports`
    /// (already a restricted view — see
    /// `speet_plugin_host::PluginRegistry::restricted_view` and §2.7's
    /// trust note) as its [`HostImportsFfi`].
    pub fn load(
        path: &Path,
        role: ImportRole,
        imports: Arc<dyn HostImports>,
    ) -> Result<Self, DylibLoadError> {
        let lib = unsafe { Library::new(path) }
            .map_err(|e| DylibLoadError(format!("loading {}: {e}", path.display())))?;
        let role_name = role_symbol_name(role);

        let create: CreateFn = unsafe { resolve(&lib, &format!("speet_plugin_create_{role_name}")) }?;
        let call: CallFn = unsafe { resolve(&lib, &format!("speet_plugin_call_{role_name}")) }?;
        let free_buffer: FreeBufferFn =
            unsafe { resolve(&lib, &format!("speet_plugin_free_buffer_{role_name}")) }?;
        let destroy: DestroyFn = unsafe { resolve(&lib, &format!("speet_plugin_destroy_{role_name}")) }?;

        // Boxed so the `Arc<dyn HostImports>` has a stable heap address to
        // hand the dylib as `ctx` — taken before the box moves into `Inner`,
        // which is fine: a raw pointer doesn't keep a borrow-checker
        // borrow alive.
        let imports_ctx = Box::new(imports);
        let ctx = &*imports_ctx as *const Arc<dyn HostImports> as *mut c_void;
        let ffi = HostImportsFfi {
            ctx,
            call: host_imports_call,
            free: host_imports_free,
        };
        let handle = unsafe { create(ffi) };

        Ok(Self {
            inner: Mutex::new(Inner {
                handle,
                symbols: Symbols {
                    call,
                    free_buffer,
                    destroy,
                },
                _lib: lib,
                _imports_ctx: imports_ctx,
            }),
        })
    }

    /// Marshal `payload` (a `speet_plugin_api::remote::XRequest` encoding)
    /// through `speet_plugin_call_<role>` and return the response bytes,
    /// freeing the dylib-allocated buffer afterward via
    /// `speet_plugin_free_buffer_<role>`.
    pub fn call_raw(&self, payload: &[u8]) -> Result<Vec<u8>, DylibLoadError> {
        let guard = self.inner.lock().unwrap();
        let buf = unsafe { (guard.symbols.call)(guard.handle, payload.as_ptr(), payload.len()) };
        let bytes = unsafe { buf.as_slice() }.to_vec();
        unsafe { (guard.symbols.free_buffer)(buf) };
        Ok(bytes)
    }
}

unsafe fn resolve<T: Copy>(lib: &Library, name: &str) -> Result<T, DylibLoadError> {
    let mut symbol_name = name.as_bytes().to_vec();
    symbol_name.push(0);
    let sym = unsafe { lib.get::<T>(&symbol_name) }
        .map_err(|e| DylibLoadError(format!("missing symbol {name}: {e}")))?;
    Ok(*sym)
}
