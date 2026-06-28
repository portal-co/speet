//! Registers `host.speet_host_call` (§2.7's WASM host-entity-import
//! mechanism) into the engine's [`wasmi::Linker`]. Only ever reachable from
//! the guest side if the guest declares the import — wasmi simply won't
//! link a module that imports something the `Linker` hasn't defined, so an
//! unlisted import fails at instantiation, not at call time. See
//! `docs/guides/plugin-api.md` §7.

use std::sync::Arc;

use speet_plugin_api::remote::dispatch_import;
use speet_plugin_api::HostImports;
use wasmi::{Caller, Linker};

use crate::abi::{self, ImportRole};

/// The engine `Store`'s user data: just the restricted [`HostImports`] view
/// this guest was granted (§2.7's trust note) — never the caller's
/// `Context`/`E`, preserving the Context/E-agnostic invariant end to end.
pub struct HostState {
    pub imports: Arc<dyn HostImports>,
}

impl HostState {
    pub fn new(imports: Arc<dyn HostImports>) -> Self {
        Self { imports }
    }
}

/// Register `host.speet_host_call` into `linker`. Always registered — a
/// guest that doesn't declare the import simply never links against it, so
/// this is not itself a privilege grant; the *granted* allowlist lives in
/// `HostState::imports` (a [`speet_plugin_host::RestrictedHostImports`] in
/// practice), supplied at load time.
pub fn register(linker: &mut Linker<HostState>) -> Result<(), wasmi::errors::LinkerError> {
    linker
        .func_wrap(
            abi::IMPORT_MODULE,
            abi::IMPORT_HOST_CALL,
            |mut caller: Caller<'_, HostState>,
             role: i32,
             name_ptr: i32,
             name_len: i32,
             req_ptr: i32,
             req_len: i32|
             -> i64 { host_call(&mut caller, role, name_ptr, name_len, req_ptr, req_len) },
        )
        .map(|_| ())
}

fn host_call(
    caller: &mut Caller<'_, HostState>,
    role: i32,
    name_ptr: i32,
    name_len: i32,
    req_ptr: i32,
    req_len: i32,
) -> i64 {
    let memory = match caller
        .get_export(abi::EXPORT_MEMORY)
        .and_then(|e| e.into_memory())
    {
        Some(m) => m,
        None => return abi::pack(0, 0),
    };

    let mut name_bytes = vec![0u8; name_len.max(0) as usize];
    if memory.read(&caller, name_ptr as usize, &mut name_bytes).is_err() {
        return abi::pack(0, 0);
    }
    let Ok(name) = core::str::from_utf8(&name_bytes) else {
        return abi::pack(0, 0);
    };

    let mut req_bytes = vec![0u8; req_len.max(0) as usize];
    if memory.read(&caller, req_ptr as usize, &mut req_bytes).is_err() {
        return abi::pack(0, 0);
    }

    let Some(role) = u8::try_from(role).ok().and_then(ImportRole::from_u8) else {
        return abi::pack(0, 0);
    };

    let resp_bytes = match dispatch_import(caller.data().imports.as_ref(), role, name, &req_bytes)
    {
        Some(bytes) => bytes,
        // Not granted, unknown name, or malformed request: there is no
        // wire-level "denied" response (a denied import simply isn't
        // reachable, per the trust note in `docs/guides/plugin-api.md` §7),
        // so the guest sees an empty buffer and must treat that as failure.
        None => return abi::pack(0, 0),
    };

    let Ok(resp_ptr) = call_guest_alloc(caller, resp_bytes.len() as u32) else {
        return abi::pack(0, 0);
    };
    if memory.write(&mut *caller, resp_ptr as usize, &resp_bytes).is_err() {
        return abi::pack(0, 0);
    }
    abi::pack(resp_ptr, resp_bytes.len() as u32)
}

fn call_guest_alloc(caller: &mut Caller<'_, HostState>, len: u32) -> Result<u32, wasmi::Error> {
    let alloc = caller
        .get_export(abi::EXPORT_ALLOC)
        .and_then(|e| e.into_func())
        .ok_or_else(|| wasmi::Error::new("guest missing speet_plugin_alloc export"))?;
    let typed = alloc.typed::<i32, i32>(&*caller)?;
    let ptr = typed.call(&mut *caller, len as i32)?;
    Ok(ptr as u32)
}
