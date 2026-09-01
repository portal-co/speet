//! Rust guest SDK for `speet-plugin-host-wasm`.
//!
//! Provides the bump allocator and `speet_plugin_alloc` / `speet_plugin_call`
//! / `speet_plugin_dealloc` exports a `.wasm` plugin must implement, plus
//! dispatch into a user-supplied [`speet_plugin_api`] trait object. `#![no_std]`
//! + alloc; no serde. See `docs/guides/plugin-api.md` §5.2.

#![no_std]

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

use speet_plugin_api::remote::{
    AddressMapperRequest, ArchRequest, MemoryAccessRequest, ObjectModelRequest, TableRequest,
    TargetRequest, dispatch_address_mapper, dispatch_arch, dispatch_memory_access,
    dispatch_object_model, dispatch_table, dispatch_target,
};
use speet_plugin_api::wire::{WireDecode, WireEncode};
use speet_plugin_api::{
    AddressMapperPlugin, ArchPlugin, MemoryAccessPlugin, ObjectModelPlugin, PluginKind,
    TablePlugin, TargetPlugin,
};

/// Pack `(ptr, len)` into one `i64` — high 32 bits `ptr`, low 32 bits `len`.
/// Matches `speet_plugin_host_wasm::abi::pack`.
pub fn pack(ptr: u32, len: u32) -> i64 {
    ((ptr as i64) << 32) | (len as i64)
}

pub fn unpack(packed: i64) -> (u32, u32) {
    let ptr = ((packed as u64) >> 32) as u32;
    let len = (packed as u64 & 0xFFFF_FFFF) as u32;
    (ptr, len)
}

/// 8 MiB bump heap in guest linear memory. Large enough for a translate
/// request/response (guest `.text` + produced megabinary) without growing.
const HEAP_SIZE: usize = 8 * 1024 * 1024;

struct Heap(UnsafeCell<[u8; HEAP_SIZE]>);
unsafe impl Sync for Heap {}

static HEAP: Heap = Heap(UnsafeCell::new([0; HEAP_SIZE]));
static BUMP: AtomicUsize = AtomicUsize::new(0);

/// Reserve `len` bytes from the bump heap. Aligns to 8.
pub fn bump_alloc(len: usize) -> *mut u8 {
    let aligned = (len + 7) & !7;
    let off = BUMP.fetch_add(aligned, Ordering::Relaxed);
    if off.saturating_add(aligned) > HEAP_SIZE {
        return core::ptr::null_mut();
    }
    unsafe { (*HEAP.0.get()).as_mut_ptr().add(off) }
}

pub fn bump_reset() {
    BUMP.store(0, Ordering::Relaxed);
}

/// A guest implements exactly one plugin role.
pub enum GuestPlugin {
    Arch(Arc<dyn ArchPlugin>),
    AddressMapper(Arc<dyn AddressMapperPlugin>),
    MemoryAccess(Arc<dyn MemoryAccessPlugin>),
    Table(Arc<dyn TablePlugin>),
    ObjectModel(Arc<dyn ObjectModelPlugin>),
    Target(Arc<dyn TargetPlugin>),
}

impl GuestPlugin {
    pub fn kind(&self) -> PluginKind {
        match self {
            GuestPlugin::Arch(_) => PluginKind::Arch,
            GuestPlugin::AddressMapper(_) | GuestPlugin::MemoryAccess(_) => PluginKind::Memory,
            GuestPlugin::Table(_) => PluginKind::Table,
            GuestPlugin::ObjectModel(_) => PluginKind::ObjectModel,
            GuestPlugin::Target(_) => PluginKind::Target,
        }
    }

    /// Decode `req` as this role's `XRequest`, dispatch, encode the response.
    pub fn handle_call(&self, req: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            GuestPlugin::Arch(p) => {
                if let Ok((r, _)) = ArchRequest::decode(req) {
                    dispatch_arch(p.as_ref(), r).encode(&mut out);
                }
            }
            GuestPlugin::AddressMapper(p) => {
                if let Ok((r, _)) = AddressMapperRequest::decode(req) {
                    dispatch_address_mapper(p.as_ref(), r).encode(&mut out);
                }
            }
            GuestPlugin::MemoryAccess(p) => {
                if let Ok((r, _)) = MemoryAccessRequest::decode(req) {
                    dispatch_memory_access(p.as_ref(), r).encode(&mut out);
                }
            }
            GuestPlugin::Table(p) => {
                if let Ok((r, _)) = TableRequest::decode(req) {
                    dispatch_table(p.as_ref(), r).encode(&mut out);
                }
            }
            GuestPlugin::ObjectModel(p) => {
                if let Ok((r, _)) = ObjectModelRequest::decode(req) {
                    dispatch_object_model(p.as_ref(), r).encode(&mut out);
                }
            }
            GuestPlugin::Target(p) => {
                if let Ok((r, _)) = TargetRequest::decode(req) {
                    dispatch_target(p.as_ref(), r).encode(&mut out);
                }
            }
        }
        out
    }
}

/// Process-wide plugin installed by the cdylib's `#[no_mangle] init`.
pub static PLUGIN: spin_once::OnceLock<GuestPlugin> = spin_once::OnceLock::new();

mod spin_once {
    use super::GuestPlugin;
    use core::sync::atomic::{AtomicBool, Ordering};

    pub struct OnceLock<T> {
        set: AtomicBool,
        val: core::cell::UnsafeCell<Option<T>>,
    }
    unsafe impl<T: Send + Sync> Sync for OnceLock<T> {}
    impl<T> OnceLock<T> {
        pub const fn new() -> Self {
            Self {
                set: AtomicBool::new(false),
                val: core::cell::UnsafeCell::new(None),
            }
        }
        pub fn set(&self, v: T) -> Result<(), T> {
            if self.set.swap(true, Ordering::SeqCst) {
                return Err(v);
            }
            unsafe { *self.val.get() = Some(v) };
            Ok(())
        }
        pub fn get(&self) -> Option<&T> {
            if self.set.load(Ordering::SeqCst) {
                unsafe { (*self.val.get()).as_ref() }
            } else {
                None
            }
        }
    }
    impl OnceLock<GuestPlugin> {}
}

pub fn install(plugin: GuestPlugin) {
    let _ = PLUGIN.set(plugin);
}

/// Shared body of `speet_plugin_call`: run the installed plugin, copy the
/// response into the bump heap, return packed `(ptr, len)`.
pub fn call_installed(req_ptr: i32, req_len: i32) -> i64 {
    if req_ptr < 0 || req_len < 0 {
        return pack(0, 0);
    }
    let req = unsafe { core::slice::from_raw_parts(req_ptr as *const u8, req_len as usize) };
    let resp = match PLUGIN.get() {
        Some(p) => p.handle_call(req),
        None => Vec::new(),
    };
    let ptr = bump_alloc(resp.len());
    if ptr.is_null() {
        return pack(0, 0);
    }
    unsafe {
        core::ptr::copy_nonoverlapping(resp.as_ptr(), ptr, resp.len());
    }
    pack(ptr as u32, resp.len() as u32)
}

/// Guest-side `speet_plugin_alloc`. Safe to re-export from a cdylib.
pub fn wasm_alloc(len: i32) -> i32 {
    if len <= 0 {
        return 0;
    }
    let p = bump_alloc(len as usize);
    if p.is_null() { 0 } else { p as i32 }
}

/// Guest-side `speet_plugin_call`. Safe to re-export from a cdylib.
pub fn wasm_call(req_ptr: i32, req_len: i32) -> i64 {
    call_installed(req_ptr, req_len)
}

#[cfg(target_arch = "wasm32")]
mod wasm_exports {
    use super::*;

    #[no_mangle]
    pub extern "C" fn speet_plugin_alloc(len: i32) -> i32 {
        wasm_alloc(len)
    }

    #[no_mangle]
    pub extern "C" fn speet_plugin_call(req_ptr: i32, req_len: i32) -> i64 {
        wasm_call(req_ptr, req_len)
    }

    #[no_mangle]
    pub extern "C" fn speet_plugin_dealloc(_ptr: i32, _len: i32) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use speet_plugin_api::remote::TargetRequest;
    use speet_plugin_api::target::{ModuleManifest, PluginSyscallTable, TargetPlugin};

    struct EmptyTarget;
    impl TargetPlugin for EmptyTarget {
        fn module_manifest(&self) -> ModuleManifest {
            ModuleManifest::default()
        }
        fn syscall_table(&self) -> PluginSyscallTable {
            PluginSyscallTable::default()
        }
    }

    #[test]
    fn pack_unpack() {
        assert_eq!(unpack(pack(0x1000, 42)), (0x1000, 42));
    }

    #[test]
    fn target_dispatch_roundtrip() {
        let g = GuestPlugin::Target(Arc::new(EmptyTarget));
        let mut req = Vec::new();
        TargetRequest::ModuleManifest.encode(&mut req);
        let resp = g.handle_call(&req);
        assert!(!resp.is_empty());
        let _ = String::from("ok");
    }
}
