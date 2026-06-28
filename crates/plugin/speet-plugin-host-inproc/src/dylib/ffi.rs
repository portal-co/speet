//! The fixed `extern "C"`/`#[repr(C)]` surface a dylib plugin exports, and
//! the matching surface the host hands it for host-entity imports (§2.7).
//! This module *is* the spec — see `docs/guides/plugin-api.md` §5 / §7 and
//! `docs/plugin-api.md` §5.1.
//!
//! No Rust trait object, `Box<dyn Trait>`, `Vec<T>`, or `String` crosses
//! this boundary — only raw pointers/lengths and `speet_plugin_api::remote`
//! `XRequest`/`XResponse` byte encodings (self-describing, no separate
//! method tag). This is what lets a dylib plugin stay binary-stable across
//! `rustc` versions, instead of depending on the host and plugin sharing
//! one.
//!
//! ## Exported symbols, one set per role (`arch`, `address_mapper`,
//! `memory_access`, `table`, `object_model`, `target` — see
//! [`role_symbol_name`])
//!
//! | Symbol | Signature |
//! |---|---|
//! | `speet_plugin_create_<role>` | `extern "C" fn(imports: HostImportsFfi) -> *mut c_void` |
//! | `speet_plugin_call_<role>` | `extern "C" fn(handle: *mut c_void, payload_ptr: *const u8, payload_len: usize) -> PluginBuffer` |
//! | `speet_plugin_free_buffer_<role>` | `extern "C" fn(buf: PluginBuffer)` |
//! | `speet_plugin_destroy_<role>` | `extern "C" fn(handle: *mut c_void)` |
//!
//! A plugin that needs no host-entity imports simply ignores the `imports`
//! parameter to `create_<role>`.
//!
//! ## Buffer ownership: "the producer frees its own allocation"
//!
//! A buffer `call_<role>` returns was allocated by the *dylib's* own
//! allocator, so the *host* asks the *dylib* to free it
//! (`free_buffer_<role>`) — never call the host's allocator on
//! dylib-allocated memory or vice versa. Symmetrically, a buffer
//! [`HostImportsFfi::call`] returns was allocated by the *host*, so the
//! *dylib* asks the host to free it via [`HostImportsFfi::free`].

use std::ffi::c_void;

use speet_plugin_api::remote::ImportRole;

/// A buffer crossing the dylib FFI boundary: a raw pointer + length, no
/// allocator metadata beyond "whoever produced it frees it" (see module
/// docs).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PluginBuffer {
    pub ptr: *mut u8,
    pub len: usize,
}

impl PluginBuffer {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }

    /// Leak `bytes` into a buffer the *other* side is responsible for
    /// freeing via the matching `free`/`free_buffer` call.
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        let mut bytes = bytes.into_boxed_slice();
        let ptr = bytes.as_mut_ptr();
        let len = bytes.len();
        std::mem::forget(bytes);
        Self { ptr, len }
    }

    /// Borrow this buffer's bytes without taking ownership. Caller is
    /// responsible for freeing it afterward via whichever side allocated it.
    ///
    /// # Safety
    /// `self` must be either [`PluginBuffer::empty`] or a value produced by
    /// [`PluginBuffer::from_vec`] that has not yet been freed.
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() || self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Reconstruct and drop the `Box<[u8]>` this buffer was leaked from.
    ///
    /// # Safety
    /// `self` must have been produced by [`PluginBuffer::from_vec`] *on this
    /// side of the boundary* — never call this on a buffer the other side
    /// allocated.
    pub unsafe fn free_as_owned(self) {
        if !self.ptr.is_null() {
            unsafe {
                drop(Box::from_raw(std::slice::from_raw_parts_mut(
                    self.ptr, self.len,
                )));
            }
        }
    }
}

/// Handed to `speet_plugin_create_<role>` so a dylib plugin can resolve a
/// host-entity import (§2.7) — the dylib realization of
/// `speet_plugin_api::HostImports`. `ctx` is host-owned; the dylib must
/// treat it as opaque and pass it back into `call`/`free` unmodified.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HostImportsFfi {
    pub ctx: *mut c_void,
    pub call: extern "C" fn(
        ctx: *mut c_void,
        role: u8,
        name_ptr: *const u8,
        name_len: usize,
        payload_ptr: *const u8,
        payload_len: usize,
    ) -> PluginBuffer,
    pub free: extern "C" fn(buf: PluginBuffer),
}

pub type CreateFn = unsafe extern "C" fn(HostImportsFfi) -> *mut c_void;
pub type CallFn = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> PluginBuffer;
pub type FreeBufferFn = unsafe extern "C" fn(PluginBuffer);
pub type DestroyFn = unsafe extern "C" fn(*mut c_void);

/// The symbol-name fragment for each role — `speet_plugin_create_arch`,
/// `speet_plugin_create_address_mapper`, etc. A free function (not an
/// inherent `impl` on `ImportRole`) because `ImportRole` is defined in
/// `speet_plugin_api`, a foreign crate.
pub fn role_symbol_name(role: ImportRole) -> &'static str {
    match role {
        ImportRole::Arch => "arch",
        ImportRole::AddressMapper => "address_mapper",
        ImportRole::MemoryAccess => "memory_access",
        ImportRole::Table => "table",
        ImportRole::ObjectModel => "object_model",
        ImportRole::Target => "target",
    }
}
