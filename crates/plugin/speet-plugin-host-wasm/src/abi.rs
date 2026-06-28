//! The fixed export/import surface a `.wasm` plugin must provide (and may
//! additionally import). This module *is* the spec a non-Rust plugin author
//! implements against — see `docs/guides/plugin-api.md` §5.2.
//!
//! ## Guest exports (required)
//!
//! | Export | Signature | Purpose |
//! |---|---|---|
//! | `memory` | (memory) | Linear memory the host reads/writes request and response bytes into |
//! | `speet_plugin_alloc` | `(len: i32) -> i32` | Reserve `len` bytes in guest memory; returns the pointer |
//! | `speet_plugin_call` | `(req_ptr: i32, req_len: i32) -> i64` | Dispatch one call; the request bytes are a `speet_plugin_api::remote::XRequest` encoding (self-describing — the method tag is the request's own leading byte, so no separate tag parameter is needed) |
//!
//! ## Guest exports (optional)
//!
//! | Export | Signature | Purpose |
//! |---|---|---|
//! | `speet_plugin_dealloc` | `(ptr: i32, len: i32) -> ()` | Best-effort cleanup after the host reads a response; skipped if absent |
//!
//! ## Host import (only linked if the plugin's manifest declares imports — §2.7)
//!
//! | Import | Signature | Purpose |
//! |---|---|---|
//! | `host.speet_host_call` | `(role: i32, name_ptr: i32, name_len: i32, req_ptr: i32, req_len: i32) -> i64` | Call back into a manifest-granted host (or other-plugin) entity, identified by `role` (see [`ImportRole`]) and name |
//!
//! `speet_plugin_call`'s and `speet_host_call`'s `i64` return value packs a
//! `(ptr, len)` pair into the guest's own memory holding the response bytes
//! — see [`pack`]/[`unpack`]. The host always writes that buffer via the
//! guest's own `speet_plugin_alloc`, so allocator ownership never crosses
//! the trust boundary in the other direction.

pub const EXPORT_MEMORY: &str = "memory";
pub const EXPORT_ALLOC: &str = "speet_plugin_alloc";
pub const EXPORT_DEALLOC: &str = "speet_plugin_dealloc";
pub const EXPORT_CALL: &str = "speet_plugin_call";

pub const IMPORT_MODULE: &str = "host";
pub const IMPORT_HOST_CALL: &str = "speet_host_call";

/// Pack a `(ptr, len)` pair into one `i64`: high 32 bits `ptr`, low 32 bits
/// `len`. Avoids relying on the WASM multi-value proposal for the one place
/// this ABI would otherwise need two return values.
pub fn pack(ptr: u32, len: u32) -> i64 {
    ((ptr as i64) << 32) | (len as i64)
}

/// Inverse of [`pack`].
pub fn unpack(packed: i64) -> (u32, u32) {
    let ptr = ((packed as u64) >> 32) as u32;
    let len = (packed as u64 & 0xFFFF_FFFF) as u32;
    (ptr, len)
}

/// Re-exported from `speet_plugin_api::remote`, which a guest fixture and a
/// future subprocess host both also need — it's a wire-level concept, not a
/// WASM-specific one. See that module's docs for why it's finer-grained than
/// [`speet_plugin_api::PluginKind`].
pub use speet_plugin_api::remote::ImportRole;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        for (ptr, len) in [(0u32, 0u32), (0x1000, 42), (u32::MAX, u32::MAX)] {
            let packed = pack(ptr, len);
            assert_eq!(unpack(packed), (ptr, len));
        }
    }
}
