//! Runtime shim sources for recompiled host binaries.
//!
//! The full-binary recompiler emits a `.o` of recompiled guest code that
//! references a few runtime symbols (`__wasm_mem`, `__wasm_mem_pages`,
//! `__wasm_memory_grow`) and a guest entry export (`__guest_entry`). This crate
//! ships the C source of the shim that defines those symbols and bootstraps the
//! process, so the driver can compile and link it alongside the generated `.o`.
//!
//! See [`RUNTIME_C`] for the source and [`write_runtime_c`] to materialize it.

mod shim;
mod entry_bridge;
mod guest_stubs;
mod data_segments;

pub use shim::{generate_memory_tu, generate_shim};

pub use entry_bridge::{entry_bridge_c, entry_bridge_direct_c};
pub use guest_stubs::{
    entry_stub_symbol, generate_guest_stubs_c, halt_stub_symbol, GuestStubEntry,
};
pub use data_segments::{generate_data_segments_c, DataSegmentBytes};

/// The C source of the runtime shim. Compile with the system C compiler and link
/// with the generated guest object plus the tunneled host dylib (`-lc` on Linux,
/// `-lSystem` on macOS).
pub const RUNTIME_C: &str = include_str!("runtime.c");

/// The export name the backend must assign to the recompiled guest entry so the
/// shim's `main` can transfer control to it.
pub const GUEST_ENTRY_SYMBOL: &str = "__guest_entry";

/// The export name for the optional passive-data initializer.
pub const DATA_INIT_SYMBOL: &str = "__speet_data_init";

/// Bytes reserved at the very top of `__wasm_mem`, above where the guest
/// stack starts, for host-import stubs that need to hand the guest a
/// *translated* copy of host-owned data (e.g. `getenv`'s result) rather than
/// a raw host pointer — returning a raw host pointer would hit the exact
/// truncation bug the guest-stack/`__wasm_mem` address-space fix (see
/// `entry_bridge`'s module doc) already fixed once, just for a return value
/// instead of a seeded register. `entry_bridge`'s `sp_init` computation
/// subtracts this from `__wasm_mem_pages * 65536` before seeding SP, so the
/// stack (which only ever grows *down* from its starting point) can never
/// collide with this region; `shim.rs`'s stubs that copy host data in write
/// to `[__wasm_mem_pages * 65536 - HOST_STR_SCRATCH_BYTES, __wasm_mem_pages *
/// 65536)` and return a wasm-relative offset into it. Not a general-purpose
/// allocator — callers must not assume a copy survives past the next call
/// into a stub that also uses this region.
pub const HOST_STR_SCRATCH_BYTES: u32 = 4096;

/// Write the runtime C source to `path`. Returns the bytes written.
pub fn write_runtime_c(path: &std::path::Path) -> std::io::Result<usize> {
    std::fs::write(path, RUNTIME_C)?;
    Ok(RUNTIME_C.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_source_defines_required_symbols() {
        assert!(RUNTIME_C.contains("__wasm_mem"));
        assert!(RUNTIME_C.contains("__wasm_mem_pages"));
        assert!(RUNTIME_C.contains("__wasm_memory_grow"));
        assert!(RUNTIME_C.contains(GUEST_ENTRY_SYMBOL));
        assert!(RUNTIME_C.contains(DATA_INIT_SYMBOL));
    }
}
