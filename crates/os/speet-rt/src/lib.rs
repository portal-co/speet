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

pub use shim::{generate_memory_tu, generate_shim};

pub use entry_bridge::entry_bridge_c;

/// The C source of the runtime shim. Compile with the system C compiler and link
/// with the generated guest object plus the tunneled host dylib (`-lc` on Linux,
/// `-lSystem` on macOS).
pub const RUNTIME_C: &str = include_str!("runtime.c");

/// The export name the backend must assign to the recompiled guest entry so the
/// shim's `main` can transfer control to it.
pub const GUEST_ENTRY_SYMBOL: &str = "__guest_entry";

/// The export name for the optional passive-data initializer.
pub const DATA_INIT_SYMBOL: &str = "__speet_data_init";

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
