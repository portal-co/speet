//! Host scratch memory index (guest linear memory stays at 0).
pub use crate::link::HOST_MEMORY_INDEX;

/// Translate aarch64 Darwin `.text` and assemble a WASI megabinary.
pub fn recompile_aarch64_darwin_wasi_to_wasm(text: &[u8], start_addr: u64) -> alloc::vec::Vec<u8> {
    crate::link::link_wasi_megabinary(text, start_addr)
}
