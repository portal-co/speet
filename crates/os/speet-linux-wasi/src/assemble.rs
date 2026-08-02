//! Host scratch memory index (guest linear memory stays at 0).
pub use crate::link::HOST_MEMORY_INDEX;

/// Translate RV64 Linux `.text` and assemble a WASI megabinary.
pub fn recompile_rv64_wasi_to_wasm(text: &[u8], start_addr: u64) -> alloc::vec::Vec<u8> {
    crate::link::link_wasi_megabinary(text, start_addr)
}
