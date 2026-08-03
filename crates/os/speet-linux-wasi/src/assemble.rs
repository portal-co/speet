//! Host scratch memory index (guest linear memory stays at 0).
pub use crate::link::HOST_MEMORY_INDEX;

/// Translate RV64 Linux `.text` and assemble a WASI megabinary.
pub fn recompile_rv64_wasi_to_wasm(text: &[u8], start_addr: u64) -> alloc::vec::Vec<u8> {
    crate::link::link_wasi_megabinary(text, start_addr)
}

/// Translate Linux aarch64 `.text` into a WASI megabinary.
pub fn recompile_aarch64_linux_wasi_to_wasm(text: &[u8], start_addr: u64) -> alloc::vec::Vec<u8> {
    crate::link::link_aarch64_linux_wasi_megabinary(text, start_addr)
}

/// Translate Linux x86-64 `.text` into a WASI megabinary.
pub fn recompile_x86_64_linux_wasi_to_wasm(text: &[u8], start_addr: u64) -> alloc::vec::Vec<u8> {
    crate::link::link_x86_64_linux_wasi_megabinary(text, start_addr)
}

/// Translate Linux MIPS N64 `.text` into a WASI megabinary.
pub fn recompile_mips_linux_wasi_to_wasm(text: &[u8], start_addr: u64) -> alloc::vec::Vec<u8> {
    crate::link::link_mips_linux_wasi_megabinary(text, start_addr)
}
