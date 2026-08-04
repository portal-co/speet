//! Host scratch memory index (guest linear memory stays at 0).
pub use crate::link::HOST_MEMORY_INDEX;

/// Translate aarch64 Darwin `.text` and assemble a WASI megabinary.
pub fn recompile_aarch64_darwin_wasi_to_wasm(text: &[u8], start_addr: u64) -> alloc::vec::Vec<u8> {
    recompile_aarch64_darwin_wasi_to_wasm_with_escape(text, start_addr, yecta::SpeculativeEscape::JUMP)
}

/// Recompile with an explicit [`yecta::SpeculativeEscape`] policy.
pub fn recompile_aarch64_darwin_wasi_to_wasm_with_escape(
    text: &[u8],
    start_addr: u64,
    speculative: yecta::SpeculativeEscape,
) -> alloc::vec::Vec<u8> {
    crate::link::link_wasi_megabinary_with_escape(text, start_addr, speculative)
}

/// Translate a Mach-O-style aarch64 guest whose external calls already point
/// at virtual redirect-shim PCs. `targets` supplies the address-to-symbol
/// table used to realize `read`/`write`/`close`/`exit` through the shared WASI
/// handlers; raw `svc #0x80` remains active as a secondary path.
///
/// Each supported target address must be the virtual shim PC allocated after
/// `.text` (`start_addr + text.len() + (shim_index + 1) * 4`). A Mach-O loader
/// normally establishes this by patching its GOT/lazy-pointer cells before the
/// guest starts. This API deliberately does not pretend a text-only caller has
/// enough information to patch those cells itself.
pub fn recompile_aarch64_darwin_wasi_to_wasm_with_targets(
    text: &[u8],
    start_addr: u64,
    targets: &speet_plugin_api::external_target::ExternalTargetTable,
) -> alloc::vec::Vec<u8> {
    crate::link::link_wasi_megabinary_with_targets(text, start_addr, targets)
}
