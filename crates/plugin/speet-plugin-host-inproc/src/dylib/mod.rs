//! Dylib mode: a runtime-loaded shared library behind a stable
//! `extern "C"`/`#[repr(C)]` surface. See [`ffi`] for the exact symbol
//! contract and `docs/guides/plugin-api.md` §5 for the design rationale
//! (no Rust trait object crosses this boundary, so a dylib plugin stays
//! binary-stable across `rustc` versions rather than depending on the host
//! and plugin sharing one).

pub mod ffi;
mod adapters;
mod engine;
mod transport;

pub use adapters::{
    DylibAddressMapperPlugin, DylibArchPlugin, DylibMemoryAccessPlugin, DylibObjectModelPlugin,
    DylibTablePlugin, DylibTargetPlugin,
};
pub use engine::{DylibLoadError, DylibPlugin};
pub use transport::{DylibDescriptor, DylibTransport};
