//! Compatibility shim for `os-host-api`.
//!
//! `speet-host-api` now lives in `os-emulation/crates/runtime/os-host-api`.
//! This crate re-exports the same public API so existing `@speet`
//! consumers keep compiling until they migrate to `os-host-api` directly.

pub use os_host_api::*;
