//! Compatibility shim for `os-ctx`.
//!
//! `osctx` now lives in `os-emulation/crates/runtime/os-ctx`. This crate
//! re-exports the same public API so existing `@speet` consumers keep
//! compiling until they migrate to `os-ctx` directly.

pub use os_ctx::*;
