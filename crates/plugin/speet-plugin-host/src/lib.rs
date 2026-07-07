//! `speet-plugin-host` — the [`PluginTransport`] seam, [`PluginHandle`], and
//! [`PluginRegistry`] shared by every host backend.
//!
//! This crate has no engine-specific dependencies — only `speet-plugin-api`
//! types and a transport trait object. Backend crates
//! (`speet-plugin-host-inproc`/`-wasm`/`-subprocess`) depend on this crate to
//! implement [`PluginTransport`]; this crate never depends on them. That is
//! what lets a 4th host be added later as a new leaf crate without touching
//! this one. See `docs/guides/plugin-api.md`.

#![no_std]

extern crate alloc;

mod manifest;
mod registry;
mod transport;

pub use manifest::{parse_manifest, ManifestEntry, ManifestError};
pub use registry::{PluginRegistry, RestrictedHostImports};
pub use transport::{PluginHandle, PluginLoadError, PluginTransport};

pub use speet_plugin_api::PluginKind;
