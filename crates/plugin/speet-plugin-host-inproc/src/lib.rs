//! `speet-plugin-host-inproc` — the in-process plugin host. **Trusted /
//! first-party path** (see `docs/guides/plugin-api.md` §5): the host process
//! grants the plugin its own full privileges, speet does nothing to contain
//! it.
//!
//! Two modes:
//! - **Static** (always available, zero extra deps): the embedder
//!   constructs their plugin struct directly and registers it — see
//!   [`register_arch`] and friends. No `PluginTransport` involved; there is
//!   no boundary to cross.
//! - **Dylib** (`feature = "dylib"`, off by default): a separately-compiled
//!   `cdylib` loaded at runtime via `libloading`, stabilized through a
//!   plain `extern "C"`/`#[repr(C)]` surface — never a Rust trait object —
//!   so it does not depend on matching `rustc` versions. See [`dylib`].

#[cfg(all(
    feature = "dylib",
    not(any(target_os = "linux", target_os = "macos", target_os = "windows"))
))]
compile_error!(
    "speet-plugin-host-inproc's \"dylib\" feature requires a platform with \
     dynamic-linking support (linux/macos/windows). Disable the feature for \
     this target instead of relying on a runtime failure."
);

pub mod static_mode;

#[cfg(feature = "dylib")]
pub mod dylib;

pub use static_mode::*;
