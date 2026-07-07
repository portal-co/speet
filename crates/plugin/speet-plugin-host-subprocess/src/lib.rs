//! Subprocess plugin host — drives a plugin implemented as a separate OS
//! process, speaking the same five `speet-plugin-api` traits over the
//! [`frame`] protocol on its `stdin`/`stdout` pipes. **Trusted-unless-
//! externally-sandboxed path** — speet does not itself sandbox a subprocess
//! plugin; treat it as trusted unless the embedder wraps it in OS-level
//! sandboxing (seccomp/Landlock/containers). For plugins from sources you
//! do not fully trust, use `speet-plugin-host-wasm` instead. See
//! `docs/guides/plugin-api.md` §5.3.
//!
//! Built on `std::process::Command` only — no dependency beyond `std`.
//! Unsupported wherever `std::process` is unavailable (`wasm32-unknown-
//! unknown`, `no_std` embedded); an embedder simply omits this crate.

pub mod frame;
mod adapters;
mod process;
mod transport;

pub use adapters::{
    SubprocessAddressMapperPlugin, SubprocessArchPlugin, SubprocessMemoryAccessPlugin,
    SubprocessObjectModelPlugin, SubprocessTablePlugin, SubprocessTargetPlugin,
};
pub use process::{SubprocessError, SubprocessPlugin};
pub use transport::{SubprocessDescriptor, SubprocessTransport};
