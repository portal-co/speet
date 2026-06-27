//! WASM plugin host — sandboxes a `.wasm` guest module behind the same five
//! `speet-plugin-api` traits the in-process host implements directly.
//! **Untrusted / third-party path** — recommended default for plugins from
//! sources you do not fully trust. See `docs/guides/plugin-api.md` §5.2.
//!
//! Engine: `wasmi` (default, lighter interpreter, no_std-friendlier, faster
//! cold start) — mirrors `crates/test/speet-e2e`'s `run_module`/
//! `run_module_wasmtime` dual-engine precedent. The `wasmtime`/`"exceptions"`
//! engine for plugins specifically needing exception-handling-proposal
//! support is a later phase, not required alongside this one.
//!
//! See [`abi`] for the exact export/import surface a `.wasm` plugin must
//! implement.

pub mod abi;
mod adapters;
mod engine;
mod host_calls;
mod transport;

pub use adapters::{
    WasmAddressMapperPlugin, WasmArchPlugin, WasmMemoryAccessPlugin, WasmObjectModelPlugin,
    WasmTablePlugin, WasmTargetPlugin,
};
pub use engine::{WasmLoadError, WasmPlugin};
pub use host_calls::HostState;
pub use transport::{WasmDescriptor, WasmTransport};
