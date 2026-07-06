//! Pluggable host API for the thin runtime.
//!
//! The default [`TunneledHostApi`] resolves guest externals through the
//! [`tunnel`] allowlist. Embeddings swap in [`FilteredHostApi`] or future
//! backends without changing the recompiler pipeline.

mod filtered;
mod manifest;
mod redirecting;
mod registry;
mod tunneled;

pub use filtered::{FilteredHostApi, HostPolicy};
pub use manifest::{FuncImport, ImportManifest, LinkRecipe};
pub use redirecting::{PltRedirect, RedirectingHostApi};
pub use registry::HostApiRegistry;
pub use tunneled::TunneledHostApi;

use binary_io::{BinArch, BinOs};
use tunnel::TunnelResolution;

/// Boundary between recompiled guest code and the host operating system.
pub trait HostApi: Send + Sync {
    /// WASM func imports the C link shim must define (e.g. `env__exit`).
    fn import_manifest(&self) -> ImportManifest;

    /// Resolve a guest external symbol for link-time alias emission.
    fn resolve_ambient(&self, guest_name: &str) -> Option<TunnelResolution>;

    /// Dylib and linker flags for the final link step.
    fn link_recipe(&self) -> LinkRecipe;

    /// Compile-time redirect for a guest PLT/external symbol (integrated hooks).
    fn resolve_plt_redirect(&self, _guest_symbol: &str) -> Option<PltRedirect> {
        None
    }

    /// Optional dynamic syscall dispatch for unknown numbers (vkernel path).
    fn syscall(&mut self, _nr: u64, _args: &[u64]) -> i64 {
        -1 // ENOSYS
    }
}

/// Pick the default tunnel for the host OS.
pub fn default_tunnel() -> Box<dyn tunnel::Tunnel + Send + Sync> {
    if cfg!(target_os = "macos") {
        Box::new(tunnel::MacLibSystemTunnel)
    } else {
        Box::new(tunnel::LinuxLibcTunnel)
    }
}

/// Default [`HostApi`] for the current host platform.
pub fn default_host_api() -> TunneledHostApi {
    TunneledHostApi::for_host()
}

/// Default [`HostApi`] for the integrated thin runtime (with PLT hooks).
pub fn integrated_host_api() -> RedirectingHostApi<TunneledHostApi> {
    RedirectingHostApi::integrated(
        TunneledHostApi::for_host().with_manifest(ImportManifest::integrated_native()),
    )
}

/// Target arch/OS for linking recompiled output on this host.
pub fn host_link_target() -> (BinArch, BinOs) {
    let os = if cfg!(target_os = "macos") {
        BinOs::MacOs
    } else {
        BinOs::Linux
    };
    let arch = if cfg!(target_arch = "aarch64") {
        BinArch::AArch64
    } else {
        BinArch::X86_64
    };
    (arch, os)
}
