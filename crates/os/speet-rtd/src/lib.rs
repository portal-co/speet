//! `speet-rtd` daemon library.
//!
//! Thin wrapper around the generic `os-daemon`, registering speet's
//! existing AOT-recompile pipeline (`IntegratedNativeRuntime`) as the
//! `"integrated"` backend. The daemon itself no longer hardcodes which
//! transformation strategy produced a runnable artifact — future backends
//! (e.g. a dylib/so rewriter) register into the same `os_daemon::Daemon`
//! registry alongside this one.

mod simple_rewrite;
#[cfg(feature = "jit")]
mod vane_jit;

use os_daemon::Daemon as GenericDaemon;
use speet_host_api::integrated_host_api;
use speet_runtime::IntegratedNativeRuntime;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::Arc;

pub fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("speet/rt-artifacts");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache/speet/rt-artifacts");
    }
    PathBuf::from("/tmp/speet-rt-artifacts")
}

pub struct Daemon(GenericDaemon);

impl Daemon {
    pub fn new() -> Self {
        let root = cache_root();
        let _ = std::fs::create_dir_all(&root);
        let rt = IntegratedNativeRuntime::new(Arc::new(integrated_host_api())).with_disk_cache(root);
        let mut inner = GenericDaemon::new();
        inner.register(Box::new(rt));
        maybe_register_simple_rewrite(&mut inner);
        #[cfg(feature = "jit")]
        {
            inner.register_jit_backend(Box::new(vane_jit::VaneWasmJitBackend));
            inner.register_jit_backend(Box::new(vane_jit::VaneWasmAarch64JitBackend));
            inner.register_jit_backend(Box::new(vane_jit::VaneBlitzJitBackend));
        }
        Self(inner)
    }

    pub fn handle_frame(&self, frame: &[u8]) -> Vec<u8> {
        self.0.handle_frame(frame)
    }

    pub fn run_on_listener(listener: UnixListener) -> Result<(), String> {
        Daemon::new().0.run_on_listener(listener)
    }
}

impl Default for Daemon {
    fn default() -> Self {
        Self::new()
    }
}

pub use os_daemon::bind_socket;

pub fn default_listen_path() -> PathBuf {
    speet_runtime::default_socket_path()
}

/// Resolve the interposer shim path: explicit `SIMPLE_REWRITE_SHIM`, or the
/// built-in `os-interposer` artifact when `OS_USE_BUILTIN_INTERPOSER=1`.
fn simple_rewrite_shim_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SIMPLE_REWRITE_SHIM") {
        return Some(PathBuf::from(path));
    }
    if std::env::var_os("OS_USE_BUILTIN_INTERPOSER").is_some_and(|v| !v.is_empty() && v != "0") {
        let path = os_interposer::lib_path();
        if path.exists() {
            return Some(path);
        }
        eprintln!(
            "simple-rewrite: OS_USE_BUILTIN_INTERPOSER set but {} missing (build os-interposer first)",
            path.display()
        );
    }
    None
}

/// Registers the `"simple-rewrite"` backend only if a shim path is configured.
#[cfg(target_os = "macos")]
fn maybe_register_simple_rewrite(daemon: &mut GenericDaemon) {
    use simple_rewrite::{SimpleRewriteBackend, SimpleRewriteConfig};

    let Some(shim_path) = simple_rewrite_shim_path() else {
        return;
    };
    let Ok(identity) = std::env::var("SANDBOX_CODESIGN_IDENTITY") else {
        eprintln!("simple-rewrite: SIMPLE_REWRITE_SHIM is set but SANDBOX_CODESIGN_IDENTITY is not; skipping");
        return;
    };
    let identity = if identity == "-" {
        os_codesign_macho::SigningIdentity::AdHoc
    } else {
        os_codesign_macho::SigningIdentity::Real(identity)
    };
    let config = SimpleRewriteConfig {
        shim_path: PathBuf::from(shim_path),
        cache_root: cache_root().join("rewrite"),
        identity,
        keychain: std::env::var("SANDBOX_CODESIGN_KEYCHAIN").ok(),
        allow_get_task_allow: std::env::var("SANDBOX_ALLOW_GET_TASK_ALLOW").as_deref() == Ok("1"),
    };
    daemon.register(Box::new(SimpleRewriteBackend::new(config)));
}

/// Registers the `"simple-rewrite"` backend when a shim path is available.
#[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd", target_os = "netbsd"))]
fn maybe_register_simple_rewrite(daemon: &mut GenericDaemon) {
    use simple_rewrite::{SimpleRewriteBackend, SimpleRewriteConfig};

    let Some(shim_path) = simple_rewrite_shim_path() else {
        return;
    };
    let config = SimpleRewriteConfig {
        shim_path: PathBuf::from(shim_path),
        cache_root: cache_root().join("rewrite"),
    };
    daemon.register(Box::new(SimpleRewriteBackend::new(config)));
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
)))]
fn maybe_register_simple_rewrite(_daemon: &mut GenericDaemon) {}
