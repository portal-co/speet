//! `speet-rtd` daemon library.
//!
//! Thin wrapper around the generic `os-daemon`, registering speet's
//! existing AOT-recompile pipeline (`IntegratedNativeRuntime`) as the
//! `"integrated"` backend. The daemon itself no longer hardcodes which
//! transformation strategy produced a runnable artifact — future backends
//! (e.g. a dylib/so rewriter) register into the same `os_daemon::Daemon`
//! registry alongside this one.

pub mod ctl;
#[cfg(feature = "hot-recompiler")]
pub mod hot;
#[cfg(feature = "mcp")]
pub mod mcp;
mod simple_rewrite;
#[cfg(feature = "jit")]
mod vane_jit;

use os_daemon::Daemon as GenericDaemon;
use os_daemon_protocol::{decode_request, encode_response, Request, Response};
use speet_host_api::integrated_host_api;
use speet_runtime::{IntegratedNativeRuntime, RecompileReport, SharedIntegratedRuntime};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(feature = "hot-recompiler")]
use crate::hot::HotLoader;

pub fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("speet/rt-artifacts");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache/speet/rt-artifacts");
    }
    PathBuf::from("/tmp/speet-rt-artifacts")
}

pub struct Daemon {
    inner: GenericDaemon,
    rt: Arc<Mutex<IntegratedNativeRuntime>>,
    #[cfg(feature = "hot-recompiler")]
    hot: Mutex<HotLoader>,
}

impl Daemon {
    pub fn new() -> Self {
        let root = cache_root();
        let _ = std::fs::create_dir_all(&root);
        let rt = Arc::new(Mutex::new(
            IntegratedNativeRuntime::new(Arc::new(integrated_host_api())).with_disk_cache(root),
        ));
        let mut inner = GenericDaemon::new();
        inner.register(Box::new(SharedIntegratedRuntime::new(rt.clone())));
        maybe_register_simple_rewrite(&mut inner);
        #[cfg(feature = "jit")]
        {
            inner.register_jit_backend(Box::new(vane_jit::VaneWasmJitBackend));
            inner.register_jit_backend(Box::new(vane_jit::VaneWasmAarch64JitBackend));
            inner.register_jit_backend(Box::new(vane_jit::VaneBlitzJitBackend));
        }
        Self {
            inner,
            rt,
            #[cfg(feature = "hot-recompiler")]
            hot: Mutex::new(HotLoader::new()),
        }
    }

    #[cfg(feature = "hot-recompiler")]
    pub fn set_hot_watch_paths(&self, rec: PathBuf, stubs: PathBuf) {
        *self.hot.lock().unwrap() = HotLoader::with_paths(rec, stubs);
    }

    fn prepare(&self) {
        #[cfg(feature = "hot-recompiler")]
        {
            let mut hot = self.hot.lock().unwrap();
            if let Err(e) = hot.ensure_fresh() {
                eprintln!("hot-recompiler ensure_fresh: {e}");
            }
            self.rt.lock().unwrap().set_hot_plugins(
                hot.hashes(),
                hot.extra_wired(),
                hot.frontend(),
            );
        }
    }

    pub fn handle_frame(&self, frame: &[u8]) -> Vec<u8> {
        self.prepare();
        let req = match decode_request(frame) {
            Ok(r) => r,
            Err(_) => {
                return self.inner.handle_frame(frame);
            }
        };
        match req {
            Request::LastReport => encode_response(&Response::Report {
                json: self
                    .last_report()
                    .map(|r| r.to_json())
                    .unwrap_or_else(|| "{}".into()),
            }),
            Request::GuestInfo { path } => {
                let info = self.guest_info(Path::new(&path));
                encode_response(&Response::GuestInfo {
                    json: info.to_json(),
                })
            }
            Request::Run { path, argv } => match self.run_guest(Path::new(&path), &argv) {
                Ok((exit_code, stdout, stderr)) => encode_response(&Response::Run {
                    exit_code,
                    stdout,
                    stderr,
                }),
                Err(message) => encode_response(&Response::Error { message }),
            },
            _ => self.inner.handle_frame(frame),
        }
    }

    pub fn last_report(&self) -> Option<RecompileReport> {
        self.rt.lock().unwrap().last_report()
    }

    pub fn guest_info(&self, path: &Path) -> speet_runtime::GuestInfo {
        self.rt.lock().unwrap().guest_info(path)
    }

    pub fn analyze_report(&self, path: &Path) -> Result<RecompileReport, String> {
        self.prepare();
        self.rt.lock().unwrap().analyze_report(path)
    }

    pub fn recompile_report(&self, path: &Path, link: bool) -> RecompileReport {
        self.prepare();
        self.rt.lock().unwrap().recompile_report(path, link)
    }

    pub fn run_guest(&self, path: &Path, argv: &[String]) -> Result<(i32, String, String), String> {
        self.prepare();
        self.rt.lock().unwrap().run_guest(path, argv)
    }

    pub fn serve(&self, listener: UnixListener) -> Result<(), String> {
        for stream in listener.incoming() {
            let stream = stream.map_err(|e| e.to_string())?;
            let _ = handle_client(self, stream);
        }
        Ok(())
    }

    pub fn run_on_listener(listener: UnixListener) -> Result<(), String> {
        Daemon::new().serve(listener)
    }
}

fn handle_client(daemon: &Daemon, stream: std::os::unix::net::UnixStream) -> Result<(), String> {
    use os_daemon_protocol::{read_frame, write_frame};
    let mut reader = stream.try_clone().map_err(|e| e.to_string())?;
    let frame = read_frame(&mut reader)?;
    let resp = daemon.handle_frame(&frame);
    let mut sock = stream;
    write_frame(&mut sock, &resp)?;
    Ok(())
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
#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
))]
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
