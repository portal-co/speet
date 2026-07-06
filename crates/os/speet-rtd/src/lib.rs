//! `speet-rtd` daemon library.

use speet_host_api::integrated_host_api;
use speet_runtime::{
    default_socket_path, IntegratedNativeRuntime, NativeRuntime, ObtainError, SuitabilityReport,
};
use speet_runtime::rtd_protocol::{
    decode_request, encode_response, read_frame, write_frame, Request, Response,
};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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
    rt: Mutex<IntegratedNativeRuntime>,
}

impl Daemon {
    pub fn new() -> Self {
        let root = cache_root();
        let _ = std::fs::create_dir_all(&root);
        let rt = IntegratedNativeRuntime::new(Arc::new(integrated_host_api())).with_disk_cache(root);
        Self {
            rt: Mutex::new(rt),
        }
    }

    pub fn handle_frame(&self, frame: &[u8]) -> Vec<u8> {
        let req = match decode_request(frame) {
            Ok(r) => r,
            Err(_) => {
                return encode_response(&Response::Error {
                    message: "invalid request".into(),
                });
            }
        };
        encode_response(&self.dispatch(req))
    }

    fn dispatch(&self, req: Request) -> Response {
        match req {
            Request::Ping => Response::Pong,
            Request::Analyze { path } => self.handle_analyze(&path),
            Request::Obtain { path, host_id: _ } => self.handle_obtain(&path),
        }
    }

    fn handle_analyze(&self, path: &str) -> Response {
        let rt = self.rt.lock().unwrap();
        match rt.analyze(Path::new(path)) {
            Ok(report) if report.suitable => Response::Suitable,
            Ok(report) => unsuitable_response(&report),
            Err(e) => Response::Error { message: e },
        }
    }

    fn handle_obtain(&self, path: &str) -> Response {
        let mut rt = self.rt.lock().unwrap();
        match rt.obtain_executable(Path::new(path)) {
            Ok(exe) => Response::Ready {
                exe_path: exe.display().to_string(),
                cache_hit: false,
            },
            Err(ObtainError::Unsuitable(report)) => unsuitable_response(&report),
            Err(ObtainError::RecompileFailed(e)) => Response::Error { message: e },
        }
    }

    pub fn run_on_listener(listener: UnixListener) -> Result<(), String> {
        let daemon = Daemon::new();
        for stream in listener.incoming() {
            let stream = stream.map_err(|e| e.to_string())?;
            let _ = handle_client(&daemon, stream);
        }
        Ok(())
    }
}

fn handle_client(daemon: &Daemon, stream: UnixStream) -> Result<(), String> {
    let mut reader = stream.try_clone().map_err(|e| e.to_string())?;
    let frame = read_frame(&mut reader)?;
    let resp = daemon.handle_frame(&frame);
    let mut sock = stream;
    write_frame(&mut sock, &resp)?;
    Ok(())
}

fn unsuitable_response(report: &SuitabilityReport) -> Response {
    Response::Unsuitable {
        unresolved_deps: report.unresolved_deps.clone(),
        fn_ptr_deps: report.fn_ptr_deps.clone(),
    }
}

pub fn bind_socket(path: &Path) -> Result<UnixListener, String> {
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    UnixListener::bind(path).map_err(|e| e.to_string())
}

pub fn default_listen_path() -> PathBuf {
    default_socket_path()
}
