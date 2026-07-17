//! IPC client for the recompile daemon (`speet-rtd`).

use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use crate::integrated::SuitabilityReport;
use crate::rtd_protocol::{
    decode_response, encode_request, read_frame, write_frame, Request, Response,
};

/// Backend id speet's own AOT recompiler registers under; matches
/// `os_transform_core::BackendId::INTEGRATED_RECOMPILE`.
const INTEGRATED_BACKEND: &str = "integrated";

/// Default Unix socket path when `SPEET_RTD_SOCK` is unset. Delegates to the
/// generic daemon's own fallback chain (`os_daemon::default_listen_path`) so
/// a `speet-rtd` binary and this client always agree on where to connect
/// when no socket env var is set.
pub fn default_socket_path() -> PathBuf {
    os_daemon::default_listen_path()
}

pub fn socket_path() -> PathBuf {
    std::env::var_os("SPEET_RTD_SOCK")
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}

fn send_request(sock: &mut UnixStream, req: &Request) -> Result<Response, String> {
    let frame = encode_request(req);
    write_frame(sock, &frame)?;
    let resp_frame = read_frame(sock)?;
    decode_response(&resp_frame).map_err(|_| "invalid rtd response".to_string())
}

/// Ping the daemon; returns false if unreachable.
pub fn ping() -> bool {
    let Ok(mut sock) = UnixStream::connect(socket_path()) else {
        return false;
    };
    send_request(&mut sock, &Request::Ping)
        .map(|r| matches!(r, Response::Pong))
        .unwrap_or(false)
}

/// Ask the daemon to analyze `path` against speet's own AOT-recompile backend.
pub fn analyze_remote(path: &Path) -> Result<Result<SuitabilityReport, SuitabilityReport>, String> {
    let mut sock = UnixStream::connect(socket_path())
        .map_err(|e| format!("connect {}: {e}", socket_path().display()))?;
    let resp = send_request(
        &mut sock,
        &Request::Analyze {
            path: path.display().to_string(),
            backend: Some(INTEGRATED_BACKEND.to_string()),
        },
    )?;
    parse_analyze_response(resp)
}

/// Ask the daemon for a cached/recompiled executable path from `backend`.
pub fn obtain_remote(path: &Path, backend: &str) -> Result<ObtainResponse, String> {
    let mut sock = UnixStream::connect(socket_path())
        .map_err(|e| format!("connect {}: {e}", socket_path().display()))?;
    let resp = send_request(
        &mut sock,
        &Request::Obtain {
            path: path.display().to_string(),
            backend: backend.to_string(),
        },
    )?;
    parse_obtain_response(resp)
}

#[derive(Debug, Clone)]
pub enum ObtainResponse {
    Ready { exe_path: PathBuf, cache_hit: bool },
    /// Free-form reasons the backend gave for rejecting the input. The wire
    /// protocol no longer splits these into typed `unresolved_deps`/
    /// `fn_ptr_deps` buckets (see `os-transform-core::Suitability`); nothing
    /// in this crate destructures those buckets from a remote response, so
    /// nothing is lost by carrying the flat list here.
    Unsuitable(Vec<String>),
    Error(String),
}

fn parse_analyze_response(
    resp: Response,
) -> Result<Result<SuitabilityReport, SuitabilityReport>, String> {
    match resp {
        Response::Suitable { .. } => Ok(Ok(SuitabilityReport {
            suitable: true,
            unresolved_deps: vec![],
            fn_ptr_deps: vec![],
        })),
        Response::Unsuitable { reasons, .. } => Ok(Err(SuitabilityReport {
            suitable: false,
            unresolved_deps: reasons,
            fn_ptr_deps: vec![],
        })),
        Response::Error { message } => Err(message),
        other => Err(format!("unexpected analyze response: {other:?}")),
    }
}

fn parse_obtain_response(resp: Response) -> Result<ObtainResponse, String> {
    match resp {
        Response::Ready { exe_path, cache_hit, .. } => Ok(ObtainResponse::Ready {
            exe_path: PathBuf::from(exe_path),
            cache_hit,
        }),
        Response::Unsuitable { reasons, .. } => Ok(ObtainResponse::Unsuitable(reasons)),
        Response::Error { message } => Ok(ObtainResponse::Error(message)),
        other => Err(format!("unexpected obtain response: {other:?}")),
    }
}

/// Emit C source for `__speet_execve_hook` that consults the daemon over the
/// generic `os-daemon-protocol` wire format, requesting speet's own
/// `"integrated"` AOT-recompile backend specifically.
pub fn generate_execve_hook_c() -> String {
    os_daemon_hook::generate_execve_hook_c("__speet_execve_hook", INTEGRATED_BACKEND, "SPEET_RTD_SOCK")
}
