//! `speet-rtd` daemon library.

use speet_host_api::integrated_host_api;
use speet_runtime::{
    default_socket_path, IntegratedNativeRuntime, NativeRuntime, ObtainError, SuitabilityReport,
};
use std::io::{BufRead, BufReader, Write};
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

    pub fn handle_line(&self, line: &str) -> String {
        let line = line.trim();
        if line.is_empty() {
            return error_response("empty request");
        }
        if line.contains(r#""op":"ping""#) {
            return r#"{"status":"pong"}"#.to_string();
        }
        if line.contains(r#""op":"analyze""#) {
            return self.handle_analyze(line);
        }
        if line.contains(r#""op":"obtain""#) {
            return self.handle_obtain(line);
        }
        error_response("unknown op")
    }

    fn handle_analyze(&self, line: &str) -> String {
        let Some(path) = extract_json_string_field(line, "path") else {
            return error_response("missing path");
        };
        let rt = self.rt.lock().unwrap();
        match rt.analyze(Path::new(&path)) {
            Ok(report) if report.suitable => r#"{"status":"suitable"}"#.to_string(),
            Ok(report) => unsuitable_response(&report),
            Err(e) => error_response(&e),
        }
    }

    fn handle_obtain(&self, line: &str) -> String {
        let Some(path) = extract_json_string_field(line, "path") else {
            return error_response("missing path");
        };
        let mut rt = self.rt.lock().unwrap();
        match rt.obtain_executable(Path::new(&path)) {
            Ok(exe) => {
                let cache_hit = false;
                format!(
                    r#"{{"status":"ready","exe_path":"{}","cache_hit":{}}}"#,
                    escape_json(&exe.display().to_string()),
                    cache_hit
                )
            }
            Err(ObtainError::Unsuitable(report)) => unsuitable_response(&report),
            Err(ObtainError::RecompileFailed(e)) => error_response(&e),
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
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let resp = daemon.handle_line(&line);
    let mut sock = stream;
    sock.write_all(resp.as_bytes())
        .and_then(|_| sock.write_all(b"\n"))
        .map_err(|e| e.to_string())?;
    sock.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn unsuitable_response(report: &SuitabilityReport) -> String {
    format!(
        r#"{{"status":"unsuitable","unresolved_deps":[{}],"fn_ptr_deps":[{}]}}"#,
        string_array_json(&report.unresolved_deps),
        string_array_json(&report.fn_ptr_deps)
    )
}

fn error_response(msg: &str) -> String {
    format!(r#"{{"status":"error","message":"{}"}}"#, escape_json(msg))
}

fn string_array_json(items: &[String]) -> String {
    items
        .iter()
        .map(|s| format!("\"{}\"", escape_json(s)))
        .collect::<Vec<_>>()
        .join(",")
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn extract_json_string_field(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => break,
            '\\' => match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    Some(out)
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
