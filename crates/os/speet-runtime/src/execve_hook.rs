//! IPC client for the recompile daemon (`speet-rtd`).

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use crate::integrated::SuitabilityReport;

/// Default Unix socket path when `SPEET_RTD_SOCK` is unset.
pub fn default_socket_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(dir).join("speet-rtd.sock");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache/speet")
            .join("rtd.sock");
    }
    PathBuf::from("/tmp/speet-rtd.sock")
}

pub fn socket_path() -> PathBuf {
    std::env::var_os("SPEET_RTD_SOCK")
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}

fn send_request(sock: &mut UnixStream, line: &str) -> Result<String, String> {
    sock.write_all(line.as_bytes())
        .and_then(|_| sock.write_all(b"\n"))
        .map_err(|e| e.to_string())?;
    sock.flush().map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(sock.try_clone().map_err(|e| e.to_string())?);
    let mut resp = String::new();
    reader.read_line(&mut resp).map_err(|e| e.to_string())?;
    Ok(resp.trim().to_string())
}

/// Ping the daemon; returns false if unreachable.
pub fn ping() -> bool {
    let Ok(mut sock) = UnixStream::connect(socket_path()) else {
        return false;
    };
    send_request(&mut sock, r#"{"op":"ping"}"#)
        .map(|r| r.contains("pong"))
        .unwrap_or(false)
}

/// Ask the daemon to analyze `path`.
pub fn analyze_remote(path: &Path) -> Result<Result<SuitabilityReport, SuitabilityReport>, String> {
    let mut sock = UnixStream::connect(socket_path())
        .map_err(|e| format!("connect {}: {e}", socket_path().display()))?;
    let req = format!(
        r#"{{"op":"analyze","path":{}}}"#,
        serde_json_string(path.display().to_string())
    );
    let resp = send_request(&mut sock, &req)?;
    parse_analyze_response(&resp)
}

/// Ask the daemon for a cached/recompiled executable path.
pub fn obtain_remote(path: &Path, host_id: &str) -> Result<ObtainResponse, String> {
    let mut sock = UnixStream::connect(socket_path())
        .map_err(|e| format!("connect {}: {e}", socket_path().display()))?;
    let req = format!(
        r#"{{"op":"obtain","path":{},"host_id":{}}}"#,
        serde_json_string(path.display().to_string()),
        serde_json_string(host_id.to_string())
    );
    let resp = send_request(&mut sock, &req)?;
    parse_obtain_response(&resp)
}

#[derive(Debug, Clone)]
pub enum ObtainResponse {
    Ready { exe_path: PathBuf, cache_hit: bool },
    Unsuitable(SuitabilityReport),
    Error(String),
}

fn serde_json_string(s: String) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn parse_analyze_response(
    line: &str,
) -> Result<Result<SuitabilityReport, SuitabilityReport>, String> {
    if line.contains(r#""status":"suitable""#) {
        return Ok(Ok(SuitabilityReport {
            suitable: true,
            unresolved_deps: vec![],
            fn_ptr_deps: vec![],
        }));
    }
    if line.contains(r#""status":"unsuitable""#) {
        let report = parse_unsuitable_fields(line);
        return Ok(Err(report));
    }
    Err(format!("unexpected analyze response: {line}"))
}

fn parse_obtain_response(line: &str) -> Result<ObtainResponse, String> {
    if line.contains(r#""status":"ready""#) {
        let path = extract_json_string_field(line, "exe_path").ok_or_else(|| {
            format!("missing exe_path in ready response: {line}")
        })?;
        let cache_hit = line.contains(r#""cache_hit":true"#);
        return Ok(ObtainResponse::Ready {
            exe_path: PathBuf::from(path),
            cache_hit,
        });
    }
    if line.contains(r#""status":"unsuitable""#) {
        return Ok(ObtainResponse::Unsuitable(parse_unsuitable_fields(line)));
    }
    if line.contains(r#""status":"error""#) {
        let msg = extract_json_string_field(line, "message").unwrap_or_else(|| line.to_string());
        return Ok(ObtainResponse::Error(msg));
    }
    Err(format!("unexpected obtain response: {line}"))
}

fn parse_unsuitable_fields(line: &str) -> SuitabilityReport {
    SuitabilityReport {
        suitable: false,
        unresolved_deps: extract_json_array_field(line, "unresolved_deps"),
        fn_ptr_deps: extract_json_array_field(line, "fn_ptr_deps"),
    }
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
                'r' => out.push('\r'),
                't' => out.push('\t'),
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    Some(out)
}

fn extract_json_array_field(line: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\":[");
    let Some(start) = line.find(&needle) else {
        return vec![];
    };
    let rest = &line[start + needle.len()..];
    let end = rest.find(']').unwrap_or(rest.len());
    let inner = &rest[..end];
    inner
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                Some(s[1..s.len() - 1].to_string())
            } else {
                None
            }
        })
        .collect()
}

/// Emit C source for `__speet_execve_hook` that consults the daemon.
pub fn generate_execve_hook_c() -> String {
    format!(
        r#"#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

static const char *rtd_socket_path(void) {{
    const char *env = getenv("SPEET_RTD_SOCK");
    if (env && env[0]) return env;
    const char *xdg = getenv("XDG_RUNTIME_DIR");
    if (xdg && xdg[0]) {{
        static char buf[512];
        snprintf(buf, sizeof(buf), "%s/speet-rtd.sock", xdg);
        return buf;
    }}
    const char *home = getenv("HOME");
    if (home && home[0]) {{
        static char buf[512];
        snprintf(buf, sizeof(buf), "%s/.cache/speet/rtd.sock", home);
        return buf;
    }}
    return "/tmp/speet-rtd.sock";
}}

static int rtd_obtain_path(const char *guest_path, char *out, size_t out_len) {{
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    const char *sock = rtd_socket_path();
    if (strlen(sock) >= sizeof(addr.sun_path)) {{ close(fd); return -1; }}
    strncpy(addr.sun_path, sock, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {{ close(fd); return -1; }}
    char req[1024];
    int n = snprintf(req, sizeof(req), "{{\"op\":\"obtain\",\"path\":\"%s\",\"host_id\":\"tunneled\"}}\n", guest_path);
    if (write(fd, req, (size_t)n) != (ssize_t)n) {{ close(fd); return -1; }}
    char resp[2048];
    ssize_t r = read(fd, resp, sizeof(resp) - 1);
    close(fd);
    if (r <= 0) return -1;
    resp[r] = '\0';
    const char *key = "\"exe_path\":\"";
    char *p = strstr(resp, key);
    if (!p) return -1;
    p += strlen(key);
    char *end = strchr(p, '"');
    if (!end) return -1;
    size_t len = (size_t)(end - p);
    if (len + 1 > out_len) return -1;
    memcpy(out, p, len);
    out[len] = '\0';
    return 0;
}}

int __speet_execve_hook(const char *path, char *const argv[], char *const envp[]) {{
    char cached[1024];
    if (rtd_obtain_path(path, cached, sizeof(cached)) == 0) {{
        return execve(cached, argv, envp);
    }}
    return execve(path, argv, envp);
}}
"#
    )
}
