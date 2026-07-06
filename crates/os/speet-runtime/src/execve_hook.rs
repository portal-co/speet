//! IPC client for the recompile daemon (`speet-rtd`).

use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use crate::integrated::SuitabilityReport;
use crate::rtd_protocol::{
    decode_response, encode_request, read_frame, write_frame, Request, Response,
};

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

/// Ask the daemon to analyze `path`.
pub fn analyze_remote(path: &Path) -> Result<Result<SuitabilityReport, SuitabilityReport>, String> {
    let mut sock = UnixStream::connect(socket_path())
        .map_err(|e| format!("connect {}: {e}", socket_path().display()))?;
    let resp = send_request(
        &mut sock,
        &Request::Analyze {
            path: path.display().to_string(),
        },
    )?;
    parse_analyze_response(resp)
}

/// Ask the daemon for a cached/recompiled executable path.
pub fn obtain_remote(path: &Path, host_id: &str) -> Result<ObtainResponse, String> {
    let mut sock = UnixStream::connect(socket_path())
        .map_err(|e| format!("connect {}: {e}", socket_path().display()))?;
    let resp = send_request(
        &mut sock,
        &Request::Obtain {
            path: path.display().to_string(),
            host_id: host_id.to_string(),
        },
    )?;
    parse_obtain_response(resp)
}

#[derive(Debug, Clone)]
pub enum ObtainResponse {
    Ready { exe_path: PathBuf, cache_hit: bool },
    Unsuitable(SuitabilityReport),
    Error(String),
}

fn parse_analyze_response(
    resp: Response,
) -> Result<Result<SuitabilityReport, SuitabilityReport>, String> {
    match resp {
        Response::Suitable => Ok(Ok(SuitabilityReport {
            suitable: true,
            unresolved_deps: vec![],
            fn_ptr_deps: vec![],
        })),
        Response::Unsuitable {
            unresolved_deps,
            fn_ptr_deps,
        } => Ok(Err(SuitabilityReport {
            suitable: false,
            unresolved_deps,
            fn_ptr_deps,
        })),
        Response::Error { message } => Err(message),
        other => Err(format!("unexpected analyze response: {other:?}")),
    }
}

fn parse_obtain_response(resp: Response) -> Result<ObtainResponse, String> {
    match resp {
        Response::Ready { exe_path, cache_hit } => Ok(ObtainResponse::Ready {
            exe_path: PathBuf::from(exe_path),
            cache_hit,
        }),
        Response::Unsuitable {
            unresolved_deps,
            fn_ptr_deps,
        } => Ok(ObtainResponse::Unsuitable(SuitabilityReport {
            suitable: false,
            unresolved_deps,
            fn_ptr_deps,
        })),
        Response::Error { message } => Ok(ObtainResponse::Error(message)),
        other => Err(format!("unexpected obtain response: {other:?}")),
    }
}

/// Emit C source for `__speet_execve_hook` that consults the daemon (binary protocol).
pub fn generate_execve_hook_c() -> String {
    format!(
        r#"#include <stdint.h>
#include <stdio.h>
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

static int wire_write_u32(uint8_t *out, size_t cap, size_t *pos, uint32_t v) {{
    if (*pos + 4 > cap) return -1;
    out[*pos + 0] = (uint8_t)(v);
    out[*pos + 1] = (uint8_t)(v >> 8);
    out[*pos + 2] = (uint8_t)(v >> 16);
    out[*pos + 3] = (uint8_t)(v >> 24);
    *pos += 4;
    return 0;
}}

static int wire_write_str(uint8_t *out, size_t cap, size_t *pos, const char *s) {{
    size_t slen = strlen(s);
    if (wire_write_u32(out, cap, pos, (uint32_t)slen) != 0) return -1;
    if (*pos + slen > cap) return -1;
    memcpy(out + *pos, s, slen);
    *pos += slen;
    return 0;
}}

static int wire_read_u32(const uint8_t *in, size_t len, size_t *pos, uint32_t *out) {{
    if (*pos + 4 > len) return -1;
    *out = (uint32_t)in[*pos]
         | ((uint32_t)in[*pos + 1] << 8)
         | ((uint32_t)in[*pos + 2] << 16)
         | ((uint32_t)in[*pos + 3] << 24);
    *pos += 4;
    return 0;
}}

static int wire_read_str(const uint8_t *in, size_t len, size_t *pos, char *out, size_t out_cap) {{
    uint32_t slen = 0;
    if (wire_read_u32(in, len, pos, &slen) != 0) return -1;
    if (*pos + slen > len || slen + 1 > out_cap) return -1;
    memcpy(out, in + *pos, slen);
    out[slen] = '\0';
    *pos += slen;
    return 0;
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

    uint8_t req[1024];
    size_t pos = 0;
    req[pos++] = 1; /* PROTOCOL_VERSION */
    req[pos++] = 2; /* OP_OBTAIN */
    size_t payload_start = pos;
    pos += 4;
    size_t payload_pos = 0;
    uint8_t payload[512];
    if (wire_write_str(payload, sizeof(payload), &payload_pos, guest_path) != 0) {{ close(fd); return -1; }}
    if (wire_write_str(payload, sizeof(payload), &payload_pos, "integrated") != 0) {{ close(fd); return -1; }}
    if (wire_write_u32(req, sizeof(req), &payload_start, (uint32_t)payload_pos) != 0) {{ close(fd); return -1; }}
    if (pos + payload_pos > sizeof(req)) {{ close(fd); return -1; }}
    memcpy(req + pos, payload, payload_pos);
    pos += payload_pos;

    if (write(fd, req, pos) != (ssize_t)pos) {{ close(fd); return -1; }}

    uint8_t hdr[6];
    ssize_t r = read(fd, hdr, sizeof(hdr));
    if (r != (ssize_t)sizeof(hdr)) {{ close(fd); return -1; }}
    uint32_t plen = 0;
    size_t p = 2;
    if (wire_read_u32(hdr, sizeof(hdr), &p, &plen) != 0) {{ close(fd); return -1; }}
    if (hdr[0] != 1 || hdr[1] != 3) {{ close(fd); return -1; }} /* STATUS_READY */

    uint8_t body[2048];
    if (plen > sizeof(body)) {{ close(fd); return -1; }}
    r = read(fd, body, plen);
    close(fd);
    if (r != (ssize_t)plen) return -1;

    size_t bp = 0;
    return wire_read_str(body, plen, &bp, out, out_len);
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
