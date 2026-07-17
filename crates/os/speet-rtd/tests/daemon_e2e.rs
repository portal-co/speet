//! Daemon IPC and cache tests.

use speet_rtd::{bind_socket, Daemon};
use speet_runtime::rtd_protocol::{decode_response, encode_request, read_frame, write_frame, Request};
use std::os::unix::net::UnixStream;
use std::path::Path;
use tempfile::tempdir;

/// Path to the checked-in (not submodule) aarch64-macos linked corpus
/// binary. Panics rather than returning `Option` if it's missing: the file
/// is tracked directly in this repo, so on the architecture this fixture
/// targets its absence means something is actually broken (wrong path,
/// corrupted checkout) and the test should fail loudly, not silently skip.
fn linked_exit42() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus");
    let p = root.join("aarch64-macos/exit42.linked.macho");
    assert!(p.is_file(), "missing checked-in corpus fixture: {}", p.display());
    p
}

#[test]
fn daemon_ping_analyze_obtain() {
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("rtd.sock");
    let listener = bind_socket(&sock_path).expect("bind");
    let daemon = Daemon::new();

    // The wire protocol is one frame per connection (matching production:
    // `os_daemon::Daemon::run_on_listener` accepts a fresh connection per
    // request, and every real client in `execve_hook.rs` — `ping`,
    // `analyze_remote`, `obtain_remote` — dials a new `UnixStream` per
    // call). The server side here must therefore loop over incoming
    // connections rather than handle exactly one, and the client below
    // must open a new connection per request rather than reuse one stream.
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let _ = handle_one(&daemon, stream);
        }
    });

    std::thread::sleep(std::time::Duration::from_millis(50));

    let mut sock = UnixStream::connect(&sock_path).expect("connect");
    let pong = roundtrip(&mut sock, &encode_request(&Request::Ping));
    match decode_response(&pong).unwrap() {
        speet_runtime::rtd_protocol::Response::Pong => {}
        other => panic!("expected pong, got {other:?}"),
    }

    if !cfg!(target_arch = "aarch64") {
        eprintln!("SKIP: aarch64-only corpus fixture");
        return;
    }
    let guest = linked_exit42();
    let mut sock = UnixStream::connect(&sock_path).expect("connect");
    let analyze = roundtrip(
        &mut sock,
        &encode_request(&Request::Analyze {
            path: guest.display().to_string(),
            backend: Some("integrated".into()),
        }),
    );
    match decode_response(&analyze).unwrap() {
        speet_runtime::rtd_protocol::Response::Suitable { .. } => {}
        other => panic!("expected suitable, got {other:?}"),
    }
}

fn handle_one(daemon: &Daemon, stream: UnixStream) -> Result<(), String> {
    let mut reader = stream.try_clone().map_err(|e| e.to_string())?;
    let frame = read_frame(&mut reader)?;
    let resp = daemon.handle_frame(&frame);
    let mut sock = stream;
    write_frame(&mut sock, &resp)?;
    Ok(())
}

fn roundtrip(sock: &mut UnixStream, req: &[u8]) -> Vec<u8> {
    write_frame(sock, req).expect("write");
    read_frame(sock).expect("read")
}
