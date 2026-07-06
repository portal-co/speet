//! Daemon IPC and cache tests.

use speet_rtd::{bind_socket, Daemon};
use speet_runtime::rtd_protocol::{decode_response, encode_request, read_frame, write_frame, Request};
use std::os::unix::net::UnixStream;
use std::path::Path;
use tempfile::tempdir;

fn linked_exit42() -> Option<std::path::PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/c-corpus");
    if cfg!(target_arch = "aarch64") {
        let p = root.join("aarch64-macos/exit42.linked.macho");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[test]
fn daemon_ping_analyze_obtain() {
    let dir = tempdir().unwrap();
    let sock = dir.path().join("rtd.sock");
    let listener = bind_socket(&sock).expect("bind");
    let daemon = Daemon::new();

    std::thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            let _ = handle_one(&daemon, stream);
        }
    });

    std::thread::sleep(std::time::Duration::from_millis(50));

    let mut sock = UnixStream::connect(&sock).expect("connect");
    let pong = roundtrip(&mut sock, &encode_request(&Request::Ping));
    match decode_response(&pong).unwrap() {
        speet_runtime::rtd_protocol::Response::Pong => {}
        other => panic!("expected pong, got {other:?}"),
    }

    let Some(guest) = linked_exit42() else {
        eprintln!("SKIP: no exit42 artifact");
        return;
    };
    let analyze = roundtrip(
        &mut sock,
        &encode_request(&Request::Analyze {
            path: guest.display().to_string(),
        }),
    );
    match decode_response(&analyze).unwrap() {
        speet_runtime::rtd_protocol::Response::Suitable => {}
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
