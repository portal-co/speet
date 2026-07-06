//! Daemon IPC and cache tests.

use speet_rtd::{bind_socket, Daemon};
use std::io::{Read, Write};
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
    writeln!(sock, r#"{{"op":"ping"}}"#).unwrap();
    sock.flush().unwrap();
    let pong = read_line(&mut sock);
    assert!(pong.contains("pong"), "pong={pong}");

    let Some(guest) = linked_exit42() else {
        eprintln!("SKIP: no exit42 artifact");
        return;
    };
    let path_json = guest.display().to_string().replace('\\', "\\\\").replace('"', "\\\"");
    writeln!(sock, r#"{{"op":"analyze","path":"{path_json}"}}"#).unwrap();
    sock.flush().unwrap();
    let analyze = read_line(&mut sock);
    assert!(
        analyze.contains("suitable"),
        "analyze={analyze}"
    );
}

fn handle_one(daemon: &Daemon, stream: UnixStream) -> Result<(), String> {
    let mut reader = std::io::BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    std::io::BufRead::read_line(&mut reader, &mut line).map_err(|e| e.to_string())?;
    let resp = daemon.handle_line(&line);
    let mut sock = stream;
    sock.write_all(resp.as_bytes())
        .and_then(|_| sock.write_all(b"\n"))
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn read_line(sock: &mut UnixStream) -> String {
    let mut buf = [0u8; 4096];
    let n = sock.read(&mut buf).expect("read");
    String::from_utf8_lossy(&buf[..n]).trim().to_string()
}
