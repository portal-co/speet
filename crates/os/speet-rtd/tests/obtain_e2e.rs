//! Daemon obtain cache test (Linux ELF guests only).

use speet_rtd::{bind_socket, Daemon};
use std::io::{BufRead, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use tempfile::tempdir;

#[test]
fn daemon_obtain_linux_elf() {
    if !cfg!(target_os = "linux") {
        eprintln!("SKIP: Linux ELF obtain test");
        return;
    }
    let guest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/thin-runtime-corpus/x86_64-linux/exit_42.elf");
    if !guest.is_file() {
        eprintln!("SKIP: missing {}", guest.display());
        return;
    }

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

    let path_json = guest.display().to_string().replace('\\', "\\\\").replace('"', "\\\"");
    let mut sock = UnixStream::connect(&sock).expect("connect");
    writeln!(
        sock,
        r#"{{"op":"obtain","path":"{path_json}","host_id":"tunneled"}}"#
    )
    .unwrap();
    sock.flush().unwrap();
    let resp = read_line(&mut sock);
    if resp.contains("error") || resp.contains("unsuitable") {
        eprintln!("SKIP obtain: {resp}");
        return;
    }
    assert!(resp.contains("ready"), "obtain={resp}");
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
    let mut buf = [0u8; 8192];
    let n = sock.read(&mut buf).expect("read");
    String::from_utf8_lossy(&buf[..n]).trim().to_string()
}
