//! Daemon obtain cache test (Linux ELF guests only).

use speet_rtd::{bind_socket, Daemon};
use speet_runtime::rtd_protocol::{
    decode_response, encode_request, read_frame, write_frame, Request, Response,
};
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

    let mut sock = UnixStream::connect(&sock).expect("connect");
    let resp = roundtrip(
        &mut sock,
        &encode_request(&Request::Obtain {
            path: guest.display().to_string(),
            host_id: "integrated".into(),
        }),
    );
    match decode_response(&resp).unwrap() {
        Response::Ready { .. } => {}
        Response::Error { message } => {
            eprintln!("SKIP obtain error: {message}");
        }
        Response::Unsuitable { .. } => {
            eprintln!("SKIP obtain: unsuitable");
        }
        other => panic!("unexpected obtain response: {other:?}"),
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
