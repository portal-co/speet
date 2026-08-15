//! Daemon-side JIT backend registration (Phase 4). Only compiled with
//! `--features jit` (and requires nightly, transitively from vane-arch).

#![cfg(feature = "jit")]

use speet_rtd::Daemon;
use speet_runtime::rtd_protocol::{decode_response, encode_request, JitArtifactWire, Request, Response};

#[test]
fn vane_wasm_backend_compiles_a_real_instruction() {
    let daemon = Daemon::new();
    let frame = encode_request(&Request::Compile {
        guest_pc: 0x1000,
        guest_bytes: 0x0000_8067u32.to_le_bytes().to_vec(), // `ret` / jalr x0, 0(x1)
        num_regs: 32,
        backend: "vane-wasm".into(),
    });
    let resp = decode_response(&daemon.handle_frame(&frame)).unwrap();
    match resp {
        Response::Compiled { artifact: JitArtifactWire::Wasm(bytes), backend } => {
            assert_eq!(backend, "vane-wasm");
            assert_eq!(&bytes[0..4], b"\0asm", "should be a well-formed WASM module");
        }
        other => panic!("unexpected response: {other:?}"),
    }
}

#[test]
fn vane_wasm_backend_rejects_empty_guest_bytes() {
    let daemon = Daemon::new();
    let frame = encode_request(&Request::Compile {
        guest_pc: 0x1000,
        guest_bytes: vec![],
        num_regs: 32,
        backend: "vane-wasm".into(),
    });
    match decode_response(&daemon.handle_frame(&frame)).unwrap() {
        Response::Error { message } => assert!(message.contains("decode")),
        other => panic!("unexpected response: {other:?}"),
    }
}

#[test]
fn vane_wasm_aarch64_backend_compiles_a_real_instruction() {
    let daemon = Daemon::new();
    let frame = encode_request(&Request::Compile {
        guest_pc: 0x1000,
        guest_bytes: 0xD65F_03C0u32.to_le_bytes().to_vec(), // `ret` (x30)
        num_regs: 73,
        backend: "vane-wasm-aarch64".into(),
    });
    let resp = decode_response(&daemon.handle_frame(&frame)).unwrap();
    match resp {
        Response::Compiled { artifact: JitArtifactWire::Wasm(bytes), backend } => {
            assert_eq!(backend, "vane-wasm-aarch64");
            assert_eq!(&bytes[0..4], b"\0asm", "should be a well-formed WASM module");
        }
        other => panic!("unexpected response: {other:?}"),
    }
}

#[test]
fn vane_wasm_aarch64_backend_rejects_wrong_num_regs() {
    let daemon = Daemon::new();
    let frame = encode_request(&Request::Compile {
        guest_pc: 0x1000,
        guest_bytes: 0xD65F_03C0u32.to_le_bytes().to_vec(),
        num_regs: 32, // AArch64's full architectural state is 73, not caller-adjustable
        backend: "vane-wasm-aarch64".into(),
    });
    match decode_response(&daemon.handle_frame(&frame)).unwrap() {
        Response::Error { message } => assert!(message.contains("73")),
        other => panic!("unexpected response: {other:?}"),
    }
}

#[test]
fn vane_blitz_backend_is_registered_but_reports_unsupported() {
    let daemon = Daemon::new();
    let frame = encode_request(&Request::Compile {
        guest_pc: 0x1000,
        guest_bytes: 0x0000_8067u32.to_le_bytes().to_vec(),
        num_regs: 32,
        backend: "vane-blitz".into(),
    });
    match decode_response(&daemon.handle_frame(&frame)).unwrap() {
        // Registered (not "unknown backend") but not yet functional -- see
        // vane_jit::VaneBlitzJitBackend's doc comment for why.
        Response::Error { message } => {
            assert!(!message.contains("unknown backend"));
            assert!(message.contains("unsupported"));
        }
        other => panic!("unexpected response: {other:?}"),
    }
}

#[test]
fn unregistered_backend_is_a_clear_unknown_backend_error() {
    let daemon = Daemon::new();
    let frame = encode_request(&Request::Compile {
        guest_pc: 0,
        guest_bytes: vec![1],
        num_regs: 1,
        backend: "does-not-exist".into(),
    });
    match decode_response(&daemon.handle_frame(&frame)).unwrap() {
        Response::Error { message } => assert!(message.contains("unknown backend")),
        other => panic!("unexpected response: {other:?}"),
    }
}
