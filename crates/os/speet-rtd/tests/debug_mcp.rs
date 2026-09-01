//! MCP, CLI, LastReport, and hot-reload tests against an in-process daemon.

use speet_rtd::{bind_socket, Daemon};
use speet_runtime::rtd_protocol::{
    decode_response, encode_request, read_frame, write_frame, Request, Response,
};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

fn linked_exit42() -> Option<std::path::PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus");
    if cfg!(target_arch = "aarch64") {
        let p = root.join("aarch64-macos/exit42.linked.macho");
        if p.is_file() {
            return Some(p);
        }
    }
    if cfg!(target_arch = "x86_64") {
        let p = root.join("x86_64-macos/exit42.linked.macho");
        if p.is_file() {
            return Some(p);
        }
    }
    None
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

fn spawn_daemon(sock_path: &Path) -> Arc<Daemon> {
    let listener = bind_socket(sock_path).expect("bind");
    let daemon = Arc::new(Daemon::new());
    let d = daemon.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let _ = handle_one(&d, stream);
        }
    });
    std::thread::sleep(std::time::Duration::from_millis(50));
    daemon
}

#[test]
fn last_report_and_guest_info_roundtrip() {
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("rtd.sock");
    let _daemon = spawn_daemon(&sock_path);
    std::env::set_var("SPEET_RTD_SOCK", &sock_path);

    let mut sock = UnixStream::connect(&sock_path).expect("connect");
    let pong = roundtrip(&mut sock, &encode_request(&Request::Ping));
    match decode_response(&pong).unwrap() {
        Response::Pong => {}
        other => panic!("expected pong, got {other:?}"),
    }

    let Some(guest) = linked_exit42() else {
        eprintln!("SKIP: no same-platform corpus fixture");
        return;
    };

    let mut sock = UnixStream::connect(&sock_path).expect("connect");
    let info = roundtrip(
        &mut sock,
        &encode_request(&Request::GuestInfo {
            path: guest.display().to_string(),
        }),
    );
    match decode_response(&info).unwrap() {
        Response::GuestInfo { json } => {
            assert!(json.contains("\"arch\""), "{json}");
            assert!(json.contains("\"imports\""), "{json}");
        }
        other => panic!("expected guest info, got {other:?}"),
    }

    let mut sock = UnixStream::connect(&sock_path).expect("connect");
    let _ = roundtrip(
        &mut sock,
        &encode_request(&Request::Analyze {
            path: guest.display().to_string(),
            backend: Some("integrated".into()),
        }),
    );

    let mut sock = UnixStream::connect(&sock_path).expect("connect");
    let report = roundtrip(&mut sock, &encode_request(&Request::LastReport));
    match decode_response(&report).unwrap() {
        Response::Report { json } => {
            assert!(json.contains("unresolved_deps"), "{json}");
            assert!(json.contains("fn_ptr_deps"), "{json}");
            assert!(json.contains("unsupported_insns"), "{json}");
            assert!(json.contains("\"plugins\""), "{json}");
        }
        other => panic!("expected report, got {other:?}"),
    }

    let rc = speet_rtd::ctl::run(&["analyze".into(), guest.display().to_string()]);
    assert!(rc == 0 || rc == 1, "ctl analyze exit {rc}");
}

#[cfg(feature = "mcp")]
#[test]
fn mcp_handle_message_roundtrip() {
    let daemon = Arc::new(Daemon::new());
    let server = speet_rtd::mcp::build_mcp_server(daemon.clone());
    let sid = server.new_session();

    let init = speet_rtd::mcp::handle_mcp_message(
        &server,
        &sid,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
    )
    .expect("init response");
    assert!(init.contains("speet-recompiler-debug"), "{init}");

    let listed = speet_rtd::mcp::handle_mcp_message(
        &server,
        &sid,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
    )
    .expect("tools/list");
    assert!(listed.contains("analyze_binary"), "{listed}");
    assert!(listed.contains("recompile"), "{listed}");
    assert!(listed.contains("last_report"), "{listed}");
    assert!(listed.contains("guest_info"), "{listed}");
    assert!(listed.contains("run_guest"), "{listed}");
    assert!(!listed.contains("reload"), "{listed}");

    let Some(guest) = linked_exit42() else {
        eprintln!("SKIP: no same-platform corpus fixture");
        return;
    };
    let path_json = serde_json::to_string(&guest.display().to_string()).unwrap();
    let call = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"analyze_binary","arguments":{{"path":{path_json}}}}}}}"#
    );
    let resp = speet_rtd::mcp::handle_mcp_message(&server, &sid, &call).expect("analyze");
    assert!(
        resp.contains("unresolved_deps") || resp.contains("suitable"),
        "{resp}"
    );
}

#[cfg(feature = "hot-recompiler")]
mod hot {
    use super::*;
    use speet_host_api::integrated_host_api;
    use speet_plugin_api::remote::TargetResponse;
    use speet_plugin_api::target::{FuncImportDecl, ModuleManifest};
    use speet_plugin_api::wire::WireEncode;
    use speet_recompiler_guest::TranslateResponse;
    use speet_runtime::IntegratedNativeRuntime;
    use wasm_encoder::{
        CodeSection, DataSection, ExportKind, ExportSection, Function, FunctionSection,
        GlobalSection, GlobalType, Instruction, MemorySection, MemoryType, Module, TypeSection,
        ValType,
    };

    fn bump_alloc_fn() -> Function {
        let mut f = Function::new([(1, ValType::I32)]);
        let ptr_local = 1u32;
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::LocalSet(ptr_local));
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::GlobalSet(0));
        f.instruction(&Instruction::LocalGet(ptr_local));
        f.instruction(&Instruction::End);
        f
    }

    fn constant_call_module(resp: &[u8]) -> Vec<u8> {
        let resp_addr = 0u32;
        let bump_init = 8192i32;
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I32]);
        types
            .ty()
            .function([ValType::I32, ValType::I32], [ValType::I64]);
        let mut funcs = FunctionSection::new();
        funcs.function(0);
        funcs.function(1);
        let mut mem = MemorySection::new();
        mem.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        let mut globals = GlobalSection::new();
        globals.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: true,
                shared: false,
            },
            &wasm_encoder::ConstExpr::i32_const(bump_init),
        );
        let mut exports = ExportSection::new();
        exports.export("memory", ExportKind::Memory, 0);
        exports.export("speet_plugin_alloc", ExportKind::Func, 0);
        exports.export("speet_plugin_call", ExportKind::Func, 1);
        let mut code = CodeSection::new();
        code.function(&bump_alloc_fn());
        let mut call_fn = Function::new([]);
        call_fn.instruction(&Instruction::I64Const(speet_plugin_host_wasm::abi::pack(
            resp_addr,
            resp.len() as u32,
        )));
        call_fn.instruction(&Instruction::End);
        code.function(&call_fn);
        let mut data = DataSection::new();
        data.active(
            0,
            &wasm_encoder::ConstExpr::i32_const(resp_addr as i32),
            resp.iter().copied(),
        );
        let mut module = Module::new();
        module.section(&types);
        module.section(&funcs);
        module.section(&mem);
        module.section(&globals);
        module.section(&exports);
        module.section(&code);
        module.section(&data);
        let bytes = module.finish();
        wasmparser::validate(&bytes).expect("fixture wasm");
        bytes
    }

    fn recompiler_wasm(unsupported: &[&str], error: &str) -> Vec<u8> {
        let resp = TranslateResponse {
            wasm: b"\0asm\x01\0\0\0".to_vec(),
            unsupported: unsupported.iter().map(|s| (*s).to_string()).collect(),
            error: error.into(),
        };
        let mut buf = Vec::new();
        resp.encode(&mut buf);
        constant_call_module(&buf)
    }

    fn stubs_wasm(fields: &[&str]) -> Vec<u8> {
        let manifest = ModuleManifest {
            func_imports: fields
                .iter()
                .map(|f| FuncImportDecl {
                    module: "stub".into(),
                    field: (*f).into(),
                    params: vec![],
                    results: vec![],
                })
                .collect(),
            ..Default::default()
        };
        let mut manifest_bytes = Vec::new();
        TargetResponse::ModuleManifest { manifest }.encode(&mut manifest_bytes);
        // Tag 0 = ModuleManifest; the fixture ignores the request and always
        // returns the manifest blob (enough for extra_wired extraction).
        constant_call_module(&manifest_bytes)
    }

    #[test]
    fn auto_reload_on_file_rewrite_no_reload_rpc() {
        let Some(guest) = linked_exit42() else {
            eprintln!("SKIP: no same-platform corpus fixture");
            return;
        };
        let dir = tempdir().unwrap();
        let rec = dir.path().join("recompiler.wasm");
        let stubs = dir.path().join("stubs.wasm");
        std::fs::write(&rec, recompiler_wasm(&["undef:deadbeef"], "")).unwrap();
        std::fs::write(&stubs, stubs_wasm(&[])).unwrap();

        let daemon = Daemon::new();
        daemon.set_hot_watch_paths(rec.clone(), stubs.clone());
        let r1 = daemon.recompile_report(&guest, false);
        assert!(
            r1.unsupported_insns.iter().any(|s| s.contains("deadbeef"))
                || r1.translate_error.is_some(),
            "guest A should surface missing insn: {r1:?}"
        );
        let hash_a = r1.plugins.recompiler.clone();
        assert_ne!(hash_a, speet_runtime::PluginHashes::NATIVE);

        std::fs::write(&rec, recompiler_wasm(&[], "")).unwrap();
        let r2 = daemon.recompile_report(&guest, false);
        assert_ne!(
            r2.plugins.recompiler, hash_a,
            "next request must hash the rewritten artifact"
        );
        assert!(
            !r2.unsupported_insns.iter().any(|s| s.contains("deadbeef")),
            "guest B must not keep A's unsupported list: {:?}",
            r2.unsupported_insns
        );
    }

    #[test]
    fn stubs_hash_changes_cache_key() {
        let Some(guest) = linked_exit42() else {
            eprintln!("SKIP: no same-platform corpus fixture");
            return;
        };
        let dir = tempdir().unwrap();
        let rec = dir.path().join("recompiler.wasm");
        let stubs = dir.path().join("stubs.wasm");
        std::fs::write(&rec, recompiler_wasm(&[], "")).unwrap();
        std::fs::write(&stubs, stubs_wasm(&[])).unwrap();

        let daemon = Daemon::new();
        daemon.set_hot_watch_paths(rec.clone(), stubs.clone());
        let _ = daemon.analyze_report(&guest);
        let h1 = daemon.last_report().unwrap().plugins.stubs;
        let rt = IntegratedNativeRuntime::new(Arc::new(integrated_host_api()));
        // Drive cache-key helper with the hashes the daemon just loaded.
        let report = daemon.last_report().unwrap();
        rt.set_hot_plugins(report.plugins.clone(), vec![], None);
        let k1 = rt.artifact_cache_key(&guest).unwrap();

        std::fs::write(&stubs, stubs_wasm(&["not_a_real_symbol"])).unwrap();
        let _ = daemon.analyze_report(&guest);
        let h2 = daemon.last_report().unwrap().plugins.stubs;
        assert_ne!(h1, h2, "stubs content-hash must change after overwrite");
        rt.set_hot_plugins(
            daemon.last_report().unwrap().plugins,
            vec!["not_a_real_symbol".into()],
            None,
        );
        let k2 = rt.artifact_cache_key(&guest).unwrap();
        assert_ne!(k1, k2, "cache key must include stubs hash");
        assert!(k2.contains("stubs="), "{k2}");
    }

    fn wasm32_target_available() -> bool {
        std::process::Command::new("rustc")
            .args(["--print", "cfg", "--target", "wasm32-unknown-unknown"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn skip_closed_if_wasm32_target_missing() {
        if !wasm32_target_available() {
            eprintln!("SKIP: wasm32-unknown-unknown rustc target missing");
            return;
        }
        // Target present: hand-encoded fixtures already cover wasmi load.
        // A full `cargo build -p speet-recompiler-guest --target wasm32-unknown-unknown`
        // is the agent rebuild step, not this unit test.
    }
}
