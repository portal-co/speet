//! Dual-lane harness: Lane A (WASI / wasmi) + optional Lane B-native (thin runtime).
//!
//! Shared scenarios run under WASI on every host. When `target_os = "macos"` and
//! LLVM is available, the same guest also runs through the thin-runtime native
//! path for exit-code parity.

use wasmi::AsContext;

/// Outcome of a Lane A (WASI) run.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WasiLaneResult {
    stdout: Vec<u8>,
    exit_code: Option<i32>,
}

#[derive(Default)]
struct HostState {
    stdout: Vec<u8>,
    exit_code: Option<i32>,
}

/// Run a megabinary under wasmi with preview1 stubs.
///
/// Seeds `memory` at `seed_addr` with `seed_bytes` before calling `_start`.
fn run_wasi_lane(wasm: &[u8], seed_addr: usize, seed_bytes: &[u8]) -> WasiLaneResult {
    wasmparser::validate(wasm).expect("megabinary must validate");

    use wasmi::{Engine, Linker, Module as WasmiModule, Store};

    let engine = Engine::default();
    let mut store = Store::new(&engine, HostState::default());
    let mut linker = Linker::<HostState>::new(&engine);

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_write",
            |mut caller: wasmi::Caller<'_, HostState>,
             fd: i32,
             iovs: i32,
             iovs_len: i32,
             nwritten_ptr: i32|
             -> i32 {
                if fd == 1 || fd == 2 {
                    let mem = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .unwrap();
                    let mut total_written = 0;
                    for i in 0..iovs_len {
                        let base = (iovs + i * 8) as usize;
                        let mut buf_ptr_bytes = [0u8; 4];
                        let mut buf_len_bytes = [0u8; 4];
                        mem.read(caller.as_context(), base, &mut buf_ptr_bytes)
                            .unwrap();
                        mem.read(caller.as_context(), base + 4, &mut buf_len_bytes)
                            .unwrap();
                        let buf_ptr = u32::from_le_bytes(buf_ptr_bytes) as usize;
                        let buf_len = u32::from_le_bytes(buf_len_bytes) as usize;
                        let mut buf = vec![0u8; buf_len];
                        mem.read(caller.as_context(), buf_ptr, &mut buf).unwrap();
                        caller.data_mut().stdout.extend_from_slice(&buf);
                        total_written += buf_len as i32;
                    }
                    let nwritten_bytes = total_written.to_le_bytes();
                    mem.write(&mut caller, nwritten_ptr as usize, &nwritten_bytes)
                        .unwrap();
                }
                0
            },
        )
        .unwrap();

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_read",
            |_caller: wasmi::Caller<'_, HostState>,
             _fd: i32,
             _iovs: i32,
             _iovs_len: i32,
             _nread_ptr: i32|
             -> i32 { 0 },
        )
        .unwrap();

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_close",
            |_caller: wasmi::Caller<'_, HostState>, _fd: i32| -> i32 { 0 },
        )
        .unwrap();

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "proc_exit",
            |mut caller: wasmi::Caller<'_, HostState>, code: i32| {
                caller.data_mut().exit_code = Some(code);
            },
        )
        .unwrap();

    let module = WasmiModule::new(&engine, wasm).expect("valid wasm module");
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiation failed");

    if !seed_bytes.is_empty() {
        let mem = instance
            .get_export(&store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();
        mem.write(&mut store, seed_addr, seed_bytes).unwrap();
    }

    let entry_func = instance.get_func(&store, "_start").expect("_start export");
    let mut results = vec![wasmi::Val::I32(0); entry_func.ty(&store).results().len()];
    let call_params: Vec<wasmi::Val> = entry_func
        .ty(&store)
        .params()
        .iter()
        .map(|ty| match ty {
            wasmi::ValType::I32 => wasmi::Val::I32(0),
            wasmi::ValType::I64 => wasmi::Val::I64(0),
            wasmi::ValType::F32 => wasmi::Val::F32(wasmi::F32::from_bits(0)),
            wasmi::ValType::F64 => wasmi::Val::F64(wasmi::F64::from_bits(0)),
            _ => wasmi::Val::I64(0),
        })
        .collect();
    let _ = entry_func.call(&mut store, &call_params, &mut results);

    let state = store.into_data();
    WasiLaneResult {
        stdout: state.stdout,
        exit_code: state.exit_code,
    }
}

/// RV64 Linux: `exit(42)`.
const EXIT_42: &[u8] = &[
    0x13, 0x05, 0xA0, 0x02, // addi a0, x0, 42
    0x93, 0x08, 0xD0, 0x05, // addi a7, x0, 93
    0x73, 0x00, 0x00, 0x00, // ecall
];

#[test]
fn dual_lane_linux_exit_42_wasi() {
    let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm(EXIT_42, 0x1000);
    let result = run_wasi_lane(&wasm, 0, &[]);
    assert_eq!(
        result,
        WasiLaneResult {
            stdout: vec![],
            exit_code: Some(42),
        }
    );
}

/// Same scenario from thin-runtime corpus ELF `.text` (Lane A on every host).
#[test]
fn dual_lane_linux_corpus_exit_42_wasi() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/thin-runtime-corpus/rv64-linux/exit_42.elf");
    let data = std::fs::read(&path).expect("corpus exit_42.elf");
    let obj = object::File::parse(&*data).expect("parse ELF");
    use object::{Object, ObjectSection};
    let text = obj
        .section_by_name(".text")
        .expect(".text")
        .data()
        .expect(".text data");
    let addr = obj.section_by_name(".text").unwrap().address();
    assert_eq!(text, EXIT_42);
    let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm(text, addr);
    let result = run_wasi_lane(&wasm, 0, &[]);
    assert_eq!(result.exit_code, Some(42));
}

#[cfg(target_os = "macos")]
#[test]
fn dual_lane_linux_exit_42_native() {
    use binary_io::{BinArch, BinOs};
    use speet_runtime::{default_host_api, Runtime};
    use std::sync::Arc;

    let mut rt = Runtime::new(Arc::new(default_host_api()));
    if !rt.llvm_available() {
        eprintln!("SKIP: LLVM not available");
        return;
    }

    #[cfg(target_arch = "aarch64")]
    let arch = BinArch::AArch64;
    #[cfg(not(target_arch = "aarch64"))]
    let arch = BinArch::X86_64;

    let status = rt
        .recompile_rv64_and_run(EXIT_42, 0x1000, arch, BinOs::MacOs)
        .expect("native pipeline");
    assert_eq!(status.code(), Some(42));

    // Lane A parity on the same host.
    let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm(EXIT_42, 0x1000);
    let wasi = run_wasi_lane(&wasm, 0, &[]);
    assert_eq!(wasi.exit_code, Some(42));
}
