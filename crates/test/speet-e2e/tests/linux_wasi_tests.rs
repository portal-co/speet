//! Integration tests for the Linux-to-WASI syscall translation bridge.

use wasmi::AsContext;

#[test]
fn test_linux_to_wasi_write_and_exit() {
    let text = [
        0x13, 0x05, 0x10, 0x00, // addi a0, x0, 1
        0x93, 0x05, 0x80, 0x20, // addi a1, x0, 520
        0x13, 0x06, 0x60, 0x00, // addi a2, x0, 6
        0x93, 0x08, 0x00, 0x04, // addi a7, x0, 64
        0x73, 0x00, 0x00, 0x00, // ecall (write)
        0x13, 0x05, 0x00, 0x00, // addi a0, x0, 0
        0x93, 0x08, 0xd0, 0x05, // addi a7, x0, 93
        0x73, 0x00, 0x00, 0x00, // ecall (exit)
    ];

    let start_addr = 0x1000u64;
    let wasm = speet_linux_wasi::recompile_rv64_wasi_to_wasm(&text, start_addr);
    wasmparser::validate(&wasm).expect("WASM module is invalid");

    use wasmi::{Engine, Linker, Module as WasmiModule, Store};

    #[derive(Default)]
    struct HostState {
        stdout: Vec<u8>,
        exit_code: Option<i32>,
    }

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
            |_caller: wasmi::Caller<'_, HostState>, _fd: i32, _iovs: i32, _iovs_len: i32, _nread_ptr: i32| -> i32 {
                0
            },
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

    let module = WasmiModule::new(&engine, &wasm).expect("valid wasm module");
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiation failed");

    let mem = instance
        .get_export(&store, "memory")
        .unwrap()
        .into_memory()
        .unwrap();
    mem.write(&mut store, 520, b"hello\n").unwrap();

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
    assert_eq!(std::str::from_utf8(&state.stdout).unwrap(), "hello\n");
    assert_eq!(state.exit_code, Some(0));
}
