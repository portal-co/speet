//! Integration tests for the Linux-to-WASI syscall translation bridge.

#[path = "harness/mod.rs"]
mod harness;

use std::convert::Infallible;
use rv_asm::Xlen;
use speet_riscv::RiscVRecompiler;
use yecta::{LocalPool, Reactor, TypeIdx};
use wasm_encoder::Function;
use speet_link_core::ReactorContext;
use wasmi::AsContext;
use wasm_encoder::Encode;

#[test]
fn test_linux_to_wasi_write_and_exit() {
    // 1. Machine code for minimal RISC-V binary (RV64):
    // addi a0, x0, 1      -> write to fd 1 (stdout)
    // addi a1, x0, 520    -> buffer ptr = 520 (0x208)
    // addi a2, x0, 6      -> length = 6
    // addi a7, x0, 64     -> syscall 64 (write)
    // ecall
    // addi a0, x0, 0      -> exit code 0
    // addi a7, x0, 93     -> syscall 93 (exit)
    // ecall
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

    // 2. Set up the recompiler, reactor, and contexts.
    let start_addr = 0x1000u64;
    let mut recompiler = RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
        start_addr, false, true, false,
    );
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let base_func_offset = 4u32; // Leave indices 0..3 for WASI imports
    let mut rctx = harness::make_rctx(&mut reactor, base_func_offset, TypeIdx(0), harness::Eh::None);
    let mut ctx = ();
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let params = harness::collect_rv_params(&rctx);

    // 3. Initialize the WASI imports and the LinuxToWasi table.
    let wasi_imports = speet_linux_wasi::WasiImports {
        fd_write: 0,
        fd_read: 1,
        fd_close: 2,
        proc_exit: 3,
    };
    let linux_to_wasi = speet_linux_wasi::LinuxToWasi::new(wasi_imports);
    let syscall_table = linux_to_wasi.build_table(|n| n as u32);

    // 4. Define the dynamic TestEcallCallback.
    struct TestEcallCallback<'t> {
        dispatcher: speet_syscall::WasmSyscallDispatcher<'t>,
        base_func_offset: u32,
        start_pc: u32,
    }

    impl<'t> speet_riscv::EcallCallback<(), Infallible> for TestEcallCallback<'t> {
        fn call(
            &mut self,
            ecall: &speet_riscv::EcallInfo,
            ctx: &mut (),
            cb: &mut speet_riscv::CallbackContext<'_, (), Infallible>,
        ) {
            let next_pc_offset = (ecall.pc - self.start_pc) / 4 + 1;
            let target = self.base_func_offset + 2 * next_pc_offset;
            self.dispatcher.next_pc_func = if target < self.base_func_offset + 16 {
                target
            } else {
                self.base_func_offset
            };
            speet_riscv::EcallCallback::call(&mut self.dispatcher, ecall, ctx, cb);
        }
    }

    let mut callback = TestEcallCallback {
        dispatcher: speet_syscall::WasmSyscallDispatcher {
            table: &syscall_table,
            syscall_num_local: 17, // a7 = x17
            syscall_num_is_i64: true,
            num_params: params.len() as u32,
            next_pc_func: 0, // Set dynamically in call()
        },
        base_func_offset,
        start_pc: start_addr as u32,
    };
    recompiler.set_ecall_callback(&mut callback);

    // 5. Translate instructions into WASM functions.
    recompiler.translate_bytes(
        &mut ctx,
        &mut rctx,
        &text,
        start_addr as u32,
        Xlen::Rv64,
        &mut |a| Function::new(a.collect::<Vec<_>>()),
    ).expect("translate_bytes failed");

    let fns = rctx.drain_fns();

    // 6. Assemble the module.
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, FunctionSection, ImportSection,
        MemorySection, MemoryType, Module, TableSection, TypeSection, ValType,
        TableType, RefType, ElementSection, Elements, ConstExpr,
    };

    let mut types = TypeSection::new();
    // Type 0: translated function signature: [I64; 32] -> []
    types.ty().function(params.clone(), vec![]);
    // Type 1: fd_write/fd_read signature: [I32, I32, I32, I32] -> [I32]
    types.ty().function(vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32], vec![ValType::I32]);
    // Type 2: fd_close signature: [I32] -> [I32]
    types.ty().function(vec![ValType::I32], vec![ValType::I32]);
    // Type 3: proc_exit signature: [I32] -> []
    types.ty().function(vec![ValType::I32], vec![]);

    let mut imports = ImportSection::new();
    imports.import("wasi_snapshot_preview1", "fd_write", wasm_encoder::EntityType::Function(1));
    imports.import("wasi_snapshot_preview1", "fd_read", wasm_encoder::EntityType::Function(1));
    imports.import("wasi_snapshot_preview1", "fd_close", wasm_encoder::EntityType::Function(2));
    imports.import("wasi_snapshot_preview1", "proc_exit", wasm_encoder::EntityType::Function(3));

    let mut funcs = FunctionSection::new();
    for _ in &fns {
        funcs.function(0);
    }

    let total_fns = fns.len() as u32;
    let table_size = 4 + total_fns;
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: table_size as u64,
        maximum: Some(table_size as u64),
        table64: true,
        shared: false,
    });

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("_start", ExportKind::Func, 4);

    let all_indices: Vec<u32> = (4..4 + total_fns).collect();
    let mut elems = ElementSection::new();
    elems.active(Some(0), &ConstExpr::i64_const(4), Elements::Functions(std::borrow::Cow::Borrowed(&all_indices)));

    let mut code = CodeSection::new();
    for f in &fns {
        code.function(f);
    }

    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    module.section(&tables);
    module.section(&mems);
    module.section(&exports);
    module.section(&elems);
    module.section(&code);
    let wasm = module.finish();

    println!("Total fns: {}", fns.len());
    for (i, f) in fns.iter().enumerate() {
        let mut bytes = Vec::new();
        f.encode(&mut bytes);
        println!("Function {}: length = {} bytes", i, bytes.len());
    }

    // 7. Validate the WASM using wasmparser.
    let parser = wasmparser::Parser::new(0);
    let mut func_idx = 0;
    for payload in parser.parse_all(&wasm) {
        match payload {
            Ok(wasmparser::Payload::CodeSectionEntry(body)) => {
                if func_idx == 0 {
                    println!("--- FIRST FUNCTION BODY ---");
                    let mut reader = body.get_operators_reader().unwrap();
                    loop {
                        let pos = reader.original_position();
                        match reader.read() {
                            Ok(op) => println!("  [{pos}] {:?}", op),
                            Err(_) => break,
                        }
                    }
                }
                func_idx += 1;
            }
            _ => {}
        }
    }
    wasmparser::validate(&wasm).expect("WASM module is invalid");

    // 8. Execute in wasmi with mock WASI preview1.
    use wasmi::{Engine, Linker, Module as WasmiModule, Store};

    #[derive(Default)]
    struct HostState {
        stdout: Vec<u8>,
        exit_code: Option<i32>,
    }

    let engine = Engine::default();
    let mut store = Store::new(&engine, HostState::default());
    let mut linker = Linker::<HostState>::new(&engine);

    linker.func_wrap("wasi_snapshot_preview1", "fd_write",
        |mut caller: wasmi::Caller<'_, HostState>, fd: i32, iovs: i32, iovs_len: i32, nwritten_ptr: i32| -> i32 {
            println!("fd_write called: fd={fd} iovs={iovs:#x} iovs_len={iovs_len} nwritten_ptr={nwritten_ptr:#x}");
            if fd == 1 || fd == 2 {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let mut total_written = 0;
                for i in 0..iovs_len {
                    let base = (iovs + i * 8) as usize;
                    let mut buf_ptr_bytes = [0u8; 4];
                    let mut buf_len_bytes = [0u8; 4];
                    mem.read(caller.as_context(), base, &mut buf_ptr_bytes).unwrap();
                    mem.read(caller.as_context(), base + 4, &mut buf_len_bytes).unwrap();
                    let buf_ptr = u32::from_le_bytes(buf_ptr_bytes) as usize;
                    let buf_len = u32::from_le_bytes(buf_len_bytes) as usize;

                    let mut buf = vec![0u8; buf_len];
                    mem.read(caller.as_context(), buf_ptr, &mut buf).unwrap();
                    caller.data_mut().stdout.extend_from_slice(&buf);
                    total_written += buf_len as i32;
                }
                let nwritten_bytes = total_written.to_le_bytes();
                mem.write(&mut caller, nwritten_ptr as usize, &nwritten_bytes).unwrap();
            }
            0
        }).unwrap();

    linker.func_wrap("wasi_snapshot_preview1", "fd_read",
        |_caller: wasmi::Caller<'_, HostState>, _fd: i32, _iovs: i32, _iovs_len: i32, _nread_ptr: i32| -> i32 {
            0
        }).unwrap();

    linker.func_wrap("wasi_snapshot_preview1", "fd_close",
        |_caller: wasmi::Caller<'_, HostState>, _fd: i32| -> i32 {
            0
        }).unwrap();

    linker.func_wrap("wasi_snapshot_preview1", "proc_exit",
        |mut caller: wasmi::Caller<'_, HostState>, code: i32| {
            caller.data_mut().exit_code = Some(code);
        }).unwrap();

    let module = WasmiModule::new(&engine, &wasm).expect("valid wasm module");
    let instance = linker.instantiate_and_start(&mut store, &module).expect("instantiation failed");

    // Write "hello\n" into guest memory at address 520 (0x208) as expected by the guest write syscall.
    let mem = instance.get_export(&store, "memory").unwrap().into_memory().unwrap();
    mem.write(&mut store, 520, b"hello\n").unwrap();

    let entry_func = instance.get_func(&store, "_start").ok_or("no _start export").unwrap();
    let mut results = vec![wasmi::Val::I32(0); entry_func.ty(&store).results().len()];

    // Build params that match the actual exported function signature.
    // The translated function has 32 int regs + 32 fp regs + PC + trap params;
    // hardcoding 32 zeros causes a silent type-mismatch. Derive from the type instead.
    let call_params: Vec<wasmi::Val> = entry_func.ty(&store).params().iter().map(|ty| {
        match ty {
            wasmi::ValType::I32 => wasmi::Val::I32(0),
            wasmi::ValType::I64 => wasmi::Val::I64(0),
            wasmi::ValType::F32 => wasmi::Val::F32(wasmi::F32::from_bits(0)),
            wasmi::ValType::F64 => wasmi::Val::F64(wasmi::F64::from_bits(0)),
            _ => wasmi::Val::I64(0),
        }
    }).collect();
    let _ = entry_func.call(&mut store, &call_params, &mut results);

    let state = store.into_data();
    assert_eq!(std::str::from_utf8(&state.stdout).unwrap(), "hello\n");
    assert_eq!(state.exit_code, Some(0));
}
