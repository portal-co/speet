//! Shared wasmi + WASI preview1 runner (Lane A for linux-wasi / darwin-wasi).

use wasmi::{AsContext, Engine, Linker, Module as WasmiModule, Store};

/// Outcome of a preview1 Lane A run.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RunOutcome {
    pub stdout: Vec<u8>,
    pub exit_code: Option<i32>,
}

#[derive(Default)]
struct HostState {
    stdout: Vec<u8>,
    exit_code: Option<i32>,
}

/// Run a megabinary under wasmi with preview1 stubs.
///
/// Optionally seeds `memory` at `seed_addr` with `seed_bytes` before `_start`.
pub fn run_preview1(
    wasm: &[u8],
    entry: &str,
    seed_addr: usize,
    seed_bytes: &[u8],
) -> Result<RunOutcome, String> {
    wasmparser::validate(wasm).map_err(|e| e.to_string())?;

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
        .map_err(|e| e.to_string())?;

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
        .map_err(|e| e.to_string())?;

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_close",
            |_caller: wasmi::Caller<'_, HostState>, _fd: i32| -> i32 { 0 },
        )
        .map_err(|e| e.to_string())?;

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "proc_exit",
            |mut caller: wasmi::Caller<'_, HostState>, code: i32| {
                caller.data_mut().exit_code = Some(code);
            },
        )
        .map_err(|e| e.to_string())?;

    let module = WasmiModule::new(&engine, wasm).map_err(|e| e.to_string())?;
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(|e| e.to_string())?;

    if !seed_bytes.is_empty() {
        let mem = instance
            .get_export(&store, "memory")
            .and_then(|e| e.into_memory())
            .ok_or("no memory export")?;
        mem.write(&mut store, seed_addr, seed_bytes)
            .map_err(|e| e.to_string())?;
    }

    let func = instance
        .get_func(&mut store, entry)
        .ok_or_else(|| format!("no {entry} export"))?;
    let ty = func.ty(&store);
    let params: Vec<wasmi::Val> = ty
        .params()
        .iter()
        .map(|vt| match vt {
            wasmi::ValType::I32 => wasmi::Val::I32(0),
            wasmi::ValType::I64 => wasmi::Val::I64(0),
            wasmi::ValType::F32 => wasmi::Val::F32(wasmi::F32::from_bits(0)),
            wasmi::ValType::F64 => wasmi::Val::F64(wasmi::F64::from_bits(0)),
            _ => panic!("unexpected param type"),
        })
        .collect();
    let mut results = vec![wasmi::Val::I32(0); ty.results().len()];
    let _ = func.call(&mut store, &params, &mut results);

    let data = store.into_data();
    Ok(RunOutcome {
        stdout: data.stdout,
        exit_code: data.exit_code,
    })
}
