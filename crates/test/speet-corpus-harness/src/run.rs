//! Execute instrumented corpus modules under wasmi.

use wasmi::AsContext;

const RUN_FUEL: u64 = 50_000_000;

#[derive(Debug, Default)]
pub struct RunState {
    pub unreachable_trap_hits: Vec<i32>,
    pub exit_code: Option<i32>,
    pub main_return: Option<i32>,
    pub stdout: Vec<u8>,
}

#[derive(Default)]
struct HostState {
    unreachable_trap_hits: Vec<i32>,
    exit_code: Option<i32>,
    stdout: Vec<u8>,
}

pub fn run_corpus_module(wasm: &[u8], entry: &str) -> Result<RunState, String> {
    let mut config = wasmi::Config::default();
    config.consume_fuel(true);
    let engine = wasmi::Engine::new(&config);
    let mut store = wasmi::Store::new(&engine, HostState::default());
    store.set_fuel(RUN_FUEL).map_err(|e| e.to_string())?;
    let mut linker: wasmi::Linker<HostState> = wasmi::Linker::new(&engine);

    linker
        .func_wrap("env", "__speet_hint", |_caller: wasmi::Caller<'_, HostState>, _id: i32| Ok(()))
        .map_err(|e| e.to_string())?;
    linker
        .func_wrap(
            "env",
            "write",
            |mut caller: wasmi::Caller<'_, HostState>, fd: i32, ptr: i32, len: i32| -> i32 {
                if (fd == 1 || fd == 2) && len > 0 {
                    if let Some(mem) = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                    {
                        let mut buf = vec![0u8; len as usize];
                        let _ = mem.read(caller.as_context(), ptr as usize, &mut buf);
                        caller.data_mut().stdout.extend_from_slice(&buf);
                    }
                }
                len
            },
        )
        .map_err(|e| e.to_string())?;
    linker
        .func_wrap(
            "env",
            "exit",
            |mut caller: wasmi::Caller<'_, HostState>, code: i32| {
                caller.data_mut().exit_code = Some(code);
            },
        )
        .map_err(|e| e.to_string())?;
    linker
        .func_wrap(
            "env",
            "__speet_unreachable_trap",
            |mut caller: wasmi::Caller<'_, HostState>, func_idx: i32| {
                eprintln!("speet: unreachable trap at func {func_idx}");
                caller.data_mut().unreachable_trap_hits.push(func_idx);
            },
        )
        .map_err(|e| e.to_string())?;

    let module = wasmi::Module::new(&engine, wasm).map_err(|e| e.to_string())?;
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(|e| e.to_string())?;

    let func = instance
        .get_func(&mut store, entry)
        .ok_or_else(|| format!("no export {entry}"))?;
    let ty = func.ty(&store);
    let params: Vec<wasmi::Val> = ty
        .params()
        .iter()
        .map(|vt| match vt {
            wasmi::ValType::I32 => wasmi::Val::I32(0),
            wasmi::ValType::I64 => wasmi::Val::I64(0),
            wasmi::ValType::F32 => wasmi::Val::F32(wasmi::F32::from_bits(0)),
            wasmi::ValType::F64 => wasmi::Val::F64(wasmi::F64::from_bits(0)),
            _ => wasmi::Val::I32(0),
        })
        .collect();
    let mut results = vec![wasmi::Val::I32(0); ty.results().len()];
    let _ = func.call(&mut store, &params, &mut results);

    let main_return = if results.len() == 1 {
        match results[0] {
            wasmi::Val::I32(v) => Some(v),
            _ => None,
        }
    } else {
        None
    };

    let host = store.into_data();
    Ok(RunState {
        unreachable_trap_hits: host.unreachable_trap_hits,
        exit_code: host.exit_code,
        main_return,
        stdout: host.stdout,
    })
}
