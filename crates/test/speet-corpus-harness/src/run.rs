//! Execute instrumented corpus modules under wasmi.

use std::sync::{Arc, Mutex};

const RUN_FUEL: u64 = 50_000_000;

#[derive(Debug, Default)]
pub struct RunState {
    pub unreachable_trap_hits: Vec<i32>,
    pub exit_code: Option<i32>,
}

#[derive(Default)]
struct HostInner {
    unreachable_trap_hits: Vec<i32>,
    exit_code: Option<i32>,
}

pub fn run_corpus_module(wasm: &[u8], entry: &str) -> Result<RunState, String> {
    let inner = Arc::new(Mutex::new(HostInner::default()));
    let inner_trap = inner.clone();
    let inner_exit = inner.clone();

    let mut config = wasmi::Config::default();
    config.consume_fuel(true);
    let engine = wasmi::Engine::new(&config);
    let mut store = wasmi::Store::new(&engine, ());
    store.set_fuel(RUN_FUEL).map_err(|e| e.to_string())?;
    let mut linker: wasmi::Linker<()> = wasmi::Linker::new(&engine);

    linker
        .func_wrap("env", "__speet_hint", |_caller: wasmi::Caller<'_, ()>, _id: i32| Ok(()))
        .map_err(|e| e.to_string())?;
    linker
        .func_wrap(
            "env",
            "write",
            |_caller: wasmi::Caller<'_, ()>, _fd: i32, _ptr: i32, len: i32| Ok(len),
        )
        .map_err(|e| e.to_string())?;
    linker
        .func_wrap("env", "exit", move |caller: wasmi::Caller<'_, ()>, code: i32| {
            if let Ok(mut g) = inner_exit.lock() {
                g.exit_code = Some(code);
            }
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    linker
        .func_wrap(
            "env",
            "__speet_unreachable_trap",
            move |_caller: wasmi::Caller<'_, ()>, func_idx: i32| {
                eprintln!("speet: unreachable trap at func {func_idx}");
                if let Ok(mut g) = inner_trap.lock() {
                    g.unreachable_trap_hits.push(func_idx);
                }
                Ok(())
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

    let g = inner.lock().map_err(|e| e.to_string())?;
    Ok(RunState {
        unreachable_trap_hits: g.unreachable_trap_hits.clone(),
        exit_code: g.exit_code,
    })
}
