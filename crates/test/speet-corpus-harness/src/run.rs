//! Execute instrumented corpus modules under wasmi.
//!
//! **Formerly a known gap, now fixed:** the `arith`/`pairs`/`frame` corpus
//! programs — plain `noinline`-separated-function C, with no crt0/libc call
//! chain, no `exit`/syscall, terminating via a bare `ret` from `main` — used
//! to fail with `main_return: None`, for two reasons:
//!
//! 1. [`assemble_corpus_module`](crate::assemble::assemble_corpus_module)
//!    gave every translated function (including the exported entry) a
//!    shared `params -> []` type, so even when the call graph completed, the
//!    final return value (x0/rax/a0) was never surfaced as a WASM result.
//!    Fixed by principle 4 (`docs/guides/thin-runtime-genericity.md`):
//!    every speet-emitted function now shares type `(registers) ->
//!    (registers)`, and a reserved **halt stub** past the last translated
//!    function surfaces the live register file as the call's results when a
//!    guest `ret` lands past the end of the translated set — exactly the
//!    `main`-returns-with-no-crt0 case these corpus programs exercise. See
//!    [`run_corpus_module`]'s `halt_addr` seeding, below.
//! 2. `arith`/`pairs` used to trap with `uninitialized element N` from a
//!    `call_indirect` — an address/table-index mismatch in indirect-jump
//!    target computation, since fixed by adding `base_func_offset` to the
//!    affected snippets (see the `set_base_func_offset` call each corpus
//!    setup function makes).

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

/// WASM param index of the guest SP register for `arch`, and the module's
/// memory param type it's expressed in (I32 vs I64, matching `memory64`).
/// See `docs/guides/thin-runtime-genericity.md`: guest registers are WASM
/// params, so calling the entry with all-zero params (as a naive wasmi
/// driver would) leaves SP at address 0 — the guest's first `stp`/`push`-
/// style prologue then corrupts low linear memory (or wasmi silently traps
/// on the resulting out-of-bounds access, which `run_corpus_module` used to
/// swallow via `let _ = func.call(...)`, surfacing as a missing exit code
/// rather than a clear error).
fn sp_param_index(arch: crate::CorpusArch) -> u32 {
    match arch {
        crate::CorpusArch::AArch64 => speet_aarch64::AArch64Recompiler::<(), ()>::SP_PARAM_INDEX,
        crate::CorpusArch::X86_64 => speet_x86_64::X86Recompiler::<(), ()>::SP_PARAM_INDEX,
        crate::CorpusArch::Riscv => speet_riscv::RV64_SP_PARAM_INDEX,
    }
}

/// The WASM result index holding the guest's ABI return-value register for
/// `arch`. **Not always index 0**: AArch64 (X0) and x86-64 (RAX) both put
/// their return-value register first in their own register-file layout, so
/// index 0 happens to coincide there, but RISC-V's `int_reg_slot` is laid
/// out in natural register-number order (x0..x31) — x0 is the
/// *hardwired-zero* register, never a return value, while the real
/// return-value register (a0) is x10. Reading a hardcoded index 0 here
/// previously always read RISC-V's permanently-zero x0 instead of a0,
/// making every RISC-V corpus program that legitimately returns a nonzero
/// value from `main` look like it always returned 0.
fn return_reg_index(arch: crate::CorpusArch) -> u32 {
    match arch {
        crate::CorpusArch::AArch64 | crate::CorpusArch::X86_64 => 0,
        crate::CorpusArch::Riscv => 10, // a0 = x10
    }
}

/// The guest link-register's WASM param index for `arch`, or `None` for an
/// arch whose `ret` reads the return address from guest memory (x86-64) —
/// see `speet_rt::entry_bridge_c`'s `lr_param_index` (the native-link
/// counterpart of this same seeding) and
/// `docs/guides/thin-runtime-genericity.md` principle 4.
fn lr_param_index(arch: crate::CorpusArch) -> Option<u32> {
    match arch {
        crate::CorpusArch::AArch64 => Some(speet_aarch64::AArch64Recompiler::<(), ()>::LR_PARAM_INDEX),
        crate::CorpusArch::X86_64 => None,
        crate::CorpusArch::Riscv => Some(speet_riscv::RV64_RA_PARAM_INDEX),
    }
}

/// Seed the guest's initial return address to the **halt sentinel** (see
/// `speet_recompile::frontend::halt_addr`) before invoking the entry
/// function — a *normal* feature (principle 4), not a test-only
/// convenience: real guest programs may return out of `main` with no
/// crt0/libc chain to call `exit`, and the sentinel is what makes that land
/// on the reserved halt stub instead of on whatever garbage the
/// return-address register/slot held at entry. Mirrors
/// `speet_rt::entry_bridge_c`'s native-link seeding: register-based archs
/// get the sentinel directly as a param; x86-64 (stack-based `ret`) gets it
/// written 8 bytes below the reserved stack top, matching what a real
/// `call` would have pushed.
pub fn run_corpus_module(
    wasm: &[u8],
    entry: &str,
    arch: crate::CorpusArch,
    halt_addr: u64,
) -> Result<RunState, String> {
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

    // Guest SP must point at real, freshly reserved linear memory (see
    // `sp_param_index`'s doc) — top of the module's own memory, 16-byte
    // aligned, growing down like a real stack. Stack-based-return archs
    // (x86-64) additionally reserve 8 bytes below that for the halt
    // sentinel "return address" (see this function's doc), matching what a
    // real `call` would have pushed.
    let sp_idx = sp_param_index(arch);
    let lr_idx = lr_param_index(arch);
    let mem = instance.get_export(&store, "memory").and_then(|e| e.into_memory());
    let mem_top: u64 = mem.map(|m| (m.data(&store).len() as u64) & !0xF).unwrap_or(0);
    let guest_sp: u64 = if lr_idx.is_none() { mem_top.saturating_sub(8) } else { mem_top };
    if lr_idx.is_none() {
        if let Some(m) = mem {
            let _ = m.write(&mut store, guest_sp as usize, &halt_addr.to_le_bytes());
        }
    }

    let params: Vec<wasmi::Val> = ty
        .params()
        .iter()
        .enumerate()
        .map(|(i, vt)| {
            if i as u32 == sp_idx {
                match vt {
                    wasmi::ValType::I64 => return wasmi::Val::I64(guest_sp as i64),
                    wasmi::ValType::I32 => return wasmi::Val::I32(guest_sp as i32),
                    _ => {}
                }
            }
            if Some(i as u32) == lr_idx {
                match vt {
                    wasmi::ValType::I64 => return wasmi::Val::I64(halt_addr as i64),
                    wasmi::ValType::I32 => return wasmi::Val::I32(halt_addr as i32),
                    _ => {}
                }
            }
            match vt {
                wasmi::ValType::I32 => wasmi::Val::I32(0),
                wasmi::ValType::I64 => wasmi::Val::I64(0),
                wasmi::ValType::F32 => wasmi::Val::F32(wasmi::F32::from_bits(0)),
                wasmi::ValType::F64 => wasmi::Val::F64(wasmi::F64::from_bits(0)),
                _ => wasmi::Val::I32(0),
            }
        })
        .collect();
    let mut results = vec![wasmi::Val::I32(0); ty.results().len()];
    // Trapping guests (e.g. an uncaught `unreachable`, or a genuine
    // out-of-bounds access) surface as an `Err` here; only swallow it after
    // logging, since a silently-discarded trap previously looked identical
    // to "ran to completion without calling exit" (a missing exit/main_return
    // with no diagnostic at all).
    if let Err(e) = func.call(&mut store, &params, &mut results) {
        eprintln!("speet-corpus-harness: guest call trapped: {e}");
    }
    // Every speet-emitted function has type `(registers) -> (registers)`
    // (principle 4), but which *result index* holds the guest's ABI
    // return-value register is architecture-specific (see
    // `return_reg_index`'s doc) — it is NOT always index 0.
    let main_return = results
        .get(return_reg_index(arch) as usize)
        .and_then(|v| match v {
            wasmi::Val::I32(v) => Some(*v),
            wasmi::Val::I64(v) => Some(*v as i32),
            _ => None,
        });

    let host = store.into_data();
    Ok(RunState {
        unreachable_trap_hits: host.unreachable_trap_hits,
        exit_code: host.exit_code,
        main_return,
        stdout: host.stdout,
    })
}
