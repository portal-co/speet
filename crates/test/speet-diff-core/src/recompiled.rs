//! Speet side of the comparison: translate the case with the production
//! x86-64 frontend path (`speet_recompile::frontend::translate_with_plt`,
//! slot-assigner wired), assemble via the corpus-harness flow (halt stub +
//! unreachable instrumentation), and run under wasmi — then read the final
//! register file back from the halt stub's results.

use crate::case::{Arch, ExecOutcome, ExitKind, FuzzCase, RegState};
use speet_aarch64::AArch64Recompiler;
use speet_arm::ArmRecompiler;
use speet_corpus_harness::{
    assemble_corpus_module, corpus_manifest, corpus_n_imports, instrument_functions,
};
use speet_link_core::image_layout::MemoryModel;
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext};
use speet_memory::memory_access_for_model;
use speet_mips::MipsRecompiler;
use speet_riscv::RiscVRecompiler;
use rv_asm::Xlen;
use speet_x86::X86_32Recompiler;
use speet_x86_64::X86Recompiler;
use wasmparser;
use wasmi::{Engine, Linker, Module as WasmiModule, Store};
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};
use wasm_encoder::{Function, ValType};

/// WASM param indices of the x86-64 register file, per `speet-x86_64`'s
/// layout: GPRs 0–15 (i64; RSP = 4 per `X86Recompiler::SP_PARAM_INDEX`),
/// RIP = 16 (i32), flags 17–21 (i32, ZF SF CF OF PF), temps 22–25, XMM
/// 26–41. The halt stub returns the register file in this order.
#[derive(Debug)]
pub enum RecompiledError {
    /// Executed an unsupported instruction — the `__speet_unreachable_trap`
    /// instrumentation fired (execution-time skip; translation never gates).
    UnsupportedExecuted { func_idx: u32 },
    /// Guest trapped (e.g. store to the read-only page, out-of-bounds
    /// access, `unreachable` from a control-flow hole).
    Trapped(String),
    /// Step budget exhausted under wasmi (`OutOfFuel`).
    StepBudgetExceeded,
    /// Module assembly/validation failed — a harness bug, not a case
    /// outcome; surfaces as an error so it's never mistaken for a skip.
    Internal(String),
}

struct HostState {
    unreachable_hits: Vec<u32>,
    guest_exit: bool,
}

/// Translate + assemble the case into a runnable module (exposed so tests
/// and examples can inspect the WASM directly).
///
/// Deliberately does NOT install a slot assigner (plan §3): the x86-64
/// frontend's indirect-return table math (`ReturnAddressSnippet`) is
/// byte-granular — `(return_addr - base_rip) + base_func_offset` — which
/// only matches the table layout when every decoded byte offset is a slot.
/// The gated `PcSlotMap` layout (one slot per instruction start) severs
/// that correspondence. Byte-granular slots are also what the e2e corpus
/// harness (`translate_x86`) and the halt-sentinel convention assume.
pub fn build_case_module(case: &FuzzCase) -> Result<Vec<u8>, RecompiledError> {
    let (fns, params, _unsupported) = translate_case(case);
    let n_imports = n_imports_cached();

    // Instrument `unreachable` opcodes to call `__speet_unreachable_trap`,
    // mirroring the corpus-harness flow — this is what converts an
    // executed-unsupported instruction into a skip-not-fail signal (plan §5).
    // `SPEET_DIFF_NO_INSTRUMENT=1` skips this (debugging: dump raw bodies).
    let mut fns = fns;
    let manifest = corpus_manifest();
    if std::env::var("SPEET_DIFF_NO_INSTRUMENT").is_err() {
        let trap_idx = manifest
            .index_of("env", "__speet_unreachable_trap")
            .expect("corpus manifest declares env.__speet_unreachable_trap");
        instrument_functions(&mut fns, n_imports, trap_idx);
    }

    // All three archs' slot layouts put slot 0 at the entry PC (x86:
    // byte-granular; aarch64: one slot per 4-byte word; riscv64: one slot
    // per 2-byte halfword).
    let entry_func_idx = 0;
    let wasm = assemble_corpus_module(&fns, &params, entry_func_idx);
    wasmparser::validate(&wasm)
        .map_err(|e| RecompiledError::Internal(format!("validate: {e}")))?;
    Ok(wasm)
}

/// Translate + assemble + run the case. Returns the final architectural
/// state, or the execution-time skip reason.
pub fn run_recompiled(case: &FuzzCase) -> Result<ExecOutcome, RecompiledError> {
    let wasm = build_case_module(case)?;
    exec_module(case, &wasm)
}

fn exec_module(case: &FuzzCase, wasm: &[u8]) -> Result<ExecOutcome, RecompiledError> {
    let mut config = wasmi::Config::default();
    config.consume_fuel(true);
    let engine = Engine::new(&config);
    let mut store =
        Store::new(&engine, HostState { unreachable_hits: Vec::new(), guest_exit: false });
    // Budget: wasmi fuel ≈ instructions here (each guest instruction is a
    // handful of WASM ops; the fuel unit is per-op, so scale the cap up).
    store
        .set_fuel(case.step_budget.saturating_mul(64))
        .map_err(|_| RecompiledError::StepBudgetExceeded)?;
    let mut linker: Linker<HostState> = Linker::new(&engine);

    linker
        .func_wrap(
            "env",
            "__speet_unreachable_trap",
            |mut caller: wasmi::Caller<'_, HostState>, func_idx: i32| {
                // Best-effort guest-PC recovery for diagnostics: the
                // instrumentation's trap doesn't carry it, but the module
                // memory holds nothing useful here either — record the func
                // slot only (slot → instruction start via the slot map is
                // available to the caller).
                caller.data_mut().unreachable_hits.push(func_idx as u32);
            },
        )
        .map_err(|e| RecompiledError::Internal(e.to_string()))?;
    linker
        .func_wrap("env", "__speet_hint", |_caller: wasmi::Caller<'_, HostState>, _id: i32| {
            Ok(())
        })
        .map_err(|e| RecompiledError::Internal(e.to_string()))?;
    linker
        .func_wrap(
            "env",
            "write",
            |_caller: wasmi::Caller<'_, HostState>, _fd: i32, _p: i32, _n: i32| -> i32 { 0 },
        )
        .map_err(|e| RecompiledError::Internal(e.to_string()))?;
    linker
        .func_wrap(
            "env",
            "exit",
            |mut caller: wasmi::Caller<'_, HostState>, _code: i32| {
                caller.data_mut().guest_exit = true;
            },
        )
        .map_err(|e| RecompiledError::Internal(e.to_string()))?;

    let module = WasmiModule::new(&engine, wasm)
        .map_err(|e| RecompiledError::Internal(format!("compile: {e}")))?;
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(|e| RecompiledError::Internal(format!("instantiate: {e}")))?;

    // Seed guest memory with the case's images. OwnedLinear identity
    // mapping means guest VA == WASM offset, and the generator's layout
    // constants keep every region inside the module's default 4 MiB memory.
    let mem = instance
        .get_export(&store, "memory")
        .and_then(|e| e.into_memory())
        .ok_or_else(|| RecompiledError::Internal("no memory export".into()))?;
    mem.write(&mut store, case.data_base as usize, &case.data)
        .map_err(|e| RecompiledError::Internal(format!("seed data: {e}")))?;
    mem.write(&mut store, case.stack_base as usize, &case.stack)
        .map_err(|e| RecompiledError::Internal(format!("seed stack: {e}")))?;
    // Halt sentinel: already embedded in `case.stack` at [SP] by the
    // generator (x86 convention: at entry, [rsp] holds the return
    // address) — a final `ret` pops it, lands past the translated set on
    // the reserved halt stub, and the stub surfaces the live register file
    // as the call's results (thin-runtime-genericity §4).

    // Invoke the entry with the register file seeded from the case.
    let func = instance
        .get_func(&store, "_start")
        .ok_or_else(|| RecompiledError::Internal("no _start".into()))?;
    let ty = func.ty(&store);
    let params: Vec<wasmi::Val> = ty
        .params()
        .iter()
        .enumerate()
        .map(|(i, vt)| seed_param(i, *vt, case))
        .collect();
    let n_results = ty.results().len();
    let mut results = vec![wasmi::Val::I32(0); n_results];
    let call_res = func.call(&mut store, &params, &mut results);

    // Snapshot final memory before dropping the store.
    let mut data_after = vec![0u8; case.data.len()];
    let mut stack_after = vec![0u8; case.stack.len()];
    mem.read(&store, case.data_base as usize, &mut data_after)
        .map_err(|e| RecompiledError::Internal(format!("read data: {e}")))?;
    mem.read(&store, case.stack_base as usize, &mut stack_after)
        .map_err(|e| RecompiledError::Internal(format!("read stack: {e}")))?;

    // Read-only-page detection: the wasmi module has no per-page protection,
    // so a guest store to the RO window silently succeeds. Diff the window —
    // a change means the guest stored to RO, which under the oracle faults
    // (UC_ERR_WRITE_PROT). Classify as a trap so the comparison layer skips
    // it (plan §1: RO store → skip for now). If the RO window is untouched
    // while the oracle faulted on RO, that's a genuine behavioral difference
    // and stays comparable.
    for (off, len) in &case.read_only {
        // `off` is already data-region-relative (see `FuzzCase.read_only`).
        let start = *off as usize;
        let end = start + *len as usize;
        if case.data[start..end] != data_after[start..end] {
            return Err(RecompiledError::Trapped("store to read-only page".into()));
        }
    }

    // ── Skip classification first (plan §5): skips are never failures ──
    let host = store.into_data();
    if let Some(&func_idx) = host.unreachable_hits.first() {
        if let Ok(path) = std::env::var("SPEET_DIFF_DUMP_ON_TRAP") {
            let _ = std::fs::write(&path, wasm);
        }
        return Err(RecompiledError::UnsupportedExecuted { func_idx });
    }
    if host.guest_exit {
        return Err(RecompiledError::Trapped("guest called exit".into()));
    }
    match call_res {
        Err(e) => {
            let msg = e.to_string();
            if msg.to_lowercase().contains("fuel") {
                Err(RecompiledError::StepBudgetExceeded)
            } else {
                // Includes out-of-bounds stores to the RO-page window (the
                // RO region is *unmapped* in the module) — the comparison
                // layer classifies OOB in the RO window as StoreToReadOnly.
                Err(RecompiledError::Trapped(msg))
            }
        }
        Ok(()) => {
            // Halt stub returned the live register file as results.
            Ok(ExecOutcome {
                regs: regs_from_results(case, &results),
                data: data_after,
                stack: stack_after,
                exit: ExitKind::Completed,
            })
        }
    }
}

/// Map case register state onto the WASM entry's parameter list (per arch).
fn seed_param(i: usize, vt: wasmi::ValType, case: &FuzzCase) -> wasmi::Val {
    match case.arch {
        Arch::X86_64 => match (i, vt) {
            (idx, wasmi::ValType::I64) if idx < 16 => {
                wasmi::Val::I64(case.regs.gprs[idx] as i64)
            }
            // RIP local (16) — see `speet_recompile::instrument::GuestPcRef::LocalI64(16)`.
            (16, wasmi::ValType::I32) => wasmi::Val::I32(case.entry_pc as i32),
            // Flags 17–21 (ZF SF CF OF PF).
            (17, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.zf as i32),
            (18, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.sf as i32),
            (19, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.cf as i32),
            (20, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.of as i32),
            (21, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.pf as i32),
            // Temps 22–25 and XMM 26–41: zero.
            (_, wasmi::ValType::I64) => wasmi::Val::I64(0),
            (_, wasmi::ValType::I32) => wasmi::Val::I32(0),
            (_, other) => panic!("unexpected param type {other:?}"),
        },
        Arch::AArch64 => match (i, vt) {
            (idx, wasmi::ValType::I64) if idx < 31 => {
                wasmi::Val::I64(case.regs.gprs[idx] as i64)
            }
            // 31 GPRs + PC i64 (31) + 4 NZCV i32 (32–35, in N Z C V order
            // per `nzcv_slot`'s declaration) + 3 scratch i64 (36–38) + 32 FP
            // (39–70) + expected_RA (71) + SP (72) — see
            // `AArch64Recompiler::BASE_PARAMS`/`SP_PARAM_INDEX`.
            (31, wasmi::ValType::I64) => wasmi::Val::I64(case.entry_pc as i64),
            (32, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.sf as i32), // N
            (33, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.zf as i32), // Z
            (34, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.cf as i32), // C
            (35, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.of as i32), // V
            (71, wasmi::ValType::I64) => wasmi::Val::I64(0), // expected_RA
            (72, wasmi::ValType::I64) => wasmi::Val::I64(case.regs.sp as i64),
            (_, wasmi::ValType::I64) => wasmi::Val::I64(0),
            (_, wasmi::ValType::I32) => wasmi::Val::I32(0),
            // FP registers 39–70 are F64 params — M4 is integer-only.
            (_, wasmi::ValType::F64) => wasmi::Val::F64(0f64.into()),
            (_, other) => panic!("unexpected param type {other:?}"),
        },
        Arch::RiscV64 => match (i, vt) {
            (idx, wasmi::ValType::I64) if idx < 32 => {
                wasmi::Val::I64(case.regs.gprs[idx] as i64)
            }
            // 32 int i64 (0–31) + 32 FP f64 (32–63) + PC i64 (64) +
            // expected_RA i64 (65) — see `RiscVRecompiler::BASE_PARAMS`.
            (64, wasmi::ValType::I64) => wasmi::Val::I64(case.entry_pc as i64),
            (65, wasmi::ValType::I64) => wasmi::Val::I64(0), // expected_RA
            (_, wasmi::ValType::I64) => wasmi::Val::I64(0),
            (_, wasmi::ValType::F64) => wasmi::Val::F64(0f64.into()),
            (_, other) => panic!("unexpected param type {other:?}"),
        },
        Arch::Arm => match (i, vt) {
            // 16 GPR i64 (0–15, R13=SP, R14=LR) + expected_ra i64 (16) +
            // CPSR i32 (17) — see `ArmRecompiler::BASE_PARAMS` (18).
            (idx, wasmi::ValType::I64) if idx < 16 => {
                wasmi::Val::I64(case.regs.gprs[idx] as i64)
            }
            (16, wasmi::ValType::I64) => wasmi::Val::I64(0), // expected_ra
            (17, wasmi::ValType::I32) => {
                // CPSR: N Z C V at bits 31/30/29/28.
                wasmi::Val::I32(
                    ((case.regs.sf as i32) << 31)
                        | ((case.regs.zf as i32) << 30)
                        | ((case.regs.cf as i32) << 29)
                        | ((case.regs.of as i32) << 28),
                )
            }
            (_, wasmi::ValType::I64) => wasmi::Val::I64(0),
            (_, wasmi::ValType::I32) => wasmi::Val::I32(0),
            (_, other) => panic!("unexpected param type {other:?}"),
        },
        Arch::X86_32 => match (i, vt) {
            // 8 GPR i64 (0–7) + EIP i64 (8) + flags i32 9–12 (ZF SF CF OF)
            // + expected_ra i64 (13) + tmp i64 (14–15).
            (idx, wasmi::ValType::I64) if idx < 8 => {
                wasmi::Val::I64(case.regs.gprs[idx] as i64)
            }
            (8, wasmi::ValType::I64) => wasmi::Val::I64(case.entry_pc as i64),
            (9, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.zf as i32),
            (10, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.sf as i32),
            (11, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.cf as i32),
            (12, wasmi::ValType::I32) => wasmi::Val::I32(case.regs.of as i32),
            (13, wasmi::ValType::I64) => wasmi::Val::I64(0), // expected_ra
            (_, wasmi::ValType::I64) => wasmi::Val::I64(0),
            (_, wasmi::ValType::I32) => wasmi::Val::I32(0),
            (_, other) => panic!("unexpected param type {other:?}"),
        },
        Arch::RiscV32 => match (i, vt) {
            // 32 int i32 (0–31) + 32 FP (32–63) + PC i32 (64) +
            // expected_RA i32 (65).
            (idx, wasmi::ValType::I32) if idx < 32 => {
                wasmi::Val::I32(case.regs.gprs[idx] as i32)
            }
            (64, wasmi::ValType::I32) => wasmi::Val::I32(case.entry_pc as i32),
            (65, wasmi::ValType::I32) => wasmi::Val::I32(0), // expected_RA
            (_, wasmi::ValType::I32) => wasmi::Val::I32(0),
            // memory64 address-mapper scratch params are i64 even on RV32
            // (guest addresses are 64-bit WASM offsets under OwnedLinear).
            (_, wasmi::ValType::I64) => wasmi::Val::I64(0),
            (_, wasmi::ValType::F64) => wasmi::Val::F64(0f64.into()),
            (_, other) => panic!("unexpected param type {other:?}"),
        },
        Arch::Mips => match (i, vt) {
            // 32 GPR i32 (0–31) + HI/LO i32 (32–33) + PC i32 (34) +
            // expected_RA i32 (35) — see `MipsRecompiler::BASE_PARAMS` (36).
            (idx, wasmi::ValType::I32) if idx < 32 => {
                wasmi::Val::I32(case.regs.gprs[idx] as i32)
            }
            (34, wasmi::ValType::I32) => wasmi::Val::I32(case.entry_pc as i32),
            (35, wasmi::ValType::I32) => wasmi::Val::I32(0), // expected_RA
            (_, wasmi::ValType::I32) => wasmi::Val::I32(0),
            // memory64 address-mapper scratch params (i64) — same note as
            // the RV32 branch.
            (_, wasmi::ValType::I64) => wasmi::Val::I64(0),
            (_, other) => panic!("unexpected param type {other:?}"),
        },
    }
}

fn regs_from_results(case: &FuzzCase, results: &[wasmi::Val]) -> RegState {
    let mut regs = RegState::default();
    match case.arch {
        Arch::X86_64 => {
            for g in 0..16 {
                regs.gprs[g] = read_i64(results, g);
            }
            // Final RIP: the halt stub returns the register file as it
            // stood when the guest jumped past the translated set — the
            // sentinel.
            regs.rip = 0; // compared loosely (both engines end "at halt")
            regs.zf = read_i64(results, 17) != 0;
            regs.sf = read_i64(results, 18) != 0;
            regs.cf = read_i64(results, 19) != 0;
            regs.of = read_i64(results, 20) != 0;
            regs.pf = read_i64(results, 21) != 0;
        }
        Arch::AArch64 => {
            for g in 0..31 {
                regs.gprs[g] = read_i64(results, g);
            }
            regs.rip = 0;
            // NZCV param order: N, Z, C, V (per `nzcv_slot` declaration).
            regs.sf = read_i64(results, 32) != 0;
            regs.zf = read_i64(results, 33) != 0;
            regs.cf = read_i64(results, 34) != 0;
            regs.of = read_i64(results, 35) != 0;
            regs.sp = read_i64(results, 72);
        }
        Arch::RiscV64 => {
            for g in 0..32 {
                regs.gprs[g] = read_i64(results, g);
            }
            regs.gprs[0] = 0; // hardwired zero — both engines keep it 0
            regs.rip = 0;
        }
        Arch::Arm => {
            for g in 0..16 {
                regs.gprs[g] = read_i64(results, g);
            }
            regs.rip = 0;
            let cpsr = read_i64(results, 17);
            regs.sf = cpsr & (1 << 31) != 0;
            regs.zf = cpsr & (1 << 30) != 0;
            regs.cf = cpsr & (1 << 29) != 0;
            regs.of = cpsr & (1 << 28) != 0;
        }
        Arch::X86_32 => {
            for g in 0..8 {
                regs.gprs[g] = read_i64(results, g);
            }
            regs.rip = 0;
            regs.zf = read_i64(results, 9) != 0;
            regs.sf = read_i64(results, 10) != 0;
            regs.cf = read_i64(results, 11) != 0;
            regs.of = read_i64(results, 12) != 0;
        }
        Arch::RiscV32 => {
            for g in 0..32 {
                regs.gprs[g] = read_i64(results, g);
            }
            regs.gprs[0] = 0;
            regs.rip = 0;
        }
        Arch::Mips => {
            for g in 0..32 {
                regs.gprs[g] = read_i64(results, g);
            }
            regs.gprs[0] = 0; // $0 hardwired zero
            regs.rip = 0;
        }
    }
    regs
}

fn read_i64(results: &[wasmi::Val], idx: usize) -> u64 {
    match results.get(idx) {
        Some(wasmi::Val::I64(v)) => *v as u64,
        Some(wasmi::Val::I32(v)) => *v as i64 as u64,
        _ => 0,
    }
}

/// Number of imports in the corpus manifest (read from the manifest, never
/// hand-counted — thin-runtime-genericity §1).
static N_IMPORTS_ONCE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

fn n_imports_cached() -> u32 {
    *N_IMPORTS_ONCE.get_or_init(corpus_n_imports)
}

/// Translate the case with the production x86-64 frontend: returns
/// (function bodies, register-file params, unsupported-insn coverage signal).
pub fn translate_case(case: &FuzzCase) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    match case.arch {
        Arch::X86_64 => translate_case_x86(case),
        Arch::AArch64 => translate_case_a64(case),
        Arch::RiscV64 => translate_case_rv64(case),
        Arch::Arm => translate_case_arm(case),
        Arch::X86_32 => translate_case_x86_32(case),
        Arch::RiscV32 => translate_case_rv32(case),
        Arch::Mips => translate_case_mips(case),
    }
}

fn make_rctx(
    reactor: &mut Reactor<(), core::convert::Infallible, Function, LocalPool>,
) -> ReactorAdapter<'_, (), core::convert::Infallible, Function, LocalPool> {
    static T: TableIdx = TableIdx(0);
    let mut rctx = ReactorAdapter {
        reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark { slot_count: 0, total_locals: 0 },
        injected_start: yecta::Mark { slot_count: 0, total_locals: 0 },
        layout_params: speet_link_core::RuntimeLayoutParams::new(),
        pool: yecta::Pool { handler: &T, ty: TypeIdx(0) },
        escape: yecta::CallEscape::Jump,
    };
    rctx.set_base_func_offset(n_imports_cached());
    rctx
}

fn collect_params(rctx: &ReactorAdapter<'_, (), core::convert::Infallible, Function, LocalPool>) -> Vec<ValType> {
    // Snapshot the register-file param list right after setup_traps (the
    // mark includes trap + memory-access scratch params) — same as the e2e
    // harness `collect_rv_params`, which never hand-counts.
    rctx.layout()
        .iter_before(&rctx.locals_mark())
        .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
        .collect()
}

fn translate_case_x86(case: &FuzzCase) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    let mut rc = X86Recompiler::<(), core::convert::Infallible>::new_with_base_rip(case.entry_pc);
    rc.set_memory_access(memory_access_for_model::<(), core::convert::Infallible>(
        MemoryModel::OwnedLinear,
    ));
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.bind_memory_layout(&rctx);
    let params = collect_params(&rctx);
    rc.translate_bytes(&mut ctx, &mut rctx, &case.code, case.entry_pc, &mut |a| {
        Function::new(a.collect::<Vec<_>>())
    })
    .expect("translate_bytes");
    let unsupported = rc.unsupported_insns().iter().cloned().collect();
    (rctx.drain_fns(), params, unsupported)
}

fn translate_case_a64(case: &FuzzCase) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    let mut rc = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(case.entry_pc);
    rc.set_memory_access(memory_access_for_model::<(), core::convert::Infallible>(
        MemoryModel::OwnedLinear,
    ));
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.bind_memory_layout(&rctx);
    let params = collect_params(&rctx);
    rc.translate_bytes(&mut ctx, &mut rctx, &case.code, case.entry_pc, &mut |a| {
        Function::new(a.collect::<Vec<_>>())
    })
    .expect("translate_bytes");
    let unsupported = rc.unsupported_insns().iter().cloned().collect();
    (rctx.drain_fns(), params, unsupported)
}

fn translate_case_rv(case: &FuzzCase, xlen: Xlen) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    let rv64 = xlen == Xlen::Rv64;
    // (base_pc, track_hints, enable_rv64, use_memory64) — same config as
    // the e2e harness `translate_rv`.
    let mut rc = RiscVRecompiler::<(), core::convert::Infallible, Function>::new_with_full_config(
        case.entry_pc, false, rv64, true,
    );
    // Memory access: like the e2e harness, RV frontends keep the raw
    // emission path (memory64's i64 loads/stores ARE the identity mapping
    // under OwnedLinear). Binding the address-mapper here mismatches the
    // RV32 i32 value widths (validate: "expected i32, found i64").
    if rv64 {
        rc.set_memory_access(memory_access_for_model::<(), core::convert::Infallible>(
            MemoryModel::OwnedLinear,
        ));
    }
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.bind_memory_layout(&rctx);
    let params = collect_params(&rctx);
    rc.translate_bytes(&mut ctx, &mut rctx, &case.code, case.entry_pc as u32, xlen, &mut |a| {
        Function::new(a.collect::<Vec<_>>())
    })
    .expect("translate_bytes");
    let unsupported = rc.unsupported_insns().iter().cloned().collect();
    (rctx.drain_fns(), params, unsupported)
}

fn translate_case_rv64(case: &FuzzCase) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    translate_case_rv(case, Xlen::Rv64)
}

fn translate_case_rv32(case: &FuzzCase) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    translate_case_rv(case, Xlen::Rv32)
}

fn translate_case_arm(case: &FuzzCase) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    let mut rc = ArmRecompiler::<(), core::convert::Infallible>::new_with_base_pc(case.entry_pc);
    rc.set_memory_access(memory_access_for_model::<(), core::convert::Infallible>(
        MemoryModel::OwnedLinear,
    ));
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.bind_memory_layout(&rctx);
    let params = collect_params(&rctx);
    rc.translate_bytes(&mut ctx, &mut rctx, &case.code, case.entry_pc, &mut |a| {
        Function::new(a.collect::<Vec<_>>())
    })
    .expect("translate_bytes");
    let unsupported = rc.unsupported_insns().iter().cloned().collect();
    (rctx.drain_fns(), params, unsupported)
}

fn translate_case_x86_32(case: &FuzzCase) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    let mut rc = X86_32Recompiler::<(), core::convert::Infallible>::new_with_base_eip(case.entry_pc);
    rc.set_memory_access(memory_access_for_model::<(), core::convert::Infallible>(
        MemoryModel::OwnedLinear,
    ));
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.bind_memory_layout(&rctx);
    let params = collect_params(&rctx);
    rc.translate_bytes(&mut ctx, &mut rctx, &case.code, case.entry_pc, &mut |a| {
        Function::new(a.collect::<Vec<_>>())
    })
    .expect("translate_bytes");
    let unsupported = rc.unsupported_insns().iter().cloned().collect();
    (rctx.drain_fns(), params, unsupported)
}

fn translate_case_mips(case: &FuzzCase) -> (Vec<Function>, Vec<ValType>, Vec<String>) {
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    // MipsRecompiler::new_with_base_pc takes u32 (mips32 addresses).
    let mut rc = MipsRecompiler::<(), core::convert::Infallible, Function>::new_with_base_pc(
        case.entry_pc as u32,
    );
    rc.set_memory64(true);
    // Byte order stays little-endian for now (matches historical behavior;
    // the oracle is BE, so byte-order-sensitive accesses still diverge).
    // Enabling the BE swaps trips a yecta const-fold/peephole unsoundness:
    // the swap term sequences get rewritten stack-imbalanced when the stored
    // value is a known constant — minimal repro in mips-plan.md §4b.
    // Delay-slot support: the fetcher is installed below and branch/jump
    // translation re-runs the delay word inside the taken arm (see
    // docs/mips-plan.md §2). Delay words stay real slots so slot indices
    // stay 1:1 with decode positions.
    // Raw emission path (see the RV note): memory64's i64 loads/stores are
    // the identity mapping under OwnedLinear; binding the address-mapper
    // mismatches MIPS's i32 value widths.
    // rc.set_memory_access(...);
    // Delay-slot fetcher: hand the recompiler the big-endian word stream so
    // branch/jump translation can re-run the delay word in its taken arm.
    let entry = case.entry_pc as u32;
    rc.set_delay_slot_fetcher(Box::new(move |pc: u32| {
        let off = (pc.wrapping_sub(entry)) as usize;
        if off + 4 > case.code.len() {
            return None;
        }
        let bytes = [
            case.code[off],
            case.code[off + 1],
            case.code[off + 2],
            case.code[off + 3],
        ];
        Some(rabbitizer::Instruction::new(
            u32::from_be_bytes(bytes),
            pc,
            rabbitizer::InstrCategory::CPU,
        ))
    }));
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.bind_memory_layout(&rctx);
    let params = collect_params(&rctx);
    // MIPS words are big-endian; translate per word (same as the e2e
    // harness `translate_mips`).
    let words = case.code.len() / 4;
    for i in 0..words {
        let offset = i * 4;
        let word = u32::from_be_bytes([
            case.code[offset],
            case.code[offset + 1],
            case.code[offset + 2],
            case.code[offset + 3],
        ]);
        let pc = (case.entry_pc as u32).wrapping_add(offset as u32);
        let inst = rabbitizer::Instruction::new(word, pc, rabbitizer::InstrCategory::CPU);
        rc.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        })
        .expect("translate_instruction");
    }
    let _ = rctx.seal_remaining(&mut ctx);
    // MIPS tracks no unsupported_insns (unrecognized words emit a bare
    // `unreachable` with no name recorded) — empty coverage signal.
    (rctx.drain_fns(), params, Vec::new())
}

/// Debug helper: which mnemonics the recompiler flagged as unsupported
/// (coverage signal only — never a gate; plan §1).
pub fn unsupported_for(case: &FuzzCase) -> Vec<String> {
    translate_case(case).2
}

/// Debug: number of imports in the corpus manifest.
pub fn n_imports_dbg() -> u32 {
    n_imports_cached()
}
