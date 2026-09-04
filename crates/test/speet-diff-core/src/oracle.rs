//! Unicorn oracle (plan §3.1 `oracle` module).
//!
//! Maps the case's regions (code RX at `CODE_BASE`, data RW with a
//! read-only page, stack RW), seeds the halt sentinel, runs with a
//! step cap, and snapshots final architectural state. All scope-filter
//! outcomes (fault, RO store, budget) map to the case-level skip kinds —
//! the oracle is never "wrong", just out of scope.

use crate::case::{ExecOutcome, ExitKind, FuzzCase, RegState};
use crate::generator::{CODE_BASE, STACK_BASE};
use unicorn_engine::unicorn_const::{Arch, Mode, Prot};
use unicorn_engine::{RegisterX86, Unicorn};

fn err_str(e: unicorn_engine::uc_error) -> String {
    format!("{e:?}")
}

/// Run the case under Unicorn. `Err` = oracle-side skip (plan §5: the
/// comparison layer turns this into `SkipReason::OracleFault` unless the
/// error itself names a scope filter).
pub fn run_oracle(case: &FuzzCase) -> Result<ExecOutcome, String> {
    let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).map_err(err_str)?;

    // Code region: RX (plan §1 scope rule — memory mapped RX).
    let code_size = ((case.code.len() + 0xFFF) & !0xFFF).max(0x1000) as u64;
    uc.mem_map(CODE_BASE, code_size, Prot::READ | Prot::EXEC)
        .map_err(err_str)?;
    uc.mem_write(CODE_BASE, &case.code).map_err(err_str)?;

    // Data region: RW, with the read-only page `mem_protect`ed down.
    let data_size = (case.data.len() as u64).max(0x1000);
    uc.mem_map(case.data_base, data_size, Prot::READ | Prot::WRITE)
        .map_err(err_str)?;
    uc.mem_write(case.data_base, &case.data).map_err(err_str)?;
    for (off, len) in &case.read_only {
        uc.mem_protect(case.data_base + off, *len, Prot::READ)
            .map_err(err_str)?;
    }

    // Stack region: RW + the halt-sentinel "return address" below SP.
    let stack_size = case.stack.len() as u64;
    uc.mem_map(case.stack_base, stack_size, Prot::READ | Prot::WRITE)
        .map_err(err_str)?;
    uc.mem_write(case.stack_base, &case.stack).map_err(err_str)?;

    // Initial register file.
    let names = [
        RegisterX86::RAX, RegisterX86::RCX, RegisterX86::RDX, RegisterX86::RBX,
        RegisterX86::RSP, RegisterX86::RBP, RegisterX86::RSI, RegisterX86::RDI,
        RegisterX86::R8, RegisterX86::R9, RegisterX86::R10, RegisterX86::R11,
        RegisterX86::R12, RegisterX86::R13, RegisterX86::R14, RegisterX86::R15,
    ];
    for (i, reg) in names.iter().enumerate() {
        uc.reg_write(*reg, case.regs.gprs[i]).map_err(err_str)?;
    }
    // RIP is set by `emu_start`'s `begin` argument — writing the RIP
    // register directly returns `uc_error::ARG` on the x86 backend.

    // Stop address = the halt sentinel: one past the last code byte. A final
    // `ret` popping it ends the emulation cleanly (exit-at-sequence-end).
    let until = case.halt_sentinel();
    // emu_start(begin, until, timeout_us=0 (none), count=step budget).
    let started = uc.emu_start(case.entry_pc, until, 0, case.step_budget as usize);

    match started {
        Ok(()) => {}
        Err(unicorn_engine::uc_error::WRITE_PROT) => {
            return Ok(outcome_of(&uc, case, ExitKind::StoreToReadOnly));
        }
        Err(unicorn_engine::uc_error::OK) => unreachable!(),
        Err(e) => {
            let name = format!("{e:?}");
            // Invalid opcode ⇒ executed-unsupported skip. FETCH_PROT /
            // anything else ⇒ oracle fault.
            if name.contains("INSN_INVALID") || name.contains("INVALID") {
                return Ok(outcome_of(&uc, case, ExitKind::UnsupportedExecuted));
            }
            if name.contains("FETCH_PROT") {
                return Ok(outcome_of(&uc, case, ExitKind::Completed));
            }
            return Err(name);
        }
    }

    Ok(outcome_of(&uc, case, ExitKind::Completed))
}

/// Snapshot the post-run state. Final RIP: `emu_start` stopping at `until`
/// reports the stop address; a `ret`-to-sentinel exit lands RIP at the
/// sentinel, matching the recompiled halt-stub semantics.
fn outcome_of(uc: &Unicorn<'_, ()>, case: &FuzzCase, exit: ExitKind) -> ExecOutcome {
    let names = [
        RegisterX86::RAX, RegisterX86::RCX, RegisterX86::RDX, RegisterX86::RBX,
        RegisterX86::RSP, RegisterX86::RBP, RegisterX86::RSI, RegisterX86::RDI,
        RegisterX86::R8, RegisterX86::R9, RegisterX86::R10, RegisterX86::R11,
        RegisterX86::R12, RegisterX86::R13, RegisterX86::R14, RegisterX86::R15,
    ];
    let mut regs = RegState::default();
    for (i, reg) in names.iter().enumerate() {
        regs.gprs[i] = uc.reg_read(*reg).unwrap_or(0);
    }
    regs.rip = uc.reg_read(RegisterX86::RIP).unwrap_or(case.halt_sentinel());
    let eflags = uc.reg_read(RegisterX86::EFLAGS).unwrap_or(0) as u64;
    // EFLAGS bit positions: CF=0, PF=2, ZF=6, SF=7, OF=11.
    regs.cf = eflags & (1 << 0) != 0;
    regs.pf = eflags & (1 << 2) != 0;
    regs.zf = eflags & (1 << 6) != 0;
    regs.sf = eflags & (1 << 7) != 0;
    regs.of = eflags & (1 << 11) != 0;

    let mut data = vec![0u8; case.data.len()];
    let _ = uc.mem_read(case.data_base, &mut data);
    let mut stack = vec![0u8; case.stack.len()];
    let _ = uc.mem_read(STACK_BASE, &mut stack);

    ExecOutcome { regs, data, stack, exit }
}
