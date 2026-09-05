//! Unicorn oracle (plan §3.1 `oracle` module).
//!
//! Maps the case's regions (code RX at `CODE_BASE`, data RW with a
//! read-only page, stack RW), seeds the halt sentinel, runs with a
//! step cap, and snapshots final architectural state. All scope-filter
//! outcomes (fault, RO store, budget) map to the case-level skip kinds —
//! the oracle is never "wrong", just out of scope.

use crate::case::{Arch, ExecOutcome, ExitKind, FuzzCase, RegState};
use crate::generator::{CODE_BASE, STACK_BASE};
use unicorn_engine::unicorn_const::{Arch as UArch, Mode, Prot};
use unicorn_engine::{RegisterARM, RegisterARM64, RegisterMIPS, RegisterRISCV, RegisterX86, Unicorn};

fn err_str(e: unicorn_engine::uc_error) -> String {
    format!("{e:?}")
}

/// Run the case under Unicorn. `Err` = oracle-side skip (plan §5: the
/// comparison layer turns this into `SkipReason::OracleFault` unless the
/// error itself names a scope filter).
pub fn run_oracle(case: &FuzzCase) -> Result<ExecOutcome, String> {
    match case.arch {
        Arch::X86_64 => run_oracle_x86(case),
        Arch::AArch64 => run_oracle_a64(case),
        Arch::RiscV64 => run_oracle_rv64(case),
        Arch::Arm => run_oracle_arm(case),
        Arch::X86_32 => run_oracle_x86_32(case),
        Arch::RiscV32 => run_oracle_rv32(case),
        Arch::Mips => run_oracle_mips(case),
    }
}

const ARM_REGS: [RegisterARM; 16] = [
    RegisterARM::R0, RegisterARM::R1, RegisterARM::R2, RegisterARM::R3,
    RegisterARM::R4, RegisterARM::R5, RegisterARM::R6, RegisterARM::R7,
    RegisterARM::R8, RegisterARM::R9, RegisterARM::R10, RegisterARM::R11,
    RegisterARM::R12, RegisterARM::SP, RegisterARM::LR, RegisterARM::PC,
];

const MIPS_REGS: [RegisterMIPS; 32] = [
    RegisterMIPS::R0, RegisterMIPS::R1, RegisterMIPS::R2, RegisterMIPS::R3,
    RegisterMIPS::R4, RegisterMIPS::R5, RegisterMIPS::R6, RegisterMIPS::R7,
    RegisterMIPS::R8, RegisterMIPS::R9, RegisterMIPS::R10, RegisterMIPS::R11,
    RegisterMIPS::R12, RegisterMIPS::R13, RegisterMIPS::R14, RegisterMIPS::R15,
    RegisterMIPS::R16, RegisterMIPS::R17, RegisterMIPS::R18, RegisterMIPS::R19,
    RegisterMIPS::R20, RegisterMIPS::R21, RegisterMIPS::R22, RegisterMIPS::R23,
    RegisterMIPS::R24, RegisterMIPS::R25, RegisterMIPS::R26, RegisterMIPS::R27,
    RegisterMIPS::R28, RegisterMIPS::R29, RegisterMIPS::R30, RegisterMIPS::R31,
];

/// AArch32 oracle. Halt: sentinel seeded in LR (R14), terminator BX LR.
fn run_oracle_arm(case: &FuzzCase) -> Result<ExecOutcome, String> {
    let mut uc = Unicorn::new(UArch::ARM, Mode::ARM).map_err(err_str)?;
    map_regions(&mut uc, case)?;
    for (i, reg) in ARM_REGS.iter().enumerate().take(15) {
        uc.reg_write(*reg, case.regs.gprs[i] & 0xFFFF_FFFF).map_err(err_str)?;
    }
    // CPSR: N Z C V at bits 31/30/29/28 (flag_names: Z→zf N→sf C→cf V→of).
    let cpsr = ((case.regs.sf as u32) << 31)
        | ((case.regs.zf as u32) << 30)
        | ((case.regs.cf as u32) << 29)
        | ((case.regs.of as u32) << 28);
    uc.reg_write(RegisterARM::CPSR, cpsr as u64).map_err(err_str)?;
    let until = case.halt_sentinel();
    let exit = match uc.emu_start(case.entry_pc, until, 0, case.step_budget as usize) {
        Ok(()) => ExitKind::Completed,
        Err(e) => classify_start_err(e)?,
    };
    let mut regs = RegState::default();
    for (i, reg) in ARM_REGS.iter().enumerate().take(15) {
        regs.gprs[i] = uc.reg_read(*reg).unwrap_or(0);
    }
    regs.rip = uc.reg_read(RegisterARM::PC).unwrap_or(until);
    let cpsr = uc.reg_read(RegisterARM::CPSR).unwrap_or(0) as u64;
    regs.sf = cpsr & (1 << 31) != 0;
    regs.zf = cpsr & (1 << 30) != 0;
    regs.cf = cpsr & (1 << 29) != 0;
    regs.of = cpsr & (1 << 28) != 0;
    let (data, stack) = snapshot_mem(&uc, case);
    Ok(ExecOutcome { regs, data, stack, exit })
}

/// i686 oracle. Same register names as x86-64 (MODE_32), 32-bit EIP via
/// `emu_start`'s begin; the halt sentinel is popped by RET.
fn run_oracle_x86_32(case: &FuzzCase) -> Result<ExecOutcome, String> {
    let mut uc = Unicorn::new(UArch::X86, Mode::MODE_32).map_err(err_str)?;
    map_regions(&mut uc, case)?;
    for (i, reg) in [
        RegisterX86::EAX, RegisterX86::ECX, RegisterX86::EDX, RegisterX86::EBX,
        RegisterX86::ESP, RegisterX86::EBP, RegisterX86::ESI, RegisterX86::EDI,
    ].iter().enumerate() {
        uc.reg_write(*reg, case.regs.gprs[i] & 0xFFFF_FFFF).map_err(err_str)?;
    }
    let until = case.halt_sentinel();
    let exit = match uc.emu_start(case.entry_pc, until, 0, case.step_budget as usize) {
        Ok(()) => ExitKind::Completed,
        Err(e) => classify_start_err(e)?,
    };
    let mut regs = RegState::default();
    for (i, reg) in [
        RegisterX86::EAX, RegisterX86::ECX, RegisterX86::EDX, RegisterX86::EBX,
        RegisterX86::ESP, RegisterX86::EBP, RegisterX86::ESI, RegisterX86::EDI,
    ].iter().enumerate() {
        regs.gprs[i] = uc.reg_read(*reg).unwrap_or(0) & 0xFFFF_FFFF;
    }
    regs.rip = uc.reg_read(RegisterX86::EIP).unwrap_or(until);
    let eflags = uc.reg_read(RegisterX86::EFLAGS).unwrap_or(0) as u64;
    regs.cf = eflags & (1 << 0) != 0;
    regs.pf = eflags & (1 << 2) != 0;
    regs.zf = eflags & (1 << 6) != 0;
    regs.sf = eflags & (1 << 7) != 0;
    regs.of = eflags & (1 << 11) != 0;
    let (data, stack) = snapshot_mem(&uc, case);
    Ok(ExecOutcome { regs, data, stack, exit })
}

/// RV32 oracle.
fn run_oracle_rv32(case: &FuzzCase) -> Result<ExecOutcome, String> {
    let mut uc = Unicorn::new(UArch::RISCV, Mode::RISCV32).map_err(err_str)?;
    map_regions(&mut uc, case)?;
    for (i, reg) in RV64_REGS.iter().enumerate() {
        uc.reg_write(*reg, case.regs.gprs[i] & 0xFFFF_FFFF).map_err(err_str)?;
    }
    let until = case.halt_sentinel();
    let exit = match uc.emu_start(case.entry_pc, until, 0, case.step_budget as usize) {
        Ok(()) => ExitKind::Completed,
        Err(e) => classify_start_err(e)?,
    };
    let mut regs = RegState::default();
    for (i, reg) in RV64_REGS.iter().enumerate() {
        regs.gprs[i] = uc.reg_read(*reg).unwrap_or(0) & 0xFFFF_FFFF;
    }
    regs.gprs[0] = 0;
    regs.rip = uc.reg_read(RegisterRISCV::PC).unwrap_or(until);
    let (data, stack) = snapshot_mem(&uc, case);
    Ok(ExecOutcome { regs, data, stack, exit })
}

/// MIPS32 big-endian oracle. Halt: sentinel seeded in $31 (ra),
/// terminator JR $ra.
fn run_oracle_mips(case: &FuzzCase) -> Result<ExecOutcome, String> {
    let mut uc = Unicorn::new(UArch::MIPS, Mode::MIPS32 | Mode::BIG_ENDIAN).map_err(err_str)?;
    map_regions(&mut uc, case)?;
    for (i, reg) in MIPS_REGS.iter().enumerate() {
        uc.reg_write(*reg, case.regs.gprs[i] & 0xFFFF_FFFF).map_err(err_str)?;
    }
    let until = case.halt_sentinel();
    let exit = match uc.emu_start(case.entry_pc, until, 0, case.step_budget as usize) {
        Ok(()) => ExitKind::Completed,
        Err(e) => classify_start_err(e)?,
    };
    let mut regs = RegState::default();
    for (i, reg) in MIPS_REGS.iter().enumerate() {
        regs.gprs[i] = uc.reg_read(*reg).unwrap_or(0) & 0xFFFF_FFFF;
    }
    regs.gprs[0] = 0;
    regs.rip = uc.reg_read(RegisterMIPS::PC).unwrap_or(until);
    let (data, stack) = snapshot_mem(&uc, case);
    Ok(ExecOutcome { regs, data, stack, exit })
}

/// Shared region mapping (code RX / data RW+RO page / stack RW).
fn map_regions(
    uc: &mut Unicorn<'_, ()>,
    case: &FuzzCase,
) -> Result<(), String> {
    let code_size = ((case.code.len() + 0xFFF) & !0xFFF).max(0x1000) as u64;
    uc.mem_map(CODE_BASE, code_size, Prot::READ | Prot::EXEC)
        .map_err(err_str)?;
    uc.mem_write(CODE_BASE, &case.code).map_err(err_str)?;
    let data_size = (case.data.len() as u64).max(0x1000);
    uc.mem_map(case.data_base, data_size, Prot::READ | Prot::WRITE)
        .map_err(err_str)?;
    uc.mem_write(case.data_base, &case.data).map_err(err_str)?;
    for (off, len) in &case.read_only {
        uc.mem_protect(case.data_base + off, *len, Prot::READ)
            .map_err(err_str)?;
    }
    let stack_size = case.stack.len() as u64;
    uc.mem_map(case.stack_base, stack_size, Prot::READ | Prot::WRITE)
        .map_err(err_str)?;
    uc.mem_write(case.stack_base, &case.stack).map_err(err_str)?;
    Ok(())
}

/// Interpret an `emu_start` error into exit-kind semantics shared by archs.
fn classify_start_err(e: unicorn_engine::uc_error) -> Result<ExitKind, String> {
    match e {
        unicorn_engine::uc_error::WRITE_PROT => Ok(ExitKind::StoreToReadOnly),
        unicorn_engine::uc_error::OK => unreachable!(),
        other => {
            let name = format!("{other:?}");
            if name.contains("INSN_INVALID") || name.contains("INVALID") {
                Ok(ExitKind::UnsupportedExecuted)
            } else if name.contains("FETCH_PROT") {
                Ok(ExitKind::Completed)
            } else {
                Err(name)
            }
        }
    }
}

/// Snapshot memory regions shared by archs.
fn snapshot_mem(uc: &Unicorn<'_, ()>, case: &FuzzCase) -> (Vec<u8>, Vec<u8>) {
    let mut data = vec![0u8; case.data.len()];
    let _ = uc.mem_read(case.data_base, &mut data);
    let mut stack = vec![0u8; case.stack.len()];
    let _ = uc.mem_read(STACK_BASE, &mut stack);
    (data, stack)
}

fn run_oracle_x86(case: &FuzzCase) -> Result<ExecOutcome, String> {
    let mut uc = Unicorn::new(UArch::X86, Mode::MODE_64).map_err(err_str)?;
    map_regions(&mut uc, case)?;
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
    let until = case.halt_sentinel();
    let exit = match uc.emu_start(case.entry_pc, until, 0, case.step_budget as usize) {
        Ok(()) => ExitKind::Completed,
        Err(e) => classify_start_err(e)?,
    };
    Ok(outcome_of_x86(&uc, case, exit))
}


/// Snapshot the post-run state (x86-64). Final RIP: `emu_start` stopping at
/// `until` reports the stop address; a `ret`-to-sentinel exit lands RIP at
/// the sentinel, matching the recompiled halt-stub semantics.
fn outcome_of_x86(uc: &Unicorn<'_, ()>, case: &FuzzCase, exit: ExitKind) -> ExecOutcome {
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

    let (data, stack) = snapshot_mem(uc, case);
    ExecOutcome { regs, data, stack, exit }
}

/// X0–X28 are contiguous (199..=227); X29/X30 sit at 1/2. Keep an explicit
/// table — never compute register ids by arithmetic (thin-runtime-genericity
/// §1's "never hand-count" applied to register enums).
const A64_REGS: [RegisterARM64; 31] = [
    RegisterARM64::X0, RegisterARM64::X1, RegisterARM64::X2, RegisterARM64::X3,
    RegisterARM64::X4, RegisterARM64::X5, RegisterARM64::X6, RegisterARM64::X7,
    RegisterARM64::X8, RegisterARM64::X9, RegisterARM64::X10, RegisterARM64::X11,
    RegisterARM64::X12, RegisterARM64::X13, RegisterARM64::X14, RegisterARM64::X15,
    RegisterARM64::X16, RegisterARM64::X17, RegisterARM64::X18, RegisterARM64::X19,
    RegisterARM64::X20, RegisterARM64::X21, RegisterARM64::X22, RegisterARM64::X23,
    RegisterARM64::X24, RegisterARM64::X25, RegisterARM64::X26, RegisterARM64::X27,
    RegisterARM64::X28, RegisterARM64::X29, RegisterARM64::X30,
];

/// AArch64 oracle. Halt: the sentinel is seeded in X30 (LR) and the case
/// terminates with `RET`; `emu_start`'s `until` = sentinel. NZCV is a
/// single u32: N=31, Z=30, C=29, V=28.
fn run_oracle_a64(case: &FuzzCase) -> Result<ExecOutcome, String> {
    let mut uc = Unicorn::new(UArch::ARM64, Mode::LITTLE_ENDIAN).map_err(err_str)?;
    map_regions(&mut uc, case)?;
    for (i, reg) in A64_REGS.iter().enumerate() {
        uc.reg_write(*reg, case.regs.gprs[i]).map_err(err_str)?;
    }
    let sp = case.regs.sp;
    uc.reg_write(RegisterARM64::SP, sp).map_err(err_str)?;
    // Seed NZCV from the case — unicorn's ARM64 power-on default is
    // Z=1 (NZCV = 0x40000000), which would silently diverge from the
    // recompiled side's seeded flags.
    let nzcv = ((case.regs.sf as u32) << 31)
        | ((case.regs.zf as u32) << 30)
        | ((case.regs.cf as u32) << 29)
        | ((case.regs.of as u32) << 28);
    uc.reg_write(RegisterARM64::NZCV, nzcv as u64).map_err(err_str)?;
    let until = case.halt_sentinel();
    let exit = match uc.emu_start(case.entry_pc, until, 0, case.step_budget as usize) {
        Ok(()) => ExitKind::Completed,
        Err(e) => classify_start_err(e)?,
    };
    let mut regs = RegState::default();
    for (i, reg) in A64_REGS.iter().enumerate() {
        regs.gprs[i] = uc.reg_read(*reg).unwrap_or(0);
    }
    regs.rip = uc.reg_read(RegisterARM64::PC).unwrap_or(until);
    regs.sp = uc.reg_read(RegisterARM64::SP).unwrap_or(sp);
    let nzcv = uc.reg_read(RegisterARM64::NZCV).unwrap_or(0) as u64;
    regs.sf = nzcv & (1 << 31) != 0; // N
    regs.zf = nzcv & (1 << 30) != 0; // Z
    regs.cf = nzcv & (1 << 29) != 0; // C
    regs.of = nzcv & (1 << 28) != 0; // V
    let (data, stack) = snapshot_mem(&uc, case);
    Ok(ExecOutcome { regs, data, stack, exit })
}

/// X0–X31 are contiguous (1..=32) but keep an explicit table — same rule
/// as `A64_REGS`.
const RV64_REGS: [RegisterRISCV; 32] = [
    RegisterRISCV::X0, RegisterRISCV::X1, RegisterRISCV::X2, RegisterRISCV::X3,
    RegisterRISCV::X4, RegisterRISCV::X5, RegisterRISCV::X6, RegisterRISCV::X7,
    RegisterRISCV::X8, RegisterRISCV::X9, RegisterRISCV::X10, RegisterRISCV::X11,
    RegisterRISCV::X12, RegisterRISCV::X13, RegisterRISCV::X14, RegisterRISCV::X15,
    RegisterRISCV::X16, RegisterRISCV::X17, RegisterRISCV::X18, RegisterRISCV::X19,
    RegisterRISCV::X20, RegisterRISCV::X21, RegisterRISCV::X22, RegisterRISCV::X23,
    RegisterRISCV::X24, RegisterRISCV::X25, RegisterRISCV::X26, RegisterRISCV::X27,
    RegisterRISCV::X28, RegisterRISCV::X29, RegisterRISCV::X30, RegisterRISCV::X31,
];

/// RV64 oracle. Halt: the sentinel is seeded in x1 (ra) and the case
/// terminates with `jalr x0, x1, 0`; `emu_start`'s `until` = sentinel.
/// x0 reads as 0 (hardwired).
fn run_oracle_rv64(case: &FuzzCase) -> Result<ExecOutcome, String> {
    let mut uc = Unicorn::new(UArch::RISCV, Mode::RISCV64).map_err(err_str)?;
    map_regions(&mut uc, case)?;
    for (i, reg) in RV64_REGS.iter().enumerate() {
        uc.reg_write(*reg, case.regs.gprs[i]).map_err(err_str)?;
    }
    let until = case.halt_sentinel();
    let exit = match uc.emu_start(case.entry_pc, until, 0, case.step_budget as usize) {
        Ok(()) => ExitKind::Completed,
        Err(e) => classify_start_err(e)?,
    };
    let mut regs = RegState::default();
    for (i, reg) in RV64_REGS.iter().enumerate() {
        regs.gprs[i] = uc.reg_read(*reg).unwrap_or(0);
    }
    regs.gprs[0] = 0;
    regs.rip = uc.reg_read(RegisterRISCV::PC).unwrap_or(until);
    let (data, stack) = snapshot_mem(&uc, case);
    Ok(ExecOutcome { regs, data, stack, exit })
}
