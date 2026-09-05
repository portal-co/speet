//! Fuzz case model: instruction bytes + initial state + observed outcome.

/// Target architecture of a case. Each arch fixes the register-file shape
/// (register count, modeled flags) and the halt convention — see [`Arch`]'s
/// methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Arch {
    /// x86-64: 16 GPRs (RAX..R15), 5 flags (ZF SF CF OF PF), stack-based
    /// `ret` (halt sentinel read from `[SP]`).
    X86_64,
    /// AArch64: 31 GPRs (X0–X30, X30 = LR), NZCV flags, link-register
    /// `ret` (halt sentinel seeded directly into X30), dedicated SP.
    AArch64,
    /// RV64: 32 GPRs (X0–X31, X0 hardwired zero, X1 = RA, X2 = SP),
    /// no modeled flags, link-register `ret`.
    RiscV64,
    /// AArch32 (ARMv7 A32): 16 GPRs (R0–R15, R13 = SP, R14 = LR,
    /// R15 = PC), CPSR flags, link-register `ret`. Word-aligned (4).
    Arm,
    /// i686: 8 GPRs (EAX..EDI), 5 flags (ZF SF CF OF PF), stack-based
    /// `ret`, byte-granular slots.
    X86_32,
    /// RV32: 32 GPRs (i32), no modeled flags, link-register `ret`.
    RiscV32,
    /// MIPS32 (big-endian words): 32 GPRs + HI/LO, no modeled flags,
    /// register conventions GPR-side ($31 = ra, $29 = sp).
    Mips,
}

impl Arch {
    /// Number of architectural GPRs the case's register file models.
    pub fn n_gprs(self) -> usize {
        match self {
            Arch::X86_64 => 16,
            Arch::X86_32 => 8,
            Arch::Arm => 16,
            Arch::AArch64 => 31,
            Arch::RiscV64 | Arch::RiscV32 => 32,
            Arch::Mips => 32,
        }
    }

    /// Modeled flag names in `RegState`'s (zf, sf, cf, of, pf) slot order —
    /// the mapping each recompiler uses for its own flag bits. RISC-V has
    /// no modeled flags.
    pub fn flag_names(self) -> &'static [(&'static str, usize)] {
        match self {
            // x86: (name, RegState field index) — zf sf cf of pf.
            Arch::X86_64 | Arch::X86_32 => {
                &[("ZF", 0), ("SF", 1), ("CF", 2), ("OF", 3), ("PF", 4)]
            }
            // AArch64 NZCV — N→sf, Z→zf, C→cf, V→of. PF unused.
            Arch::AArch64 => &[("Z", 0), ("N", 1), ("C", 2), ("V", 3)],
            // ARM CPSR — same NZCV mapping.
            Arch::Arm => &[("Z", 0), ("N", 1), ("C", 2), ("V", 3)],
            Arch::RiscV64 | Arch::RiscV32 | Arch::Mips => &[],
        }
    }

    /// Whether a final `ret` pops the halt sentinel from the stack
    /// (`true`, x86-64) or reads it from the link register (`false`,
    /// AArch64 / RV64).
    pub fn stack_based_ret(self) -> bool {
        matches!(self, Arch::X86_64 | Arch::X86_32)
    }

    /// Instruction alignment. Generators must keep `code.len()` a multiple
    /// of it and the entry PC aligned.
    pub fn code_align(self) -> u64 {
        match self {
            Arch::X86_64 | Arch::X86_32 => 1,
            Arch::Arm | Arch::AArch64 | Arch::RiscV64 | Arch::RiscV32 | Arch::Mips => 4,
        }
    }

    /// True for archs whose linear memory is i32-addressed (32-bit guest
    /// pointers) — affects oracle register types and wasmi param types.
    pub fn is_i32_addr(self) -> bool {
        matches!(self, Arch::Arm | Arch::X86_32 | Arch::RiscV32 | Arch::Mips)
    }
}

/// Initial / final architectural register state.
///
/// `gprs` is sized for the widest arch (RV64's 32); only `[0, n_gprs())`
/// are meaningfully set/compared (x86 uses 0–15 RAX..R15, AArch64 0–30
/// X0–X30 with X30 = LR, RV64 0–31 with X0 hardwired zero). Flags are the
/// recompilers' modeled slots, mapped per arch by [`Arch::flag_names`]
/// (AArch64 NZCV: N→sf, Z→zf, C→cf, V→of).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct RegState {
    pub gprs: [u64; 32],
    pub rip: u64,
    /// Zero flag (x86 ZF / AArch64 Z).
    pub zf: bool,
    /// Sign flag (x86 SF / AArch64 N).
    pub sf: bool,
    /// Carry flag (x86 CF / AArch64 C).
    pub cf: bool,
    /// Overflow flag (x86 OF / AArch64 V).
    pub of: bool,
    /// Parity flag (x86 only).
    pub pf: bool,
    /// AArch64's dedicated SP (x86-64 and RV64 keep SP in `gprs`).
    pub sp: u64,
}

impl RegState {
    /// Flag values in the (name, index) order [`Arch::flag_names`] returns.
    pub fn flags(&self) -> [bool; 5] {
        [self.zf, self.sf, self.cf, self.of, self.pf]
    }
}

/// How an execution ended. Only `Completed` cases are compared; every other
/// kind maps to a skip (see `compare::SkipReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitKind {
    /// Ran to the end of the generated instruction sequence (or hit the
    /// halt-stub via an indirect return past the translated set — the
    /// "clean exit at sequence end" case).
    Completed,
    /// Executed an instruction the frontend couldn't translate (speet:
    /// `unreachable`), or an invalid opcode under the oracle.
    UnsupportedExecuted,
    /// A trap fired (guest exception, halt, …).
    Trapped,
    /// Store to a read-only page.
    StoreToReadOnly,
    /// Instruction-count budget exhausted (runaway loop) — discarded.
    StepBudgetExceeded,
}

/// Byte-range diff of a memory region after a run.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryDiff {
    /// (offset_within_region, old_bytes, new_bytes)
    pub changes: Vec<(u64, Vec<u8>, Vec<u8>)>,
}

impl MemoryDiff {
    /// Diff `before`/`after` snapshots of a region. Coalesces contiguous
    /// changed bytes into single runs so a 1000-case log isn't a million
    /// one-byte entries.
    pub fn compute(before: &[u8], after: &[u8]) -> Self {
        debug_assert_eq!(before.len(), after.len());
        let mut changes = Vec::new();
        let mut i = 0;
        while i < before.len() {
            if before[i] != after[i] {
                let start = i;
                while i < before.len() && before[i] != after[i] {
                    i += 1;
                }
                changes.push((
                    start as u64,
                    before[start..i].to_vec(),
                    after[start..i].to_vec(),
                ));
            } else {
                i += 1;
            }
        }
        Self { changes }
    }
}

/// One fuzz case: bytes + initial machine state.
#[derive(Debug, Clone)]
pub struct FuzzCase {
    /// Target architecture — fixes the register-file shape, flag mapping,
    /// halt convention, and which generator/runner pair replays `seed`.
    pub arch: Arch,
    /// Instruction bytes placed at `entry_pc` in the RX code region.
    pub code: Vec<u8>,
    pub entry_pc: u64,
    /// Initial register file.
    pub regs: RegState,
    /// Initial contents of the RW data region.
    pub data: Vec<u8>,
    /// Address of the data region.
    pub data_base: u64,
    /// Initial contents of the RW stack region (region-relative).
    pub stack: Vec<u8>,
    pub stack_base: u64,
    /// Pages (region-relative byte offsets + length) mapped read-only in
    /// both engines. A store here faults the oracle and `unreachable`s
    /// speet — both → skip.
    pub read_only: Vec<(u64, u64)>,
    /// Instruction-count budget for both engines.
    pub step_budget: u64,
    /// Generator seed — recorded in every artifact so any case replays
    /// deterministically via `generate_case(arch, seed)`.
    pub seed: u64,
}

/// Terminal state of one execution.
#[derive(Debug, Clone)]
pub struct ExecOutcome {
    pub regs: RegState,
    /// Data-region contents after the run.
    pub data: Vec<u8>,
    /// Stack-region contents after the run.
    pub stack: Vec<u8>,
    pub exit: ExitKind,
}

impl ExecOutcome {
    /// Memory diff vs the case's initial state.
    pub fn memory_diff(&self, case: &FuzzCase) -> (MemoryDiff, MemoryDiff) {
        (
            MemoryDiff::compute(&case.data, &self.data),
            MemoryDiff::compute(&case.stack, &self.stack),
        )
    }
}

impl FuzzCase {
    /// Initial stack pointer. Register-file archs (x86-64, RV64) keep SP in
    /// `gprs`; AArch64 seeds it as its dedicated param (see
    /// `AArch64Recompiler::SP_PARAM_INDEX`). Top of the stack region,
    /// 16-byte aligned; stack-based-ret archs reserve 8 bytes below for the
    /// halt sentinel.
    pub fn initial_sp(&self) -> u64 {
        let top = self.stack_base + ((self.stack.len() as u64) & !0xF);
        if self.arch.stack_based_ret() { top - 8 } else { top }
    }

    /// Seed the halt sentinel (guest address one past the last code byte):
    /// at `[SP]` for stack-based-ret archs (x86 convention: at entry,
    /// `[rsp]` holds the return address), or returned for link-register
    /// archs (seeded directly into X30 / X1 by the runner).
    pub fn halt_sentinel(&self) -> u64 {
        self.entry_pc + self.code.len() as u64
    }
}
