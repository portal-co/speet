//! Fuzz case model: instruction bytes + initial state + observed outcome.

/// Initial / final architectural register state (x86-64 integer subset).
///
/// GPR order follows the x86-64 numbering used throughout speet
/// (`speet-x86_64::X86Recompiler` resolve order): RAX..R15, indices 0–15.
/// Flags are the five the recompiler models: ZF, SF, CF, OF, PF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub struct RegState {
    pub gprs: [u64; 16],
    pub rip: u64,
    /// Zero Flag.
    pub zf: bool,
    /// Sign Flag.
    pub sf: bool,
    /// Carry Flag.
    pub cf: bool,
    /// Overflow Flag.
    pub of: bool,
    /// Parity Flag.
    pub pf: bool,
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
    /// deterministically via `generate_case(seed)`.
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
    /// Initial stack pointer: top of the stack region, 16-byte aligned,
    /// 8 bytes reserved below for the halt-sentinel "return address" on
    /// stack-based-return archs (x86-64) — same convention as
    /// `speet_corpus_harness::run_corpus_module`.
    pub fn initial_sp(&self) -> u64 {
        self.stack_base + ((self.stack.len() as u64) & !0xF) - 8
    }

    /// Seed the halt sentinel (guest address one past the last code byte)
    /// 8 bytes below SP so a final `ret` lands past the translated set —
    /// the "clean exit at sequence end" case — instead of on garbage.
    pub fn halt_sentinel(&self) -> u64 {
        self.entry_pc + self.code.len() as u64
    }
}
