//! Divergence / skip accounting (comparison-fuzzing-plan.md §5).
//!
//! Skips are counted per reason and reported alongside pass/fail counts —
//! they are never failures. Divergences are recorded as JSON artifacts under
//! `test-data/diff-fuzz/<arch>/` for minimization and regression replay.

use crate::case::{FuzzCase, RegState};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Why a case was skipped (per the plan's scope rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SkipReason {
    /// An unsupported instruction was actually executed (speet `unreachable` /
    /// oracle invalid opcode). Translation never gates — frontends overtranslate.
    UnsupportedExecuted,
    /// A trap fired under either engine.
    Trapped,
    /// Store to a read-only page (oracle fault / speet `unreachable`).
    StoreToReadOnly,
    /// Oracle faulted for a non-scope reason.
    OracleFault,
    /// Instruction budget exhausted (runaway loop).
    StepBudgetExceeded,
}

/// Aggregate counters for one fuzz run.
#[derive(Debug, Default, Clone, Serialize)]
pub struct RunStats {
    pub passed: u64,
    pub divergences: u64,
    pub skip_unsupported_executed: u64,
    pub skip_trapped: u64,
    pub skip_store_ro: u64,
    pub skip_oracle_fault: u64,
    pub skip_budget: u64,
}

impl RunStats {
    pub fn record_skip(&mut self, reason: SkipReason) {
        match reason {
            SkipReason::UnsupportedExecuted => self.skip_unsupported_executed += 1,
            SkipReason::Trapped => self.skip_trapped += 1,
            SkipReason::StoreToReadOnly => self.skip_store_ro += 1,
            SkipReason::OracleFault => self.skip_oracle_fault += 1,
            SkipReason::StepBudgetExceeded => self.skip_budget += 1,
        }
    }

    pub fn total_skipped(&self) -> u64 {
        self.skip_unsupported_executed
            + self.skip_trapped
            + self.skip_store_ro
            + self.skip_oracle_fault
            + self.skip_budget
    }

    pub fn total(&self) -> u64 {
        self.passed + self.divergences + self.total_skipped()
    }
}

/// A recorded divergence: the full case + both outcomes' key state.
#[derive(Debug, Clone, Serialize)]
pub struct DivergenceReport {
    /// Human-readable divergence facets (compare_outcomes' description).
    pub description: String,
    pub case: CaseRecord,
    pub oracle: OutcomeRecord,
    pub recompiled: OutcomeRecord,
}

/// Serializable projection of a `FuzzCase`.
#[derive(Debug, Clone, Serialize)]
pub struct CaseRecord {
    pub code_hex: String,
    pub entry_pc: u64,
    pub regs: RegState,
    pub data_hex: String,
    pub data_base: u64,
    pub stack_hex: String,
    pub stack_base: u64,
    pub read_only: Vec<(u64, u64)>,
    pub step_budget: u64,
}

impl CaseRecord {
    pub fn of(case: &FuzzCase) -> Self {
        Self {
            code_hex: hex(&case.code),
            entry_pc: case.entry_pc,
            regs: case.regs,
            data_hex: hex(&case.data),
            data_base: case.data_base,
            stack_hex: hex(&case.stack),
            stack_base: case.stack_base,
            read_only: case.read_only.clone(),
            step_budget: case.step_budget,
        }
    }
}

/// Serializable projection of an `ExecOutcome`.
#[derive(Debug, Clone, Serialize)]
pub struct OutcomeRecord {
    pub regs: RegState,
    pub exit: String,
    pub data_changes: Vec<(u64, String, String)>,
    pub stack_changes: Vec<(u64, String, String)>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn dir_for(arch: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/diff-fuzz")
        .join(arch)
}

/// Write a divergence artifact; returns the artifact path.
pub fn record_divergence(arch: &str, label: &str, report: &DivergenceReport) -> PathBuf {
    let dir = dir_for(arch);
    std::fs::create_dir_all(&dir).expect("create divergence dir");
    let path = dir.join(format!("{label}.json"));
    let json = serde_json::to_string_pretty(report).expect("serialize divergence");
    std::fs::write(&path, json).expect("write divergence artifact");
    path
}

/// Append a skip to the run log (best-effort; logging failures don't fail runs).
pub fn record_skip(arch: &str, label: &str, reason: SkipReason) {
    let dir = dir_for(arch);
    std::fs::create_dir_all(&dir).ok();
    let line = format!("{label}\t{reason:?}\n");
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("skips.log"))
    {
        let _ = f.write_all(line.as_bytes());
    }
}
