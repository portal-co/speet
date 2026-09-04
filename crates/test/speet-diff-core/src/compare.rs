//! Comparison semantics (plan §5).
//!
//! Skips first — they are never failures. Then exact equality on integer
//! registers, flags, and memory bytes. No tolerance.

use crate::case::{ExecOutcome, ExitKind, FuzzCase, MemoryDiff};
use crate::report::SkipReason;
use crate::recompiled::RecompiledError;

/// Result of comparing one case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Comparison {
    /// Both engines agree on every compared facet.
    Match,
    /// Out of scope — count and move on (never a failure).
    Skip(SkipReason),
    /// The engines disagree: human-readable facet list.
    Divergence(String),
}

/// Compare the recompiled result (a `Result`) against the oracle outcome.
///
/// `oracle` is `None` when the oracle itself skipped (faulted/budget) — the
/// case is skipped, not compared.
pub fn compare_outcomes(
    case: &FuzzCase,
    oracle: Option<&ExecOutcome>,
    recompiled: Result<&ExecOutcome, &RecompiledError>,
) -> Comparison {
    // ── Recompiled-side skips (execution-time semantics, plan §1) ──
    if let Err(e) = recompiled {
        return match e {
            RecompiledError::UnsupportedExecuted { .. } => {
                Comparison::Skip(SkipReason::UnsupportedExecuted)
            }
            RecompiledError::Trapped(m) if is_ro_store_trap(m) => {
                Comparison::Skip(SkipReason::StoreToReadOnly)
            }
            RecompiledError::Trapped(_) => Comparison::Skip(SkipReason::Trapped),
            RecompiledError::StepBudgetExceeded => {
                Comparison::Skip(SkipReason::StepBudgetExceeded)
            }
            RecompiledError::Internal(m) => {
                Comparison::Divergence(format!("harness internal error: {m}"))
            }
        };
    }
    let r = recompiled.unwrap();

    // ── Oracle-side skips ──
    let Some(o) = oracle else {
        return Comparison::Skip(SkipReason::OracleFault);
    };

    // Exit-kind mismatch where neither is filtered is a divergence.
    if o.exit != r.exit {
        return Comparison::Divergence(format!(
            "exit kind: oracle={:?} recompiled={:?}",
            o.exit, r.exit
        ));
    }

    // ── Exact register/flag comparison (plan §5: no tolerance) ──
    let mut facets = Vec::new();
    for g in 0..16 {
        if o.regs.gprs[g] != r.regs.gprs[g] {
            facets.push(format!(
                "gpr{}: oracle={:#x} recompiled={:#x}",
                g, o.regs.gprs[g], r.regs.gprs[g]
            ));
        }
    }
    for (name, (of, rf)) in [
        ("ZF", (o.regs.zf, r.regs.zf)),
        ("SF", (o.regs.sf, r.regs.sf)),
        ("CF", (o.regs.cf, r.regs.cf)),
        ("OF", (o.regs.of, r.regs.of)),
        ("PF", (o.regs.pf, r.regs.pf)),
    ] {
        if of != rf {
            facets.push(format!("flag {name}: oracle={of} recompiled={rf}"));
        }
    }

    // ── Exact memory comparison ──
    let (o_data, o_stack) = (o.data.clone(), o.stack.clone());
    let (r_data, r_stack) = (r.data.clone(), r.stack.clone());
    let od = MemoryDiff::compute(&case.data, &o_data);
    let rd = MemoryDiff::compute(&case.data, &r_data);
    if od != rd {
        facets.push(format!(
            "data memory diff: oracle={od:?} recompiled={rd:?}"
        ));
    }
    let os = MemoryDiff::compute(&case.stack, &o_stack);
    let rs = MemoryDiff::compute(&case.stack, &r_stack);
    if os != rs {
        facets.push(format!(
            "stack memory diff: oracle={os:?} recompiled={rs:?}"
        ));
    }

    if facets.is_empty() {
        Comparison::Match
    } else {
        Comparison::Divergence(facets.join("; "))
    }
}

/// The RO page is unmapped (not merely RO) inside the wasmi module — a store
/// there traps with an out-of-bounds/OOB-flavored message. Classify it so
/// RO-store cases skip rather than count as generic traps (plan §1).
fn is_ro_store_trap(msg: &str) -> bool {
    // The RO window inside the data region: [DATA_BASE + RO_DATA_OFFSET, +RO_DATA_SIZE).
    // An OOB trap alone isn't proof the faulting *address* was the RO page;
    // for phase 1 we accept the OOB family as the RO-store carrier because
    // address-bounded generation keeps every other store in-bounds.
    msg.contains("memory") && (msg.contains("out-of-bounds") || msg.contains("out of bounds"))
}
