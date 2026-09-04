//! Comparison-fuzzing driver: run N random cases through both engines and
//! report pass/skip/divergence counts (plan §3, §5, §6).
//!
//! Usage: `cargo run -p speet-diff-core --bin diff-fuzz -- [N] [--seed S]`
//! Divergences land in `test-data/diff-fuzz/x86_64/` as JSON artifacts with
//! the case seed, so every finding replays deterministically.

use speet_diff_core::generator::generate_case;
use speet_diff_core::report::{
    record_divergence, record_skip, CaseRecord, DivergenceReport, OutcomeRecord, RunStats,
};
use speet_diff_core::{compare_outcomes, run_oracle, run_recompiled, Comparison, ExecOutcome};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u64 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(100);
    let mut seed: Option<u64> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--seed" {
            seed = args.get(i + 1).and_then(|a| a.parse().ok());
            i += 2;
        } else {
            i += 1;
        }
    }
    let base_seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    let mut stats = RunStats::default();
    for k in 0..n {
        let case_seed = base_seed.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let case = generate_case(case_seed);
        let label = format!("seed-{case_seed:016x}");

        // Oracle side.
        let oracle: Result<ExecOutcome, String> = run_oracle(&case);

        // Recompiled side.
        let recompiled: Result<ExecOutcome, speet_diff_core::RecompiledError> =
            run_recompiled(&case);

        match compare_outcomes(&case, oracle.as_ref().ok(), recompiled.as_ref()) {
            Comparison::Match => stats.passed += 1,
            Comparison::Skip(reason) => {
                stats.record_skip(reason);
                record_skip("x86_64", &label, reason);
            }
            Comparison::Divergence(desc) => {
                stats.divergences += 1;
                // Minimize first (plan §6.2): the artifact carries the
                // minimized case — the smallest case we know still diverges.
                let minimized = speet_diff_core::minimize(&case);
                let report = DivergenceReport {
                    description: desc.clone(),
                    case: CaseRecord::of(&minimized),
                    oracle: OutcomeRecord {
                        regs: oracle.as_ref().map(|o| o.regs).unwrap_or_default(),
                        exit: format!("{:?}", oracle.as_ref().map(|o| o.exit)),
                        data_changes: vec![],
                        stack_changes: vec![],
                    },
                    recompiled: OutcomeRecord {
                        regs: recompiled.as_ref().map(|o| o.regs).unwrap_or_default(),
                        exit: match &recompiled {
                            Ok(o) => format!("{:?}", o.exit),
                            Err(e) => format!("{e:?}"),
                        },
                        data_changes: vec![],
                        stack_changes: vec![],
                    },
                };
                // The divergence description is the key evidence — embed it.
                let path = record_divergence("x86_64", &label, &report);
                eprintln!(
                    "DIVERGENCE {label}: {desc}\n  minimized: {} bytes (was {})\n  artifact: {}",
                    minimized.code.len(),
                    case.code.len(),
                    path.display()
                );
            }
        }
    }

    println!(
        "x86_64 diff-fuzz: {} cases | {} pass | {} divergences | {} skipped \
         (unsupported-exec {} / trap {} / ro-store {} / oracle-fault {} / budget {})",
        stats.total(),
        stats.passed,
        stats.divergences,
        stats.total_skipped(),
        stats.skip_unsupported_executed,
        stats.skip_trapped,
        stats.skip_store_ro,
        stats.skip_oracle_fault,
        stats.skip_budget,
    );
    if stats.divergences > 0 {
        std::process::exit(1);
    }
}
