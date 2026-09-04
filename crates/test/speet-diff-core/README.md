# speet-diff-core

Comparison (differential) fuzzing core for speet: random x86-64 cases are
executed under a Unicorn oracle and under speet's recompiled output (wasmi
Lane A), then compared bit-exactly. See `docs/comparison-fuzzing-plan.md`.

## Layout

- `src/case.rs` — case model (bytes, initial state, outcome, memory diffs)
- `src/generator.rs` — structure-aware x86-64 generator (frontend-vocabulary forms,
  bounded addressing, iced-verified encodings)
- `src/oracle.rs` — Unicorn oracle (RX code, RW data with RO page, step budget)
- `src/recompiled.rs` — production translate path (`X86Recompiler`, byte-granular
  slots), corpus-harness assembly (halt stub + unreachable instrumentation), wasmi run
- `src/compare.rs` — skip classification (never failures) + exact comparison
- `src/report.rs` — skip counters + JSON divergence artifacts under
  `test-data/diff-fuzz/x86_64/` (each artifact records its seed; cases replay
  deterministically via `generate_case(seed)`)

## Run

```
cargo run -p speet-diff-core --bin diff-fuzz -- 200          # random seeds
cargo run -p speet-diff-core --bin diff-fuzz -- 200 --seed 1 # reproducible base seed
```

Exit code 1 iff divergences were found.

## Diagnostic examples

- `dis <seed>` — disassemble the generated case
- `dumpfn <seed> [from_fn]` — dump translated WASM bodies
- `why <seed>` — recompiled outcome + translation-time unsupported list
  (`unsupported_for` is a coverage signal only — never a gate; plan §1)
- `one <seed>` — full comparison verdict for one seed
- `pf` — minimal flag-repro probes

## Findings so far (see plan §7)

- **Constant folding drops flag side effects**: yecta's const-fold optimizer
  folds ALU ops on known-constant operands and skips the flag computation —
  observable whenever a folded op's flags are live in the final compared state.
  Repro: `cargo run -p speet-diff-core --example pf` (`add 3+3`: oracle PF=1,
  recompiled PF=0).
- x86-64 frontend gaps surfaced (coverage signal): AND/OR/XOR `reg,[mem]`
  (ADD/SUB are translated), INC/DEC/NEG.
