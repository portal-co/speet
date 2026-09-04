# Comparison (Differential) Fuzzing Plan — speet vs Unicorn

**Status:** plan / not started
**Areas touched:** `crates/test/` (new fuzz crates), all architecture frontends (`crates/native/speet-*`), `speet-wasm` / `speet-interp` (execution lanes)
**Related:** [goals/arch.md](../goals/arch.md), [TESTING.md](../TESTING.md), [docs/guides/arch-recompilers.md](guides/arch-recompilers.md), [docs/guides/recompiler-debug-mcp.md](guides/recompiler-debug-mcp.md)

---

## 1. Goal

Build **comparison fuzzing** harnesses that prove the recompiled output implements the
same semantics as the emulated ISA. Each fuzz harness:

1. Generates **random instructions** (plus a random-but-structured initial machine state).
2. Executes the same instruction sequence under **Unicorn** (the oracle emulator).
3. Executes the same sequence under **speet** (instructions translated by the arch
   recompiler, then run in the recompiled form).
4. **Compares the final architectural state** — registers, PC, and written memory.

A divergence between recompiled output and Unicorn on a *supported, in-scope* input is a
functional recompiler bug and must be captured as a minimized reproducible artifact.

### Scope rules (harness filters)

These filters keep the comparison meaningful and avoid fighting known non-goals:

- **Memory is mapped RX** (readable + executable). The fuzz memory model is a flat
  code+data region with RX pages; no self-modifying-code, no page-table simulation.
- **Ignore unsupported instructions at execution, not translation.** The frontends
  **overtranslate**: translation does not reject unsupported instructions, so the fuzzer
  does **not** consult `unsupported_insns` at all. An unsupported instruction is skipped
  only if the case actually **executes** one — it surfaces as an `unreachable` at runtime,
  which counts as a skip. (Per
  [thin-runtime-genericity.md](guides/thin-runtime-genericity.md),
  `unsupported_insns` remains a debugging/coverage signal only.)
- **Ignore traps.** If execution reaches a speet trap — including the `unreachable` from
  an executed unsupported instruction or a guest exception — the case is skipped. The
  fuzzer compares **clean runs to completion of the generated sequence only**; the
  trap/ROP machinery ([speet-traps.md](guides/speet-traps.md)) is deliberately out of
  scope here.
- **Ignore stores to read-only memory.** A store targeting a page mapped read-only in
  the oracle faults the oracle run; in speet it traps to `unreachable`. Both outcomes are
  **counted as a skip for now** (a future phase may instead model RO stores as no-ops on
  both sides to keep such cases comparable).

Everything else — register values, flags, memory contents, final PC — must match exactly.

### What this buys us

End-to-end corpus tests (see `crates/test/speet-program-equivalence`) compare whole
programs and only catch bugs on paths real programs exercise. Comparison fuzzing probes
**arbitrary instruction encodings and arbitrary state**, including encodings real
compilers never emit (which is where decoder/translator gaps hide — see the
[asm-arch-instruction-sync.md](guides/asm-arch-instruction-sync.md) invariant).

---

## 2. Existing building blocks

| Piece | Location | Role in the fuzzer |
|---|---|---|
| Arch frontends | `crates/native/speet-{x86_64,aarch64,riscv,mips,arm,x86}` | translate generated instruction bytes into slots |
| WASM execution lane | `crates/managed/speet-wasm` (`FuncSchedule` + `IndexOffsets` flow) | run translated WASM (wasmi) — primary comparison lane |
| Interpreter lane | `crates/helper/speet-interp` | optional fast lane for triage before WASM execution |
| Runner/executor plumbing | `crates/test/speet-guest-runner` | pattern for runner abstraction; fuzz oracle gets its own executor |
| Outcome comparison | `crates/test/speet-program-equivalence` | pattern for "same program, same outcome" asserts |
| Debug MCP | `speet-rtd` mcp ([guide](guides/recompiler-debug-mcp.md)) | diagnose failing artifacts without restarting speet-rtd |
| **Missing** | — | Unicorn bindings (`unicorn-engine` crate — add as dev-dependency only), instruction/state generators, comparison core |

Unicorn is a **test-only dependency**: it must never appear in the dependency tree of
any shipped crate. Fuzz crates live under `crates/test/` and gate Unicorn behind a
feature so `cargo test` without the feature still builds everything else.

---

## 3. Architecture

Two new crates under `crates/test/`:

```text
crates/test/speet-diff-core        # shared: generator, oracle, recompiled runner, comparator
crates/test/speet-diff-fuzz        # harnesses: proptest + cargo-fuzz targets, per arch
```

### 3.1 `speet-diff-core`

```rust
/// One fuzz case: bytes + initial state.
pub struct FuzzCase {
    pub arch: GuestArch,
    /// Instruction bytes placed at `entry_pc` in the RX region.
    pub code: Vec<u8>,
    pub entry_pc: u64,
    /// Initial register file (x0..n / pc; flags modeled per-arch).
    pub regs: RegState,
    /// Initial contents of the data region.
    pub data: Vec<u8>,
    /// Stack contents + initial SP.
    pub stack: Vec<u8>,
    pub sp: u64,
}

/// Terminal state of one execution.
pub struct ExecOutcome {
    pub regs: RegState,
    pub pc: u64,
    /// Page-diff of data/stack regions after the run.
    pub mem: MemoryDiff,
    /// Clean end vs trap/fault/timeout — used by the scope filters, not compared.
    pub exit: ExitKind,
}
```

Modules:

- **`generate`** — instruction generator (§4).
- **`oracle`** — Unicorn executor. Maps one RX code region, one RW data region, one RW
  stack; installs hooks to snapshot final state; maps selected data pages read-only so
  RO-store cases fault the oracle (→ skip).
- **`recompiled`** — speet executor. Translates the case's code with the arch frontend
  (overtranslation: unsupported instructions are translated through, with no
  `unsupported_insns` gate), links it via the normal `FuncSchedule`/`IndexOffsets` flow,
  runs it in the WASM lane (wasmi), and snapshots the same `ExecOutcome`. A store to a
  read-only page traps to `unreachable` (→ skip). Uses the standard translate API with
  its `ctx: &mut Context` parameter — no special-cased test path.
- **`compare`** — the scope-filter logic (§5) and state equality.
- **`minimize`** — artifact reduction: shrink the instruction sequence and random state
  while the divergence persists (delta-debugging over case length first, then over
  register/data entropy).

### 3.2 Harnesses (`speet-diff-fuzz`)

Both harnesses share `speet-diff-core`; they differ only in driver:

- **`proptest`** (`cargo test -p speet-diff-fuzz`): property strategies generate
  `FuzzCase` values; a shrinking `TestCaseError::Fail` on divergence gives readable
  minimized cases in CI. Good default for CI because it needs no corpus and shrinks
  well.
- **`libfuzzer`** (`cargo fuzz run speet-diff-fuzz/<arch>`): the same generator wrapped
  in `Arbitrary`, with a checked-in **seed corpus** of hand-written interesting cases
  (each instruction family from the sync matrix, flag-crossing sequences, all branch
  forms, unaligned/odd offsets). libfuzzer's coverage guidance + corpus persistence is
  better for long-running local campaigns; crashes land in `crash-<arch>/` as artifacts
  that `speet-diff-core::minimize` further reduces.

A generator built on `Arbitrary` + a hand-written structure-aware strategy (not raw
bytes — see §4) serves both drivers.

---

## 4. Instruction generator

Raw random bytes are almost useless — most decodes as unsupported or traps. The
generator is **structure-aware and oracle-filtered**:

1. **Instruction vocabulary per arch**: an explicit list of encodable instruction
   *forms* (opcode + operand patterns), generated from the decode tables the frontend
   already uses, weighted toward arithmetic/load-store/branch families that the
   frontend claims to support. For x86-64 include encodings the backend can emit per
   the asm-arch sync matrix, including prefix variants.
2. **Valid operands only**: register numbers within the modeled register file; memory
   operands resolved to addresses inside the mapped data/stack regions at generation
   time (displacement-bounded) *or* left indirect and later filtered (below).
3. **Bounded basic blocks**: sequences are split into short blocks terminated by a
   branch whose target is another generated block (never outside the code region).
   No backward edges beyond the case window; no self-loops that never terminate
   (step budget, below).
4. **Step budget**: both engines run with an instruction-count cap. A case that hits
   the cap under either engine is discarded (loops), not compared.

Because the vocabulary is derived from what the frontend *claims* to support, most
generated cases avoid executed-unsupported skips; the skip counters per reason (§5)
report how much of the generated space is actually comparable.

**Per-arch state model:** registers mapped to the frontend's canonical model (e.g.
x86-64 GPRs + RFLAGS as bits; AArch64 X0–30 + NZCV). FP is in scope for scalar only
initially (x86-64 XMM lanes are scalar-modeled today — see
[arch-recompilers.md](guides/arch-recompilers.md) §6); packed SIMD cases are generated
but executed-unsupported cases land in the skip bucket until the frontend models them.

**Do not force flag materialization.** For x86, the translator's partial-flag laziness
is left exactly as-is: no flag-materializing instructions are inserted after arithmetic
ops. The comparison at instruction boundaries must hold regardless — if flag laziness
is not observable at boundaries, that's a real divergence and the fuzzer should find it,
not mask it.

---

## 5. Comparison semantics

```
run oracle(case)        -> Option<ExecOutcome>   // None = oracle faulted (incl. RO store,
                                                 // fault, timeout) -> skip
run recompiled(case)    -> Option<ExecOutcome>   // None = `unreachable` at runtime
                                                 // (executed unsupported instruction,
                                                 // trap, RO store) -> skip
if exit kinds differ (and neither is filtered)     -> DIVERGENCE
else compare regs, pc, mem diff                     -> DIVERGENCE on any mismatch
```

- **Exact equality** for integer registers, flags, and memory bytes. No tolerance.
- **PC equality** means "same next slot", i.e. the same instruction boundary — compare
  against the mapped guest PC, not internal slot indices.
- **Skip accounting:** skips are counted per reason (executed-unsupported, trap,
  RO-store, oracle fault, step-budget) and reported alongside pass/fail counts, but are
  never failures. A case where **translation** of an unsupported instruction occurs but
  the instruction is never *executed* is fully in scope and compared normally —
  overtranslation means dead unsupported code must not corrupt the translated output.
- **Floating point:** bit-exact comparison including NaN payloads where the frontend
  models exact bits (x86-64 scalar XMM does — i64 round-trip). Where an arch's FP
  rounding is known-different (flush-to-zero, FMA contraction), record it in a per-arch
  "known divergences" table that the comparator consults; anything not in the table and
  not bit-exact is a bug. The table starts empty; entries require a written justification.
- **Divergence report:** minimized `FuzzCase` serialized as JSON under
  `test-data/diff-fuzz/<arch>/`, with oracle trace and speet trace attached, ready to
  feed into the [recompiler-debug MCP](guides/recompiler-debug-mcp.md) loop.
- **Regression corpus:** every confirmed-and-fixed divergence is added to
  `crates/test/speet-diff-fuzz/tests/regression/` as a plain `#[test]` replaying the
  minimized case, so fixed bugs can never silently regress.

---

## 6. Arch rollout order

| Phase | Arch | Notes |
|---|---|---|
| 1 | x86-64 | Largest sync matrix, scalar SSE in place; highest expected bug yield |
| 2 | aarch64 | Fixed 4-byte slots simplify generation; NZCV modeling first |
| 3 | riscv (RV64, then RV32) | Weak-memory ordering: run single-hart only, `MemOrder::Relaxed` observable behavior — do not fuzz multi-hart interleavings |
| 4 | mips | Delay-slot semantics: generator must model `pc+8` branch targets and the separate delay-slot decode slot ([arch-recompilers.md](guides/arch-recompilers.md) §5) |
| 5 | arm, x86 (thin 32-bit set) | After main-four soak |

DEX (`speet-dex`) is managed bytecode and out of scope for Unicorn comparison; it would
need a different oracle (Art/d8 pipeline) — separate plan if pursued.

---

## 7. Invariants to respect (from the component guides)

These are the mistakes the guides exist to prevent; the fuzzer must not reintroduce them:

- Do not create a "one function = one instruction" fast path. Slots exist at every
  possible alignment ([yecta.md](guides/yecta.md) §1a) — the generator deliberately
  produces unaligned/overlapping encodings (RVC, x86 prefixes) and expects them to
  decode via the real `PcSlotMap` path.
- Do not translate through a simplified single-loop translator for fuzzing; the
  comparison must exercise the production `return_call`-per-edge structure.
- Do not bypass the normal link path (`FuncSchedule` two-pass,
  [linker.md](guides/linker.md)); fuzz cases link like any other artifact.
- Do not gate translation on `unsupported_insns`. The frontends overtranslate; the
  fuzzer relies on this — unsupported instructions translate through and only fail (→
  `unreachable` → skip) if actually executed. Dead unsupported code must translate to
  harmless slots.
- A fuzz case ending in an indirect return should hit the halt-stub path and surface the
  register file ([thin-runtime-genericity.md](guides/thin-runtime-genericity.md)), which
  the comparator treats as "clean exit at sequence end", not an error.
- Unicorn is dev-dependency-only and feature-gated; nothing in `crates/` outside
  `crates/test/` may depend on it.

---

## 8. Milestones

1. **M1 — x86-64 proptest, integer only.** Generator (ALU + mov + jcc + a few loads/
   stores), Unicorn oracle, WASM lane, comparator, divergence JSON. Exit: 10k cases/run
   green on CI, no unfixed divergences.
2. **M2 — minimization + regression corpus.** `minimize` module, replay tests, artifact
   format frozen. Exit: a hand-seeded "known bug" (temporarily introduced) is minimized
   and replayed correctly.
3. **M3 — cargo-fuzz targets + seed corpus.** libfuzzer drivers per arch, seed corpus
   checked in, skip-reason counters surfaced in CI output; `unsupported_insns` reporting
   stays a separate coverage signal (never a gate).
4. **M4 — aarch64 + riscv** (single-hart), FP bit-exact comparison where modeled.
5. **M5 — mips + 32-bit set.**
6. **M6 — CI integration.** proptest suites run per-PR; a nightly libfuzzer soak job
   uploads any crash artifacts.

## 9. Resolved decisions

Former open questions, with the decisions made:

- **Unicorn version pinning — pin to latest.** Track the latest `unicorn-engine`
  release in the workspace lockfile and bump on upgrade rather than freezing an old
  pin. A suspected oracle bug after an upgrade gets a second opinion from qemu-user via
  `speet-guest-runner` before being dismissed as a speet divergence.
- **RO stores — trap on RO store and `unreachable`; count as a skip (for now).** No
  RO-write no-op masking in the memory model: a store to a read-only page traps to
  `unreachable` in speet and faults the oracle; both count as a skip. A future phase may
  model RO stores as no-ops on both sides to make such cases comparable.
- **Unsupported instructions — execution-time skip only; do not check
  `unsupported_insns`.** The frontends overtranslate, so translation never rejects an
  instruction and the fuzzer must not gate on `unsupported_insns` (it stays a
  debugging/coverage signal). A case skips only if an unsupported instruction is
  actually executed, surfacing as `unreachable` at runtime. Dead unsupported code
  translating cleanly is in scope and compared normally.
