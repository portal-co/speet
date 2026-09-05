# MIPS recompiler: delay slots, big-endian memory, store-path correctness

**Status: Planned** — supersedes the open MIPS findings in
[comparison-fuzzing-plan.md](comparison-fuzzing-plan.md) (findings #6, #7 and
the endianness sub-finding of #4).
**Crates affected:** `yecta`, `speet-mips`, `speet-ordering`,
`speet-linux-wasi`, `speet-diff-core`.
**Evidence base:** `speet-diff-core` differential fuzzing vs Unicorn
(`docs/comparison-fuzzing-plan.md`), fixed-seed runs (`--seed 42`), minimal
artifacts under `test-data/diff-fuzz/mips/`.

---

## 1. Problem statement

Three correctness gaps remain in `speet-mips`, all found by differential
fuzzing and all currently shipped as known divergences:

1. **Branch/jump delay slots are ignored.** Every MIPS control transfer
   (BEQ/BNE/BLEZ/BGTZ/BLTZ/BGEZ/J/JAL/JR/JALR) architecturally executes the
   instruction at `pc+4` *before* the transfer takes effect, on both the
   taken and not-taken paths. The recompiler emits the transfer alone, so
   any case whose delay slot has observable effects diverges from the
   oracle. This is the largest remaining MIPS divergence class.
2. **Data memory byte order.** WASM linear memory is little-endian. The raw
   (no-mapper) load/store path therefore reads/writes LE values for a
   big-endian guest: `sw $t3, 0xf78($t0)` writes `0x20` at byte `0xaf8`
   where the oracle writes `0x00 0x20 …` starting at `0xaf8` (BE). Loads
   mirror the error. Round-tripping programs (store then load the same
   address, no byte-level comparison) are unaffected, which is why the
   corpus e2e never caught this.
3. **Register clobbers around stores** (finding #7): minimal repros like
   `sw $t0, 0xaf8($t0)` + `jr $ra` diverge in *unrelated* registers
   (`gpr17: oracle=0x7 recompiled=0xffffffff`, `gpr1: oracle=0x0
   recompiled=0x40000000`, `gpr27`, `gpr6`, `gpr14`…). Root cause not yet
   identified; hypotheses in §5.

Items 1 and 2 have complete infrastructure already merged (commit
`16f54ea`): `emit_delay_slot`, `set_delay_slot_fetcher`,
`is_absorbed_delay_pc`, precomputed branch predicates, and
`big_endian_memory` swap emission — all **gated off** because the naive
enablement produces wrong code (§2.3, §4.3). This plan describes what must
change in `yecta` (generically, not MIPS-specifically) and in `speet-mips`
to turn those gates on.

Non-goals:

- MIPS16/microMIPS (no delay slots there — out of scope entirely).
- Branch-likely instructions (ISAv1 deprecated, removed in r6) — emit
  `unreachable` as today.
- Changing the BE instruction-fetch convention (all callers already decode
  words big-endian).

---

## 2. Delay slots via generic `yecta` retargeting

### 2.1 Why this belongs in `yecta`

The `Reactor`'s whole model is "one WASM function per possible instruction
slot, every edge a `return_call`, straight-line runs merged" (yecta.md
§1a–§1c). MIPS delay slots are not a MIPS quirk in this model — they are an
instance of a *general* requirement: **the edge a control transfer takes may
need to execute a bounded prefix of the architectural fall-through path
first.** Any ISA feature with that shape (delay slots; shadowed link
registers that must be spilled along a particular edge; hardware loops with
a "first iteration" prologue) wants the same lowering. So the mechanism
lands on `Reactor::ji`, parameterized, and `speet-mips` becomes its first
consumer.

### 2.2 Current lowering, precisely

Today a conditional branch at slot `pc` emits (via `ji` →
`emit_conditional_jump` → per-entry `emit_conditional_arm`):

```text
slot(pc) body:
  <condition snippet>          ;; BranchCondition: compares GPR locals
  if                           ;; BlockType::Empty
    <params>                   ;; emit_params_with_fixups: register file
    return_call target         ;; taken edge — SOLE_EXIT
  else
    …                          ;; nothing today
  end                          ;; closed at seal time (if_stmts)
  <merged continuation>        ;; slot(pc+4)'s body, inlined by merging
```

The lens-queue speculative fall-through edge (yecta.md §1c, `next_with`)
has already adopted slot(pc) as a predecessor of slot(pc+4) before the
branch is even decoded, so the *not-taken* arm merges slot(pc+4)'s body
inline and reaches slot(pc+8) by construction. Consequences:

- **Not-taken path: delay slot executes correctly** (it is the merged
  body), by accident of merging. This is why the fuzzer's *not-taken*
  cases mostly pass.
- **Taken path: delay slot never executes.** `return_call target` leaves
  the function before the merged continuation. This is the bug.
- **Predicate timing is also wrong today** if the delay slot writes a
  register the branch compares: the condition snippet is evaluated at the
  top of the body, *before* the merged continuation (delay slot) runs — so
  this one is accidentally correct too, but only because the condition is
  emitted first. The merged `emit_delay_slot`-inside-slot(pc) approach
  tried in `16f54ea` broke this ordering (condition was read after the
  inline delay body), which is why branch cases regressed when it was
  enabled; the precomputed-predicate half of that commit (evaluate the
  comparison into a temp before anything else, jump on the temp) is the
  correct invariant and stays.

### 2.3 The generic mechanism: taken-arm prefix ("retargeting")

Add one optional parameter to `Reactor::ji` / `ji_internal` (and plumb it
through `emit_conditional_jump` / `emit_conditional_arm` /
`emit_conditional_call`):

```rust
pub fn ji(
    …,
    condition: Option<&(dyn Snippet<Context, E> + '_)>,
    /// Emitted at the top of the *taken* arm, before the params/return_call.
    /// This is the generic hook for MIPS delay slots: the frontend passes a
    /// snippet that re-emits the delay instruction's translation, so the
    /// taken edge executes [delay body][transfer] exactly like the
    /// not-taken edge executes [delay body][fall-through].
    taken_prefix: Option<&(dyn Snippet<Context, E> + '_)>,
    tail_idx: usize,
) -> Result<(), E>
```

Emission shape in `emit_conditional_arm`:

```text
  <condition snippet>
  if
    <taken_prefix>             ;; NEW — delay-slot body
    <params + fixups>
    return_call target
  else
  end
  <merged continuation>
```

Design constraints, each tied to an existing invariant:

- **Re-run, don't capture.** The prefix is a `&dyn Snippet` (the same
  seam as `BranchCondition`/`ExpectedRaSnippet`/`TableIndexSnippet`)
  whose `emit_snippet` **re-runs the frontend's `emit_insn` on the delay
  instruction at emission time**, feeding every emitted op through the
  `go` sink yecta hands it. It must *not* pre-translate the body into a
  `Vec<wasm_encoder::Instruction>` at branch-translation time: the
  passed-in sink may carry ambient capabilities
  (`push_ambient_addr`/`call_ambient` — link-time data addresses, host
  call plumbing) that a frozen op stream silently drops, and a
  flattened second copy of the emission logic would drift from the tail
  path it duplicates. Re-running keeps one source of truth and lets the
  sink do whatever it does for inline runs.
- **`emit_insn` needs a sink parameter.** Today it feeds the current
  tail through `rctx`. Factor the feed target out:
  `emit_insn_to(ctx, rctx, sink, insn)` where `sink` is the same
  closure type snippet emission already uses
  (`&mut dyn FnMut(&mut Context, &Instruction<'_>) -> Result<(), E>`),
  with the existing `emit_insn` delegating to it with the current-tail
  feed; the gpr/memory/swap helpers route through the same parameter.
  The inline delay copy fires no `on_instruction` hook (unchanged
  decision). Where the snippet gets its `&MipsRecompiler` depends on
  the reactor-context split: if the snippet's `Context` can reach the
  frontend, it holds only the decoded delay word; otherwise box it with
  the same `'cb` lifetime as the existing `delay_slot_fetcher`.
- **Const folding sees the body — and that's sound.** Emitted through
  the real sink, the delay body flows through the Entry's optimizer
  exactly like any inline run (`feed_one`). This is correct because the
  absorbed slot means this is the *only* copy of the delay instruction —
  no duplicate slot(pc+4) exists to fold inconsistently with — and the
  optimizer already handles structured nesting from condition snippets
  and alias-check `If`/`End` frames. `flush_bundles` at `ji` entry
  still orders pending lazy stores before the arm (keep that ordering).
- **Alias checks stay live.** `emit_load`'s runtime alias check inside
  the delay body emits its `If`/`End` frames through the same sink —
  balanced frames, so the taken arm stays structurally valid. No
  special case. If the delay body's memory emission can route through
  ambient addresses (link-time data references in the linker path),
  yecta must hand snippets the ambient-capable sink surface
  (`Entry::instruction`/`as_ambient_sink`) rather than bare
  `feed_one` — unify with however condition snippets emit today.
- **`SOLE_EXIT` stays.** The taken arm's `return_call` is still the
  function's only exit edge to `target`; the Else arm merges the
  continuation. No `ExitId` proliferation is required for this feature —
  `ExitId` groundwork exists precisely so a *later* native-condition
  optimizer (yecta.md §1e) can distinguish edges; delay slots do not need
  it.
- **`max_ifs_per_fn` / `max_insts_per_fn` accounting.** The prefix raises
  the body size of branch slots. Because the body flows through the real
  sink, `inst_count` grows naturally and saturation splits still fire;
  `increment_if_stmts_for_predecessors` is unaffected (the prefix adds no
  `if` frames of its own beyond the alias-check frames, which are counted
  the same way they are in straight-line loads today).
- **Hoisted call regions.** The prefix must not open a call region
  (`close_call_region` runs before the `Else`). Delay bodies are never
  control transfers (rejected below), so they never open one. Enforce
  with a debug assert in `emit_conditional_arm`.
- **Static condition resolution.** `try_resolve_condition` may fold the
  branch to known-taken/known-not-taken per entry. Known-taken must still
  emit the prefix (it rides along in `emit_unconditional_jump_body` — add
  the same parameter there); known-not-taken drops the prefix along with
  the jump.

### 2.4 `speet-mips` integration

With the yecta hook in place, `translate_branch`/`J`/`JAL`/`JR`/`JALR` all
follow one recipe (replacing the disabled `emit_delay_slot` call sites from
`16f54ea`):

1. **Predicate first.** Evaluate the branch condition into a dedicated
   temp (already implemented: `temps_slot` local 0 via
   `PrecomputedCondition`). Architecturally the comparison uses
   pre-delay-slot register values.
2. **Fetch the delay word** through `set_delay_slot_fetcher` (callers hand
   the recompiler a `(pc) -> Option<rabbitizer::Instruction>` closure; both
   `speet-linux-wasi`'s `emit_mips_unit` and `speet-diff-core` have the
   word list in hand). Missing fetcher → record a `delay_slot_notes` entry
   and emit the branch without a slot (current behavior, coverage signal).
   Out-of-range slot → same.
3. **Reject control transfers in the slot.** A BRANCH/CALL-class delay
   instruction is UNPREDICTABLE per the ISA — emit the branch
   `unreachable`-terminated with a note (fuzzer treats it as a skip).
4. **Wrap the delay word in a re-run snippet**: set the PC local to
   `pc+4` (the architectural PC while the slot executes), then pass a
   snippet that calls `emit_insn_to` on the decoded delay word through
   the sink yecta provides, as `taken_prefix` to `ji`. No buffer, no
   second emission path — the tail slot and the taken arm share one
   `emit_insn`.
5. **Record `absorbed_delay_pcs.push(pc+4)`** so the sequential
   translation loop skips it (already implemented). Slot(pc+4) therefore
   never exists *unless* something else branches directly at it —
   branching into a delay slot is architecturally legal but
   pathological; `pc_to_func_idx` will resolve it to the wrong slot (the
   next created slot), so detect it: when translating any branch whose
   *target* is an absorbed PC, drop to unsupported with a note. The fuzz
   generators already never target delay words.
6. **JAL/JALR link register.** Already correct: `$ra` is written to
   `pc+8` (delay-slot-relative return) before the prefix runs; the prefix
   must not clobber `$ra` — reject delay instructions that write the link
   register of an enclosing JAL/JALR (note + unsupported). Writes to
   *other* registers are fine and are the whole point.

Speculative native-stack JAL/JALR (`ji_with_params`) needs the same prefix
parameter on `JumpCallParams` (or a sibling `ji_with_params_and_prefix`) so
the taken path of a speculative call also executes the slot first.

### 2.5 Why not the alternatives

- **Retarget the taken arm to slot(pc+4).** Wrong: slot(pc+4)'s merged
  body continues to slot(pc+8) — the taken edge would transfer to the
  delay slot's *continuation*, not to the branch target. Two different
  continuations from one slot is exactly what merging cannot express.
- **Split slot(pc+4) in two** (delay-only slot ending in
  `return_call target` for the taken edge + full slot for fall-through).
  Doubles slots on every branch path and duplicates the delay body —
  strictly worse than the prefix, and it breaks the "slot exists at every
  alignment for indirect branches" invariant (which of the two is
  `pc_to_func_idx(pc+4)`?).
- **Trap to the interpreter for branch instructions.** Correctness by
  giving up; defeats the reactor for the hottest guest construct (loop
  back-edges).
- **Pre-translate the delay body into a `Vec<Instruction>`** at branch
  translation time (an earlier draft of this plan). Two defects: the
  frozen op stream bypasses whatever the passed-in sink adds on top of
  plain instructions — ambient capabilities like `push_ambient_addr` /
  `call_ambient` (link-time data addresses, host-call plumbing) are
  silently dropped, so a body that emits cleanly on the tail path can
  mis-link on the taken arm; and it forks the emission logic into a
  flattened copy that must be kept in lockstep with `emit_insn`'s real
  path (folding, alias-check frames, hook behavior). Re-running the
  recompiler inside the snippet keeps a single source of truth.

### 2.6 Enablement order

1. yecta: `taken_prefix` parameter + tests (a synthetic two-slot reactor
   case asserting the taken arm's body order: prefix → params →
   return_call; plus saturation/`seal_for_split` interaction, and an
   ambient-capability test asserting `push_ambient_addr` ops emitted
   inside the prefix reach the entry's ambient sink rather than being
   dropped).
2. `speet-linux-wasi` `emit_mips_unit` installs the fetcher (first
   production consumer; WASI corpus provides broad-coverage execution).
3. `speet-diff-core` installs it; regenerate `test-data/diff-fuzz/mips/`
   artifacts (compare-semantics change ⇒ wipe per-arch artifacts, per the
   established rule); fixed-seed A/B against the gated build.
4. Only then remove the `delay_slot_notes`-only fallback path.

---

## 3. Big-endian data memory

### 3.1 Design (already merged, default-off)

`big_endian_memory: bool` on `MipsRecompiler` (default `false` = today's
LE behavior, byte-order-neutral for round-tripping guests). When set:

| insn | raw-path lowering (addr already widened per `use_memory64`) |
|------|--------------------------------------------------------------|
| LB/LBU/SB | unchanged (single byte) |
| LH | `I32Load16S`, swap low 16, re-sign-extend (`<<16 >>s 16`) |
| LHU | `I32Load16U`, swap16 |
| SH | swap16(value), `I32Store16` |
| LW | `I32Load`, swap32, (`I64ExtendI32U` under MIPS64) |
| SW | (`I32WrapI64` under MIPS64), swap32, `I32Store` |
| LD/SD | swap64 via `I64Load`/`I64Store` |

Swaps use dedicated per-function scratch locals (`swap_tmp_slot` i32,
`swap_tmp_i64_slot` i64, appended in `init_function` *after* the pool
slots) — deliberately **not** pool locals, which the lazy-store machinery
recycles. Unaligned LWL/LWR/SWL/SWR remain unsupported (emit
`unreachable`), unchanged.

The `memory_access` mapper path (`bind_memory_access`) is documented as
responsible for its own byte order; the WASM-runtime OS lane must be
audited separately before flipping any default there (AGENTS §10: one
redirect policy, no per-lane forks — but byte order is a *guest ABI*
property, so the flag is per-recompiler-instance, set by the embedder).

### 3.2 The swap lowering is correct but the module doesn't validate

With `big_endian_memory = true` the fuzzer produces modules that fail
`wasmparser::validate` with `type mismatch: expected i64, found i32`
(repro: 60 fixed-seed cases → 0 pass / 56 div, almost all harness-internal
validate errors). The swap sequences themselves check out by hand; the
failure is an interaction with the *lazy-store alias-check flush*:

- `emit_store` with `MemOrder::Relaxed` defers stores via `feed_lazy`,
  saving address/value into pool locals; a later load flushes them behind
  a runtime equality check. Those flush bodies read pool locals whose
  types were recorded at `feed_lazy` time.
- The swaps sit *between* the address computation and `emit_store`, on the
  value stack. Under `Strong` (the current `speet-diff-core`/link.rs
  default) no deferral happens — yet the validate failures still appear,
  so the suspect is the flush machinery *emitted for loads* whose saved
  locals were re-indexed when `swap_tmp_slot`/`swap_tmp_i64_slot` were
  appended to the per-function layout (init order in `init_function`
  changed relative to `declare_trap_locals`/`alloc_cell`).

**Debug plan (timebox: one session):**

1. Reproduce with a single `SW` + `JR` case (the
   `examples/mkmod.rs` + `mpdump.rs` pair from the session dumps the
   exact slot).
2. Diff the slot's op stream swaps-on vs swaps-off; locate the op where
   the i64/i32 expectation diverges. Confirm or refute the
   layout-reindexing hypothesis by moving the swap-local append *after*
   `declare_trap_locals`/`alloc_cell` in `init_function`.
3. If the flush hypothesis is wrong, walk `feed_lazy`/`flush_bundles_for_load`
   with `MemOrder::Relaxed` explicitly set — the swap must never be
   captured into a deferred bundle (deferred stores save the *value*
   local, and the swap's result lives on the stack, not in a local). If
   that's the break: either force eager emission for swapped stores
   (`emit_store` bypass under `big_endian_memory`), or route swapped
   values through `swap_tmp` and pass that local to `feed_lazy`.

### 3.3 Enablement order

1. §3.2 fix lands; `big_endian_memory = true` in `speet-diff-core`'s
   `translate_case_mips` (oracle is BE).
2. Fixed-seed sweep: the SW/LW "data memory diff" class must vanish;
   `ro-store` and `trap` skip classes unchanged.
3. `speet-linux-wasi` flips its MIPS lane to BE data memory (guest ABI
   decision, matches the BE instruction fetch all callers already use).
4. Regenerate `test-data/diff-fuzz/mips/` artifacts; proptest gate gets
   the new "endian" facet only if any class survives.

---

## 4. Register clobbers around stores (finding #7)

Minimal repros from the corpus (`ad0a02d8` `sw $t0, 0x2d8($t0)` diverging
in `$17`; fresh fixed-seed runs diverging in `$1`/`$6`/`$14`/`$27` with
values like `0x40000000`, `0x400`, `0xffffffff`) all store to the data
anchor with no other stateful instruction in the case. The store's raw
lowering touches no GPR destination, so the clobbered values must come
from one of:

1. **Trap-parameter mis-indexing**: `setup_traps` appends trap params
   after `BASE_PARAMS=36`; if any raw-path helper writes a GPR local
   computed from a *post-trap* index into a *pre-trap* context (or vice
   versa) the wrong local gets a garbage value. The suspicious values
   (`0x400` = bit 10, `0x40000000` = bit 30) smell like shifted flag or
   scratch locals.
2. **`addr_scratch_slot` reuse across the trap boundary**: local 44's
   documented value ("36 + 8") is a hand-count; if trap locals were ever
   declared before it, the scratch aliases a GPR.
3. **`seal_remaining` sealing the trailing slot with values from a
   different register-file layout** when the case's last word is `JR`
   (most repros end in `jr $ra`).

**Debug plan:** replay one clobber repro with
`SPEET_DIFF_DUMP_ON_TRAP=1`, dump the store slot and the halt stub,
map every `LocalGet`/`LocalSet` index against the layout snapshot
(`iter_before(&locals_mark())` — never hand-count, per AGENTS §9), and
bisect the three hypotheses. Fix the indexing at the source; add a
`debug_assert` that GPR writes never target `>= gpr_slot + 32`.

Acceptance: all checked-in clobber artifacts replay `fixed`; the "gprN:
oracle=X recompiled=Y" divergence class disappears from fixed-seed MIPS
runs.

---

## 5. Testing and acceptance

- **yecta unit tests** (§2.6 step 1) run without any MIPS involvement.
- **speet-mips tests**: the existing 11 plus new delay-slot cases:
  taken/not-taken slot execution order, predicate uses pre-slot values
  (slot writes the compared register), JAL `$ra` = `pc+8`, control
  transfer in slot → unsupported, branch targeting an absorbed PC →
  unsupported.
- **speet-diff-core**: `examples/sbfm.rs`-style probe tables are the
  durable regression record; add `examples/delayslot.rs` with hand-assembled
  BE words cross-checked by `examples/mipsdis.rs` (rabbitizer) and llvm-mc.
- **Differential**: fixed-seed (`--seed 42`, 60+ cases) A/B at each
  enablement step; `PROPTEST_CASES=300` soak must hold the
  documented-facet gate. Fresh artifacts land only when a *new* facet
  class appears.
- **Corpus**: WASI MIPS corpus binaries through `speet-e2e` (wasmi lane)
  exercise delay slots from real compiler-generated code, not just
  fuzz-synthetic shapes.

## 6. Milestones

| # | Scope | Done when |
|---|-------|-----------|
| MS-1 | yecta `taken_prefix` + tests | unit tests green; no consumer changes yet |
| MS-2 | `speet-mips` prefix integration + rejects | speet-mips tests incl. new delay cases pass; fetcher still unused |
| MS-3 | BE memory validate fix (§3.2) | swaps-on module validates for every generated case shape |
| MS-4 | Enable fetcher in link.rs + diff-core; BE on | fixed-seed sweep: delay + endian classes gone; artifacts regenerated |
| MS-5 | Clobber root-cause (§4) | clobber artifacts replay `fixed` |
| MS-6 | Corpus/e2e pass + docs | comparison-fuzzing-plan findings #6/#7 closed; guides updated if invariants changed |

MS-1/MS-2 are independent of MS-3; MS-4 depends on both; MS-5 is
independent and can land in any order.
