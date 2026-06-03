# Large Binary & Performance

← [goals.md](../goals.md)

---

## Active

- [ ] **Parallelizable recompilation** — unlocked once the `&mut self` → interior mutability migration is complete; see [active.md](active.md) and [docs/guides/parallel-api.md](../docs/guides/parallel-api.md)

## Planned

- [ ] **Omit trivial / undefined instructions from output** — instructions that always produce WASM `unreachable` or no-ops can be elided; reduces output binary size
- [ ] **Dead-code elimination** — global DCE pass over the megabinary after all binaries are linked
- [ ] **Support large binaries end-to-end** — currently tested only on small ELF sections; needs end-to-end test with a real-world binary (see [workloads.md](workloads.md) for candidates)
- [ ] **Native control-flow optimization inside `yecta` `Entry`ies (faithful-by-default)** — today every conditional branch lowers to a conditional `return_call` (`if <cond> { return_call target } else …`) fanned into each reachable function. The goal is to let *opt-in* `Optimizer`/`Entry` variants emit native WASM control flow (real `if`/`br_if`/blocks, native condition emission) instead, while preserving the one-function-per-possible-instruction model and full faithfulness by default. Groundwork has landed in `crates/helper/yecta/src/lib.rs`:
  - the per-`Entry` optimizer state is a one-variant `enum Optimizer { ConstFold(..) }`, ready for additional variants;
  - per-instruction and conditional-branch emission are `Entry` methods (`Entry::feed_one`, `Entry::emit_conditional_arm`) driven by a Reactor↔Entry handshake — the Reactor owns the predecessor graph, each `Entry` decides *how* it emits;
  - predecessor edges carry an `ExitId` (today always `SOLE_EXIT`) so a future variant can distinguish a function's multiple exits (e.g. taken-branch vs fall-through).

  Remaining work is the optimized variant itself: a new `Optimizer` variant whose `emit_conditional_arm` lowering emits native control flow and registers distinct `ExitId`s via `add_pred`, plus a per-region opt-in knob.

## Completed

- [x] **Function slot omission for unreachable instructions** — requires reachable count passed to `FuncSchedule::push` before emission; implemented in `speet-reach` + `FuncSchedule`
