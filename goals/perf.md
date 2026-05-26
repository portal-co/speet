# Large Binary & Performance

← [goals.md](../goals.md)

---

## Active

- [ ] **Parallelizable recompilation** — unlocked once the `&mut self` → interior mutability migration is complete; see [active.md](active.md) and [docs/guides/parallel-api.md](../docs/guides/parallel-api.md)

## Planned

- [ ] **Omit trivial / undefined instructions from output** — instructions that always produce WASM `unreachable` or no-ops can be elided; reduces output binary size
- [ ] **Dead-code elimination** — global DCE pass over the megabinary after all binaries are linked
- [ ] **Support large binaries end-to-end** — currently tested only on small ELF sections; needs end-to-end test with a real-world binary (see [workloads.md](workloads.md) for candidates)

## Completed

- [x] **Function slot omission for unreachable instructions** — requires reachable count passed to `FuncSchedule::push` before emission; implemented in `speet-reach` + `FuncSchedule`
