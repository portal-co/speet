# Active Work

← [goals.md](../goals.md)

---

## Parallel API Migration — remaining work

The original 8-step migration (see [docs/parallel-api-migration.md](../docs/parallel-api-migration.md)) is mostly complete. The build is no longer broadly broken. The remaining blocker for parallel schedule emission is that plugin/hook traits take `&mut self`, making emit closures non-`Send`.

See [docs/guides/parallel-api.md](../docs/guides/parallel-api.md) for direction on interior mutability patterns.

- [ ] Audit which steps from `docs/parallel-api-migration.md` are fully done; mark them
- [ ] Identify all `&mut self` occurrences in `InstructionTrap`, `JumpTrap`, `TrapConfig` across `speet-traps` and callers
- [ ] Replace `&mut self` on `RopDetectTrap` depth counter with `AtomicU32`
- [ ] Replace `&mut self` on other hook traits with `Mutex<T>` or `RwLock<T>` as appropriate
- [ ] Verify emit closures passed to `FuncSchedule::execute` become `Send + Sync`
- [ ] Implement `FuncSchedule::execute_parallel` (Rayon or similar)
- [ ] Parallelism benchmark vs. sequential baseline

---

## Trap integration into architecture recompilers

The trap trait infrastructure in `speet-traps` is complete. Integration into each architecture recompiler is specified in [docs/trap-hooks.md](../docs/trap-hooks.md) §8 but not yet implemented.

- [ ] `speet-riscv`: add `TrapConfig` field + setters, `setup_traps` call, per-instruction trap firing, jump-site trap firing, `classify_insn`
- [ ] `speet-mips`: same as above
- [ ] `speet-x86_64`: same as above
- [ ] `speet-dex`: same as above

---

## Documentation for undocumented crates

- [ ] Write `docs/speet-memory.md`
- [ ] Write `docs/speet-reach.md`
- [ ] Write `docs/speet-interp.md`
- [ ] Write `docs/speet-wasm.md`
- [ ] Write `docs/speet-object.md`
- [ ] Write `docs/osctx.md`
- [ ] Write `docs/speet-syscall.md`
- [ ] Write `docs/speet-linux-wasi.md`
- [ ] Write `docs/module-builder.md`
- [ ] Add `speet-schedule` section to `docs/recompiler-guide.md` §3
- [ ] Add `dex-bytecode` crate README
