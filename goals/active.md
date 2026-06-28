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

## Plugin API — remaining work

The external plugin system (`crates/plugin/`) is built and documented: see [docs/guides/plugin-api.md](../docs/guides/plugin-api.md) and [docs/plugin-api.md](../docs/plugin-api.md). All five plugin trait families and all three MVP hosts (in-process static mode, WASM via `wasmi`, subprocess) are implemented and tested end-to-end, including the host-entity-import mechanism in both directions and its negative (denied-import) case. Two pieces remain:

- [ ] `speet-plugin-host-inproc` dylib mode: the `extern "C"`/`#[repr(C)]` surface + `libloading`-based loader behind the already-scaffolded `"dylib"` feature (see `docs/plugin-api.md` §5.1).
- [ ] `ArchPluginRecompiler` in `speet-plugin-adapter`: drives a real `Recompile<Context, E, F>` implementation from an `ArchPlugin`'s `step()` calls, activating `crates/os/speet-recompile/src/frontend.rs`'s `RecompilerChoice::Plugin` arm (currently `unimplemented!()`). Sequenced last — see `docs/plugin-api.md` §3.6 for why.
- [ ] `speet-plugin-host-wasm` `"exceptions"` feature (`wasmtime` engine) — only if/when a real plugin needs the exception-handling proposal; not required alongside the `wasmi` default.

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
