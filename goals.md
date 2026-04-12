# Speet — Goals

Prefer dynamically creating subgoals over handling entire goals at a time.
AI agents: add subgoals here and to memory files as you work.

---

## Active Work

- [ ] **Parallel API migration** — the build is currently broken (~96 errors across downstream crates).
  See `docs/parallel-api-migration.md` for full migration steps.
  - [ ] Add `feed`, `seal`, `barrier`, `jmp_tail`, `with_local_pool` wrappers to `yecta`
  - [ ] Add `NoopHandler` + `StaticPool`/`as_pool()` to `yecta`
  - [ ] Update `speet-ordering`: add `tail_idx` param to all public fns
  - [ ] Update `speet-link`: fix `Pool` fields, wire convenience wrappers
  - [ ] Update `speet-x86_64` and `speet-powerpc`
  - [ ] Update `speet-riscv` and `speet-mips`
  - [ ] (Stretch) Deprecate convenience wrappers; replace all call sites with explicit `tail_idx`

---

## Architecture Frontends

- [ ] Finish `speet-x86_64` — FP/SIMD currently stubbed as `unreachable`
- [ ] Implement `speet-aarch64` — decoder (`disarm64`) is in workspace deps but unused
- [ ] Implement `speet-powerpc` — crate exists as a stub; no translation logic
- [ ] Implement WASM-GC frontend (urgent: needed for Claude Code / jsaw + speet integration)
- [ ] `speet-dex` — in progress; bring to feature parity with `speet-riscv`

---

## Large Binary & Performance

- [ ] Parallelizable recompilation (unlocked once API migration is complete; see `docs/parallel-api-migration.md`)
- [ ] Omit trivial / undefined instructions from output
- [ ] Dead-code elimination
  - [x] Function slot omission for unreachable instructions (requires reachable count passed to `FuncSchedule::push` before emission)
- [ ] Support large binaries end-to-end (currently tested only on small ELF sections)

---

## OS Emulation & Container Megabinary

Goal: translate entire container images into a single, AOT-compiled WASM megabinary.
See `docs/container-plan.md` for architecture, formats, and phased rollout.

- [ ] Unified `MegabinaryContext`: shared `Reactor` + global `FuncIdx` registry across all binaries
- [ ] Multi-architecture dispatch inside a single megabinary
- [ ] Linux syscall ABI — `osctx` currently only has trait stubs
- [ ] Container lifting pipeline: scan → lift → merge → sign
- [ ] `manifest.json` hash-mapping format (hash → megabinary entry point)
- [ ] Megabinary dispatcher (`_dispatch(hash_id, argc, argv)`)
- [ ] vkernel: host-side process exposing Linux-compatible syscall surface to WASM
- [ ] `execve` interception with hash verification (fail on mismatch)
- [ ] Shared polyfills in megabinary (WASM-compiled CPython, QuickJS for Node)
- [ ] containerd / Kubernetes shim integration

---

## Security

- [ ] No-JIT enforcement in vkernel (block `mmap`/`mprotect` with `PROT_EXEC`)
- [ ] Signed megabinary + manifest; vkernel verifies before any execution
- [ ] App-level security hardening (`speet-traps`: `RopDetectTrap`, `CfiReturnTrap`)
- [ ] Antimalware hooks
- [ ] Agentic AI safety: agents can only invoke pre-approved tools already in the megabinary
- [ ] Syscall whitelisting per binary (stored in manifest)

---

## Trace & Profiling

- [ ] Trace support, including transfers to assembly traces
