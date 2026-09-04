# Speet — Goals

Prefer dynamically creating subgoals over handling entire goals at a time.
AI agents: add subgoals to the relevant file in `goals/` as you work.

---

## Index

| Area | File | Status |
|------|------|--------|
| **Active work** | [goals/active.md](goals/active.md) | Parallel API interior mutability; trap integration; new docs; plugin API dylib mode + `ArchPluginRecompiler` |
| **Architecture frontends** | [goals/arch.md](goals/arch.md) | Main four implemented (aarch64/riscv/x86/mips); MIPS Flag/thin + x86 SIMD in progress; powerpc not started; WASM-GC urgent |
| **OS emulation & container megabinary** | [goals/os.md](goals/os.md) | Phase 0 target: ls/cat/echo megabinary demo |
| **Security** | [goals/security.md](goals/security.md) | No-JIT; signed megabinary; ROP/CFI traps; syscall whitelisting |
| **Large binary & performance** | [goals/perf.md](goals/perf.md) | Parallel recompilation (blocked); DCE; large binary e2e |
| **Trace & profiling** | [goals/trace.md](goals/trace.md) | Not yet started |
| **Candidate workloads** | [goals/workloads.md](goals/workloads.md) | Bun, Electron, COBOL, EHR, embedded OpenSSL, retro |
| **Ambient library integration** | [goals/ambient.md](goals/ambient.md) | `AmbientInfo`/`AmbientSink` API; recompilers calling unrecompiled libs |

---

## Key references

- [docs/parallel-api-migration.md](docs/parallel-api-migration.md) — API migration history and invariants
- [docs/comparison-fuzzing-plan.md](docs/comparison-fuzzing-plan.md) — differential fuzzing (speet vs Unicorn) plan
- [docs/container-plan.md](docs/container-plan.md) — container megabinary architecture and threat model
- [docs/guides/](docs/guides/) — per-component alignment guides
- [docs/plugin-api.md](docs/plugin-api.md) — external plugin system design doc (architecture/memory/table/object-model/target plugins; in-process/WASM/subprocess hosts)
