# Trace & Profiling

← [goals.md](../goals.md)

---

- [ ] **Trace support** — emit per-instruction trace hooks that can record PC, register state, and memory accesses; hook into `TraceLogTrap` in `speet-traps`
- [ ] **Transfers to assembly traces** — allow trace data to feed into external profiling tools (perf, Instruments); requires mapping WASM function indices back to guest PCs
- [ ] **Per-binary trace filtering** — enable/disable tracing per binary in the megabinary; stored in `manifest.json`
