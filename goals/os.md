# OS Emulation & Container Megabinary

← [goals.md](../goals.md)

Full architecture, formats, pipeline, security model, and phased rollout: [docs/container-plan.md](../docs/container-plan.md).

**Thin runtime** (native-to-native, on-demand recompilation): [docs/thin-runtime-plan.md](../docs/thin-runtime-plan.md). Future enhancements: [docs/future-features.md](../docs/future-features.md).

---

## Container megabinary — phased rollout

### Phase 0 — Hello Megabinary
- [ ] Recompile `ls`, `cat`, `echo` into one megabinary
- [ ] Implement `_dispatch(hash_id, argc, argv)` entry point
- [ ] Hash-router: compute binary hash → look up in `manifest.json` → jump to entry
- [ ] `manifest.json` generation in `speet-link` / build tooling

### Phase 1 — AI Tool-Sandbox
- [ ] Agent containers with Python, Bash, Git; demonstrate <10 ms tool invocation
- [ ] Polyfill layer: Python → WASM-compiled CPython/MicroPython; Bash → QuickJS or equivalent
- [ ] Shared polyfill linking via `yecta`'s `Pool` + `Target::Static`

### Phase 2 — containerd Shim
- [ ] Hash-interception shim in a containerd-compatible plugin
- [ ] Run "megabinary-fied" OCI images under the shim

### Phase 3 — Library Deduplication
- [ ] Deduplicate libc/openssl across all binaries in the megabinary
- [ ] Measure memory and code-size savings

### Phase 4 — Production Canary
- [ ] Deploy NGINX and Redis on dedicated WASM nodes in a Kubernetes cluster

---

## Infrastructure

- [ ] **vkernel** — host-side process exposing Linux-compatible syscall surface to WASM instances
  - Intercepts `execve`: compute hash → manifest lookup → dispatch or `EACCES`
  - Blocks `mmap`/`mprotect` with `PROT_EXEC`
  - Enforces per-binary syscall whitelists from manifest
  - Snapshot/restore of WASM memory for agent pause/resume
- [ ] **Unified `MegabinaryContext`** — shared reactor + global `FuncIdx` registry across all binaries
- [ ] **Multi-architecture dispatch** inside a single megabinary
- [ ] **Linux syscall ABI** — `osctx` currently has trait stubs only; needs concrete implementations
- [ ] **Container lifting pipeline**: scan → lift → merge → sign CLI (`recompiler-build`)

---

## Speet integration points

- `speet-link`: `MegabinaryBuilder` needs multi-binary `base_func_offset` registry
- `osctx`: `OS::syscall` / `OS::osfuncall` need concrete vkernel implementations
- `speet-traps`: `RopDetectTrap` + `CfiReturnTrap` as standard hooks in all frontends
- `speet-linux-wasi`: WASI preview1 bindings for the Linux target

---

## Thin runtime — phased rollout

### Phase 0 — LLVM corpus MVP
- [x] `test-data/thin-runtime-corpus/` with ELF + Mach-O guests (LLVM `compile_corpus.sh`)
- [x] `speet-host-api` + `speet-runtime` driver (tunneled default, LLVM link, spawn child)
- [x] Corpus roundtrip: RV64 `exit(42)` → x86_64/ELF (link) and aarch64/Mach-O (run)

### Phase 1 — Real binary loading + cache
- [x] `binary_io::load_auto` wrapper in `speet-runtime::load_binary`
- [ ] `ExternalTargets` PLT tunneling in driver
- [x] Artifact cache (`ArtifactCache` keyed by input hash + pipeline version)
- [x] Expand `speet-host-syscall` (x86_64 Linux table)

### Phase 2 — Pluggable HostApi
- [x] `HostApiRegistry` + `FilteredHostApi` + `HostPolicy` (manifest-style lists)

### Phase 3 — Convergence
- [ ] `OsctxHostApi` bridge to vkernel; `TargetPlugin` at recompile time

### Future (documented, not implemented)
- [ ] Host JIT asm→asm — [docs/future/host-jit.md](../docs/future/host-jit.md)
- [ ] Direct in-process linking — [docs/future/direct-linking.md](../docs/future/direct-linking.md)
- [ ] ptrace emulation — [docs/future/ptrace-emulation.md](../docs/future/ptrace-emulation.md)
