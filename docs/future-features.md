# Future Features

Umbrella index for capabilities planned across **both** the container megabinary path and the thin runtime path. Mark entries **shipped** when they land in-tree.

| Feature | Doc | Status | Consumed by |
|---|---|---|---|
| Host JIT (asm→asm) | [future/host-jit.md](future/host-jit.md) | Planned | Thin runtime |
| Direct in-process linking | [future/direct-linking.md](future/direct-linking.md) | Planned | Thin runtime |
| ptrace emulation | [future/ptrace-emulation.md](future/ptrace-emulation.md) | Planned | Container vkernel + thin runtime |
| execve interception | (thin-runtime-plan Phase 4+) | Planned | Thin runtime |
| Cross-arch recompilation | (container-plan multi-arch) | Planned | Both |
| ABI-spec redirect stubs | [future/abi-spec-redirects.md](future/abi-spec-redirects.md) | Phase 1–2 shipped (ingest + curated codegen); PC-check native emit wiring in progress | Thin runtime |
| Redirect shims via virtual GOT | [future/redirect-shim-got.md](future/redirect-shim-got.md) | Planned (deferred; interim PC-check hooks shipped) | Thin runtime |

When a feature ships, update its doc with the implementing crates and flip the status here.
