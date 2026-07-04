# ptrace Emulation

**Status: Planned** — see [future-features.md](../future-features.md).

## Purpose

Shared interception layer for syscall, signal, and debug events. Consumed by:

- **Container vkernel** — enforce manifest syscall whitelists, observe guest behavior inside WASM instances.
- **Thin runtime** — filter host calls, support transparent `execve` interception for on-the-fly recompilation.

## Sketch

```rust
// future crate: speet-ptrace or osctx extension
trait PtraceLayer {
    fn on_syscall_enter(&mut self, nr: u64, args: &[u64]) -> PtraceAction;
    fn on_syscall_exit(&mut self, nr: u64, ret: i64) -> PtraceAction;
}
```

No in-tree implementation yet. `FilteredHostApi` (thin runtime Phase 2) is the static whitelist precursor; ptrace emulation adds dynamic observation.
