# osctx

**Crate:** `crates/os/osctx`  
**Status: Active — trait definitions complete; concrete implementations pending (see [goals/os.md](../goals/os.md)).**

Defines the OS context abstraction: the boundary between translated guest code and the host operating system.

---

## Traits

### `Ctx` — per-call register/memory view

Provides a view into the guest's current register file and memory state for a single translated function call. Used by `setup_traps` and recompiler APIs that need access to the live architectural state.

### `OS` — syscall dispatch

```rust
trait OS {
    fn syscall(&mut self, ctx: &mut dyn Ctx, nr: u64, args: &[u64]) -> i64;
    fn osfuncall(&mut self, ctx: &mut dyn Ctx, addr: u64, args: &[u64]) -> i64;
}
```

`syscall` handles guest system calls by number (Linux ABI). `osfuncall` handles calls that target OS-provided functions (e.g. `__libc_start_main` shims). Both are called from the generated WASM at syscall sites.

---

## Concrete implementations

The traits have no concrete implementations yet. The vkernel (see [docs/container-plan.md](container-plan.md) §2) will provide the concrete `OS` implementation that bridges to the host Linux kernel via the vkernel's syscall filter.

---

## Usage

`osctx` traits are the integration point for the container megabinary pipeline. When an architecture frontend encounters a `syscall` instruction, it emits a WASM call to a shim that dispatches through `OS::syscall`. The vkernel hosts the concrete `OS` implementation and enforces the per-binary syscall whitelist from `manifest.json`.
