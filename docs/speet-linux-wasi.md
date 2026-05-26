# speet-linux-wasi

**Crate:** `crates/os/speet-linux-wasi`  
**Status: Active implementation.**

Registers WASI preview1 imports for the Linux target, bridging between the WASM runtime's WASI interface and the translated guest's Linux syscall ABI.

---

## Purpose

When the megabinary runs under a WASI-compatible runtime (e.g. Wasmtime, WasmEdge) rather than the custom vkernel, the translated guest's Linux syscalls must be forwarded to the WASI API. `speet-linux-wasi` provides this bridge by registering WASI preview1 imports into the megabinary's import table.

---

## Registered imports

| WASI function | Linux equivalent | Notes |
|---------------|-----------------|-------|
| `fd_write` | `write(2)` | File descriptor write |
| `fd_read` | `read(2)` | File descriptor read |
| `fd_close` | `close(2)` | File descriptor close |
| `proc_exit` | `exit(2)` | Process exit |

Additional WASI imports will be added as more syscalls are needed.

---

## Relationship to vkernel

`speet-linux-wasi` is the thin WASI-runtime path. The vkernel (see [docs/container-plan.md](container-plan.md) §2) is the production path that provides a full Linux-compatible syscall surface with security enforcement. These two paths are alternatives: a megabinary built for WASI uses `speet-linux-wasi`; a megabinary built for the vkernel uses the concrete `OS` implementation from [osctx](osctx.md).
