# speet-darwin-wasi

**Crate:** `crates/os/speet-darwin-wasi`  
**Status: Active implementation.**

Registers WASI preview1 imports for Darwin/BSD guests. Its primary Lane A path
redirects libSystem dylib calls through virtual GOT/lazy-pointer targets; raw
aarch64 `svc #0x80` remains supported for syscall-only guests. Sibling of
[`speet-linux-wasi`](speet-linux-wasi.md).

---

## Purpose

When a Mach-O / Darwin-ABI guest is translated for a WASI-compatible runtime
(Lane A), libSystem calls normally arrive through PLT/GOT or lazy pointers.
`speet-darwin-wasi` provides that bridge:

1. Embeds a host-mem-lowered guest module (`speet-darwin-wasi-guest`) with `handler_{read,write,close,exit}` and `syscall_dispatch`.
2. Links that module through `WasmFrontend` (multi-memory: mem0 private, mem1 host).
3. Resolves `read`/`write`/`close`/`exit` (and leading-underscore aliases)
   from an `ExternalTargetTable` into virtual redirect-shim PCs after `.text`.
   A Mach-O loader patches its GOT/lazy-pointer cells to those PCs; indirect
   calls then land in the shared handler path.
4. Keeps [`DarwinWasiSvc`](../crates/os/speet-darwin-wasi/src/svc.rs) →
   `call syscall_dispatch` as the secondary raw-syscall path.

---

## Guest ABI

| Convention | Value |
|------------|-------|
| Trap | `svc #0x80` (imm accepted; number still read from `x16`) |
| Syscall number | `x16` |
| Args | `x0`, `x1`, `x2`, … |
| Numbers | BSD baseline: `SYS_exit=1`, `SYS_read=3`, `SYS_write=4`, `SYS_close=6` |

Mapping table lives in `os-emulation`'s [`os-darwin-wasi`](../../os-emulation/crates/emit/os-darwin-wasi).

---

## Registered imports

| WASI function | Darwin/BSD equivalent |
|---------------|----------------------|
| `fd_write` | `write(2)` (`SYS_write=4`) |
| `fd_read` | `read(2)` (`SYS_read=3`) |
| `fd_close` | `close(2)` (`SYS_close=6`) |
| `proc_exit` | `exit(2)` (`SYS_exit=1`) |

---

## Entry point

```rust
let wasm = speet_darwin_wasi::recompile_aarch64_darwin_wasi_to_wasm(text, start_addr);
```

For dylib/GOT guests, use
`recompile_aarch64_darwin_wasi_to_wasm_with_targets(text, start_addr, &targets)`.
Target addresses must be the allocated virtual shim PCs; text-only callers
must patch their data image/GOT before execution.

E2E: `crates/test/speet-e2e/tests/darwin_wasi_tests.rs` (raw-svc and virtual
redirect write+exit).

---

## Dual-lane testing

See [`dual-backends.md`](guides/dual-backends.md) — Darwin-WASI guests share the Lane A / Lane B-native parity checklist with Linux-WASI. Existing macOS thin-runtime tests that use **RV64 Linux** guests dual-lane through `speet-linux-wasi` (`dual_lane.rs`); Mach-O/BSD guests use this crate.
