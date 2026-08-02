# speet-darwin-wasi

**Crate:** `crates/os/speet-darwin-wasi`  
**Status: Active implementation.**

Registers WASI preview1 imports for Darwin/BSD guests, bridging aarch64 `svc #0x80` syscalls onto the WASM runtime's WASI interface. Sibling of [`speet-linux-wasi`](speet-linux-wasi.md).

---

## Purpose

When a Mach-O / Darwin-ABI guest is translated for a WASI-compatible runtime (Lane A), its BSD syscalls must land on WASI preview1. `speet-darwin-wasi` provides that bridge:

1. Embeds a host-mem-lowered guest module (`speet-darwin-wasi-guest`) with `handler_{read,write,close,exit}` and `syscall_dispatch`.
2. Links that module through `WasmFrontend` (multi-memory: mem0 private, mem1 host).
3. Hooks aarch64 `svc` via [`DarwinWasiSvc`](../crates/os/speet-darwin-wasi/src/svc.rs) → `call syscall_dispatch`.

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

E2E: `crates/test/speet-e2e/tests/darwin_wasi_tests.rs` (Lane A write+exit).

---

## Dual-lane testing

See [`dual-backends.md`](guides/dual-backends.md) — Darwin-WASI guests share the Lane A / Lane B-native parity checklist with Linux-WASI. Existing macOS thin-runtime tests that use **RV64 Linux** guests dual-lane through `speet-linux-wasi` (`dual_lane.rs`); Mach-O/BSD guests use this crate.
