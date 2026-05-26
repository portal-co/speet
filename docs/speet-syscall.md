# speet-syscall

**Crate:** `crates/os/speet-syscall`  
**Status: Active implementation.**

Emits inline `br_table` dispatch over guest syscall numbers, translating them to WASM function calls into the host OS shim layer.

---

## Purpose

When an architecture frontend encounters a guest `syscall` / `int 0x80` / `ecall` instruction, the recompiler must emit WASM code that:
1. Reads the syscall number from the guest register (e.g. `a7` on RISC-V, `rax` on x86-64).
2. Dispatches to the correct host handler.
3. Returns the result into the guest return register.

`speet-syscall` generates the dispatch table as a WASM `br_table` indexed by syscall number, with each arm calling the appropriate shim function. Unknown syscall numbers fall through to a default arm that calls `OS::syscall` dynamically.

---

## Integration with `osctx`

`speet-syscall` works in conjunction with [osctx](osctx.md): the shim functions called by the `br_table` arms are generated from the `OS` trait implementations registered at link time. Each shim is a WASM function that calls `OS::syscall` with the syscall number and argument list.

The `ShimSpec` in `speet-link` coordinates between `speet-syscall` (which emits the dispatch table) and `osctx` (which provides the `OS` implementation).
