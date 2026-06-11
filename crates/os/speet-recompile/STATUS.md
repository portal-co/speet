# Full-Binary Recompiler — Implementation Status

Tracks the build-out of the recompiler described in
`/Users/g/.claude/plans/plan-to-implement-a-smooth-parrot.md`.

## Done & verified

- **M0 — asm-arch external relocation surfacing** (tests pass on both arches).
  - `portal-solutions-asm-x86-64`: `IcedWriter::into_parts_with_relocs()` returns
    `(bytes, labels, Vec<UnresolvedReloc<L>>)`; new public `AsmRelocKind`
    (`X86Branch32` for CALL/JMP/Jcc, `X86Pcrel32` for LEA). Unresolved rel32 fields
    are left zero; addend = `patch_offset - instr_end` (−4).
    `crates/asm-x86-64/src/out/iced.rs`.
  - `portal-solutions-asm-aarch64`: `AArch64Writer::into_parts_with_relocs()`; new
    `AsmRelocKind` (`A64Call26`/`A64Jump26`/`A64AdrPrel21`/`A64CondBr19`), addend 0.
    `crates/asm-aarch64/src/out/bin.rs`.
- **`tunnel` crate** (`/Users/g/Code-local/portal-hot/tunnel`, 11 tests pass).
  `Tunnel` trait + `LinuxLibcTunnel` + `MacLibSystemTunnel`; 37-symbol libc/libSystem
  allowlist; `__ambient_*` alias pairs; `dylib_link_flag` → `-lc`/`-lSystem`.
- **`binary-io` crate** (`/Users/g/Code-local/portal-hot/binary-io`, 7/7 tests pass).
  Neutral load/write over the `object` crate (ELF + Mach-O); all 10 `RelocKind`
  variants mapped on both read and write, x86_64 + aarch64.
  - **Driver gotcha — Mach-O symbol mangling**: `object::write` auto-prepends `_`
    to *defined* Text/Data symbols but NOT to *undefined* ones. So a defined
    `__guest_entry` is emitted as `___guest_entry`, while an undefined external
    stays verbatim. When targeting Mach-O the driver must pre-mangle external
    target names (e.g. pass `_write`) and account for the extra `_` on the
    exported entry / on `__ambient_*` alias definitions.
- **`speet-rt` runtime shim** (`crates/os/speet-rt`, builds; `runtime.c` compiles via clang).
  Defines `__wasm_mem`/`__wasm_mem_pages`/`__wasm_memory_grow`, weak `__speet_data_init`,
  and a C `main` that bootstraps linear memory and calls `__guest_entry`.

## Scaffolded (compiles; integration TODO)

- **`speet-recompile`** driver: `frontend` (load, `ExternalTargets`, same-platform
  guard) and `backend` (reloc-kind mapping, `__ambient_` symbol rendering) modules.
- **`speet-host-syscall`**: placeholder for the host-ambient syscall dispatcher.

## Architectural findings

1. **Memory base mismatch in the wasm-blitz native backends — RESOLVED.**
   The wasm-blitz **C** backend addresses linear memory as `__wasm_mem + (uint32_t)addr`,
   but the **x86-64/aarch64 naive** backends dereferenced the WASM address as a *raw host
   pointer* — only valid when linear memory is mapped at the WASM virtual address (the
   Unicorn test model), not in an ordinary OS process.
   - **Fix**: added a configurable `naive::MemBase` to both backends
     (`wasm-blitz/crates/blitz-x86-64/src/naive.rs`, `.../blitz-aarch64/src/naive.rs`).
     Default `MemBase::Raw` keeps the legacy behavior (existing Unicorn tests unchanged —
     the new `apply_mem_base` helper early-returns for `Raw`). New `MemBase::WasmMemSymbol`
     computes `__wasm_mem + (uint32_t)addr`, matching the C backend, threaded through SysV
     too. Verified by new `membase_tests` on both backends.
   - **Driver requirement**: the backend drive loop must set `state.mem_base =
     MemBase::WasmMemSymbol` so emitted code references `__wasm_mem` (provided by
     `speet-rt`). Memory-touching programs now work, not just `exit`-only.

2. **Entry ABI bridge.** The backend's export dispatcher (`emit_export_dispatchers`)
   makes `__guest_entry` a stub using the wasm-blitz **NaiveAbi** (custom stack/CTX
   convention), not the C ABI. The shim's C `main` cannot generally call it directly. For
   the M1 `exit`-only program the entry never returns (exits via syscall), so the mismatch
   is benign; a real version needs a SysV/AAPCS64 entry trampoline (the plan's SysVAbi path).

## Host-platform note

This dev host is **macOS / aarch64**, so the first *runnable* artifact must be
Mach-O + aarch64 + libSystem (the plan's "M4" stack), even though the x86_64/ELF paths
can be unit-tested (object round-trips) without executing.

## Next integration steps (M1)

1. Backend drive loop in `backend.rs`: feed a `MegabinaryOutput` through
   `mach_operators`/`handle_op` into one `AArch64Writer<AArch64Label>` (this host),
   `into_parts_with_relocs()`, map to `ObjReloc`, `MachOObjectWriter::write_object`.
2. Frontend drive loop in `frontend.rs`: `translate_bytes` over the `.text` of a tiny
   hand-built guest `exit` binary into a `MegabinaryBuilder`; export `__guest_entry`.
3. Link `generated.o` + `speet-rt` `runtime.o` with `clang -lSystem`; run; assert exit code.
4. Then resolve finding (1) to unlock memory-touching programs (M2+).
