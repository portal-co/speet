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

## Backend pipeline — RUNNABLE end-to-end (x86_64)

`speet-recompile::drive::compile_wasm_to_object` lowers a WASM module through
wasm-blitz's SysV codegen into one asm-arch binary writer, surfaces external relocations
via `into_parts_with_relocs`, and emits an ELF/Mach-O `.o` via `binary-io`, exporting the
entry as the C-callable `__guest_entry`. The integration test `tests/backend_e2e.rs`
recompiles a WASM `() -> i64` returning 42 into an **x86_64 Mach-O object**, links it with a
C shim (`main` → `__guest_entry`) via `clang`, runs it (Rosetta on Apple silicon), and
asserts **exit code 42**. This validates blitz codegen + binary-io object writing + the #2
C-ABI bridge + link/run together.

### Known limitation: aarch64 native SP alignment
blitz's aarch64 backend uses the hardware SP as the WASM operand stack with 8-byte
`str/ldr [sp,#±8]!` pushes. Real arm64 (macOS) enforces a 16-byte SP-alignment check on
SP-based accesses, so recompiled aarch64 code faults (SIGBUS) after the first push — even
though it passes under Unicorn (which doesn't enforce the check). Fixing it needs either
16-byte operand-stack slots or a non-SP stack register in the aarch64 backend. Until then
the runnable path is x86_64 (native on x86 hosts, Rosetta on Apple silicon).

## Scaffolded (compiles; integration TODO)

- **`speet-recompile`** `frontend` (load, `ExternalTargets`, same-platform guard) — the
  speet translate loop (real guest binary → WASM) is the next integration step.
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

2. **Entry ABI bridge — RESOLVED (mechanism).** The recompiled guest entry carries the
   whole guest register file as WASM parameters (32–70 of them). The wasm-blitz **NaiveAbi**
   needs a host runtime to bootstrap its CTX/operand-stack chain (even wasm-blitz's own
   tests can't run x86-64 naive standalone), so a NaiveAbi entry trampoline is fragile.
   Instead we compile the module with **SysVAbi/AAPCS64**, where recompiled functions follow
   the C ABI directly.
   - **Blocker found & fixed**: the SysV/AAPCS64 backends only loaded the *register*-passed
     args (`min(6)`/`min(8)`), dropping stack-passed params — so many-param functions got
     garbage. Fixed to load params 6+/8+ from the caller's stack frame
     (`wasm-blitz/crates/blitz-{x86-64,aarch64}/src/sysv.rs`). Verified by two new Unicorn
     tests (8-param x86-64, 10-param aarch64); full native suite 230 passed.
   - **Bridge**: the recompiled entry is now directly C-callable. The driver generates the
     shim's call to it (`__guest_entry(0, 0, …, guest_sp, …, 0)`) passing the initial
     register values as C arguments — the C compiler handles register+stack placement. No
     hand-written assembly trampoline required.
   - **Remaining for integration**: confirm blitz SysV marshals all N args on *internal*
     tail-calls (`Call`/`return_call`) through the register-file chain, not just at entry.

## Verification capability

`cmake` is now installed, so wasm-blitz's Unicorn-based native test suite runs here
(`cargo test -p portal-solutions-blitz-tests --test e2e native` → 230 passed). This validated
the memory-base fix (Raw default unchanged) and the SysV stack-param fix end-to-end under
emulation. Note: full ELF/Mach-O link+run still needs the driver; Unicorn covers raw native
code execution without a linker.

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
