# Agent Guide — Speet Recompiler

This file is a hub index. Each section links to a per-component guide in `docs/guides/` that documents design decisions which may look wrong or over-engineered at first glance. **Read the relevant guide before changing the component it covers** — these guides exist because agents have previously made confident-looking edits that silently broke an invariant no test caught; skimming the code without the guide reproduces the same mistake, not a faster version of the right fix.

See [docs/guides/README.md](docs/guides/README.md) for how guides work and what they are.

---

## 1. yecta reactor, speet-ordering, speet-wasm-helpers

**Guide:** [docs/guides/yecta.md](docs/guides/yecta.md)

- Do not assume "one function = one decoded instruction" — slots exist at every possible alignment.
- Do not eliminate `return_call` chains — O(1) stack depth depends on them.
- Do not remove the `locals_virtual` flush in `materialize_for`.
- Do not flush all stores before every load — use the runtime alias-check `if` blocks.

---

## 2. speet-traps

**Guide:** [docs/guides/speet-traps.md](docs/guides/speet-traps.md)

- Do not move `RopDetectTrap`'s depth counter to a local or global — it must survive `return_call`.
- Do not add `F` type param back to `InstructionTrap`/`JumpTrap`/`TrapConfig` — `dyn EmitSink` indirection is intentional.
- Do not move injected params to `()` returns — call-site trap-state preservation depends on them.

---

## 3. Linker (`speet-link-core`, `speet-linker`, `wasm-layout`, `speet-schedule`)

**Guide:** [docs/guides/linker.md](docs/guides/linker.md)

- Do not merge the two-pass `FuncSchedule` into one — cross-binary index resolution requires the layout to be final before emission.
- Do not add entity declarations inside emit closures — all indices must be registered in Pass 1.
- Do not add a `Reactor` field back to `LinkerInner` — native/WASM frontend dichotomy must stay value-level.

---

## 4. Architecture recompilers (`speet-x86_64`, `speet-riscv`, `speet-mips`, `speet-dex`)

**Guide:** [docs/guides/arch-recompilers.md](docs/guides/arch-recompilers.md)

- Do not remove `ctx: &mut Context` from `setup_traps` or any recompiler API — custom targets require it.
- Do not remove `TrapConfig` stubs — they are load-bearing placeholders for pending integration.

---

## 5. Parallel API (`yecta`, `speet-ordering`, all arch frontends)

**Guide:** [docs/guides/parallel-api.md](docs/guides/parallel-api.md)

- Do not revert `feed_to`/`feed`/`seal`/`barrier`/`jmp_tail` refactor.
- Do not simplify `Pool<'a, Context, E>` lifetime generics.
- Do not remove `tail_idx` params from `speet-ordering` public functions.
- Replace `&mut self` on hook traits with interior mutability — do not remove the parallel goal.

---

## 6. asm-arch ↔ speet instruction sync (`speet-x86_64`, `speet-aarch64`)

**Guide:** [docs/guides/asm-arch-instruction-sync.md](docs/guides/asm-arch-instruction-sync.md)

- speet's frontends must be able to decode every instruction family asm-arch's `WriterCore`
  emitter trait can produce — that's the invariant this sync maintains, not general ISA
  completeness.
- Do not assume a `WriterCore` method name matches its real encoding — verify against the
  binary writer impl (`iced.rs`/`bin.rs`) before concluding something is a gap.
- Do not assume one disarm64 enum variant = one instruction form — some merge multiple
  width/precision combinations, distinguished only by raw instruction bits.

---

## 7. Plugin API (`speet-plugin-api`, `speet-plugin-adapter`, `speet-plugin-host*`)

**Guide:** [docs/guides/plugin-api.md](docs/guides/plugin-api.md)
**Design doc:** [docs/plugin-api.md](docs/plugin-api.md)

- Do not let plugin-facing traits in `speet-plugin-api` leak `Context`/`E`/`F` generics — plugins must stay host-instantiation-agnostic.
- Do not bypass the `PluginTransport` seam to call a specific host backend directly from the adapter crate or any arch/resource crate.
- Do not let in-process dylib plugins cross the FFI boundary with anything but the fixed `extern "C"`/`#[repr(C)]` surface in the guide — no Rust trait objects, no relying on matching `rustc` versions.
- Do not add serde/bincode/rkyv to any plugin crate — the wire format is an intentional hand-rolled codec, shared by the WASM, subprocess, and dylib transports alike.
- Do not change `speet-plugin-api::arch::ArchOp`'s vocabulary or any wire format without updating `docs/plugin-api.md` and `docs/guides/plugin-api.md` in the same change.
- Do not grant a WASM or subprocess plugin a host-entity import (§2.7) outside its manifest-declared, embedder-approved allowlist.

---

## 8. Thin runtime (`speet-runtime`, `speet-host-api`, `speet-rt`)

**Guide:** [docs/thin-runtime-plan.md](docs/thin-runtime-plan.md)

- Do not conflate thin runtime with container megabinary — container is pure WASM with no host JIT; thin runtime produces native linked executables.
- Do not bypass `HostApi` for guest→host calls — tunnelled-by-default is the Phase 0 contract.
- Do not require macOS Xcode toolchain for link — LLVM `clang` + `lld` only.

---

## 9. Thin runtime genericity (`speet-recompile`, `speet-host-api`, `speet-plugin-api`, future `speet-abi-spec`/`speet-abi-codegen`)

**Guide:** [docs/guides/thin-runtime-genericity.md](docs/guides/thin-runtime-genericity.md)

- Do not hand-count a WASM import position (`N_IMPORTS`, `N_INTEGRATED_IMPORTS`, a `match name { "x" => 4, ... }` table) — allocate through `EntityIndexSpace`/`IndexSpace` and read `base()`/`total()` back; if two pieces of code need to agree on an index, share the same `IndexSpace`/manifest instance.
- Do not intercept external/PLT calls by matching a `call`/`jmp`/`bl` instruction's resolved target — check the current decode PC against the address-to-label table at every slot, since a computed jump, jump table, or fallthrough can reach the same address.
- Do not treat `unsupported_ops`/`unsupported_insns` as a correctness or suitability proof — it is a debugging/coverage signal only ("didn't immediately bail," not "translated correctly").
- Scope policy for ABI-spec-generated redirect stubs (`speet-abi-codegen`): only generate-and-check-in stubs for genuinely cross-platform behavior plus a small, deliberately curated set of easily-ported per-OS surfaces (e.g. libSystem and other easily-ported macOS stubs, low-risk Linux stubs). Do not check in generated stubs for the entirety of any OS's API surface — each addition is a deliberate, individually reviewed inclusion, never a bulk import.
- Every speet-emitted function shares one WASM type, `(registers) -> (registers)` — never `(registers) -> ()`. A guest `ret`/indirect return past the end of the translated set (most commonly `main` returning with no crt0 chain) is normal, not an error: it lands on a reserved **halt stub** (one slot past the last translated function) that surfaces the live register file as the call's results, the same way `crt0` calls `exit(main())`. Do not special-case this at runtime — the symmetric type is what lets `wasmparser::validate()` itself catch any code path that terminates without pushing the full register file first.
- Do not compute a table/`call_indirect`/`return_call_indirect` index from a runtime register/memory value without adding the space's `base_func_offset` — the table's `elem` segment populates `[n_imports, n_imports + n_fns)`, not `[0, n_fns)`; omitting the offset is the "speet emission gap" (`uninitialized element N` traps, or silently the *wrong guest function*) any multi-function guest hits on its first register-indirect `ret`/`br`/`blr`/`jalr`.
- Do not hand-maintain PLT-hook calling conventions (`match arch { X86_64 => ..., AArch64 => ... }`, per-symbol `match "execve" => { arg_locals: ... }` tables, or hardcoded import indices like `4` in tests) — use `speet_abi_stubs::plt_calling_convention(manifest, arch, guest_symbol)` (checked-in ABI stubs first, then [`ImportManifest`](crates/os/speet-host-api/src/manifest.rs) WASM param types for the intercept slot). Regenerate stub metadata via `speet-abi-codegen` (`registry.rs` + per-symbol modules under `speet-abi-stubs/src/generated/`); do not extend hand-written symbol dispatch in [`speet-abi-stubs/src/lib.rs`](crates/os/speet-abi-stubs/src/lib.rs).

---

## Compression-aware logging

Token compression proxies can sit between this tool and an LLM provider, compressing
output before it reaches the model. When a proxy is active, MORE verbose structured
output is net-cheaper than terse plaintext — the proxy reclaims the token cost and
the agent gains a richer trace.

Environment variables (set before running any binary or test in this workspace):

| Variable | Effect |
|---|---|
| `PORTAL_LOG_JSON=1` | Structured NDJSON output; also routes existing `log::` calls through the sink. Compresses ~3–5× better than plaintext. |
| `PORTAL_LOG_BATCH=1` | Group events by phase into single JSON arrays (reduces line count). |

Logger implementation: `crates/helper/speet-log/src/lib.rs`. Install it in any binary
entry point with `speet_log::install_as_global_logger(speet_log::LlmtrimLogger::from_env())`.
The `logging` feature on `speet-riscv` enables the `rlog!` macro (delegates to `log::debug!`),
captured automatically once the subscriber is installed.

These variables have no effect when unset and do not change program correctness.
