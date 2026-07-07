# Thin Runtime Genericity Guide

**Crates:** `speet-host-api`, `speet-recompile`, `speet-runtime`, `speet-rt`, `speet-abi-spec`, `speet-abi-codegen`
**Design docs:** [thin-runtime-plan.md](../thin-runtime-plan.md), [entity-index-space.md](../entity-index-space.md), [future/abi-spec-redirects.md](../future/abi-spec-redirects.md)
**Related:** [plugin-api.md](plugin-api.md) §2 (the resource-kind pattern this guide's address-to-label plugin reuses), [linker.md](linker.md) §2 (`EntityIndexSpace`, extended here), [asm-arch-instruction-sync.md](asm-arch-instruction-sync.md) (the emitter the ABI-spec codegen reuses)

---

**Read this before touching WASM import assembly, PLT/external-call redirection, the memory shim, or `HostApi`/`PltRedirect` in the thin runtime.** This component was built fast, toward an MVP, and every rule below exists because that speed produced a real, working bug or a structurally-blocked feature — not a hypothetical one. Skimming this guide and proceeding anyway is how the next one gets written.

## Core principles

### 1. The index-space abstraction is the only sanctioned way to allocate an entry

`EntityIndexSpace`/`IndexSpace` (`crates/os/speet-link-core/src/layout.rs`, see [entity-index-space.md](../entity-index-space.md)) is a two-pass allocator: Pass 1 registers how many entities of each kind exist and freezes absolute indices; Pass 2 emits bodies using those already-known indices. This discipline extends to a **host-capability space** — a slot may be realized as a WASM import today, or as a directly-injected native-call stub with no WASM import at all for a future host-JIT path. The registration/numbering discipline is shared across backends; only the realization differs.

**Never** hand-count a position (`N_IMPORTS`, `N_INTEGRATED_IMPORTS`, a `match name { "x" => 4, ... }` table) and trust that it stays in sync with a list declared somewhere else. If you need an index, `append()` a slot and read `base()` back. If two pieces of code need to agree on an index, make them share the same `IndexSpace`/manifest instance — do not give them each their own copy of the same count.

### 2. No instruction-pattern matching for control-flow interception

A guest binary that is "well-behaved" in the sense of using a plain `call`/`bl` to reach a PLT stub or library function is still free to reach that *same address* through a computed jump, an indirect call through a function pointer, a jump table, or plain fallthrough. Any interception scheme that only fires when a `call`/`jmp`/`bl` instruction's *resolved target* matches a table misses every one of those cases — and "the binary I tested with used a normal call instruction" is not evidence the next one will.

The address-to-label / PLT-redirect system must check the **current decode PC** against the redirect table at every slot the recompiler visits — the same assumption yecta already makes for "one function = one decoded instruction" (`AGENTS.md` §1: slots exist at every possible alignment, because any address can be an entry point). Detect by address, not by decoding a specific instruction shape and assuming that's the only way in.

### 3. `unsupported_ops`/`unsupported_insns` are debugging tools, never an authority

`Translated::unsupported` and `rc.unsupported_insns()` exist so a human (or the `asm-arch-instruction-sync.md` decode-coverage tests) can see what a recompiler silently fell through to `Unreachable` on. That is a *coverage signal*, not a *soundness proof*. An empty `unsupported_insns()` means "nothing decoded to Unreachable" — it does not mean "this binary was translated correctly," and it must never be substituted for an actual suitability/correctness check (e.g. `speet-runtime::suitability::analyze_imports`'s `fn_ptr_free_allowlist` gate is deliberately a separate, explicit allowlist — not "unsupported list is empty, ship it").

If you are tempted to gate anything — suitability, caching, "is this safe to run" — on `unsupported_insns().is_empty()`, stop: that list was never designed to carry that weight, and nothing currently verifies it's exhaustive.

### 4. Every speet-emitted function has type `(registers) -> (registers)` — no exceptions, and let WASM validation enforce it

The shared WASM function type every recompiled guest function is registered under (`finish_module`/`assemble_corpus_module`'s type index 0) is **symmetric**: the full guest register file in, the full guest register file out. Not `(registers) -> ()`. This is deliberate, not an oversight to "optimize away" later:

- **A guest `ret`/indirect return landing outside the translated set is a normal event, not an error.** The most common case is the *last* one: `main` returning with no further guest code to run (no crt0/libc chain in the translated binary, or a `ret` that's genuinely the end of the program). Every entry point reserves one extra table slot — the **halt stub** — one slot past the last translated function (guest address `start_addr + text.len()`, i.e. exactly where the granularity-based `(addr - base_pc) / slot_size` formula already points with no special-casing). Its body is `local.get 0 .. local.get N-1; return`: it surfaces the *live register file at the point of return* as the WASM call's results, the same way a real `crt0` would call `exit(main())` with `main`'s return register. Callers (the native C entry shim, the corpus-harness test runner) arrange for this by seeding the *initial* return-address register/slot to the halt stub's guest address before invoking the entry function — see `speet-rt::entry_bridge`'s `SPEET_HALT_ADDR` and `speet-corpus-harness::run::lr_seed` for the two current implementations. This is what makes "the program returned instead of calling `exit`" behave identically to a real process's exit path, instead of jumping into whatever garbage the return-address register/slot happened to hold.
- **Because every function shares one type, WASM's own validator is a free correctness check.** `return_call`/`return_call_indirect` require the *caller's* declared result type to equal the *callee's* — trivially true here since everyone shares type 0, regardless of that type's actual shape. The only way to violate the contract is a bare, non-tail `Instruction::Return` that doesn't push the full register file first (`n` locals, in slot order) — and if some code path does that, `wasmparser::validate()` rejects the module outright (arity/type mismatch) instead of silently miscompiling or (worse) type-punning a truncated register file into a live program's results. Don't add a runtime check for this — the module either validates or it doesn't, and that's the check.
- If you add a new architecture or a new terminal control-flow path (anything that isn't `return_call`/`return_call_indirect`/`call`+`return_call_indirect`-to-hook), it must end in "push every register-file local as the WASM function's declared results, in slot order, then `return`" — copy the halt stub's shape, don't invent a shorter one.

---

## Examples (git history)

These are cited by commit hash rather than restated in prose, so they can be inspected directly (`git show <hash>`) without disturbing the actual history:

- **`5f0cc9c`** ("big refactor") introduced `EntityIndexSpace`, replacing `MegabinaryBuilder`'s previous self-assigned, ad hoc per-kind indices with the two-pass registration/emission discipline described in principle 1. The host-capability space this guide requires (see the "Fix plan" in the thin-runtime genericity project) is a direct extension of that same commit's idea to a new entity kind — not a new abstraction invented from scratch.
- [`asm-arch-instruction-sync.md`](asm-arch-instruction-sync.md)'s `[^xchg]` footnote and its disarm64/SCVTF-merged-variant note describe the same family of mistake principle 2 warns about, one level down: trusting that a name or a pattern ("this looks like the only encoding of this instruction," "this looks like the only way to reach this address") matches reality, instead of checking the actual encoding/address space. That guide's fix was "read the binary writer, don't trust the method name"; this guide's fix is "check the PC, don't trust the call-site shape" — same lesson, different layer.
- `TODO(after-merge)`: once the thin-runtime-genericity fixes (import index-space migration, plugin-accessible address-to-label system, PC-check interception, ABI-spec redirect stubs, `PltRedirect::Ambient`) land, replace this bullet with their actual commit hashes as the canonical before/after example for this specific component. Do not fabricate hashes in the meantime.

---

## Cross-links

- [plugin-api.md](plugin-api.md) §2 — the `ArchPlugin`/`AddressMapperPlugin`/`TablePlugin`/`ObjectModelPlugin`/`TargetPlugin` resource-kind pattern that the address-to-label plugin (principle 2) is a new member of. Read this first if you're extending redirect resolution to a new plugin architecture.
- [linker.md](linker.md) §2 — `EntityIndexSpace`'s existing two-pass contract, which principle 1 extends with a host-capability kind rather than replacing.
- [asm-arch-instruction-sync.md](asm-arch-instruction-sync.md) — the `WriterCore` emitter that ABI-spec-generated redirect stubs (see [future/abi-spec-redirects.md](../future/abi-spec-redirects.md)) emit through, and the precedent for "verify against the real encoding, not the method/pattern name" that principle 2 generalizes.
- [docs/guides/README.md](README.md) — the principle → git-history-example → cross-link structure this guide follows; see it for the convention writeup.
