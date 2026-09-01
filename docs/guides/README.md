# Speet Agent Guides

This directory contains per-component alignment guides for AI agents working on the speet codebase.

---

## What guides are

Each guide documents the behaviours and invariants of one component that an agent might accidentally remove or "fix" without understanding the design. Guides are **loaded on demand** — an agent working on yecta loads `yecta.md`; an agent working on the linker loads `linker.md` — rather than being forced into every context upfront.

**Accuracy and completeness matter most.** The lesson from commit c273616: the original `AGENTS.md §1` omitted function merging and constant folding. Agents misattributed behaviour to other causes and broke those features. Omissions are as dangerous as errors. Every guide must cover *all* behaviours in its component that could be mistakenly removed or simplified.

---

## How to use guides

When touching a component, read its guide before making changes. If the guide appears to contradict what the code actually does, **trust the code and the design docs, not the guide** — then update the guide.

Guides are alignment signals, not hard stops. They work like Claude Code's contextual reminders: they surface relevant context rather than issuing rigid "STOP THIS" blocks. A goal-aligned agent that encounters a faulty or incomplete guide should reason from project goals and the relevant design docs in `docs/`. Rigidly following a wrong guide is a failure mode.

**Alignment with goals over rigidity of process.**

**Why skipping is the actual failure mode, concretely:** the mistakes these guides document did not come from agents being careless in some abstract sense — they came from agents producing edits that looked correct, compiled, and passed the tests that existed, while removing an invariant that no test exercised (a hand-counted index that only one caller currently uses; a redirect that only fires for one instruction shape; an entry point that happens to be function 0 in every binary tried so far). Reading the guide first is cheaper than finding this out from a production crash, because the guide states the invariant the passing tests don't. This is a reasoned tradeoff, not a instruction to comply with louder wording — a lower-effort model should stop here because the expected cost of skipping is high, not because the text says so twice.

---

## Available guides

| Guide | Component | Key "do not" |
|-------|-----------|--------------|
| [yecta.md](yecta.md) | `yecta` reactor, `speet-ordering`, `speet-wasm-helpers` | Do not collapse `return_call` chains; do not remove the `locals_virtual` flush; do not flush all stores before every load |
| [speet-traps.md](speet-traps.md) | `speet-traps` | Do not move `RopDetectTrap`'s depth counter to a local; do not add `F` type param back to trap traits |
| [linker.md](linker.md) | `speet-link-core`, `speet-linker`, `wasm-layout`, `speet-link`, `speet-schedule` | Do not merge two-pass into one; do not add entity declarations inside emit closures; do not add `Reactor` back to `LinkerInner` |
| [arch-recompilers.md](arch-recompilers.md) | `speet-x86_64`, `speet-riscv`, `speet-mips`, `speet-dex` | Do not remove `ctx: &mut Context` from `setup_traps` |
| [parallel-api.md](parallel-api.md) | `yecta`, `speet-ordering`, `speet-link`, all arch frontends | Do not revert `feed`/`seal`/`barrier`/`jmp_tail` refactor; replace `&mut self` on hooks with interior mutability |
| [asm-arch-instruction-sync.md](asm-arch-instruction-sync.md) | `speet-x86_64`, `speet-aarch64` | Do not assume a `WriterCore` method name matches its real encoding — verify against the binary writer impl before concluding something is in sync or a gap |
| [plugin-api.md](plugin-api.md) | `speet-plugin-api`, `speet-plugin-adapter`, `speet-plugin-host*` | Do not let plugin traits gain `Context`/`E`/`F` generics; do not bypass `PluginTransport`; do not add serde/bincode/rkyv; do not grant host-entity imports outside the manifest-declared allowlist |
| [thin-runtime-genericity.md](thin-runtime-genericity.md) | `speet-recompile`, `speet-host-api`, `speet-plugin-api`, future `speet-abi-spec`/`speet-abi-codegen` | Do not hand-count a WASM import index; do not intercept external calls by instruction shape instead of PC; do not treat `unsupported_ops` as a correctness proof |
| [dual-backends.md](dual-backends.md) | Backend A vs B, ZeroOffset / HostOffset / WASM-runtime OS, Linux-WASI / Darwin-WASI dual-lane | Do not conflate Backend A with Backend B; do not treat `speet-wasm` as Backend B; dual-lane WASI (wasmi) vs thin-runtime native for shared corpus slices; ZeroOffset text stays unmapped |
| [recompiler-debug-mcp.md](recompiler-debug-mcp.md) | `speet-rtd` `mcp`/`hot-recompiler`, `speet-plugin-guest`, `speet-recompiler-guest`, `speet-stubs-guest` | Do not stretch `ArchOp` for bulk translate; do not add serde to plugin-api; do not add a reload RPC or `grant_extern` overlay |

---

## Guide structure

New guides should follow the shape `thin-runtime-genericity.md` uses, which is the target for retrofitting older guides opportunistically (not required immediately):

1. **Core principles** — a short, numbered list of the actual invariants, each with *why* it matters (not just *what* to avoid). A principle without a reason is indistinguishable from an arbitrary style rule and gets skipped by the same reasoning that skips the whole guide.
2. **Examples drawn from git history** — cite real commits (`git show <hash>`) that show the mistake and its fix, rather than restating them in prose. This keeps the guide's evidence auditable and never requires rewriting/rebasing history to "clean up" an example.
3. **Cross-links** — where a guide's principle builds on or reuses a pattern from another guide, link it explicitly rather than restating that guide's content.

---

## Auditing a guide

If you change a component's behaviour and no guide entry covers it, add one. Guides should be audited against the actual code, not just the design docs — behaviour that exists only in code and nowhere in documentation is the most likely to be accidentally erased.

See [AGENTS.md](../../AGENTS.md) for the hub index, and `docs/` for the full design documents that guides reference.
