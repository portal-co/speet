# Agent Guide — Speet Recompiler

This file is a hub index. Each section links to a per-component guide in `docs/guides/` that documents design decisions which may look wrong or over-engineered at first glance. **Read the relevant guide before changing the component it covers.**

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
