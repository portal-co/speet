# Recompiler debug MCP and hot-pluggable WASM frontend

**Crates:** `speet-rtd` (`mcp`, `hot-recompiler`), `speet-plugin-guest`, `speet-recompiler-guest`, `speet-stubs-guest`, `speet-runtime` (`RecompileReport`)
**Skill:** [`.agents/skills/speet-recompiler-debug`](../../.agents/skills/speet-recompiler-debug/SKILL.md)
**Related:** [plugin-api.md](plugin-api.md) (do not stretch `ArchOp`; no serde on plugin-api), [thin-runtime-genericity.md](thin-runtime-genericity.md) (`unsupported_insns` is not a proof), [arch-recompilers.md](arch-recompilers.md)

---

**Read this before adding a reload RPC, an `ArchOp` variant for bulk translate, serde on plugin-api, or an in-memory `grant_extern` overlay.** Those are the edits this design exists to prevent.

## Core principles

### 1. Two off-by-default features; production is byte-for-byte unchanged

`speet-rtd` features `mcp` and `hot-recompiler` are default-off, same pattern as `jit`. With both off, `RecompilerChoice::Native` and the statically linked stub table are the only path. Do not link wasmi or thin-mcp into the production daemon.

Enable together for MCP agents: `cargo run -p speet-rtd --features mcp,hot-recompiler -- --mcp`. Skill-only agents still need `hot-recompiler` on the daemon; they skip `--mcp` and drive it with `speet-rtdctl`.

### 2. Bulk translate is a debug guest ABI, not an `ArchOp`

[`ArchOp`](../../crates/plugin/speet-plugin-api/src/arch.rs) is a narrow command stream; [`ArchPluginRecompiler`](../../crates/plugin/speet-plugin-adapter/src/arch.rs) deliberately skips yecta `jmp` merging. The thing the agent iterates is the in-tree frontend (`speet-riscv`, …), which already uses a real `ReactorContext`. Forcing that through `ArchOp` would lose function merging.

`speet-recompiler-guest` exports `speet_plugin_call` with a **Translate** request (arch, `.text`, start/entry) and returns a megabinary plus `unsupported_insns`. The daemon loads it via wasmi (`WasmPlugin::call_raw`) under the conceptual scheme `"recompiler-wasm"`. That is embedder code in `speet-rtd`, not a bypass from adapter/arch crates.

**Do not** add a Translate variant to `ArchOp` or bump `PROTOCOL_VERSION` for this workflow. **Do not** add serde/bincode/rkyv to `speet-plugin-api`.

### 3. Stubs are as hot-pluggable as the frontend

Unknown externs are host-side ([`suitability.rs`](../../crates/os/speet-runtime/src/suitability.rs) + [`speet-abi-stubs`](../../crates/os/speet-abi-stubs)). `speet-stubs-guest` is a `TargetPlugin` whose `module_manifest().func_imports[].field` names are consulted **before** the statically linked `has_wired_impl` / allowlists. The agent edits the same source the table is generated from (`WIRED_SYMBOLS` mirrors `os-abi-stubs` `STUB_SYMBOLS`), then rebuilds the wasm32 cdylib.

There is no in-memory `grant_extern` overlay. A stub the agent cannot rebuild into WASM is not a real fix.

### 4. Reload is `ensure_fresh` on the next request

Every Analyze / Obtain / MCP / CLI request content-hashes the watched files and swaps instances if either hash changed. The old `Arc` stays alive so in-flight translations finish. Missing files keep the last good instance (or Native / linked-stubs if never loaded).

Default watch paths (overridable with `SPEET_RECOMPILER_WASM` / `SPEET_STUBS_WASM`):

- `target/wasm32-unknown-unknown/release/speet_recompiler_guest.wasm`
- `target/wasm32-unknown-unknown/release/speet_stubs_guest.wasm`

**Do not** add a `reload_*` MCP tool, CLI verb, or `notify` filesystem watcher. Rebuild then rerun is the contract.

### 5. Cache keys include both hashes when the feature is on

[`IntegratedNativeRuntime::artifact_cache_key`](../../crates/os/speet-runtime/src/integrated.rs) appends `:rec={hash}:stubs={hash}` to the input hash when the loaded plugins are not the static-link sentinels (`native` / `linked-stubs`). When `hot-recompiler` is off (or never loaded a file), those segments are omitted so production keys stay as they are today.

### 6. MCP and CLI are two fronts on one internal API

`Daemon::analyze_report` / `recompile_report` / `last_report` / `guest_info` / `run_guest` are the ops. MCP tools (`crates/os/speet-rtd/src/mcp.rs`, thin-mcp `handle_message`) and `speet-rtdctl` (`crates/os/speet-rtd/src/ctl.rs`) call them. The Unix protocol adds `LastReport` / `GuestInfo` / `Run`; Analyze/Obtain stay flattened for existing clients, and the CLI fetches `LastReport` so typed buckets survive at that layer.

`speet-plugin-guest` is the missing Rust guest SDK (`#![no_std]` + alloc, bump heap, `speet_plugin_alloc` / `call` / `dealloc`). It unblocks any future ArchPlugin author, not just this debug path.

Guest OS debug sessions in `os-daemon` are for **guest-runtime** trap events. Do not overload them with recompile-time failures.
