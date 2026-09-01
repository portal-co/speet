---
name: speet-recompiler-debug
description: >
  Debug a failing speet recompile. Use when Analyze/Obtain fails, when
  unresolved-dep / fn-ptr-dep / an unknown extern appears, when
  unsupported_insns names a gap, when iterating the hot-pluggable
  recompiler.wasm or stubs.wasm without restarting speet-rtd, or when
  driving speet-rtd through MCP tools or speet-rtdctl.
---

# Speet recompiler debug

Drive rule: if MCP tools `analyze_binary`, `recompile`, `last_report`, `guest_info`, `run_guest` are present, use them. Otherwise use `speet-rtdctl` against `SPEET_RTD_SOCK` (or the daemon default socket). Same loop, same completion criteria. Do not configure MCP from this skill.

Daemon for this loop: `cargo run -p speet-rtd --features mcp,hot-recompiler -- --mcp` (MCP agents) or the same binary without `--mcp` (CLI agents). Production `speet-rtd` with both features off is unchanged.

## Loop

1. **Pin** the guest (`guest_info` / `info <path>`). Done when the JSON names path, arch, `.text` span, entry, and imports.
2. **Classify** `recompile` / `last_report` into one bucket: `unresolved_deps` / `fn_ptr_deps` (extern), `unsupported_insns` (insn), `translate_error`, `link_error`. Done when exactly one bucket is the current blocker.
3. **Patch** the single responsible crate. Extern → `crates/os/speet-abi-stubs` / `crates/os/speet-stubs-guest` `WIRED_SYMBOLS` (same names the table is generated from). Insn → the arch frontend (`speet-riscv`, `speet-aarch64`, …). Read the matching guide first: [arch-recompilers.md](../../../docs/guides/arch-recompilers.md), [thin-runtime-genericity.md](../../../docs/guides/thin-runtime-genericity.md), [plugin-api.md](../../../docs/guides/plugin-api.md). Done when the diff is confined to that crate.
4. **Rebuild** only the matching WASM guest. Extern → `cargo build -p speet-stubs-guest --target wasm32-unknown-unknown --release`. Insn / translate → `cargo build -p speet-recompiler-guest --target wasm32-unknown-unknown --release`. Leave `speet-rtd` running. The next request content-hashes the artifact (`ensure_fresh`); there is no reload tool. Done when the build writes the watched file.
5. **Rerun** `recompile` / `run_guest` (or `recompile` / `run`) on the same binary. Done when the failing bucket is empty **and** `run_guest` matches the expected exit, or the next remaining bucket is reported. Confirm `plugins.recompiler` / `plugins.stubs` hashes in the report changed after the rebuild.

`unsupported_insns` is a coverage signal, not a correctness proof. Empty does not mean the translation is right.

## Commands

| Step | MCP | CLI |
|------|-----|-----|
| pin | `guest_info` `{path}` | `speet-rtdctl info <path>` |
| classify | `recompile` `{path, link?}` then `last_report` | `speet-rtdctl recompile <path>` then `last-report` |
| typed suitability | `analyze_binary` `{path}` | `speet-rtdctl analyze <path>` |
| run | `run_guest` `{path, argv}` | `speet-rtdctl run <path> [-- args…]` |

Resources (MCP only): `speet://report/latest`, `speet://guest/imports`. CLI prints the same JSON on stdout.

## Watched artifacts

Override with `SPEET_RECOMPILER_WASM` / `SPEET_STUBS_WASM`. Defaults:

- `target/wasm32-unknown-unknown/release/speet_recompiler_guest.wasm`
- `target/wasm32-unknown-unknown/release/speet_stubs_guest.wasm`

Design: [recompiler-debug-mcp.md](../../../docs/guides/recompiler-debug-mcp.md).
