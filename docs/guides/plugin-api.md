# Plugin API Component Guide

**Crates:** `crates/plugin/speet-plugin-api`, `crates/plugin/speet-plugin-adapter`, `crates/plugin/speet-plugin-host`, `crates/plugin/speet-plugin-host-inproc`, `crates/plugin/speet-plugin-host-wasm`, `crates/plugin/speet-plugin-host-subprocess`
**Design doc:** [docs/plugin-api.md](../plugin-api.md)
**Related:** [recompiler-guide.md](../recompiler-guide.md) §3 (the internal traits this layer mirrors), [parallel-api.md](parallel-api.md) (the `Send + Sync` / interior-mutability rule this layer follows from day one)

---

## 1. Why plugins are `Context`/`E`-erased

**Code:** `crates/plugin/speet-plugin-api/src/lib.rs`, `crates/plugin/speet-plugin-api/src/snippet.rs` (`CodeSnippet`)

Every internal extension-point trait (`Recompile<Context, E, F>`, `AddressMapper<Context, E>`, `ObjectModel<C, E>`, …) is generic over the embedder's own instantiation. A plugin compiled outside this workspace — or not in Rust at all — cannot monomorphize against those parameters. `speet-plugin-api`'s traits take/return plain data instead, or a `CodeSnippet` (bare `wasm-encoder`-built instruction bytes, no function wrapper). The host-side adapter (`speet-plugin-adapter`) re-parses a returned `CodeSnippet` with `wasmparser` and forwards each instruction into the real `dyn InstructionSink<Context, E>` — this re-parse is the one bridge point between the erased plugin world and the concrete host world.

**Do not** add a `Context`/`E`/`F` generic parameter to anything in `speet-plugin-api`. **Do not** bypass `CodeSnippet` with a "fast path" that hands a plugin a live `&mut dyn InstructionSink<Context, E>` — that reintroduces exactly the coupling this design exists to avoid, and silently breaks every non-in-process host (a WASM guest or a subprocess has no way to hold a Rust trait object).

---

## 2. The five plugin trait families and what each mirrors

**Code:** `crates/plugin/speet-plugin-api/src/{arch,memory,table,object_model,target}.rs`

| Plugin trait | Mirrors | Internal trait location |
|---|---|---|
| `ArchPlugin` | `Recompile<Context, E, F>` (narrowed to a command-stream protocol, see §4) | `crates/os/speet-link-core/src/recompiler.rs` |
| `AddressMapperPlugin` / `MemoryAccessPlugin` | `AddressMapper<Context, E>` / `MemoryAccess<Context, E>` | `crates/helper/speet-memory/src/mapper.rs` |
| `TablePlugin` | `IndirectJumpHandler<Context, E>` | `crates/helper/yecta/src/lib.rs` |
| `ObjectModelPlugin` | `ObjectModel<C, E>` | `crates/managed/speet-object/src/model.rs` |
| `TargetPlugin` | `ModuleTarget<Ctx, Err>` + `speet_syscall::{SyscallTable, SyscallEntry}` | `crates/os/speet-module-target`, `crates/os/speet-syscall` |

Every method that has a stack-based contract in its internal counterpart (e.g. `ObjectModel::emit_iget`'s "stack before: `[ref]`") gets an explicit `_local` parameter in the plugin version instead — a plugin has no value-stack access, only named locals the adapter stages for it. See `crates/plugin/speet-plugin-api/src/object_model.rs`'s module doc for the full rationale.

---

## 3. The `PluginTransport` seam

**Code:** `crates/plugin/speet-plugin-host/src/transport.rs` (`PluginTransport`, `PluginHandle`), `crates/plugin/speet-plugin-host/src/registry.rs` (`PluginRegistry`)

`speet-plugin-adapter` consumes `Arc<dyn ArchPlugin>` (etc.) trait objects and nothing else — it has no dependency on `speet-plugin-host-inproc`/`-wasm`/`-subprocess`, and never will. Each host backend crate implements one trait, `PluginTransport::load(&dyn Any) -> Result<PluginHandle, PluginLoadError>`, and registers itself into a `PluginRegistry` under a scheme name (`"inproc"`, `"wasm"`, `"subprocess"`). A 4th host (say, a gRPC-based one) is a new leaf crate implementing the same trait — it requires no change to `speet-plugin-api`, `speet-plugin-adapter`, or any existing host crate.

**Do not** call a host backend crate directly from `speet-plugin-adapter` or any arch/resource crate. Always go through `PluginTransport`/`PluginHandle` — otherwise a 4th host can't be added without touching adapter logic, defeating the point of the seam.

---

## 4. The Arch plugin command-stream protocol

**Code:** `crates/plugin/speet-plugin-api/src/arch.rs` (`ArchOp`, `ArchPlugin`)

Unlike the other four resource kinds (each a request/response method call), architecture recompilation is an inherently *stateful decode loop* against `ReactorContext` — open a function, feed instructions, jump, seal, repeat. `ArchPlugin::step` exposes this as a small, closed `ArchOp` enum (`OpenFn`, `Feed`, `Jump`, `IndirectJump`, `Seal`, `RequestBytes`, `Done`) the host drives one step at a time, rather than handing the plugin a live `ReactorContext`. This is deliberately narrower than the full internal API: it does not expose `ji`/`EscapeTag`/speculative-call machinery, lazy-store lookahead, or OOB lookup-stub dispatch — those stay host-side adapter defaults a plugin can't directly invoke in v1 (see `crates/plugin/speet-plugin-api/src/arch.rs`'s module doc).

`RecompilerChoice::Plugin` in `crates/os/speet-recompile/src/frontend.rs` is the activation point for `ArchPlugin`-backed recompilation. As of this writing it is an `unimplemented!()` stub — the `ArchPluginRecompiler` that implements `Recompile<Context, E, F>` in terms of an `ArchPlugin`'s `step()` calls has not landed yet. **This is a known, tracked gap** (see `goals/active.md`), not a removed feature — do not delete the `RecompilerChoice::Plugin` variant or the stub while "cleaning up" unfinished code; the other four resource kinds' adapters (memory, table, object-model, target) are complete and exercised by tests, and `ArchPluginRecompiler` is scoped to land using the same patterns once the command-stream design above is validated against a real plugin.

**Do not** expand `ArchOp`'s vocabulary informally inside a future `ArchPluginRecompiler` without updating `speet-plugin-api::arch::ArchOp` and this guide in the same change — the op vocabulary *is* the wire contract every host (in-process, WASM, subprocess) and every plugin author has to agree on.

---

## 5. Trust model: which host for which plugin author

**Code:** `crates/plugin/speet-plugin-host-wasm/src/lib.rs`, `crates/plugin/speet-plugin-host-subprocess/src/lib.rs`, `crates/plugin/speet-plugin-host-inproc/src/lib.rs`

| Host | Trust level | Use for |
|---|---|---|
| WASM (`speet-plugin-host-wasm`) | **Untrusted / sandboxed.** wasmi/wasmtime confine the guest to its own linear memory and the narrow host-function surface speet exposes. | Arbitrary third-party or proprietary plugins, including ones you've never audited. **Recommended default.** |
| In-process (`speet-plugin-host-inproc`) | Trusted — runs with full host-process privilege, static linking or (feature `"dylib"`) a runtime-loaded shared library. | Code you already trust (internal teams, vetted vendors shipping source or a binary you can audit). |
| Subprocess (`speet-plugin-host-subprocess`) | Trusted-unless-externally-sandboxed — speet does **not** itself sandbox the child process. | Same trust tier as in-process, chosen instead when you want process isolation or a non-Rust implementation language; wrap in OS-level sandboxing (seccomp/Landlock/containers, per `docs/container-plan.md`'s threat model) if you need real isolation. |

**Do not** treat the subprocess or in-process hosts as sandboxes for untrusted plugins — only the WASM host is. Steer external/proprietary plugin authors to WASM by default.

---

## 6. The hand-rolled wire codec

**Code:** `crates/plugin/speet-plugin-api/src/wire.rs`, `crates/plugin/speet-plugin-api/src/remote.rs`

No serde, no bincode, no rkyv — `WireEncode`/`WireDecode` are hand-implemented for every plugin-api type (fixed-width LE primitives, length-prefixed `Vec`/`String`, one discriminant byte per enum variant, fields in declared positional order). This is the same codec used by the WASM host's guest-memory marshalling, the subprocess host's stdio framing, and (once built) the in-process dylib mode's `extern "C"` calls — one payload format, three transports. `speet_plugin_api::remote` builds `XRequest`/`XResponse` wire envelopes and `dispatch_x` functions on top of the raw codec, shared by the WASM and subprocess hosts so neither duplicates the other's "decode request, call trait method, encode response" logic.

Reasons for hand-rolling, not reaching for a library: serde's derive-driven flexibility is the wrong shape for a long-lived, versioned wire contract; bincode's own wire format has gone through unstable churn across versions; rkyv is Rust-only and would rule out non-Rust subprocess plugins, one of this system's explicit goals. `PROTOCOL_VERSION` (`wire.rs`) must be bumped on any field add/remove/reorder or enum-variant change.

**Do not** add serde/bincode/rkyv (or any other serialization crate) to `speet-plugin-api` or any wire-using host crate — the hand-rolled codec is intentional, not a placeholder. **Do not** let a Rust trait object, `Box<dyn Trait>`, `Vec<T>`, or `String` cross the in-process dylib boundary once it's built — only fixed `#[repr(C)]` structs/`extern "C"` signatures and §2.0-encoded byte payloads, so dylib plugins stay binary-stable across `rustc` versions.

---

## 7. Host entity imports (plugins calling back into the host)

**Code:** `crates/plugin/speet-plugin-api/src/imports.rs` (`HostImports`), `crates/plugin/speet-plugin-host/src/registry.rs` (`RestrictedHostImports`, `PluginRegistry::restricted_view`), `crates/plugin/speet-plugin-adapter/src/reverse.rs` (reverse adapters)

A plugin is not always self-contained — an `ArchPlugin` may want to delegate address translation to whatever memory plugin the embedder already configured instead of reimplementing it. `HostImports` gives a plugin named, by-kind lookup of host-resolved entities (built-in, wrapped via a reverse adapter in `speet-plugin-adapter::reverse`, or another loaded plugin — indistinguishable once registered). Realized per host:

- **In-process:** direct `Arc<dyn XPlugin>` handoff via `bind_imports(&self, imports: &dyn HostImports)` — zero overhead.
- **WASM:** an ordinary WASM host-function import (`host.speet_host_call`), registered in the engine's `Linker` before instantiation; the host writes its response into the *guest's own* linear memory via the guest's `speet_plugin_alloc` export, so allocator ownership never crosses the trust boundary. See `crates/plugin/speet-plugin-host-wasm/src/abi.rs`.
- **Subprocess:** a nested `plugin→host` frame issued while a `host→plugin` request is outstanding (a direction tag + request ID on the same framed stdio protocol). See `crates/plugin/speet-plugin-host-subprocess/src/process.rs`.

In every case the resolvable set is **exactly** the plugin's manifest-declared `imports` list, cross-checked against what the embedder is willing to grant (`PluginRegistry::restricted_view`) — never an ambient "ask for any name" capability. This restriction is load-bearing specifically for the WASM host (§5): an unlisted import must not be reachable by an untrusted guest.

**Do not** let a WASM or subprocess plugin resolve a host-entity import its manifest didn't declare and the embedder didn't grant.

---

## Keeping this guide updated

If you add a new plugin resource kind, a new host backend, or change any wire format described above, update this file and [docs/plugin-api.md](../plugin-api.md) in the same change. If a plugin trait's behavior diverges from what's written here, trust the code in `speet-plugin-api`/`speet-plugin-adapter`, fix the doc, and add a bullet to the relevant section above if the divergence is something a future agent could plausibly "fix" by reverting it. See [docs/guides/README.md](README.md) for the general policy this follows.
