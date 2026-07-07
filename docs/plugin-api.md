# Speet Plugin API — Extending Speet From Outside

This document is the design reference for **extending speet from outside the workspace**: shipping a custom architecture, memory/MMU scheme, indirect-call table strategy, object model, or OS/ABI target as an external — and potentially proprietary or closed-source — plugin, without forking or contributing source into this repository.

It is the external-facing sibling of [recompiler-guide.md](recompiler-guide.md), which documents speet's *internal* architecture for contributors working inside the workspace. Where this document says "mirrors", it means there is a 1:1 correspondence with a trait documented in `recompiler-guide.md` §3 — read that section first if you need the internal-side stack/wire contracts behind each plugin method.

For the short, alignment-oriented version of this document (the version an AI agent should read before touching this code), see [docs/guides/plugin-api.md](guides/plugin-api.md).

---

## 1. Why an external plugin system

Speet has five extension points today, each a Rust generic trait, each only implementable by adding a new crate to this workspace's `[workspace.members]`:

| Extension point | Existing trait | Crate |
|---|---|---|
| Architecture | `Recompile<Context, E, F>` | `crates/os/speet-link-core/src/recompiler.rs` |
| Memory | `AddressMapper<Context, E>` / `MemoryAccess<Context, E>` | `crates/helper/speet-memory/src/mapper.rs` |
| Indirect-call tables | `IndirectJumpHandler<Context, E>` | `crates/helper/yecta/src/lib.rs` |
| Object model | `ObjectModel<C, E>` | `crates/managed/speet-object/src/model.rs` |
| Target (OS/ABI) | `ModuleTarget<Ctx, Err>` + `SyscallTable` | `crates/os/speet-module-target`, `crates/os/speet-syscall` |

This is fine for speet's own maintainers, but has three structural costs for anyone else:

1. **Upstream-or-nothing.** A vendor with a proprietary ISA decoder, a custom MMU/paging scheme, a closed object-model ABI, or an internal OS personality has no way to plug it in without contributing source into this repository (impossible if confidential) or forking the workspace.
2. **Compile-time-only, single-language.** Every extension point above is a static Rust trait impl tied to this workspace's own build. There's no way to ship a `.wasm` blob or a separate executable implementing one of these roles.
3. **No stability boundary.** These traits carry `Context`/`E` generics tied to the *caller's* instantiation. A third party can't target "the trait" in the abstract without monomorphizing against speet's own internal generic parameters.

The plugin API solves this with a **thin, `Context`/`E`-erased trait layer** (`speet-plugin-api`) that mirrors each of the five internal traits one-for-one, plus **three MVP host backends** — in-process, WASM, and subprocess — each of which loads a plugin implementing that erased API and adapts it back into the *existing, unchanged* internal trait. A plugin-backed implementation becomes just another `Arc<dyn ExistingTrait<Context, E>>` from the perspective of every current call site.

This also feeds [container-plan.md](container-plan.md)'s roadmap: a `TargetPlugin`'s data is declarative enough to double as input to a future vkernel's `osctx::OS` impl, without that being required for this system.

---

## 2. Crate layout

```
crates/plugin/
├── speet-plugin-api/              # plugin-facing traits + data types + hand-rolled wire codec.
│                                   # #![no_std] + alloc, zero speet-internal dependencies.
├── speet-plugin-adapter/          # bridges plugin-api traits -> existing internal traits.
│                                   # zero host-crate dependencies.
├── speet-plugin-host/             # PluginTransport seam + PluginRegistry + manifest loader.
├── speet-plugin-host-inproc/      # in-process backend: static linking + dylib mode (feature "dylib").
├── speet-plugin-host-wasm/        # WASM backend: wasmi engine.
└── speet-plugin-host-subprocess/  # subprocess backend: std::process + framed stdio IPC.
```

**Why one crate per host, not one `speet-plugin-hosts`:** each host has a different platform footprint. `speet-plugin-host-wasm` pulls in a WASM engine. `speet-plugin-host-subprocess` needs `std::process`, unavailable on `no_std`/`wasm32` targets. `speet-plugin-host-inproc`'s dylib mode needs `libloading`, only meaningful where dynamic linking exists. An embedder's `Cargo.toml` simply omits the member it can't use, at zero cost to the rest.

`speet-plugin-host` (the `PluginTransport` seam + `PluginRegistry`) has no engine-specific dependencies — only `speet-plugin-api` types and a transport trait object. The backend crates depend on it; it never depends on them. That asymmetry is what lets a 4th host be added later as a new leaf crate without touching `speet-plugin-host` or `speet-plugin-api` (§6).

`speet-plugin-adapter` depends on `speet-plugin-api` plus the internal crates it bridges into (`speet-link-core`, `speet-memory`, `yecta`, `speet-object`, `speet-module-target`, `speet-syscall`, `wasm-encoder`, `wasmparser`). It does **not** depend on any host crate — it consumes `Arc<dyn ArchPlugin>` etc. regardless of which host produced them.

`speet-plugin-api` is `#![no_std]` + `extern crate alloc`, matching `yecta`/`speet-syscall`/`osctx`/`wasm-layout`, specifically so a `.wasm`-guest plugin author writing Rust can target `wasm32-unknown-unknown` without pulling in `std`. Its only dependency beyond `core`/`alloc` is `wasm-encoder` (for `CodeSnippet`'s convenience builder) — never serde, never any other speet-internal generic-trait crate.

---

## 3. The five plugin trait families

**Core invariant, repeated because it is the single most important rule in this system:** no plugin trait in `speet-plugin-api` takes or returns `Context`, `E`, or `F`. Every method either returns plain data, or a `CodeSnippet` — bare WASM instruction bytes built with `wasm-encoder`, no function wrapper, no leading locals vector, no implicit trailing `End`. The host-side adapter decodes a `CodeSnippet` with `wasmparser` and forwards each instruction into the real `dyn InstructionSink<Context, E>`.

```rust
pub struct CodeSnippet { pub wasm: Vec<u8> }
pub struct PluginError { pub code: u32, pub message: String }
pub type PResult<T> = Result<T, PluginError>;
```

Every plugin trait is `Send + Sync` and takes `&self`, not `&mut self` — implementations needing per-call mutable state use a `Mutex`/`RwLock`/atomic, never `Cell`/`RefCell`. Two reasons converge here: trait objects are shared as `Arc<dyn Trait>` so they can be handed out as [host-entity imports](#8-host-entity-imports) to other plugins, and this matches the parallel-API direction already established for internal hook traits (see [parallel-api.md](guides/parallel-api.md)).

### 3.1 `AddressMapperPlugin` / `MemoryAccessPlugin` (mirrors `AddressMapper`/`MemoryAccess`)

```rust
pub trait AddressMapperPlugin: Send + Sync {
    fn translate(&self, addr_local: u32) -> PResult<CodeSnippet>;
    fn declare_params(&self) -> Vec<PluginValType> { Vec::new() }
    fn declare_locals(&self) -> Vec<PluginValType> { Vec::new() }
    fn bind_slots(&self, param_base: u32, local_base: u32) {}
    fn chunk_size(&self) -> Option<u64> { None }
    fn bind_imports(&self, imports: &dyn HostImports) {}
}

pub trait MemoryAccessPlugin: Send + Sync {
    fn emit_load(&self, addr_local: u32, kind: PluginLoadKind) -> PResult<CodeSnippet>;
    fn emit_store_addr(&self, addr_local: u32) -> PResult<CodeSnippet>;
    fn emit_store_insn(&self, phys_addr_local: u32, kind: PluginStoreKind) -> PResult<CodeSnippet>;
    fn emit_memory_size(&self) -> PResult<CodeSnippet> { Err(PluginError::unsupported("emit_memory_size")) }
    fn emit_memory_grow(&self) -> PResult<CodeSnippet> { Err(PluginError::unsupported("emit_memory_grow")) }
    // declare_params / declare_locals / bind_slots / chunk_size / bind_imports, same shape as above
}
```

`translate`'s contract: `addr_local` holds the guest virtual address on entry; the returned snippet must leave the translated physical address on the WASM value stack. A plugin may supply only `AddressMapperPlugin` and let the adapter's built-in load/store sequencing handle the rest, or own the whole sequence via `MemoryAccessPlugin`.

### 3.2 `TablePlugin` (mirrors `IndirectJumpHandler`)

```rust
pub trait TablePlugin: Send + Sync {
    fn indirect_jump(&self, target_local: u32) -> PResult<(PluginIndirectJumpKind, CodeSnippet)>;
    fn bind_imports(&self, imports: &dyn HostImports) {}
}

pub enum PluginIndirectJumpKind { Table(u32), Ref }
```

`target_local` holds the runtime indirect-jump target; the snippet computes whatever table index or `funcref` value the returned `PluginIndirectJumpKind` needs.

### 3.3 `ObjectModelPlugin` (mirrors `ObjectModel`)

```rust
pub trait ObjectModelPlugin: Send + Sync {
    fn ref_val_type(&self) -> PluginValType;
    fn emit_new_object(&self, hash: PluginTypeHash, data_size: u32) -> PResult<CodeSnippet>;
    fn emit_new_array(&self, length_local: u32, elem_hash: PluginTypeHash, dim: u32, elem_bytes: u32) -> PResult<CodeSnippet>;
    fn emit_iget(&self, ref_local: u32, byte_offset: u32, ty: PluginFieldValType) -> PResult<CodeSnippet>;
    fn emit_iput(&self, ref_local: u32, value_local: u32, byte_offset: u32, ty: PluginFieldValType) -> PResult<CodeSnippet>;
    fn emit_aget(&self, ref_local: u32, index_local: u32, ty: PluginFieldValType) -> PResult<CodeSnippet>;
    fn emit_aput(&self, ref_local: u32, index_local: u32, value_local: u32, ty: PluginFieldValType, scratch_i32: u32, scratch_i64: u32) -> PResult<CodeSnippet>;
    fn emit_array_length(&self, ref_local: u32) -> PResult<CodeSnippet>;
    fn emit_instanceof(&self, ref_local: u32, hash: PluginTypeHash, dim: u32, scratch: u32) -> PResult<CodeSnippet>;
    fn emit_check_cast(&self, ref_local: u32, hash: PluginTypeHash, dim: u32, scratch: u32) -> PResult<CodeSnippet>;
    fn bind_imports(&self, imports: &dyn HostImports) {}
}
```

The internal `ObjectModel` trait documents each method's stack contract (e.g. `emit_iget`'s "stack before: `[ref]`") because its caller (the adapter) controls the WASM value stack directly. A plugin has no such access, so every stack-only input becomes an explicit `_local` parameter here — the adapter stages the stack value into one of a fixed set of scratch locals (supplied when the adapter is constructed) before calling the plugin. `PluginTypeHash` mirrors `speet_object::TypeHash` (a SHA3-256 of a class name); `PluginFieldValType` mirrors `speet_object::FieldValType`.

### 3.4 `TargetPlugin` (mirrors `ModuleTarget` + `SyscallTable`/`SyscallEntry`)

```rust
pub trait TargetPlugin: Send + Sync {
    fn module_manifest(&self) -> ModuleManifest;
    fn syscall_table(&self) -> PluginSyscallTable;
    fn bind_imports(&self, imports: &dyn HostImports) {}
}
```

The most data-shaped plugin kind — both methods return plain owned data (no `CodeSnippet`/`PResult` machinery), because the existing, generic `WasmSyscallDispatcher` (already built-in, used by `speet-linux-wasi`) does the actual dispatch codegen from this data. `ModuleManifest` covers func imports, globals, memories, tables, tags, and (passive/active) data and element segments — see `crates/plugin/speet-plugin-api/src/target.rs` for the full field list. `PluginSyscallTable`/`PluginSyscallEntry` mirror `speet_syscall::{SyscallTable, SyscallEntry}` field-for-field, with one twist: `PluginSyscallEntry::import_idx` indexes `ModuleManifest::func_imports` rather than naming an absolute WASM function index, so a plugin never needs the host's global index space.

### 3.5 `ArchPlugin` (mirrors `Recompile`, narrowed — see §3.6)

```rust
pub trait ArchPlugin: Send + Sync {
    fn reset_for_next_binary(&self, args: &[u8]);
    fn count_fns(&self, bytes: &[u8]) -> u32;
    fn step(&self, feedback: Option<&[u8]>) -> PResult<ArchOp>;
    fn declare_params(&self) -> Vec<PluginValType>;
    fn bind_imports(&self, imports: &dyn HostImports) {}
}

pub enum ArchOp {
    OpenFn { len: u32 },
    Feed { snippet: CodeSnippet },
    Jump { target_pc: u64, params: u32 },
    IndirectJump { snippet: CodeSnippet, params: u32 },
    Seal { snippet: CodeSnippet },
    RequestBytes { guest_addr: u64, max_len: u32 },
    Done,
}
```

### 3.6 Why `ArchPlugin` is a command stream, not a single call

Architecture recompilation is a stateful decode loop against `ReactorContext` (open a function, feed instructions, jump, seal, repeat) — not a single request/response. `ArchPlugin::step` drives this loop one `ArchOp` at a time instead of handing the plugin a live `ReactorContext` (which would require leaking `Context`/`E`, violating §3's core invariant). `step` is called repeatedly with `feedback: None` on the first call and `Some(bytes)` once the plugin has emitted `RequestBytes` and the host has fetched them.

This is **deliberately scoped narrower than the full internal `Recompile`/`ReactorContext` surface for v1**: it covers open/feed/jump/indirect-jump/seal/request-more/done, but does not expose `ji`/`EscapeTag`/speculative-call machinery, lazy-store lookahead, or out-of-bounds lookup-stub dispatch. Those stay host-side adapter defaults a plugin cannot directly invoke. Exact `ArchOp` v2 scope (speculative calls, lazy-store lookahead, OOB dispatch) is deferred until a real plugin author needs them.

**Status:** `ArchPluginRecompiler` (`crates/plugin/speet-plugin-adapter/src/arch.rs`) drives a real `dyn ReactorContext` from `ArchPlugin::step` calls, and `crates/os/speet-recompile/src/frontend.rs`'s `RecompilerChoice::Plugin` arm is wired to it — no longer a stub. One deliberate v1 simplification: `Jump`/`IndirectJump` are resolved via explicit `feed` (param forwarding) + `seal_fn` (a manually-encoded `Instruction::ReturnCall`/`ReturnCallIndirect`), **not** via `ReactorContext::jmp`/`ji_with_params`. Native recompilers use the latter, which defers to yecta's predecessor-graph machinery (and can fold/inline single-predecessor jump targets — see `docs/guides/yecta.md`'s function-merging notes); this adapter instead emits fully explicit, self-contained bytes per function, trading away that optimization for a smaller, more auditable code path for a resource kind that had no prior test coverage to validate against. See `arch.rs`'s module doc for the exact rationale and the PC↔`FuncIdx` arithmetic this implies (it *does* include `base_func_offset`, unlike `jmp`'s own relative `FuncIdx`, since a raw `ReturnCall` needs an absolute index).

---

## 4. The adapter crate and call sites

`speet-plugin-adapter` (`crates/plugin/speet-plugin-adapter/`) implements each *existing, unchanged* internal trait in terms of a `Box`/`Arc<dyn XPlugin>`:

| Adapter | Implements (internal trait) | Wraps (plugin trait) |
|---|---|---|
| `crates/plugin/speet-plugin-adapter/src/memory.rs` | `AddressMapper<Context, E>` / `MemoryAccess<Context, E>` | `AddressMapperPlugin` / `MemoryAccessPlugin` |
| `crates/plugin/speet-plugin-adapter/src/table.rs` | `IndirectJumpHandler<Context, E>` | `TablePlugin` |
| `crates/plugin/speet-plugin-adapter/src/object_model.rs` | `ObjectModel<C, E>` | `ObjectModelPlugin` |
| `crates/plugin/speet-plugin-adapter/src/target.rs` | `ModuleTarget<Ctx, Err>` + builds a `SyscallTable` | `TargetPlugin` |
| `crates/plugin/speet-plugin-adapter/src/replay.rs` | shared helper | replays a decoded `CodeSnippet` into any `dyn InstructionSink<Context, E>` |
| `crates/plugin/speet-plugin-adapter/src/reverse.rs` | wraps a *built-in* internal type so it can be granted as a host-entity import (§8) | produces an `Arc<dyn XPlugin>` |

Because each adapter only depends on `speet-plugin-api` + the relevant internal crate (never a host crate), a fully-loaded plugin — regardless of whether it came from in-process static linking, a `.wasm` file, or a subprocess — looks identical to every existing call site: just another `Arc<dyn AddressMapper<Context, E>>` (etc.), set via the same `rc.set_memory_access(...)`-style call that a hand-written internal implementation would use.

`crates/os/speet-recompile/src/frontend.rs`'s `RecompilerChoice` enum is the one call site that changed to accommodate plugins:

```rust
pub enum RecompilerChoice {
    Native(BinArch),
    Plugin(Box<dyn speet_plugin_api::ArchPlugin>),
}
```

`RecompilerChoice::Native(arch)`'s code path and output are byte-for-byte identical to the pre-plugin-API `translate(text, start_addr, arch)` function it replaced — this split added a new variant, it did not change existing behavior.

---

## 5. The three MVP host backends

### 5.1 In-process (`speet-plugin-host-inproc`) — trusted / first-party path

**Static mode** (always available): construct a plugin struct directly and register it — `registry.register_static(name, PluginHandle::arch(Arc::new(my_plugin)))`. This is the only mode where a real Rust trait object crosses any boundary; there is no boundary, since the plugin lives in the same compilation graph as the host. See `crates/plugin/speet-plugin-host-inproc/src/static_mode.rs`.

**Dylib mode** (feature `"dylib"`): a runtime-loaded shared library behind a stable `extern "C"`/`#[repr(C)]` surface — no Rust trait object, `Box<dyn Trait>`, `Vec<T>`, or `String` crosses the boundary, only raw pointers/lengths and `speet_plugin_api::remote::XRequest`/`XResponse` byte payloads (the same self-describing encoding the WASM and subprocess hosts use), so dylib plugins stay binary-stable across `rustc` versions rather than depending on the host and plugin sharing one. See `crates/plugin/speet-plugin-host-inproc/src/dylib/ffi.rs` for the exact symbol contract:

| Symbol (per role — `arch`, `address_mapper`, `memory_access`, `table`, `object_model`, `target`) | Signature |
|---|---|
| `speet_plugin_create_<role>` | `extern "C" fn(imports: HostImportsFfi) -> *mut c_void` |
| `speet_plugin_call_<role>` | `extern "C" fn(handle: *mut c_void, payload_ptr: *const u8, payload_len: usize) -> PluginBuffer` |
| `speet_plugin_free_buffer_<role>` | `extern "C" fn(buf: PluginBuffer)` |
| `speet_plugin_destroy_<role>` | `extern "C" fn(handle: *mut c_void)` |

`PluginBuffer { ptr: *mut u8, len: usize }` carries request/response bytes; `HostImportsFfi { ctx, call, free }` realizes §8's host-entity imports across this boundary, passed once at `create_<role>` time. Buffer ownership follows "the producer frees its own allocation": a buffer `call_<role>` returns was dylib-allocated, so the host frees it via `free_buffer_<role>`; symmetrically, a buffer `HostImportsFfi::call` returns was host-allocated, so the dylib frees it via `HostImportsFfi::free`. Tested in `crates/plugin/speet-plugin-host-inproc/tests/dylib_plugin.rs` against real `cdylib` fixtures compiled on the fly with a bare `rustc` invocation (no shared crate dependency with the host — proving the ABI is genuinely self-describing, not an accidental same-`rustc`-version match), including the host-entity-import mechanism in both the granted and denied cases.

Trust framing: the host process grants an in-process plugin its own full privileges; speet does nothing to contain it.

### 5.2 WASM (`speet-plugin-host-wasm`) — untrusted / sandboxed path, recommended default

Engine: `wasmi` (lighter interpreter, faster cold start, no_std-friendlier) — mirrors `crates/test/speet-e2e`'s `run_module` dual-engine precedent. A `wasmtime`/`"exceptions"` engine for plugins that specifically need the exception-handling proposal is a clean follow-up, not required alongside `wasmi`.

**ABI** (`crates/plugin/speet-plugin-host-wasm/src/abi.rs`) — the exact surface a `.wasm` plugin author (in any language that can target WASM) implements against:

| Guest export | Signature | Purpose |
|---|---|---|
| `memory` | (memory) | linear memory the host reads/writes request/response bytes into |
| `speet_plugin_alloc` | `(len: i32) -> i32` | reserve `len` bytes; returns the pointer |
| `speet_plugin_call` | `(req_ptr: i32, req_len: i32) -> i64` | dispatch one call; request bytes are a `speet_plugin_api::remote::XRequest` encoding (self-describing — the method tag is the request's own leading byte) |
| `speet_plugin_dealloc` *(optional)* | `(ptr: i32, len: i32) -> ()` | best-effort cleanup; skipped if absent |

| Host import (only linked if the manifest declares imports, §8) | Signature |
|---|---|
| `host.speet_host_call` | `(role: i32, name_ptr: i32, name_len: i32, req_ptr: i32, req_len: i32) -> i64` |

Both `i64` return values pack a `(ptr, len)` pair — high 32 bits `ptr`, low 32 bits `len` (`abi::pack`/`abi::unpack`) — deliberately avoiding any dependency on the WASM multi-value proposal for the one place this ABI would otherwise need two return values. The host always writes a response into the *guest's own* memory via the guest's `speet_plugin_alloc`, so allocator ownership never crosses the trust boundary in the other direction. This pointer+length convention mirrors WASI preview1's own `iovs`/`nread` style, already precedented in-tree by `speet-linux-wasi`'s `WasiImports`.

Platform gating: embedding `speet-plugin-host-wasm` inside a `wasm32-unknown-unknown` build is unsupported and fails to compile.

### 5.3 Subprocess (`speet-plugin-host-subprocess`) — trusted-unless-externally-sandboxed path

Built on `std::process::Command` only — no dependency beyond `std`. Framing (`crates/plugin/speet-plugin-host-subprocess/src/frame.rs`):

```
[1 byte version][1 byte FrameKind][4 bytes LE request_id][4 bytes LE payload_len][payload]
```

```rust
enum FrameKind {
    MainRequest = 0,    // host -> plugin: run this call
    MainResponse = 1,   // plugin -> host: result of the most recent MainRequest
    ImportRequest = 2,  // plugin -> host: resolve this host-entity import (§8)
    ImportResponse = 3, // host -> plugin: the result
}
```

`stdin` carries host→plugin frames, `stdout` carries plugin→host frames, `stderr` is left for the plugin's own logs and is never parsed as protocol data. `MainRequest`/`MainResponse` payloads are `speet_plugin_api::remote::XRequest`/`XResponse` encodings (same self-describing method tag as the WASM ABI). Every call is strictly sequential — the host issues one `MainRequest` and does not issue another until it has seen the matching `MainResponse`, interleaved with at most one outstanding `ImportRequest`/`ImportResponse` pair — so `request_id` isn't load-bearing for routing in this MVP, only threaded through for diagnostics and forward compatibility with a future pipelined extension. One subprocess is spawned per `PluginHandle` at `PluginTransport::load`, killed on `Drop`.

Trust framing: speet does **not** sandbox a subprocess plugin itself — treat it as trusted unless the embedder wraps it in OS-level sandboxing (seccomp/Landlock/containers, per [container-plan.md](container-plan.md)'s threat model). This is *not* the path for arbitrary untrusted third parties; that's what WASM hosting is for. Unsupported wherever `std::process` is unavailable (`wasm32-unknown-unknown`, `no_std` embedded) — an embedder simply omits this crate.

---

## 6. The `PluginTransport` seam — adding a 4th host

```rust
pub trait PluginTransport: Send + Sync {
    fn load(&self, descriptor: &dyn Any) -> Result<PluginHandle, PluginLoadError>;
}
```

`PluginHandle` is a type-erased, `PluginKind`-tagged value wrapping an `Arc<dyn XPlugin>` for whichever resource kind this is. To add a 4th host (say, a gRPC-based one reaching a plugin running on another machine):

1. New leaf crate, e.g. `speet-plugin-host-grpc`, depending only on `speet-plugin-api` + `speet-plugin-host` (+ whatever gRPC library).
2. Define a descriptor type (mirrors `WasmDescriptor`/`SubprocessDescriptor`): whatever the new transport needs to connect (an address, credentials, …) plus a role selector and a `HostImports` view for §8.
3. Implement `PluginTransport::load`: connect, then return a `PluginHandle` wrapping an adapter struct that implements the relevant plugin trait by round-tripping calls over the network using the same `speet_plugin_api::remote` request/response types and `wire` codec every other non-static host already uses.
4. Register it: `registry.register_transport("grpc", Box::new(GrpcTransport::new(...)))`, then `registry.load(name, "grpc", &descriptor)`.

No change to `speet-plugin-api`, `speet-plugin-adapter`, or any existing host crate is required — this is the entire point of the seam (see [docs/guides/plugin-api.md](guides/plugin-api.md) §3).

---

## 7. The hand-rolled wire codec

`speet-plugin-api::wire` (`crates/plugin/speet-plugin-api/src/wire.rs`):

```rust
trait WireEncode { fn encode(&self, out: &mut Vec<u8>); }
trait WireDecode: Sized { fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError>; }
```

- Primitives: fixed-width little-endian (`u8/u16/u32/u64/i8/i16/i32/i64/f32/f64`); `bool` as one byte (`0`/`1`).
- `[u8]`/`String`: `u32` LE length prefix + raw bytes (UTF-8 for `String`).
- `Vec<T>`: `u32` LE count + each element's encoding in sequence.
- `Option<T>`: one tag byte (`0`=`None`, `1`=`Some` + value).
- `(A, B)`: `A`'s encoding then `B`'s encoding, no separator.
- Enums: one discriminant byte (matching the type's documented `*Method` tag values where one exists) + variant payload fields in declared order.
- Structs: fields encoded positionally, in declared order — no field names, no self-description. This is **not** forward-compatible field-by-field; that's an accepted v1 tradeoff for implementation simplicity and cross-language portability (a non-Rust plugin author needs only this written spec, not a schema-evolution library).
- `Envelope` (`crates/plugin/speet-plugin-api/src/wire.rs`): `[1 byte protocol_version][1 byte method_tag][4 bytes LE payload length][payload]` — the WASM/dylib request/response shape; the subprocess host wraps this same payload concept in its own `Frame` (§5.3), which additionally carries a direction (`FrameKind`) and `request_id`.
- **Versioning:** `PROTOCOL_VERSION` (currently `1`) bumps on any field add/remove/reorder or enum-variant change. A host should refuse to talk to a plugin advertising an incompatible version rather than silently best-effort it.

`speet_plugin_api::remote` (`crates/plugin/speet-plugin-api/src/remote.rs`) builds on the raw codec: one `XRequest`/`XResponse` enum pair per plugin trait, each variant's wire tag matching that trait's `XMethod` enum discriminant, plus a `dispatch_x(plugin: &dyn XPlugin, req: XRequest) -> XResponse` pure function per kind. Both the WASM and subprocess hosts call the *same* `dispatch_x` functions — the "decode request, call the right trait method, encode the result" logic is written exactly once, not duplicated per transport. This module is also the precise spec a non-Rust plugin author implements against: every `XRequest`/`XResponse` variant and its field order is the wire contract, full stop.

**Worked example — `AddressMapperRequest::Translate`:**

```text
request:  [0x00][addr_local: u32 LE]                      (5 bytes; 0x00 = Translate's tag)
response: [0x00][ok_tag][wasm_len: u32 LE][wasm bytes...]  (Ok variant; ok_tag 0x00 = PResult::Ok)
```

A minimal guest (any language) implementing just this one method only needs: read the request's first byte to confirm it's `0x00`, read 4 bytes at offset 1 as a little-endian `u32` (`addr_local`), and produce that exact response shape.

---

## 8. Host entity imports

A plugin is not always self-contained. An `ArchPlugin` may want to delegate address translation to whatever memory plugin the embedder already configured (built-in or external) instead of reimplementing it; a `TargetPlugin` may want to run against a generic, arch-agnostic memory/table substrate. `speet_plugin_api::imports::HostImports` is the mechanism:

```rust
pub trait HostImports: Send + Sync {
    fn address_mapper(&self, name: &str) -> Option<Arc<dyn AddressMapperPlugin>> { None }
    fn memory_access(&self, name: &str) -> Option<Arc<dyn MemoryAccessPlugin>> { None }
    fn table(&self, name: &str) -> Option<Arc<dyn TablePlugin>> { None }
    fn object_model(&self, name: &str) -> Option<Arc<dyn ObjectModelPlugin>> { None }
    fn target(&self, name: &str) -> Option<Arc<dyn TargetPlugin>> { None }
    fn arch(&self, name: &str) -> Option<Arc<dyn ArchPlugin>> { None }
}
```

`PluginRegistry` implements `HostImports` directly. `PluginRegistry::restricted_view(&granted)` (`crates/plugin/speet-plugin-host/src/registry.rs`) returns a `RestrictedHostImports` that resolves **only** the `(PluginKind, name)` pairs in `granted` — every other name returns `None`, regardless of what else is registered. This restricted view, never the raw registry, is what gets bound to a loaded plugin.

**Reverse adapters** (`crates/plugin/speet-plugin-adapter/src/reverse.rs`) wrap a *built-in* internal implementation (e.g. a concrete `AddressMapper<Context, E>` like `standard_page_table_mapper(...)`) so it can be registered into `PluginRegistry` and granted as an import indistinguishably from an externally-loaded plugin. `Context`/`E` are fixed via closure capture at the point the embedder registers the built-in — the one place outside the embedder's own driver code where a concrete `Context`/`E` is named, deliberately confined there and never leaked into `speet-plugin-api`.

**Per-host realization** (matching `ImportRole`'s six finer-grained roles — `Arch`/`AddressMapper`/`MemoryAccess`/`Table`/`ObjectModel`/`Target` — defined in `speet_plugin_api::remote`, since `PluginKind` deliberately conflates `AddressMapper`/`MemoryAccess` under one `Memory` tag for grant bookkeeping but the wire ABI must address them separately):

- **In-process:** `bind_imports(&self, imports: &dyn HostImports)` hands the plugin a direct `Arc<dyn XPlugin>` — zero overhead, no encoding.
- **WASM:** the guest declares `host.speet_host_call` as an import; the host resolves `(role, name)` against the restricted view and, on success, calls back into the *guest's own* `speet_plugin_alloc` to write the response into guest memory before returning the packed `(ptr, len)`. On failure (denied, unknown name, or malformed request) the host returns the `(0, 0)` sentinel — a well-behaved guest checks for this and returns its own encoded `Err(PluginError)` rather than forwarding garbage. See `crates/plugin/speet-plugin-host-wasm/src/host_calls.rs`.
- **Subprocess:** the plugin emits an `ImportRequest` frame (`[role: u8][name: wire-encoded String][request bytes]`) while a `MainRequest` is outstanding; the host's read loop recognizes it, resolves against the restricted view, and replies with an `ImportResponse` frame before resuming its wait for the original `MainResponse`. See `crates/plugin/speet-plugin-host-subprocess/src/process.rs`.

In every case, the resolvable set is **exactly** the plugin's manifest-declared, embedder-approved allowlist — never an ambient "ask for any name" capability. This is load-bearing specifically for the WASM host: an unlisted import must not be reachable by an untrusted guest.

**Worked example** — an `ArchPlugin` importing a host `AddressMapperPlugin` named `"host-mmu"`:

```rust
// in-process
struct ImportingArch { memory: Mutex<Option<Arc<dyn AddressMapperPlugin>>> }
impl ArchPlugin for ImportingArch {
    fn bind_imports(&self, imports: &dyn HostImports) {
        *self.memory.lock().unwrap() = imports.address_mapper("host-mmu");
    }
    fn step(&self, _feedback: Option<&[u8]>) -> PResult<ArchOp> {
        let mem = self.memory.lock().unwrap();
        let snippet = mem.as_ref().unwrap().translate(0)?;
        Ok(ArchOp::Feed { snippet })
    }
    // ...
}
```

The WASM and subprocess equivalents of this same plugin are exercised end-to-end in `crates/plugin/speet-plugin-host-wasm/tests/wasm_plugin.rs` and `crates/plugin/speet-plugin-host-subprocess/tests/subprocess_plugin.rs` (`host_entity_import_granted_forwards_to_host_plugin` / `host_entity_import_denied_when_not_granted`), including the negative case where an ungranted import resolves to nothing.

---

## 9. Writing a plugin: one worked example per host

All three examples implement the same toy `TargetPlugin` — a reduced `speet-linux-wasi`-equivalent exposing just `proc_exit` and `fd_write` as func imports and returning a fixed `ModuleManifest`/`PluginSyscallTable`. `TargetPlugin`'s two methods take no arguments, which makes it the simplest possible plugin to study first.

**In-process** (`crates/plugin/speet-plugin-host-inproc/src/static_mode.rs`):

```rust
struct ToyTarget;
impl TargetPlugin for ToyTarget {
    fn module_manifest(&self) -> ModuleManifest { /* construct the manifest */ }
    fn syscall_table(&self) -> PluginSyscallTable { /* construct the table */ }
}

let mut registry = PluginRegistry::new();
register_target(&mut registry, "toy", ToyTarget);
let plugin = registry.target("toy").unwrap();
```

**WASM** (see `crates/plugin/speet-plugin-host-wasm/tests/wasm_plugin.rs`, `target_plugin_constant_data_roundtrip`): since `module_manifest`/`syscall_table` take no arguments, the guest's `speet_plugin_call` doesn't need to parse anything — it reads the request's one-byte method tag and returns a pointer into one of two fixed data segments, each holding the relevant `TargetResponse` pre-encoded by the *same* `speet_plugin_api::remote::TargetResponse::encode` a Rust embedder would use:

```rust
let engine = WasmPlugin::load(&wasm_bytes, Arc::new(NoImports))?;
let plugin = WasmTargetPlugin::new(engine);
assert_eq!(plugin.module_manifest(), expected_manifest);
```

**Subprocess** (see `crates/plugin/speet-plugin-host-subprocess/tests/subprocess_plugin.rs`): a child process reads `MainRequest` frames from its stdin and writes `MainResponse` frames to its stdout. For this constant-data plugin, the entire subprocess logic is "read one frame, ignore its payload, write back the precomputed `TargetResponse` bytes for whichever tag was requested" — demonstrably implementable in any language that can read/write its own stdio, which is the whole point of this host.

```rust
let descriptor = SubprocessDescriptor { role: ImportRole::Target, command: cmd, imports: Arc::new(NoImports) };
let plugin = SubprocessTransport.load(&descriptor)?;
```

---

## 10. Keeping this document updated

If you add a new plugin resource kind, a new host backend, or change any wire format described above, update this file and [docs/guides/plugin-api.md](guides/plugin-api.md) in the same change. If a plugin trait's actual behavior in `speet-plugin-api`/`speet-plugin-adapter` diverges from what's written here, trust the code, fix this document, and add a "do not" bullet to the guide if the divergence is something a future agent could plausibly "fix" by reverting it — see [docs/guides/README.md](guides/README.md)'s "Auditing a guide" section, which this document follows by the same logic even though it isn't itself a guide.
