# Parallel / Multi-Target API Migration

The build is currently broken (~96 errors across downstream crates) after a refactor of `yecta`'s
`Reactor` internals to support concurrent multi-threaded emission.  This doc describes what broke,
the required API changes, and the migration order.

---

## What Broke and Why

| Change | Old API | New API |
|---|---|---|
| `feed`/`seal` removed | `reactor.feed(ctx, &instr)` | `reactor.feed_to(tail_idx, ctx, &instr)` |
| `jmp` arity | `reactor.jmp(ctx, target, params)` | `reactor.jmp(tail_idx, ctx, target, params)` |
| `local_pool` behind Mutex | `reactor.local_pool.seed_i32(…)` | `reactor.local_pool.lock().seed_i32(…)` |
| `barrier()` removed | `reactor.barrier()` | `reactor.flush_bundles(ctx, tail_idx)` |
| `Pool` gained lifetime | `Pool` | `Pool<'a, Context, E>` |

**Root cause:** `fns` is now behind a `spin::Mutex`-guarded lock with per-entry read/write protocol
so multiple threads can emit into distinct entries concurrently.  `feed`/`seal` implicitly used
`fns.len()-1` as the tail index; that lookup now goes through `lock_global()` and the old
wrappers were removed.

---

## Affected Crates

| Crate | Error categories |
|---|---|
| `speet-ordering` | `feed` (×74), `feed_lazy` arity, `flush_bundles_for_load` arity, `barrier` |
| `speet-link` | `feed`, `seal`, `jmp` arity, `Pool` lifetime/generics |
| `speet-x86_64` | inherits `speet-link` issues; direct `reactor.feed` calls |
| `speet-riscv` | `reactor.local_pool.seed_*` (no `.lock()`), ordering call arity |
| `speet-mips` | same shape as `speet-riscv` |
| `speet-powerpc` | same shape as `speet-x86_64` |

---

## Migration Steps

| Step | Action | Checkpoint |
|---|---|---|
| 1 | Add `feed`, `seal`, `barrier`, `jmp_tail`, `with_local_pool` convenience wrappers to `yecta` | `cargo check -p yecta` clean |
| 2 | Add `NoopHandler` + `StaticPool`/`as_pool()` to `yecta` | no new errors in `speet-link` |
| 3 | Update `speet-ordering`: add `tail_idx` param to all public fns | `cargo check -p speet-ordering` clean |
| 4 | Update `speet-link`: fix `Pool` fields, wire convenience wrappers | `cargo check -p speet-link` clean |
| 5 | Update `speet-x86_64` and `speet-powerpc` | tests pass |
| 6 | Update `speet-riscv` and `speet-mips` | full test suite passes |
| 7 | (Optional) Deprecate `feed`/`seal`/`jmp_tail` wrappers; replace with explicit `tail_idx` everywhere | all crates use explicit tail_idx |
| 8 | Implement parallel `FuncSchedule::execute_parallel` | parallelism benchmark |

---

## Key API Changes to Add in `yecta` (Step 1–2)

**Convenience wrappers** (restore single-threaded ergonomics; pay one `lock_global()` per call):

```rust
// Wrappers on Reactor — sequential, not safe to call from multiple threads concurrently.
pub fn feed(&self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E>
pub fn seal(&self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E>
pub fn jmp_tail(&self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E>
pub fn barrier(&self, ctx: &mut Context) -> Result<(), E>  // replaces removed barrier()
pub fn with_local_pool<R>(&self, f: impl FnOnce(&mut P) -> R) -> R
```

**`StaticPool`** (lets structs store pool config without a lifetime parameter):

```rust
pub struct NoopHandler;  // impl IndirectJumpHandler as no-op
pub type StaticPool = (TableIdx, TypeIdx);
impl StaticPool {
    pub fn as_pool<Context, E>(&self) -> Pool<'static, Context, E> { … }
}
```

---

## Key Changes Per Crate

### `speet-ordering` (Step 3)
Add `tail_idx: usize` as last param to every public function (`emit_store`, `emit_load`,
`emit_fence`, `emit_rmw`, etc.).  Pass through to all reactor calls.

### `speet-link` (Step 4)
- Replace `pub pool: Pool` with `pub pool_table: TableIdx, pub pool_ty: TypeIdx` in `Linker`
  and `ReactorAdapter`.
- `ReactorContext::feed/seal/jmp` impls delegate to the new `reactor.feed/seal/jmp_tail` wrappers.

### `speet-riscv` / `speet-mips` (Step 6)
- `reactor.local_pool.seed_*` → `reactor.with_local_pool(|p| p.seed_*(…))`.
- Capture `tail_idx` once at the top of `translate_instruction` via `reactor.fn_count() - 1`;
  thread through all `speet-ordering` calls.  (~600 lines mechanical change for riscv.)

---

## Parallel Emission (End Goal, Step 8)

Once the single-threaded path is green, parallel emission works by:
1. Pre-allocating slots sequentially via `next_with` (structural lock, single-threaded).
2. Emitting into distinct `tail_idx` values from a thread pool (`feed_to` on distinct entries is safe concurrently).
3. Draining sequentially via `into_fns()`.

`FuncSchedule::execute_parallel` can parallelize the **emission phase** while keeping the two-pass
registration/emit invariant intact (see `AGENTS.md §5`).

---

## Invariants (do not break)
- One WASM function per guest instruction (`AGENTS.md §1`): never collapse multiple instructions into one entry.
- Trap-state in parameters, not locals (`AGENTS.md §2`): `tail_idx` forwarded to `jmp`/`feed_to` must match the live trap-parameter entry.
- Lazy store alias checking (`AGENTS.md §3`): every `feed_lazy` matched by `flush_bundles_for_load` before aliasing loads; `flush_bundles`/`barrier` at every control-flow boundary; same `tail_idx` throughout.
- Two-pass `FuncSchedule` (`AGENTS.md §5`): registration layout must be final before any emission.
