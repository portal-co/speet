# Parallel API Guide

**Crates affected:** `crates/helper/yecta`, `crates/helper/speet-ordering`, `crates/os/speet-link`, `crates/native/speet-x86_64`, `crates/native/speet-aarch64`, `crates/native/speet-riscv`, `crates/native/speet-mips`, `crates/native/speet-powerpc`  
**Migration doc:** [parallel-api-migration.md](../parallel-api-migration.md)  
**Active goals:** [goals/active.md](../../goals/active.md)

---

## Direction: interior mutability on hooks, parallel schedule emission

The end goal is to emit multiple WASM functions concurrently across a thread pool, with `FuncSchedule::execute_parallel` dispatching emit closures in parallel. The `yecta` reactor already supports concurrent emission into distinct `tail_idx` slots — `feed_to` on distinct entries is safe concurrently.

The remaining blocker is that plugin/hook traits (`InstructionTrap`, `JumpTrap`, `TrapConfig`) use `&mut self` receivers, which makes emit closures non-`Send`. **The solution is interior mutability, not reversion of the API.**

---

## What was changed and must not be reverted

The `feed`/`seal`/`barrier`/`jmp_tail` refactor changed the reactor's internal `fns` table to be behind a `spin::Mutex`-guarded lock with a per-entry read/write protocol. The old `feed`/`seal` wrappers implicitly used `fns.len()-1` as the tail index; the new API requires an explicit `tail_idx`.

**Do not:**
- Revert `feed_to` back to an implicit tail index
- Simplify `Pool<'a, Context, E>` lifetime generics back to bare references
- Remove `tail_idx` parameters from `speet-ordering` public functions
- Remove the convenience wrappers (`feed`, `seal`, `barrier`, `jmp_tail`, `with_local_pool`) — they restore single-threaded ergonomics but are implemented over the explicit API

---

## Replacing `&mut self` with interior mutability

For each hook/plugin trait implementation:

1. **Counters** (e.g. `RopDetectTrap` depth): use `AtomicU32` / `AtomicI32` with `Relaxed` ordering (counter is per-emission-chain, not shared across threads at the same instant).
2. **Read-heavy shared state**: use `Arc<RwLock<T>>` — many concurrent readers, rare writes.
3. **General mutable state**: use `Mutex<T>` as the safe default.
4. **Verify**: emit closures passed to `FuncSchedule::execute` must become `Send + Sync` after the conversion.

---

## Parallel emission mechanics (end goal)

Once emit closures are `Send`:

1. Pre-allocate all slots sequentially via `next_with` (structural lock, single-threaded).
2. Emit into distinct `tail_idx` values from a thread pool (`feed_to` on distinct entries is safe concurrently).
3. Drain sequentially via `into_fns()`.

The two-pass `FuncSchedule` invariant (see [linker.md](linker.md) §1) is preserved: registration remains single-threaded in Phase 1; only Phase 2 emission is parallelised.

---

## Current status

See [goals/active.md](../../goals/active.md) for the tracked checklist of remaining migration steps.
