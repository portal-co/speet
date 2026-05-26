# speet-reach

**Crate:** `crates/helper/speet-reach`  
**Status: Active implementation.**

Instruction reachability analysis for selective translation. Given a set of seed program counters, `speet-reach` expands the reachable set transitively by following all static control-flow edges (direct jumps, branches, fall-throughs) until a fixed point.

---

## Purpose

Not all instruction slots in a guest binary need WASM functions. `speet-reach` computes which slots are actually reachable from the binary's known entry points. The reachable count is passed to `FuncSchedule::push` so the linker can omit WASM functions for unreachable slots, reducing output binary size.

See `goals/perf.md` for the completed function-slot-omission goal this enables.

---

## Modules

- `compute` — BFS/DFS expansion from seed PCs; produces a `ReachableSet`
- `edges` — per-ISA control-flow edge extractors (direct branch targets, fall-throughs, call targets)
- `filter` — post-processing: remove slots that are inside multi-byte instructions (valid slot vs. valid instruction start)
- `spec` — reachability specification; defines what counts as a "seed" PC (entry points, exported symbols, exception vectors)
- `pc_slot_map` — mapping between guest PCs and `FuncSchedule` slot indices

---

## Integration

Reachability analysis runs before `FuncSchedule::push`. The output `ReachableSet` is consumed by the architecture frontend to skip emitting bodies for unreachable slots. The frontend still declares the slot count (for index stability) but marks unreachable slots as `unreachable` bodies.
