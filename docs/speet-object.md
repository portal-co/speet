# speet-object

**Crate:** `crates/managed/speet-object`  
**Status: Active implementation.**

Pluggable object model for managed-runtime recompilers (DEX/Dalvik, JVM, CLR). Defines the in-memory object header format and provides two implementations: a linear-memory model and a no-op model.

---

## Object header layout

Each heap object in the translated runtime starts with a 36-byte header:

```
[0..4]   type_id       — index into the megabinary's type table
[4..8]   flags         — GC mark bits, lock state, array indicator
[8..16]  vtable_ptr    — pointer to the object's virtual dispatch table
[16..24] class_ptr     — pointer to the class descriptor
[24..32] monitor_ptr   — thin-lock / monitor word
[32..36] reserved
```

The 36-byte size is fixed so the object model can be used across architectures without runtime negotiation.

---

## Implementations

### `LinearMemoryObjects`

Stores objects in a flat WASM linear memory region. Object allocation bumps a pointer; GC is out of scope for the current implementation. Used by `speet-dex` for Dalvik heap objects.

### `NoObjectModel`

A zero-overhead no-op implementation for frontends that translate to WASM GC types or that manage their own object layout externally. Satisfies the `ObjectModel` trait without emitting any header code.

---

## Usage

Architecture frontends that translate managed bytecode (`speet-dex`) use `speet-object` to:
1. Emit object allocation sequences (bump-pointer via `LinearMemoryObjects`)
2. Emit field access sequences using the fixed header offsets
3. Emit virtual dispatch via `vtable_ptr`
