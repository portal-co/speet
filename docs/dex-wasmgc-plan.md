# DEX Testing + WASMGC Support Plan

Two independent-but-related efforts:

1. **Test `speet-dex`** — it currently has zero tests and is not wired into any integration harness.
2. **Add WASMGC support** to `speet-dex` (DEX→WASM) and `speet-wasm` (WASM→WASM), and give `speet-object` a real WASMGC-backed `ObjectModel` alongside the existing linear-memory one.

Part 2 depends on Part 1 existing (you need a working test harness to trust any WASMGC change), so Part 1 should land first, but the two can be scoped and reviewed separately.

---

## 0. Ground truth (as of this writing)

- `crates/managed/speet-dex/src/lib.rs` (2661 lines): **no `#[test]`, no `#[cfg(test)]`**, and no other crate in the workspace depends on it — it is not exercised by `speet-e2e` or anything else. `DexRecompiler` does implement `Recompile<Context, E, F>` (same trait as native recompilers and `WasmFrontend`), so it is structurally able to plug into the existing e2e harness.
- `crates/managed/speet-object` already defines the pluggable `ObjectModel<C, E>` trait with a forward-looking `ref_val_type() -> ValType` hook (docstring: "`ValType::I32` for linear-memory models; `ValType::Ref(…)` for WASMGC"). Only one real implementation exists today — `LinearMemoryObjects` — plus the `unreachable`-emitting `NoObjectModel` stub. **`docs/speet-object.md` already claims `NoObjectModel` is "for frontends that translate to WASM GC types"** — that's aspirational/stale; the actual code just traps. No WASMGC `ObjectModel` exists yet.
- `DexRecompiler` hardcodes the entire DEX register file as `ValType::I32` locals/params (`lib.rs:712`, `lib.rs:2646`), and `ref_val_type()` is never called anywhere in the crate. Dalvik registers are untyped 32-bit slots reused for ints and object refs interchangeably. **Decided (§4): this stays `ValType::I32` — object refs are `i32` handles into a per-type table, not raw `ValType::Ref` locals.** `ref_val_type()` returns `I32` for every `ObjectModel` impl, including `WasmGcObjects`.
- `crates/managed/speet-wasm` (WASM→WASM transform frontend) has a catch-all pass-through for unrecognized `wasmparser::Operator`s (`Instruction::try_from(other.clone())`, `lib.rs:1439`), and the vendored `wasm-encoder`/`wasmparser` 0.240 already model GC instructions (`StructNew`, `ArrayNew*`, `RefCast*`, `BrOnCast*`, …) unconditionally — no feature flag needed. But:
  - `translate_module`'s type-section handling (`lib.rs:419-436`) only parses `CompositeInnerType::Func`; struct/array composite types are silently replaced with an empty placeholder `FuncType` and **never re-emitted**.
  - `IndexOffsets` (`lib.rs:97`) only has `func`/`global`/`table` fields — there is no `types` offset, and no rewriting of `HeapType::Concrete(idx)` inside ref-typed locals/params/results/globals/tables (`val_type_from_wasmparser`, `lib.rs:1655`, currently collapses every non-func/extern heap type to `funcref` as a fallback). Any guest GC instruction that carries a type index (`struct.new`, `array.new*`, `ref.cast`, `ref.test`, `br_on_cast*`) would pass through with the **guest's own, un-offset type index**, which is silently wrong once merged into a host module with its own type space.
  - This means today, a guest module using WASM-GC would either fail validation or (worse) validate but reference the wrong type — a correctness bug, not just a missing feature.
- The pinned test runtime `wasmi = "1"` (`crates/test/speet-e2e/Cargo.toml`) has **no WASM-GC support at all** (confirmed: no `wasm_gc`/GC-related config in `wasmi-1.1.0` source). The harness already has a second runtime, `wasmtime = "45.0.2"`, used today specifically because `wasmi` can't handle the exception-handling proposal (`run_module_wasmtime`, gated by `is_known_wasmi_exception_gap`, `harness/mod.rs:1240`). This is the exact template to reuse for GC: `wasmtime::Config` has a GC toggle (verify exact method name — recent wasmtime enables WASM-GC by default) vs. `wasmi` which cannot run GC modules at all.
- `goals/arch.md:49` already lists **"WASM-GC frontend — urgent... translates WASM-GC bytecode to the linear-memory megabinary model"**. That is the *opposite* direction from this plan (lowering a GC guest down to linear memory, rather than emitting genuine host WASM-GC objects). Flagging so the two efforts don't collide — this plan is about `speet-dex`/`speet-wasm` being able to **target** WASMGC as an output representation, plus `speet-wasm` correctly **passing through** a guest that already uses WASMGC. Confirm with whoever owns that goals-doc item before starting §4.

---

## Part 1 — Test the DEX frontend

### 1.0 Prerequisite: a way to construct a `DexRecompiler` in a test

`DexRecompiler::new(bytes)` calls `FlatMethods::parse_dex`, which needs a syntactically valid DEX container (header + string/type/proto/field/method tables + class defs) parsed by the external `dex` crate (`dex-parser`, git dependency, read-only — no writer). Even a single-instruction unit test needs a real minimal `.dex` blob; there is currently no in-repo way to produce one. This is the actual first blocker, not a testing nicety.

Two options, not mutually exclusive:

- **(a) Commit pre-built minimal fixtures**, mirroring `test-data/thin-runtime-corpus`'s pattern (committed blobs + a regen script). Source as tiny `.smali` files (or trivial `.java`), built via `d8`/`smali` in a `build.rs` step gated the same way `speet-e2e`'s C corpus already is — skip gracefully when the toolchain is absent (`smoke_c!`'s `c_obj(...)` → `None` → skip pattern, `e2e.rs`).
- **(b) Hand-roll a minimal test-only DEX encoder** (a few hundred lines: header, one string, one type, one proto, one method, one class_def, one code_item) living under `crates/managed/speet-dex/tests/` or a small internal `dex_fixture` module gated `#[cfg(test)]`. No external toolchain dependency, but real effort and it needs to stay byte-format-correct.

**Recommendation:** do (b) first for fast, toolchain-free unit tests of individual instruction translation, and (a) for the richer end-to-end/object-model corpus (real field layouts, multiple classes, inheritance-shaped names) where hand-encoding stops being worth it. Do not block Part 1 on toolchain availability in CI.

### 1.1 Unit tests (in-crate)

Add `#[cfg(test)] mod tests` to `speet-dex/src/lib.rs`, following the shape already used in `speet-wasm/src/lib.rs` (`mod tests` at `lib.rs:1711`, individual `#[test]` fns per behavior). Cover, per DEX instruction family, using the minimal fixture from §1.0:

- Arithmetic/move/const instruction groups (register-only, `NoObjectModel`, no heap).
- Branches (`if-eq`/`goto`/`packed-switch`/`sparse-switch`) — verify structured control flow lowers correctly (this crate's docs note DEX has no computed gotos, unlike native ISAs — good, narrower surface to cover).
- `invoke-*` family (direct/static/virtual/super/interface, plus `-range` variants) — verify `return_call`/param-forwarding wiring.
- Exception instructions (`throw`, `monitor-enter`/`-exit`) — check against `speet-traps`' `InsnClass::PRIVILEGED` classification already present in `insn_class` (`lib.rs:~93`).

### 1.2 Object-model coverage (`LinearMemoryObjects`)

Separate test group, `DexRecompiler::with_model(LinearMemoryObjects { .. })`, covering every `ObjectModel` method DEX actually drives:

- `new-instance` → `emit_new_object`
- `new-array` / `filled-new-array` → `emit_new_array`
- `iget*`/`iput*` → `emit_iget`/`emit_iput` across all `FieldValType` variants (wide/narrow/signed/unsigned/float/double/ref)
- `aget*`/`aput*` → `emit_aget`/`emit_aput`, same type matrix
- `array-length` → `emit_array_length`
- `instance-of` / `check-cast` → `emit_instanceof`/`emit_check_cast`, including the null-passes-checkcast case and the type-mismatch-throws case

Assert on the emitted `wasm_encoder` instruction stream directly for unit-level cases (cheap, precise), and via execution for the integration tier below (real behavior, catches interaction bugs the byte-level assertions can't).

### 1.3 `speet-e2e` integration

Wire `speet-dex` into `crates/test/speet-e2e` the same way `WasmFrontend` already is (`harness/mod.rs` imports `speet_aarch64`, `speet_riscv`, etc. directly). Add a `dex_smoke!`/`dex_run!` macro pair mirroring the existing `smoke!`/`run!` macros in `tests/e2e.rs`:

- `dex_smoke!` — translate a fixture `.dex` → assemble → `wasmparser::validate` only.
- `dex_run!` — translate → validate → execute in `wasmi` (DEX has no exception-handling proposal dependency, so plain `wasmi` should work, unlike the native-ISA EH cases).

Needs a small allocator/runtime shim (bump-pointer `alloc_object_fn`/`alloc_array_fn`/`throw_class_cast_fn`, matching `LinearMemoryObjects`'s documented signatures in `linear.rs:46-59`) supplied as host-imported or synthesized wasm functions in the test module, analogous to how `env_preview1.rs`/`env_thin.rs` supply host imports for the native corpus.

### 1.4 What "done" looks like

- `cargo test -p speet-dex` runs and passes with real assertions (not just "compiles").
- At least one `speet-e2e` DEX fixture executes end-to-end through `wasmi` and produces an expected result (e.g., a method that does object allocation + field access + arithmetic and returns a value checked in the test).
- CI picks these up automatically (no new workflow needed if `cargo test --workspace` is already the CI entry point — verify, don't assume).

---

## Part 2 — WASMGC support

### 2.1 `speet-object`: `WasmGcObjects`, a shared typed-object table

New `ObjectModel<C, E>` implementation alongside `LinearMemoryObjects`. Design, per the corrected direction above (real per-type tables, not a byte-array or an `anyref`-erased shared table):

**The trait gains a type-registration API**, additive to the existing byte-offset accessors (not a breaking change — `LinearMemoryObjects` can no-op it, since byte offsets already fully describe its layout):

- `register_struct(&mut self, fields: &[FieldValType]) -> ObjectTypeId`
- `register_array(&mut self, elem: FieldValType) -> ObjectTypeId`

`ObjectTypeId` lives in **its own disjoint ID space**, owned by the `ObjectModel` implementation — it is *not* `TypeHash` (DEX's SHA3-256 nominal class identity, which keeps being used for `instanceof`/`check-cast`, see below) and *not* a WASM guest module's type-section index. Both frontends keep their own map from their native identifiers into this table:

- `speet-dex`: `TypeHash`/class `type_id` → `ObjectTypeId`, built once per class from `field_map`/`class_sizes` (data it already computes today for `LinearMemoryObjects`'s `data_size`).
- `speet-wasm`: guest struct/array composite-type index → `ObjectTypeId`, built while parsing the guest's `TypeSection` (see §2.2 — this is what replaces raw type-index passthrough).

**Two tiers of table, not one.** A single per-type global table (every instance of a class living forever in one module-level table) is the version that leaks — see §4. The actual design has two tiers, both holding real, concretely-typed refs (never a shared `anyref` table, never a `ref.cast`):

- **Root tables** — one module-level `table (ref null $shape_i)` per registered type, per §4's original description. These back the *root set*: values that are directly reachable from a register, local, or global — i.e., values with no single "owning" object. `emit_new_object`/`emit_new_array` (`struct.new`/`array.new` against the registered composite type) initially land a freshly-allocated instance here and return the resulting **table index as an `i32`** — the value that ends up sitting in a DEX register.
- **Owner-local tables** — every place one object holds a reference to another (a struct field of object type, or an array's elements) is backed by a **mutable `array (mut ref $child_shape)` embedded inside the owner**, not the root table. For a real DEX array (`Foo[]`), this *is* the array's own backing storage — no extra layer, `register_array` already produces exactly this. For a scalar object-typed struct field, the owner gets a small (one-slot-per-such-field, statically sized at `register_struct` time — DEX field counts are fixed per class, so no growth is needed) array acting as that field's local table. Because this is a genuine WASMGC struct/array edge, the **host GC handles collection natively**: once an owner becomes unreachable, everything reachable only through its owner-local tables becomes unreachable too, with zero bookkeeping from us. This is what actually make the design non-leaking for the object graph's bulk (see §4 — root tables still need their own story).

**Loads/stores remap between these two tables at the field/array boundary — this is what keeps the `ObjectModel` API unchanged.** `emit_iget`/`emit_iput` (for `FieldValType::Ref` fields) and `emit_aget`/`emit_aput` (for ref-typed array elements) become the conversion point:

- **Store** (`iput-object`/`aput-object`): the value on the DEX register side is an `i32` root-table handle. `table.get` it from the root table to recover the real ref, then `array.set` that real ref directly into the owner's owner-local table at the field's/index's slot — the value is now reachable purely structurally, no longer needing its root-table entry to stay alive (though nothing yet clears that entry — see §4).
- **Load** (`iget-object`/`aget-object`): `array.get` the real ref out of the owner's owner-local table, then re-insert it into the *child's* root table (fresh or reused slot) and return that index as the `i32` — because the destination (a DEX register) can only speak the root-table `i32` handle representation.

Every `FieldValType::Ref` storage location — struct field or array slot — stays representable as "an `i32` index into *some* table of a specific concrete shape," exactly matching the byte-offset/`FieldValType`-driven shape the trait already has for `LinearMemoryObjects`; only *which* table (root vs. this owner's local one) differs by context, and that's resolved entirely inside `WasmGcObjects`, invisible to `speet-dex`/`speet-wasm` call sites.

**Module-level wiring**: only root tables need to be declared at the module level — analogous to how `LinearMemoryObjects` today takes externally-supplied function indices (`alloc_object_fn`/`alloc_array_fn`/`throw_class_cast_fn`, `linear.rs:60-76`). `WasmGcObjects` will need the same kind of externally-supplied-index wiring for root-table indices, one per registered type, most likely through `EntityIndexSpace`'s existing table entity-space (`docs/guides/linker.md`) rather than a new mechanism — confirm during implementation. Owner-local tables need *no* module-level wiring at all — they're ordinary fields on the registered struct/array type, declared once as part of `register_struct`/`register_array` and otherwise invisible outside `WasmGcObjects`.

**`instanceof`/`check-cast`**: keep the existing simple nominal-hash approach for the first cut. Store `TypeHash` + `array_dim` as the first two fields of every registered struct (same header shape `LinearMemoryObjects` already uses) and compare on read, exactly as today. Real WASM-native `ref.test`/`ref.cast` structural subtyping that mirrors DEX's actual class hierarchy is a genuine improvement but separate, larger work — it needs class-hierarchy modeling `speet-dex` doesn't have today (`hash.rs` — the current model is exact-nominal only, no inheritance/interfaces). Keep it explicitly out of scope here.

### 2.2 `speet-wasm`: correct WASMGC guest pass-through

Independent of §2.1 (this is about *guest* modules that already use WASM-GC, not about DEX's own object representation):

1. **Type section**: extend `Payload::TypeSection` handling (`lib.rs:419-436`) to parse `CompositeInnerType::Struct`/`Array` (and `Cont` if present in this wasmparser version) via `wasmparser`'s `SubType`/`CompositeType`, preserving rec-group boundaries and declared supertypes, and re-emit them into the host module's type section using `wasm_encoder`'s `StructType`/`ArrayType`/`SubType` builders instead of dropping them to a placeholder.
2. **`IndexOffsets`**: add a `types: u32` field. Apply it everywhere a type index currently passes through unmodified — audit needed (not yet done) for at least: `CallIndirect`/`ReturnCallIndirect`'s `type_index` (`lib.rs:1223-1246`, `1254-1272`), block types on `Block`/`Loop`/`If`/`Try`/`TryTable` when they carry `BlockType::FunctionType(idx)`, and every GC instruction operand that is a `HeapType::Concrete(idx)` or bare type index (`struct.new(_default)`, `array.new*`, `ref.cast*`, `ref.test*`, `br_on_cast*`, `call_ref`/`return_call_ref` if present).
3. **`val_type_from_wasmparser`** (`lib.rs:1655`): currently collapses every non-func/extern heap type to `funcref` as a fallback (`lib.rs:1674-1682`) — this silently corrupts any local/param/result/global/table typed with a concrete GC type. Needs a real `HeapType::Concrete(idx)` arm that applies the new `types` offset.
4. Verify whether the pre-existing `CallIndirect` path (`type_index` forwarded raw, no offset, today) is already a latent bug independent of GC, or whether `speet-link-core`'s `EntityIndexSpace` pre-declares types such that guest/host indices already coincide for the current single-frontend-per-build usage. This needs a real read of the linker before assuming either way — don't guess.

### 2.3 `speet-dex`: wiring

Once §2.1 lands, add `DexRecompiler::with_model(WasmGcObjects::new(...))` alongside the existing `with_model(LinearMemoryObjects { .. })`. Register file stays `ValType::I32` per §4 — no change to `speet-dex`'s core translation logic beyond building the class → `ObjectTypeId` map and calling `register_struct` once per class ahead of body translation (the same point where it already computes `class_sizes` today).

### 2.4 Test execution: `wasmtime`, not `wasmi`

`wasmi` cannot run WASM-GC modules at all (§0). Mirror the existing exception-handling split exactly:

- Add a GC-analog of `is_known_wasmi_exception_gap` (or extend it) so `run_module` callers know to fall back.
- Route WASMGC fixtures through `run_module_wasmtime`, after confirming/adding the right `wasmtime::Config` GC toggle (check current `wasmtime = "45.0.2"` API — GC may already be on by default in this version; don't assume the flag name without checking `wasmtime::Config`'s docs for this exact pinned version).
- New `wasmgc_smoke!`/`wasmgc_run!` macros in `speet-e2e`, fixtures under a new `test-data/wasmgc-corpus/` (hand-written `.wat` compiled with `wat2wasm`/`wasm-tools`, since these are small and don't need a full toolchain — cheaper than the DEX fixture problem in §1.0).

---

## Sequencing

1. **Part 1** end to end (§1.0–1.4) — establishes the only way to trust any later DEX change.
2. **§2.4 harness plumbing** (wasmtime GC execution path) — small, useful immediately for validating §2.2 in isolation before touching `speet-dex` at all.
3. **§2.2** (`speet-wasm` GC pass-through, lowered through `ObjectModel`) — self-contained, testable via §2.4.
4. **§2.1** (`WasmGcObjects`: type registration + per-type tables) — the shared piece both frontends drive.
5. **§4's clear-on-replace machinery** (per-register handle/scalar kind tracking in `speet-dex`, refcounted root-table slots in `WasmGcObjects`) — needed before calling `WasmGcObjects` "correct" (a version without it can still be useful for early §2.2/§2.3 testing in the meantime, as long as it's clearly labeled as leaking — don't let that quietly become the shipped version).
6. **§2.3** (wire into `speet-dex`) + DEX-side WASMGC e2e tests (reuse §1.3's harness pattern + §2.4's wasmtime execution path).
7. Real WASM-native `ref.test`/`ref.cast` subtyping (class-hierarchy-aware `instanceof`/`check-cast`) — explicitly out of scope for this plan; revisit only after the above is proven out.

---

## §4 — Register/reference representation: per-type handle tables (decided)

`DexRecompiler` types every DEX register slot as `ValType::I32` (`lib.rs:712`, `lib.rs:2646`) because Dalvik registers are untyped 32-bit cells reused for ints, floats-as-bits, and object refs interchangeably — there is no per-register static type in the bytecode itself. **Decided: this stays as-is.** No register-file redesign, no dataflow-based register retyping. `ref_val_type()` returns `ValType::I32` for every `ObjectModel` impl, `WasmGcObjects` included.

What a register holds when it's conceptually an object ref is an **`i32` handle — a table index into that object's type's root table** (§2.1: every registered `ObjectTypeId` owns a module-level `table (ref null $shape_i)` of its own, not a shared `anyref`/erased table). Allocation (`emit_new_object`/`emit_new_array`) initially lands the new instance here; field/array ops `table.get` that index to recover the real, concretely-typed ref, then operate on it directly — no `ref.cast` needed, since the table's static type already matches. This is what makes the refs *real*: once fetched from its table, a value is a genuine `(ref null $shape_i)` GC reference, not an opaque handle or an `anyref` requiring a downcast.

**Decided: object-graph edges (struct fields, array elements) do not live in the root table at all — they live in owner-local tables (§2.1), genuine WASMGC struct/array edges the host GC walks natively.** This is most of what a naive "one shared per-type table, entries appended forever" design would have leaked: once a value is stored into a field or array slot (`iput-object`/`aput-object`), it's reachable purely structurally through its owner, and an owner becoming unreachable takes everything it (transitively) owns with it — no bookkeeping needed on our side for that part.

**Decided: register-liveness-driven clearing, specifically clear-on-replace.** When a DEX instruction overwrites a register that previously held an object handle, that handle's root-table slot gets cleared (`table.set(idx, ref.null)`) as part of translating that instruction — not at some separately-computed last-use point. This is sound and needs no whole-method liveness pass to find *when* to clear: by definition, once a register is overwritten, nothing can read its old value through that register again, so "clear at the overwrite" and "clear at true last use through this register" are the same point. (Note this rules out the "clear on store into a field/array" idea considered earlier — storing a handle into a field via `iput-object` does not overwrite the *source* register, which may legitimately be read again afterward; only overwriting the register itself is a safe clear point.)

Two things this simple-sounding rule actually requires, both real, scoped work — flag clearly so "decided" doesn't read as "solved":

- **Root-table slots need a refcount, not a bare occupied/free bit.** `move-object` (and its `/from16`/`/16` variants) aliases a handle across two DEX registers pointing at the same root-table slot — extremely common (Dalvik's register allocator uses it constantly, especially shuffling values into `invoke-*`'s fixed argument registers). Clearing unconditionally on replace would null a slot a second, still-live register alias still depends on. So: increment a slot's refcount on any register write that copies an *existing* handle from another register (the `move-object` family), decrement on replace, and only actually `ref.null` the slot when the count reaches zero.
- **Knowing whether a register's *old* value was a handle at all requires a lightweight per-register kind tag, tracked locally during translation — not full register retyping.** DEX registers stay untyped storage (`ValType::I32`, per above) and the raw bytecode doesn't track this either, but DEX opcodes are self-describing about *what they write*: `new-instance`/`new-array`/the `-object` variants of `move`/`iget`/`iput`/`aget`/`aput`/`const-string`/`const-class`/`check-cast`/an object-typed `invoke-*` result all mark their destination register as "currently holds a handle"; plain `move`/arithmetic/etc. mark it "scalar." Propagate this one-bit-per-register tag forward through the instruction stream as `speet-dex` already walks it (register kind at any point is well-defined and consistent at CFG joins, since `javac`/`d8` only ever emits verifier-sound bytecode) — this is materially lighter than the "typed ref register" dataflow pass considered and set aside earlier in this doc, but it is new machinery `speet-dex` doesn't have today, and `WasmGcObjects`'s `emit_iput`/`emit_iget`/etc. call sites need it available to know when to fire the refcount/clear logic at all.

---

## Open questions to close out before/while implementing

- **Root-table entry lifetime (§4)** — mechanism decided (clear-on-replace, refcounted slots), but the two pieces it depends on are not yet built: (1) per-register handle-vs-scalar kind tracking during `speet-dex` translation, and (2) refcounted root-table slots in `WasmGcObjects` (not just occupied/free). Neither exists yet — this is real implementation work, not just wiring.
- Does `cargo test --workspace` already run in CI, or does `speet-dex`/new WASMGC tests need explicit wiring into a workflow file? (Not checked as part of this research pass.)
- Is `EntityIndexSpace` (linker, `docs/guides/linker.md`) already doing type-index renumbering across frontends in a way that changes how §2.1's per-type table indices should be wired, or how §2.2 should register guest types? Read `speet-link-core`'s linker before implementing either.
- Does the pinned `wasmtime = "45.0.2"` enable WASM-GC by default, or does it need an explicit `Config` toggle? Check before writing §2.4.
- `speet-plugin-adapter` also depends on `speet-object` (`Cargo.toml` dependents list) — check whether the new `ObjectModel` type-registration API has knock-on effects there.
- `goals/arch.md:49`'s "WASM-GC frontend... translates WASM-GC bytecode to the linear-memory megabinary model" item is no longer in tension with this plan: §2.2's "lower guest GC ops through `ObjectModel`" means pointing `speet-wasm` at `LinearMemoryObjects` instead of `WasmGcObjects` *is* that goal-doc item, using the same mechanism as genuine WASMGC passthrough. Still worth confirming with whoever owns that item that this is the intended shape.
