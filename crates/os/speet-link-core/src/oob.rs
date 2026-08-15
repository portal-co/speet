//! [`OobConfig`] — out-of-bounds jump dispatch configuration.
//!
//! When a recompiler encounters a static jump target outside the compiled set,
//! it calls [`ReactorContext::oob_jump`] instead of emitting `unreachable`.
//! `OobConfig` provides the runtime indices needed for that dispatch.
//!
//! ## Phase 1 — registration
//!
//! ```ignore
//! let oob = OobConfig::register(&mut linker.inner.entity_space);
//! linker.inner.oob_config = Some(oob);
//! ```
//!
//! ## Phase 2 — fill function indices
//!
//! After the `speet-interp` binary unit has been scheduled and its absolute
//! function indices are known, call:
//! ```ignore
//! if let Some(ref mut oob) = linker.inner.oob_config {
//!     oob.set_func_indices(lookup_stub_abs_idx, interp_abs_idx);
//! }
//! ```

use crate::layout::{EntityIndexSpace, IndexSlot};

/// Out-of-bounds jump dispatch configuration.
///
/// All compiled functions, the lookup stub, and the interpreter share the same
/// WASM function type (arch regs + `target_pc: i64` → arch regs), so they
/// can be tail-called interchangeably.
#[derive(Clone, Debug)]
pub struct OobConfig {
    /// Absolute WASM function index of the lookup stub.
    ///
    /// The lookup stub binary-searches the compiled-PC table and tail-calls
    /// the matching compiled function, or falls through to the interpreter.
    pub lookup_stub_func_idx: u32,

    /// Absolute WASM function index of the soft interpreter.
    ///
    /// Executes one guest instruction per iteration; dispatches back to
    /// compiled code via the function table when it reaches a compiled PC.
    pub interp_func_idx: u32,

    /// Pre-declared WASM table slot (registered in Phase 1).
    ///
    /// The element section populates this table with compiled function indices
    /// keyed by their PC offset, enabling `return_call_indirect` dispatch.
    pub dispatch_table_slot: IndexSlot,

    /// Optional dynamic-JIT dispatch tier, layered on top of the static
    /// binary search above. `None` (the default) reproduces today's
    /// interpreter-only OOB behaviour exactly. See [`JitConfig`].
    pub jit: Option<JitConfig>,
}

impl OobConfig {
    /// Register one WASM table in Phase 1.
    ///
    /// Returns a partially-initialised `OobConfig`; call
    /// [`set_func_indices`](Self::set_func_indices) in Phase 2.
    pub fn register(entity_space: &mut EntityIndexSpace) -> Self {
        let dispatch_table_slot = entity_space.tables.append(1);
        Self {
            lookup_stub_func_idx: 0,
            interp_func_idx: 0,
            dispatch_table_slot,
            jit: None,
        }
    }

    /// Set the absolute WASM function indices for the lookup stub and
    /// interpreter once their binary-unit slot bases are known.
    pub fn set_func_indices(&mut self, lookup_stub: u32, interp: u32) {
        self.lookup_stub_func_idx = lookup_stub;
        self.interp_func_idx = interp;
    }

    /// Absolute WASM table index of the dispatch table.
    pub fn table_idx(&self, entity_space: &EntityIndexSpace) -> u32 {
        entity_space.tables.base(self.dispatch_table_slot)
    }
}

/// Dynamic-JIT dispatch tier configuration, layered between `OobConfig`'s
/// static binary search and the interpreter fallback.
///
/// Two host-populated, fixed-capacity side tables live in guest-accessible
/// WASM memory and are scanned linearly by the JIT-aware lookup stub (see
/// `speet_interp::emit_jit_lookup_stub`):
///
/// * **Dynamic dispatch table** (`dyn_table_*`): `(guest_pc: i64, table_slot:
///   i32)` entries, 12 bytes each, `table_slot == -1` marks an empty slot.
///   Populated by a JIT backend once it compiles and links a function for a
///   PC; a hit `return_call_indirect`s into `dyn_dispatch_table_slot`, the
///   runtime-mutable sibling of [`OobConfig::dispatch_table_slot`].
/// * **Hit-count table** (`hit_table_*`): `(guest_pc: i64, count: i32)`
///   entries, 12 bytes each, `count == 0` marks an empty slot (so
///   zero-initialised memory needs no explicit setup, unlike the dynamic
///   table, which must be pre-filled with `table_slot = -1` sentinels).
///   Incremented on every OOB miss of both the static and dynamic tables;
///   crossing `hit_threshold` fires a fire-and-forget call to
///   `jit_request_func_idx`.
///
/// Both tables use bounded **linear** scans rather than hashing — a known
/// scalability tradeoff, adequate for the modest hot-PC-set sizes expected in
/// practice. The lookup stub always falls through to `oob.interp_func_idx`
/// regardless of table state or JIT-request outcome, so a slow or absent JIT
/// backend can never stall or break execution.
/// Which WASM mechanism the dynamic dispatch table's hit path uses to reach
/// a compiled function: the original untyped-table-plus-`return_call_indirect`
/// path, or a typed-funcref-table-plus-`return_call_ref` path (the [WASM
/// typed function references proposal]). Both are fully supported, selected
/// per [`JitConfig`] — `FunctionRef` is not a replacement for
/// `TableIndirect`, and choosing it does not change `TableIndirect`'s
/// emitted bytecode in any way.
///
/// `wasmi` (this codebase's reference/test-oracle engine) has no
/// function-references support, so only `TableIndirect`-mode output can run
/// there; `FunctionRef`-mode output requires `wasmtime` (see
/// `speet-dynamic-jit`'s wasmtime-gated linking path).
///
/// [WASM typed function references proposal]: https://github.com/WebAssembly/function-references
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DispatchMode {
    #[default]
    TableIndirect,
    FunctionRef,
}

#[derive(Clone, Debug)]
pub struct JitConfig {
    /// Pre-declared, runtime-mutable WASM table slot for JIT-compiled
    /// functions — distinct from [`OobConfig::dispatch_table_slot`], which
    /// stays AOT-immutable.
    pub dyn_dispatch_table_slot: IndexSlot,

    /// Which mechanism [`emit_jit_lookup_stub`](crate)'s (actually
    /// `speet_interp::emit_jit_lookup_stub`'s) dynamic-table hit path uses.
    /// Default [`DispatchMode::TableIndirect`] reproduces today's behavior
    /// byte-for-byte.
    pub dispatch_mode: DispatchMode,

    /// Pre-declared, runtime-mutable **typed funcref** WASM table slot,
    /// populated in lockstep with `dyn_dispatch_table_slot` (same slot
    /// index, same occupancy) but holding `(ref null $regfn)` values instead
    /// of untyped `funcref`s, for `DispatchMode::FunctionRef`'s
    /// `table.get`+`return_call_ref` dispatch tail. `None` when
    /// `dispatch_mode` is `TableIndirect` — no second table is registered,
    /// so `TableIndirect`'s entity numbering (and thus its compiled
    /// bytecode) is completely unaffected by `FunctionRef` support existing
    /// in the codebase at all.
    pub dyn_funcref_table_slot: Option<IndexSlot>,

    /// WASM memory index holding the dynamic dispatch side table.
    pub dyn_table_mem_idx: u32,
    /// Byte offset of the dynamic dispatch side table within its memory.
    pub dyn_table_mem_offset: u64,
    /// Number of 12-byte entries reserved in the dynamic dispatch side table.
    pub dyn_table_capacity: u32,

    /// WASM memory index holding the hit-count side table.
    pub hit_table_mem_idx: u32,
    /// Byte offset of the hit-count side table within its memory.
    pub hit_table_mem_offset: u64,
    /// Number of 12-byte entries reserved in the hit-count side table.
    pub hit_table_capacity: u32,

    /// Hit-count threshold at which a fire-and-forget JIT compile request is
    /// issued for a PC. `0` effectively disables JIT requests (the dynamic
    /// table can still be consulted/externally seeded, but the stub never
    /// asks for new compiles).
    pub hit_threshold: u32,

    /// Host import function index called (fire-and-forget, `(pc: i64)`) when
    /// a PC's hit count reaches `hit_threshold`. `None` disables JIT requests
    /// even if `hit_threshold` is nonzero.
    pub jit_request_func_idx: Option<u32>,
}

impl JitConfig {
    /// Register the dynamic dispatch table (and, for
    /// `DispatchMode::FunctionRef`, its typed-funcref sibling table) in
    /// Phase 1.
    ///
    /// Returns a partially-initialised `JitConfig` with all memory/capacity/
    /// threshold fields zeroed and no JIT-request import configured; callers
    /// must fill those in once the corresponding memories and import are
    /// resolved. Passing `DispatchMode::TableIndirect` registers exactly one
    /// table, identical to this function's behavior before `DispatchMode`
    /// existed — `FunctionRef`'s extra table registration only happens when
    /// explicitly requested, so it can never perturb `TableIndirect`
    /// callers' entity numbering.
    pub fn register(entity_space: &mut EntityIndexSpace, dispatch_mode: DispatchMode) -> Self {
        let dyn_dispatch_table_slot = entity_space.tables.append(1);
        let dyn_funcref_table_slot = match dispatch_mode {
            DispatchMode::TableIndirect => None,
            DispatchMode::FunctionRef => Some(entity_space.tables.append(1)),
        };
        Self {
            dyn_dispatch_table_slot,
            dispatch_mode,
            dyn_funcref_table_slot,
            dyn_table_mem_idx: 0,
            dyn_table_mem_offset: 0,
            dyn_table_capacity: 0,
            hit_table_mem_idx: 0,
            hit_table_mem_offset: 0,
            hit_table_capacity: 0,
            hit_threshold: 0,
            jit_request_func_idx: None,
        }
    }

    /// Absolute WASM table index of the dynamic dispatch table.
    pub fn dyn_table_idx(&self, entity_space: &EntityIndexSpace) -> u32 {
        entity_space.tables.base(self.dyn_dispatch_table_slot)
    }

    /// Absolute WASM table index of the typed-funcref sibling table, if
    /// `dispatch_mode` is `FunctionRef`.
    pub fn dyn_funcref_table_idx(&self, entity_space: &EntityIndexSpace) -> Option<u32> {
        self.dyn_funcref_table_slot.map(|slot| entity_space.tables.base(slot))
    }
}
