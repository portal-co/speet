//! [`MegabinaryBuilder`] — accumulates [`BinaryUnit`]s into one merged output.
//!
//! `MegabinaryBuilder<F>` implements [`LinkerPlugin<F>`] and is the default
//! way to build a multi-binary WASM module.  It deduplicates function types
//! across units, keeping a canonical [`TypeSection`]-ready list.
//!
//! ## Output
//!
//! Call [`finish`](MegabinaryBuilder::finish) to consume the builder and
//! obtain a [`MegabinaryOutput`]:
//!
//! ```text
//! output.types              → TypeSection  (deduplicated)
//! output.func_type_indices  → FunctionSection (per-function type index)
//! output.fns                → CodeSection
//! output.exports            → ExportSection
//! ```

use alloc::{collections::BTreeMap, string::String, vec::Vec};

use crate::linker::LinkerPlugin;
use crate::unit::{BinaryUnit, FuncType};

// ── MegabinaryOutput ──────────────────────────────────────────────────────────

/// The final output of a [`MegabinaryBuilder`].
///
/// Each field corresponds to a WASM section:
///
/// | Field | WASM section |
/// |-------|-------------|
/// | `types` | TypeSection (deduplicated) |
/// | `func_type_indices` | FunctionSection (per-function type index) |
/// | `fns` | CodeSection |
/// | `exports` | ExportSection |
/// | `passive_data` | DataSection (passive segments) |
pub struct MegabinaryOutput<F> {
    /// Deduplicated function types.  Index `i` is used by any function whose
    /// `func_type_indices` entry equals `i`.
    pub types: Vec<FuncType>,
    /// Per-function type index into `types`, parallel to `fns`.
    pub func_type_indices: Vec<u32>,
    /// All compiled WASM functions in order.
    pub fns: Vec<F>,
    /// `(symbol_name, absolute_wasm_func_index)` exports.
    pub exports: Vec<(String, u32)>,
    /// Raw byte blobs for all passive data segments accumulated from every
    /// [`BinaryUnit`].  Emit as passive WASM data segments; physical placement
    /// is performed at WASM runtime by the data-init functions.
    pub passive_data: Vec<Vec<u8>>,
    /// Data-init functions accumulated from every [`BinaryUnit`], one per
    /// unit that produced segments.  Callers may emit these as start
    /// functions or export them as `"__speet_data_init"`.
    pub data_init_fns: Vec<(F, FuncType)>,
}

// ── MegabinaryBuilder ─────────────────────────────────────────────────────────

/// Accumulates [`BinaryUnit`]s incrementally, deduplicating function types.
///
/// Implements [`LinkerPlugin<F>`]; install it on a [`Linker`](crate::linker::Linker)
/// via [`Linker::with_plugin`](crate::linker::Linker::with_plugin).
///
/// ## Type deduplication
///
/// Each new [`FuncType`] is inserted into an internal [`BTreeMap`] keyed by
/// the type itself.  Duplicate types reuse the existing index so the
/// [`TypeSection`] stays compact.
pub struct MegabinaryBuilder<F> {
    type_map: BTreeMap<FuncType, u32>,
    types: Vec<FuncType>,
    func_type_indices: Vec<u32>,
    fns: Vec<F>,
    exports: Vec<(String, u32)>,
    passive_data: Vec<Vec<u8>>,
    data_init_fns: Vec<(F, FuncType)>,
}

impl<F> Default for MegabinaryBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> MegabinaryBuilder<F> {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            type_map: BTreeMap::new(),
            types: Vec::new(),
            func_type_indices: Vec::new(),
            fns: Vec::new(),
            exports: Vec::new(),
            passive_data: Vec::new(),
            data_init_fns: Vec::new(),
        }
    }

    /// Intern a [`FuncType`] and return its deduplicated index.
    fn intern_type(&mut self, ft: FuncType) -> u32 {
        let next_idx = self.types.len() as u32;
        *self.type_map.entry(ft.clone()).or_insert_with(|| {
            self.types.push(ft);
            next_idx
        })
    }

    /// Consume the builder and return the final output.
    pub fn finish(self) -> MegabinaryOutput<F> {
        MegabinaryOutput {
            types: self.types,
            func_type_indices: self.func_type_indices,
            fns: self.fns,
            exports: self.exports,
            passive_data: self.passive_data,
            data_init_fns: self.data_init_fns,
        }
    }
}

impl<F> LinkerPlugin<F> for MegabinaryBuilder<F> {
    fn on_unit(&mut self, unit: BinaryUnit<F>) {
        // Absorb exports.
        self.exports.extend(unit.entry_points);

        // Absorb functions, interning each type.
        for (f, ft) in unit.fns.into_iter().zip(unit.func_types) {
            let idx = self.intern_type(ft);
            self.func_type_indices.push(idx);
            self.fns.push(f);
        }

        // Absorb passive data segments.
        for seg in unit.data_segments {
            self.passive_data.push(seg.data);
        }

        // Accumulate data-init functions.
        if let Some(init_fn) = unit.data_init_fn {
            self.data_init_fns.push(init_fn);
        }
    }
}
