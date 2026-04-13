//! [`MegabinaryBuilder`] — accumulates [`BinaryUnit`]s into one merged output.
//!
//! `MegabinaryBuilder<F>` implements [`LinkerPlugin<F>`] and is the default
//! way to build a multi-binary WASM module.  It deduplicates function types
//! across units, keeping a canonical [`TypeSection`]-ready list.
//!
//! Module-level entities (globals, memories, tables, tags) are accumulated
//! via the dedicated `declare_*` / `add_*` methods and propagated into
//! [`MegabinaryOutput`] by [`finish`](MegabinaryBuilder::finish).
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
//! output.memories           → MemorySection
//! output.globals            → GlobalSection
//! output.tables             → TableSection
//! output.tags               → TagSection
//! ```

use alloc::{borrow::Cow, collections::BTreeMap, string::String, vec::Vec};

use wasm_encoder::{ConstExpr, Elements, GlobalType, MemoryType, RefType, TableType, TagType};

use crate::linker::LinkerPlugin;
use crate::unit::{BinaryUnit, FuncType};

// ── ElementsOwned ─────────────────────────────────────────────────────────────

/// An owned version of [`wasm_encoder::Elements`] for storage in
/// [`MegabinaryOutput`].
///
/// [`Elements`] carries a lifetime-parameterised [`Cow`], which cannot be
/// stored directly in a struct without lifetime annotations.  `ElementsOwned`
/// clones the underlying data to remove the lifetime.
#[derive(Clone, Debug)]
pub enum ElementsOwned {
    /// A sequence of function indices.
    Functions(Vec<u32>),
    /// A sequence of reference expressions.
    Expressions(RefType, Vec<ConstExpr>),
}

impl ElementsOwned {
    /// Clone the borrowed [`Elements`] into an owned value.
    pub fn from_elements(e: Elements<'_>) -> Self {
        match e {
            Elements::Functions(cow) => ElementsOwned::Functions(cow.into_owned()),
            Elements::Expressions(rt, cow) => ElementsOwned::Expressions(rt, cow.into_owned()),
        }
    }

    /// Borrow as a [`Elements`] reference suitable for encoding.
    pub fn as_elements(&self) -> Elements<'_> {
        match self {
            ElementsOwned::Functions(v) => Elements::Functions(Cow::Borrowed(v)),
            ElementsOwned::Expressions(rt, v) => Elements::Expressions(*rt, Cow::Borrowed(v)),
        }
    }
}

// ── MegabinaryOutput ──────────────────────────────────────────────────────────

/// The final output of a [`MegabinaryBuilder`].
///
/// Each field corresponds to one or more WASM sections:
///
/// | Field | WASM section |
/// |-------|-------------|
/// | `types` | TypeSection (deduplicated) |
/// | `func_type_indices` | FunctionSection (per-function type index) |
/// | `fns` | CodeSection |
/// | `exports` | ExportSection |
/// | `passive_data` | DataSection (passive segments) |
/// | `memories` | MemorySection |
/// | `globals` | GlobalSection |
/// | `tables` | TableSection |
/// | `tags` | TagSection |
/// | `active_data` | DataSection (active segments) |
/// | `passive_elements` | ElementSection (passive segments) |
/// | `active_elements` | ElementSection (active segments) |
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

    // --- Module-level declarations ---

    /// Global variable declarations: `(type, init_expr)`.
    pub globals: Vec<(GlobalType, ConstExpr)>,
    /// Linear memory declarations.
    pub memories: Vec<MemoryType>,
    /// Table declarations: `(type, optional_init_expr)`.
    pub tables: Vec<(TableType, Option<ConstExpr>)>,
    /// Exception tag declarations.
    pub tags: Vec<TagType>,
    /// Active data segments declared directly (not from `BinaryUnit`s):
    /// `(memory_index, offset_expr, data_bytes)`.
    pub active_data: Vec<(u32, ConstExpr, Vec<u8>)>,
    /// Passive element segments.
    pub passive_elements: Vec<ElementsOwned>,
    /// Active element segments: `(table_index, offset_expr, elements)`.
    pub active_elements: Vec<(u32, ConstExpr, ElementsOwned)>,
}

// ── MegabinaryBuilder ─────────────────────────────────────────────────────────

/// Accumulates [`BinaryUnit`]s incrementally, deduplicating function types.
///
/// Implements [`LinkerPlugin<F>`]; install it on a [`Linker`](crate::linker::Linker)
/// via [`Linker::with_plugin`](crate::linker::Linker::with_plugin).
///
/// Module-level entities (globals, memories, tables, tags, data, elements) are
/// accumulated via the `declare_*` / `add_*` methods and are not part of
/// [`BinaryUnit`].
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

    globals: Vec<(GlobalType, ConstExpr)>,
    memories: Vec<MemoryType>,
    tables: Vec<(TableType, Option<ConstExpr>)>,
    tags: Vec<TagType>,
    active_data: Vec<(u32, ConstExpr, Vec<u8>)>,
    passive_elements: Vec<ElementsOwned>,
    active_elements: Vec<(u32, ConstExpr, ElementsOwned)>,
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
            globals: Vec::new(),
            memories: Vec::new(),
            tables: Vec::new(),
            tags: Vec::new(),
            active_data: Vec::new(),
            passive_elements: Vec::new(),
            active_elements: Vec::new(),
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

    /// Declare a global variable; returns its global index.
    pub fn declare_global(&mut self, ty: GlobalType, init: ConstExpr) -> u32 {
        let idx = self.globals.len() as u32;
        self.globals.push((ty, init));
        idx
    }

    /// Declare a linear memory; returns its memory index.
    pub fn declare_memory(&mut self, ty: MemoryType) -> u32 {
        let idx = self.memories.len() as u32;
        self.memories.push(ty);
        idx
    }

    /// Declare a table; returns its table index.
    ///
    /// Pass `init = Some(expr)` to use the tables-with-init-value proposal.
    pub fn declare_table(&mut self, ty: TableType, init: Option<ConstExpr>) -> u32 {
        let idx = self.tables.len() as u32;
        self.tables.push((ty, init));
        idx
    }

    /// Declare an exception tag; returns its tag index.
    pub fn declare_tag(&mut self, ty: TagType) -> u32 {
        let idx = self.tags.len() as u32;
        self.tags.push(ty);
        idx
    }

    /// Add an active data segment at a fixed address.
    pub fn add_memory_data(&mut self, memory_index: u32, offset: ConstExpr, data: Vec<u8>) {
        self.active_data.push((memory_index, offset, data));
    }

    /// Add a passive data segment; returns its data segment index
    /// (counting all passive segments accumulated so far, including those
    /// from [`BinaryUnit`]s).
    pub fn add_passive_memory_data(&mut self, data: Vec<u8>) -> u32 {
        let idx = (self.passive_data.len()) as u32;
        self.passive_data.push(data);
        idx
    }

    /// Add an active element segment targeting `table_index`.
    pub fn add_element_segment(
        &mut self,
        table_index: u32,
        offset: ConstExpr,
        elements: ElementsOwned,
    ) {
        self.active_elements.push((table_index, offset, elements));
    }

    /// Add a passive element segment; returns its element segment index.
    pub fn add_passive_element_segment(&mut self, elements: ElementsOwned) -> u32 {
        let idx = self.passive_elements.len() as u32;
        self.passive_elements.push(elements);
        idx
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
            globals: self.globals,
            memories: self.memories,
            tables: self.tables,
            tags: self.tags,
            active_data: self.active_data,
            passive_elements: self.passive_elements,
            active_elements: self.active_elements,
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
