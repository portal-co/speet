//! Concrete WASM module builders, accumulator, and complete-module assembler.
//!
//! This crate owns the `MegabinaryBuilder` / `MegabinaryOutput` types (moved
//! from `speet-link`) so that they can implement
//! [`speet_module_target::ModuleTarget`] without running into the orphan rule.
//!
//! ## Primary types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`MegabinaryBuilder<F>`] | Accumulates [`BinaryUnit`]s; deduplicates types; implements [`ModuleTarget`] |
//! | [`MegabinaryOutput<F>`] | Finished output ready for [`assemble`] |
//! | [`ElementsOwned`] | Owned form of `wasm_encoder::Elements<'_>` |
//! | [`assemble`] | Converts `MegabinaryOutput<Function>` → `wasm_encoder::Module` |
//! | [`ModuleBuilder`] | Pipeline-free declaration builder for unit tests |

use std::borrow::Cow;
use std::collections::BTreeMap;

use core::convert::Infallible;

use speet_link::linker::LinkerPlugin;
use speet_link::unit::{BinaryUnit, FuncType};
use speet_module_target::ModuleTarget;
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, Elements, ExportKind, ExportSection,
    FunctionSection, GlobalSection, GlobalType, MemorySection, MemoryType, Module, RefType,
    TableSection, TableType, TagSection, TagType, TypeSection,
};

// ── ElementsOwned ─────────────────────────────────────────────────────────────

/// An owned version of [`Elements`] for storage in [`MegabinaryOutput`].
///
/// `Elements<'a>` carries a lifetime-parameterised [`Cow`], which cannot be
/// stored directly in a struct.  `ElementsOwned` clones the data to remove
/// the lifetime.
#[derive(Clone, Debug)]
pub enum ElementsOwned {
    /// A sequence of function indices.
    Functions(Vec<u32>),
    /// A sequence of reference expressions.
    Expressions(RefType, Vec<ConstExpr>),
}

impl ElementsOwned {
    /// Clone a borrowed [`Elements`] into an owned value.
    pub fn from_elements(e: Elements<'_>) -> Self {
        match e {
            Elements::Functions(cow) => ElementsOwned::Functions(cow.into_owned()),
            Elements::Expressions(rt, cow) => ElementsOwned::Expressions(rt, cow.into_owned()),
        }
    }

    /// Borrow as [`Elements`] for encoding.
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
/// Each field corresponds to one or more WASM sections.  Pass this to
/// [`assemble`] to obtain a finished `wasm_encoder::Module`.
///
/// | Field | WASM section |
/// |-------|-------------|
/// | `types` | TypeSection (deduplicated) |
/// | `func_type_indices` | FunctionSection |
/// | `fns` | CodeSection |
/// | `exports` | ExportSection |
/// | `passive_data` | DataSection (passive) |
/// | `memories` | MemorySection |
/// | `globals` | GlobalSection |
/// | `tables` | TableSection |
/// | `tags` | TagSection |
/// | `active_data` | DataSection (active) |
/// | `passive_elements` | ElementSection (passive) |
/// | `active_elements` | ElementSection (active) |
pub struct MegabinaryOutput<F> {
    pub types: Vec<FuncType>,
    pub func_type_indices: Vec<u32>,
    pub fns: Vec<F>,
    pub exports: Vec<(String, u32)>,
    pub passive_data: Vec<Vec<u8>>,
    pub data_init_fns: Vec<(F, FuncType)>,
    pub globals: Vec<(GlobalType, ConstExpr)>,
    pub memories: Vec<MemoryType>,
    pub tables: Vec<(TableType, Option<ConstExpr>)>,
    pub tags: Vec<TagType>,
    pub active_data: Vec<(u32, ConstExpr, Vec<u8>)>,
    pub passive_elements: Vec<ElementsOwned>,
    pub active_elements: Vec<(u32, ConstExpr, ElementsOwned)>,
}

// ── MegabinaryBuilder ─────────────────────────────────────────────────────────

/// Accumulates [`BinaryUnit`]s and module-level declarations into a single
/// merged output.
///
/// Implements both [`LinkerPlugin<F>`] (for function/type/data accumulation)
/// and [`ModuleTarget<Ctx, Infallible>`] (for global/memory/table/tag
/// declarations).
///
/// ## Type deduplication
///
/// Function types are deduplicated via an internal [`BTreeMap`].  Duplicate
/// types across [`BinaryUnit`]s share a single index in the TypeSection.
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

    fn intern_type(&mut self, ft: FuncType) -> u32 {
        let next_idx = self.types.len() as u32;
        *self.type_map.entry(ft.clone()).or_insert_with(|| {
            self.types.push(ft);
            next_idx
        })
    }

    // --- Direct declaration methods (ctx-free) ---

    pub fn declare_global(&mut self, ty: GlobalType, init: ConstExpr) -> u32 {
        let idx = self.globals.len() as u32;
        self.globals.push((ty, init));
        idx
    }

    pub fn declare_memory(&mut self, ty: MemoryType) -> u32 {
        let idx = self.memories.len() as u32;
        self.memories.push(ty);
        idx
    }

    /// Pass `init = Some(expr)` to use the tables-with-init-value proposal.
    pub fn declare_table(&mut self, ty: TableType, init: Option<ConstExpr>) -> u32 {
        let idx = self.tables.len() as u32;
        self.tables.push((ty, init));
        idx
    }

    pub fn declare_tag(&mut self, ty: TagType) -> u32 {
        let idx = self.tags.len() as u32;
        self.tags.push(ty);
        idx
    }

    pub fn add_memory_data(&mut self, memory_index: u32, offset: ConstExpr, data: Vec<u8>) {
        self.active_data.push((memory_index, offset, data));
    }

    /// Returns the passive segment index.
    pub fn add_passive_memory_data(&mut self, data: Vec<u8>) -> u32 {
        let idx = self.passive_data.len() as u32;
        self.passive_data.push(data);
        idx
    }

    pub fn add_element_segment(
        &mut self,
        table_index: u32,
        offset: ConstExpr,
        elements: ElementsOwned,
    ) {
        self.active_elements.push((table_index, offset, elements));
    }

    /// Returns the passive element segment index.
    pub fn add_passive_element_segment(&mut self, elements: ElementsOwned) -> u32 {
        let idx = self.passive_elements.len() as u32;
        self.passive_elements.push(elements);
        idx
    }

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

// ── LinkerPlugin impl ─────────────────────────────────────────────────────────

impl<F> LinkerPlugin<F> for MegabinaryBuilder<F> {
    fn on_unit(&mut self, unit: BinaryUnit<F>) {
        self.exports.extend(unit.entry_points);
        for (f, ft) in unit.fns.into_iter().zip(unit.func_types) {
            let idx = self.intern_type(ft);
            self.func_type_indices.push(idx);
            self.fns.push(f);
        }
        for seg in unit.data_segments {
            self.passive_data.push(seg.data);
        }
        if let Some(init_fn) = unit.data_init_fn {
            self.data_init_fns.push(init_fn);
        }
    }
}

// ── ModuleTarget impl — generic over Ctx ─────────────────────────────────────

/// `MegabinaryBuilder` implements `ModuleTarget` for any context type.
///
/// The builder operations are infallible (indices are returned by value).
/// The `Ctx` parameter is accepted but ignored.
impl<F, Ctx> ModuleTarget<Ctx, Infallible> for MegabinaryBuilder<F> {
    fn declare_global(
        &mut self,
        _ctx: &mut Ctx,
        ty: GlobalType,
        init: &ConstExpr,
    ) -> Result<u32, Infallible> {
        Ok(self.declare_global(ty, init.clone()))
    }

    fn declare_memory(&mut self, _ctx: &mut Ctx, ty: MemoryType) -> Result<u32, Infallible> {
        Ok(self.declare_memory(ty))
    }

    fn declare_table(
        &mut self,
        _ctx: &mut Ctx,
        ty: TableType,
        init: Option<&ConstExpr>,
    ) -> Result<u32, Infallible> {
        Ok(self.declare_table(ty, init.cloned()))
    }

    fn declare_tag(&mut self, _ctx: &mut Ctx, ty: TagType) -> Result<u32, Infallible> {
        Ok(self.declare_tag(ty))
    }

    fn add_memory_data(
        &mut self,
        _ctx: &mut Ctx,
        memory_index: u32,
        offset: &ConstExpr,
        data: &[u8],
    ) -> Result<(), Infallible> {
        self.add_memory_data(memory_index, offset.clone(), data.to_vec());
        Ok(())
    }

    fn add_passive_memory_data(
        &mut self,
        _ctx: &mut Ctx,
        data: &[u8],
    ) -> Result<u32, Infallible> {
        Ok(self.add_passive_memory_data(data.to_vec()))
    }

    fn add_element_segment(
        &mut self,
        _ctx: &mut Ctx,
        table_index: u32,
        offset: &ConstExpr,
        elements: Elements<'_>,
    ) -> Result<(), Infallible> {
        self.add_element_segment(
            table_index,
            offset.clone(),
            ElementsOwned::from_elements(elements),
        );
        Ok(())
    }

    fn add_passive_element_segment(
        &mut self,
        _ctx: &mut Ctx,
        elements: Elements<'_>,
    ) -> Result<u32, Infallible> {
        Ok(self.add_passive_element_segment(ElementsOwned::from_elements(elements)))
    }
}

// ── assemble ──────────────────────────────────────────────────────────────────

/// Assemble a complete WASM module from a finished [`MegabinaryOutput`].
///
/// Sections are emitted in canonical WASM binary order
/// (verified against wasmparser's `Order` enum):
///
/// ```text
/// TypeSection       ← deduplicated function types (+ data-init fn types)
/// FunctionSection   ← per-function type indices   (+ data-init fns)
/// TableSection      ← table declarations
/// MemorySection     ← memory declarations
/// TagSection        ← exception tag declarations  (between Memory and Global)
/// GlobalSection     ← global declarations
/// ExportSection     ← named function exports + __speet_data_init exports
/// ElementSection    ← passive then active element segments
/// CodeSection       ← function bodies (+ data-init fn bodies)
/// DataSection       ← passive then active data segments
/// ```
pub fn assemble(output: MegabinaryOutput<wasm_encoder::Function>) -> Module {
    let MegabinaryOutput {
        mut types,
        mut func_type_indices,
        mut fns,
        exports,
        passive_data,
        data_init_fns,
        globals,
        memories,
        tables,
        tags,
        active_data,
        passive_elements,
        active_elements,
    } = output;

    // Append data_init_fns, interning their types.
    let data_init_start_idx = fns.len() as u32;
    for (init_fn, ft) in data_init_fns {
        let type_idx = match types.iter().position(|t| t == &ft) {
            Some(i) => i as u32,
            None => {
                let i = types.len() as u32;
                types.push(ft);
                i
            }
        };
        func_type_indices.push(type_idx);
        fns.push(init_fn);
    }
    let data_init_count = fns.len() as u32 - data_init_start_idx;

    // TypeSection
    let mut types_sec = TypeSection::new();
    for ft in &types {
        let params: Vec<_> = ft.params_val_types().collect();
        let results: Vec<_> = ft.results_val_types().collect();
        types_sec.ty().function(params, results);
    }

    // FunctionSection
    let mut funcs_sec = FunctionSection::new();
    for &ti in &func_type_indices {
        funcs_sec.function(ti);
    }

    // TableSection
    let mut table_sec = TableSection::new();
    for (tt, init) in &tables {
        if let Some(expr) = init {
            table_sec.table_with_init(*tt, expr);
        } else {
            table_sec.table(*tt);
        }
    }

    // MemorySection
    let mut mem_sec = MemorySection::new();
    for mt in &memories {
        mem_sec.memory(*mt);
    }

    // TagSection (between Memory and Global per wasmparser Order enum)
    let mut tag_sec = TagSection::new();
    for tt in &tags {
        tag_sec.tag(*tt);
    }

    // GlobalSection
    let mut global_sec = GlobalSection::new();
    for (gt, init) in &globals {
        global_sec.global(*gt, init);
    }

    // ExportSection
    let mut export_sec = ExportSection::new();
    for (name, idx) in &exports {
        export_sec.export(name, ExportKind::Func, *idx);
    }
    for i in 0..data_init_count {
        let abs_idx = data_init_start_idx + i;
        let name = if data_init_count == 1 {
            String::from("__speet_data_init")
        } else {
            format!("__speet_data_init_{}", i)
        };
        export_sec.export(&name, ExportKind::Func, abs_idx);
    }

    // ElementSection
    let mut elem_sec = ElementSection::new();
    for elems in &passive_elements {
        elem_sec.passive(elems.as_elements());
    }
    for (table_idx, offset, elems) in &active_elements {
        elem_sec.active(Some(*table_idx), offset, elems.as_elements());
    }

    // CodeSection
    let mut code_sec = CodeSection::new();
    for f in &fns {
        code_sec.function(f);
    }

    // DataSection
    let mut data_sec = DataSection::new();
    for seg in &passive_data {
        data_sec.passive(seg.iter().copied());
    }
    for (mem_idx, offset, data) in &active_data {
        data_sec.active(*mem_idx, offset, data.iter().copied());
    }

    // Assemble in canonical order:
    // Type → Function → Table → Memory → Tag → Global →
    // Export → Element → Code → Data
    let mut module = Module::new();
    module.section(&types_sec);
    module.section(&funcs_sec);
    if !tables.is_empty() {
        module.section(&table_sec);
    }
    if !memories.is_empty() {
        module.section(&mem_sec);
    }
    if !tags.is_empty() {
        module.section(&tag_sec);
    }
    if !globals.is_empty() {
        module.section(&global_sec);
    }
    module.section(&export_sec);
    if !passive_elements.is_empty() || !active_elements.is_empty() {
        module.section(&elem_sec);
    }
    module.section(&code_sec);
    if !passive_data.is_empty() || !active_data.is_empty() {
        module.section(&data_sec);
    }
    module
}

// ── ModuleBuilder ─────────────────────────────────────────────────────────────

/// Standalone module builder for use without a [`MegabinaryBuilder`] pipeline.
///
/// Implements [`ModuleTarget<Ctx, Infallible>`] and produces a
/// `wasm_encoder::Module` via [`finish`](ModuleBuilder::finish).
///
/// Unlike [`MegabinaryBuilder`], `ModuleBuilder` does not accumulate functions
/// — use it for tests or tooling that constructs module-level declarations only.
#[derive(Default)]
pub struct ModuleBuilder {
    globals:          Vec<(GlobalType, ConstExpr)>,
    memories:         Vec<MemoryType>,
    tables:           Vec<(TableType, Option<ConstExpr>)>,
    tags:             Vec<TagType>,
    active_data:      Vec<(u32, ConstExpr, Vec<u8>)>,
    passive_data:     Vec<Vec<u8>>,
    active_elements:  Vec<(u32, ConstExpr, ElementsOwned)>,
    passive_elements: Vec<ElementsOwned>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Produce a `wasm_encoder::Module` with all declared entities.
    ///
    /// No TypeSection / FunctionSection / CodeSection are emitted — those come
    /// from the function-compilation pipeline ([`MegabinaryBuilder`] +
    /// [`assemble`]).
    pub fn finish(self) -> Module {
        let mut table_sec  = TableSection::new();
        let mut mem_sec    = MemorySection::new();
        let mut tag_sec    = TagSection::new();
        let mut global_sec = GlobalSection::new();
        let mut elem_sec   = ElementSection::new();
        let mut data_sec   = DataSection::new();

        for (tt, init) in &self.tables {
            if let Some(expr) = init {
                table_sec.table_with_init(*tt, expr);
            } else {
                table_sec.table(*tt);
            }
        }
        for mt in &self.memories {
            mem_sec.memory(*mt);
        }
        for tt in &self.tags {
            tag_sec.tag(*tt);
        }
        for (gt, init) in &self.globals {
            global_sec.global(*gt, init);
        }
        for elems in &self.passive_elements {
            elem_sec.passive(elems.as_elements());
        }
        for (table_idx, offset, elems) in &self.active_elements {
            elem_sec.active(Some(*table_idx), offset, elems.as_elements());
        }
        for seg in &self.passive_data {
            data_sec.passive(seg.iter().copied());
        }
        for (mem_idx, offset, data) in &self.active_data {
            data_sec.active(*mem_idx, offset, data.iter().copied());
        }

        let mut module = Module::new();
        if !self.tables.is_empty() {
            module.section(&table_sec);
        }
        if !self.memories.is_empty() {
            module.section(&mem_sec);
        }
        if !self.tags.is_empty() {
            module.section(&tag_sec);
        }
        if !self.globals.is_empty() {
            module.section(&global_sec);
        }
        if !self.passive_elements.is_empty() || !self.active_elements.is_empty() {
            module.section(&elem_sec);
        }
        if !self.passive_data.is_empty() || !self.active_data.is_empty() {
            module.section(&data_sec);
        }
        module
    }
}

impl<Ctx> ModuleTarget<Ctx, Infallible> for ModuleBuilder {
    fn declare_global(
        &mut self,
        _ctx: &mut Ctx,
        ty: GlobalType,
        init: &ConstExpr,
    ) -> Result<u32, Infallible> {
        let idx = self.globals.len() as u32;
        self.globals.push((ty, init.clone()));
        Ok(idx)
    }

    fn declare_memory(&mut self, _ctx: &mut Ctx, ty: MemoryType) -> Result<u32, Infallible> {
        let idx = self.memories.len() as u32;
        self.memories.push(ty);
        Ok(idx)
    }

    fn declare_table(
        &mut self,
        _ctx: &mut Ctx,
        ty: TableType,
        init: Option<&ConstExpr>,
    ) -> Result<u32, Infallible> {
        let idx = self.tables.len() as u32;
        self.tables.push((ty, init.cloned()));
        Ok(idx)
    }

    fn declare_tag(&mut self, _ctx: &mut Ctx, ty: TagType) -> Result<u32, Infallible> {
        let idx = self.tags.len() as u32;
        self.tags.push(ty);
        Ok(idx)
    }

    fn add_memory_data(
        &mut self,
        _ctx: &mut Ctx,
        memory_index: u32,
        offset: &ConstExpr,
        data: &[u8],
    ) -> Result<(), Infallible> {
        self.active_data.push((memory_index, offset.clone(), data.to_vec()));
        Ok(())
    }

    fn add_passive_memory_data(
        &mut self,
        _ctx: &mut Ctx,
        data: &[u8],
    ) -> Result<u32, Infallible> {
        let idx = self.passive_data.len() as u32;
        self.passive_data.push(data.to_vec());
        Ok(idx)
    }

    fn add_element_segment(
        &mut self,
        _ctx: &mut Ctx,
        table_index: u32,
        offset: &ConstExpr,
        elements: Elements<'_>,
    ) -> Result<(), Infallible> {
        self.active_elements.push((
            table_index,
            offset.clone(),
            ElementsOwned::from_elements(elements),
        ));
        Ok(())
    }

    fn add_passive_element_segment(
        &mut self,
        _ctx: &mut Ctx,
        elements: Elements<'_>,
    ) -> Result<u32, Infallible> {
        let idx = self.passive_elements.len() as u32;
        self.passive_elements.push(ElementsOwned::from_elements(elements));
        Ok(idx)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use speet_link::unit::{BinaryUnit, DataSegment, FuncType};
    use wasm_encoder::{Function, Instruction, ValType};

    fn validate(bytes: &[u8]) {
        wasmparser::validate(bytes).expect("wasm validation failed");
    }

    fn empty_fn() -> Function {
        let mut f = Function::new([]);
        f.instruction(&Instruction::End);
        f
    }

    fn dummy_unit(n: usize, base: u32) -> BinaryUnit<Function> {
        let ft = FuncType::from_val_types(&[], &[]);
        BinaryUnit {
            fns: (0..n).map(|_| empty_fn()).collect(),
            base_func_offset: base,
            entry_points: Vec::new(),
            func_types: (0..n).map(|_| ft.clone()).collect(),
            data_segments: Vec::new(),
            data_init_fn: None,
        }
    }

    // ── MegabinaryBuilder type deduplication ──────────────────────────────────

    #[test]
    fn megabinary_type_dedup() {
        let type_a = FuncType::from_val_types(&[ValType::I32], &[]);
        let type_b = FuncType::from_val_types(&[ValType::I64], &[]);

        let mut builder: MegabinaryBuilder<u32> = MegabinaryBuilder::new();

        builder.on_unit(BinaryUnit {
            fns: vec![1u32, 2u32, 3u32],
            base_func_offset: 0,
            entry_points: vec![("foo".to_string(), 0)],
            func_types: vec![type_a.clone(), type_a.clone(), type_b.clone()],
            data_segments: vec![],
            data_init_fn: None,
        });
        builder.on_unit(BinaryUnit {
            fns: vec![4u32, 5u32],
            base_func_offset: 3,
            entry_points: vec![("bar".to_string(), 3)],
            func_types: vec![type_a.clone(), type_b.clone()],
            data_segments: vec![],
            data_init_fn: None,
        });

        let out = builder.finish();
        assert_eq!(out.types.len(), 2);
        assert_eq!(out.fns.len(), 5);

        let idx_a = out.types.iter().position(|t| *t == type_a).unwrap() as u32;
        let idx_b = out.types.iter().position(|t| *t == type_b).unwrap() as u32;
        assert_eq!(out.func_type_indices, [idx_a, idx_a, idx_b, idx_a, idx_b]);
        assert_eq!(out.exports[0].0, "foo");
        assert_eq!(out.exports[1].0, "bar");
    }

    // ── FuncSchedule integration ──────────────────────────────────────────────

    #[test]
    fn func_schedule_execute_collects_units() {
        use speet_link::{FuncSchedule, Linker};
        use yecta::LocalPool;

        type SchedErr = Infallible;

        let mut linker =
            Linker::<(), SchedErr, Function, LocalPool, MegabinaryBuilder<Function>>::with_plugin(
                MegabinaryBuilder::new(),
            );

        let mut schedule: FuncSchedule<(), SchedErr, Function> = FuncSchedule::new();
        schedule.push(2, |_, _| dummy_unit(2, 0));
        schedule.push(3, |_, _| dummy_unit(3, 2));
        schedule.execute(&mut linker, &mut ());

        let out = linker.plugin.finish();
        assert_eq!(out.fns.len(), 5);
    }

    // ── ModuleBuilder declarations ────────────────────────────────────────────

    #[test]
    fn module_builder_declarations() {
        let mut builder = ModuleBuilder::new();

        let g = builder
            .declare_global(
                &mut (),
                GlobalType { val_type: ValType::I32, mutable: true, shared: false },
                &ConstExpr::i32_const(42),
            )
            .unwrap();
        assert_eq!(g, 0);

        let m = builder
            .declare_memory(
                &mut (),
                MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None },
            )
            .unwrap();
        assert_eq!(m, 0);

        let t = builder
            .declare_table(
                &mut (),
                TableType { element_type: RefType::FUNCREF, minimum: 4, maximum: None, table64: false, shared: false },
                None,
            )
            .unwrap();
        assert_eq!(t, 0);

        builder.add_memory_data(&mut (), 0, &ConstExpr::i32_const(0), b"hello").unwrap();
        let ds = builder.add_passive_memory_data(&mut (), b"world").unwrap();
        assert_eq!(ds, 0);

        let es = builder
            .add_passive_element_segment(&mut (), Elements::Functions(Cow::Borrowed(&[])))
            .unwrap();
        assert_eq!(es, 0);

        validate(&builder.finish().finish());
    }

    // ── MegabinaryBuilder as ModuleTarget (generic Ctx) ───────────────────────

    #[test]
    fn megabinary_builder_module_target() {
        let mut builder: MegabinaryBuilder<Function> = MegabinaryBuilder::new();

        // Use &mut String as the context — just to prove it's generic.
        let mut ctx = String::from("ctx");

        <MegabinaryBuilder<Function> as ModuleTarget<String, Infallible>>::declare_global(
            &mut builder,
            &mut ctx,
            GlobalType { val_type: ValType::I64, mutable: false, shared: false },
            &ConstExpr::i64_const(0),
        )
        .unwrap();

        <MegabinaryBuilder<Function> as ModuleTarget<String, Infallible>>::declare_memory(
            &mut builder,
            &mut ctx,
            MemoryType { minimum: 2, maximum: None, memory64: false, shared: false, page_size_log2: None },
        )
        .unwrap();

        builder.on_unit(BinaryUnit {
            fns: vec![empty_fn()],
            base_func_offset: 0,
            entry_points: vec![("main".into(), 0)],
            func_types: vec![FuncType::from_val_types(&[], &[])],
            data_segments: vec![],
            data_init_fn: None,
        });

        let output = builder.finish();
        assert_eq!(output.globals.len(), 1);
        assert_eq!(output.memories.len(), 1);
        assert_eq!(output.fns.len(), 1);

        validate(&assemble(output).finish());
    }

    // ── assemble with data_init_fn ────────────────────────────────────────────

    #[test]
    fn assemble_with_data_init_fn() {
        let mut builder: MegabinaryBuilder<Function> = MegabinaryBuilder::new();

        builder.declare_memory(MemoryType {
            minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None,
        });

        let body_ft = FuncType::from_val_types(&[], &[]);
        let mut init_body = Function::new([]);
        init_body.instruction(&Instruction::End);

        builder.on_unit(BinaryUnit {
            fns: vec![empty_fn()],
            base_func_offset: 0,
            entry_points: vec![],
            func_types: vec![body_ft.clone()],
            data_segments: vec![DataSegment { data: b"test data".to_vec() }],
            data_init_fn: Some((init_body, body_ft)),
        });

        let output = builder.finish();
        assert_eq!(output.passive_data.len(), 1);
        assert_eq!(output.data_init_fns.len(), 1);

        validate(&assemble(output).finish());
    }
}
