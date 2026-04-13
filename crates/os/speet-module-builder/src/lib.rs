//! Concrete WASM module builders and a complete-module assembler.
//!
//! This crate provides three things:
//!
//! 1. **[`MegabinaryModuleTarget`]** — a newtype wrapper around
//!    [`MegabinaryBuilder`] that implements
//!    [`speet_module_target::ModuleTarget`].  Use it when you need both the
//!    [`LinkerPlugin`] pipeline and the `ModuleTarget` trait interface (e.g.
//!    [`ModuleTargetDeclarator::declare_module`]).
//!
//! 2. **[`assemble`]** — converts a completed [`MegabinaryOutput`] into a
//!    finished `wasm_encoder::Module`, assembling all sections in canonical
//!    WASM binary order.
//!
//! 3. **[`ModuleBuilder`]** — a standalone, pipeline-free builder useful for
//!    unit tests that do not go through the full [`Linker`] /
//!    [`MegabinaryBuilder`] flow.

extern crate alloc;

use core::convert::Infallible;
use core::ops::{Deref, DerefMut};

use speet_link::builder::{ElementsOwned, MegabinaryBuilder, MegabinaryOutput};
use speet_link::linker::LinkerPlugin;
use speet_link::unit::BinaryUnit;
use speet_module_target::ModuleTarget;
use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, Elements, ExportKind, ExportSection,
    FunctionSection, GlobalSection, GlobalType, MemorySection, MemoryType, Module, RefType,
    TableSection, TableType, TagSection, TagType, TypeSection,
};

// ── MegabinaryModuleTarget ────────────────────────────────────────────────────

/// A newtype wrapper around [`MegabinaryBuilder`] that also implements
/// [`ModuleTarget<(), Infallible>`].
///
/// The orphan rule prevents `impl ModuleTarget for MegabinaryBuilder` directly
/// (both types live in external crates).  Wrapping in this local newtype
/// resolves the conflict while keeping full access to the inner builder via
/// [`Deref`] / [`DerefMut`].
///
/// ## Usage
///
/// ```ignore
/// let mut target = MegabinaryModuleTarget::new();
/// // Drive module-level declarations through the trait.
/// declarator.declare_module(&mut target).unwrap();
/// // Feed BinaryUnits as normal.
/// target.on_unit(unit);
/// // Assemble the final module.
/// let module = assemble(target.finish());
/// ```
pub struct MegabinaryModuleTarget<F>(pub MegabinaryBuilder<F>);

impl<F> MegabinaryModuleTarget<F> {
    pub fn new() -> Self {
        Self(MegabinaryBuilder::new())
    }

    pub fn into_inner(self) -> MegabinaryBuilder<F> {
        self.0
    }

    pub fn finish(self) -> MegabinaryOutput<F> {
        self.0.finish()
    }
}

impl<F> Default for MegabinaryModuleTarget<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Deref for MegabinaryModuleTarget<F> {
    type Target = MegabinaryBuilder<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> DerefMut for MegabinaryModuleTarget<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F> LinkerPlugin<F> for MegabinaryModuleTarget<F> {
    fn on_unit(&mut self, unit: BinaryUnit<F>) {
        self.0.on_unit(unit);
    }
}

impl<F> ModuleTarget<(), Infallible> for MegabinaryModuleTarget<F> {
    fn declare_global(
        &mut self,
        _ctx: &mut (),
        ty: GlobalType,
        init: &ConstExpr,
    ) -> Result<u32, Infallible> {
        Ok(self.0.declare_global(ty, init.clone()))
    }

    fn declare_memory(&mut self, _ctx: &mut (), ty: MemoryType) -> Result<u32, Infallible> {
        Ok(self.0.declare_memory(ty))
    }

    fn declare_table(
        &mut self,
        _ctx: &mut (),
        ty: TableType,
        init: Option<&ConstExpr>,
    ) -> Result<u32, Infallible> {
        Ok(self.0.declare_table(ty, init.cloned()))
    }

    fn declare_tag(&mut self, _ctx: &mut (), ty: TagType) -> Result<u32, Infallible> {
        Ok(self.0.declare_tag(ty))
    }

    fn add_memory_data(
        &mut self,
        _ctx: &mut (),
        memory_index: u32,
        offset: &ConstExpr,
        data: &[u8],
    ) -> Result<(), Infallible> {
        self.0.add_memory_data(memory_index, offset.clone(), data.to_vec());
        Ok(())
    }

    fn add_passive_memory_data(
        &mut self,
        _ctx: &mut (),
        data: &[u8],
    ) -> Result<u32, Infallible> {
        Ok(self.0.add_passive_memory_data(data.to_vec()))
    }

    fn add_element_segment(
        &mut self,
        _ctx: &mut (),
        table_index: u32,
        offset: &ConstExpr,
        elements: Elements<'_>,
    ) -> Result<(), Infallible> {
        self.0.add_element_segment(table_index, offset.clone(), ElementsOwned::from_elements(elements));
        Ok(())
    }

    fn add_passive_element_segment(
        &mut self,
        _ctx: &mut (),
        elements: Elements<'_>,
    ) -> Result<u32, Infallible> {
        Ok(self.0.add_passive_element_segment(ElementsOwned::from_elements(elements)))
    }
}

// ── assemble ──────────────────────────────────────────────────────────────────

/// Assemble a complete WASM module from a finished [`MegabinaryOutput`].
///
/// Sections are emitted in canonical WASM binary order:
///
/// ```text
/// TypeSection       ← deduplicated function types (+ data-init fn types)
/// FunctionSection   ← per-function type indices   (+ data-init fns)
/// MemorySection     ← memory declarations
/// TableSection      ← table declarations
/// GlobalSection     ← global declarations
/// TagSection        ← exception tag declarations
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

    // Append data_init_fns to the function list, interning their types.
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

    // --- TypeSection ---
    let mut types_sec = TypeSection::new();
    for ft in &types {
        let params: alloc::vec::Vec<_> = ft.params_val_types().collect();
        let results: alloc::vec::Vec<_> = ft.results_val_types().collect();
        types_sec.ty().function(params, results);
    }

    // --- FunctionSection ---
    let mut funcs_sec = FunctionSection::new();
    for &ti in &func_type_indices {
        funcs_sec.function(ti);
    }

    // --- MemorySection ---
    let mut mem_sec = MemorySection::new();
    for mt in &memories {
        mem_sec.memory(*mt);
    }

    // --- TableSection ---
    let mut table_sec = TableSection::new();
    for (tt, init) in &tables {
        if let Some(expr) = init {
            table_sec.table_with_init(*tt, expr);
        } else {
            table_sec.table(*tt);
        }
    }

    // --- GlobalSection ---
    let mut global_sec = GlobalSection::new();
    for (gt, init) in &globals {
        global_sec.global(*gt, init);
    }

    // --- TagSection ---
    let mut tag_sec = TagSection::new();
    for tt in &tags {
        tag_sec.tag(*tt);
    }

    // --- ExportSection ---
    let mut export_sec = ExportSection::new();
    for (name, idx) in &exports {
        export_sec.export(name, ExportKind::Func, *idx);
    }
    for i in 0..data_init_count {
        let abs_idx = data_init_start_idx + i;
        let name = if data_init_count == 1 {
            alloc::string::String::from("__speet_data_init")
        } else {
            alloc::format!("__speet_data_init_{}", i)
        };
        export_sec.export(&name, ExportKind::Func, abs_idx);
    }

    // --- ElementSection ---
    let mut elem_sec = ElementSection::new();
    for elems in &passive_elements {
        elem_sec.passive(elems.as_elements());
    }
    for (table_idx, offset, elems) in &active_elements {
        elem_sec.active(Some(*table_idx), offset, elems.as_elements());
    }

    // --- CodeSection ---
    let mut code_sec = CodeSection::new();
    for f in &fns {
        code_sec.function(f);
    }

    // --- DataSection ---
    let mut data_sec = DataSection::new();
    for seg in &passive_data {
        data_sec.passive(seg.iter().copied());
    }
    for (mem_idx, offset, data) in &active_data {
        data_sec.active(*mem_idx, offset, data.iter().copied());
    }

    // Assemble in canonical section order.
    let mut module = Module::new();
    // Canonical section order per wasmparser:
    // Type → Import → Function → Table → Memory → Tag → Global →
    // Export → Start → Element → DataCount → Code → Data
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
/// Implements [`ModuleTarget<(), Infallible>`] and produces a
/// `wasm_encoder::Module` via [`finish`](ModuleBuilder::finish).
///
/// Unlike [`MegabinaryBuilder`], `ModuleBuilder` does not handle function
/// compilation — use it for tests or tooling that constructs module-level
/// declarations only.
pub struct ModuleBuilder {
    globals:          alloc::vec::Vec<(GlobalType, ConstExpr)>,
    memories:         alloc::vec::Vec<MemoryType>,
    tables:           alloc::vec::Vec<(TableType, Option<ConstExpr>)>,
    tags:             alloc::vec::Vec<TagType>,
    active_data:      alloc::vec::Vec<(u32, ConstExpr, alloc::vec::Vec<u8>)>,
    passive_data:     alloc::vec::Vec<alloc::vec::Vec<u8>>,
    active_elements:  alloc::vec::Vec<(u32, ConstExpr, ElementsOwned)>,
    passive_elements: alloc::vec::Vec<ElementsOwned>,
}

impl Default for ModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            globals:          alloc::vec::Vec::new(),
            memories:         alloc::vec::Vec::new(),
            tables:           alloc::vec::Vec::new(),
            tags:             alloc::vec::Vec::new(),
            active_data:      alloc::vec::Vec::new(),
            passive_data:     alloc::vec::Vec::new(),
            active_elements:  alloc::vec::Vec::new(),
            passive_elements: alloc::vec::Vec::new(),
        }
    }

    /// Produce a `wasm_encoder::Module` with all declared entities.
    ///
    /// No TypeSection / FunctionSection / CodeSection are emitted — those come
    /// from the function-compilation pipeline ([`MegabinaryModuleTarget`] +
    /// [`assemble`]).
    pub fn finish(self) -> Module {
        let mut mem_sec    = MemorySection::new();
        let mut table_sec  = TableSection::new();
        let mut global_sec = GlobalSection::new();
        let mut tag_sec    = TagSection::new();
        let mut elem_sec   = ElementSection::new();
        let mut data_sec   = DataSection::new();

        for mt in &self.memories {
            mem_sec.memory(*mt);
        }
        for (tt, init) in &self.tables {
            if let Some(expr) = init {
                table_sec.table_with_init(*tt, expr);
            } else {
                table_sec.table(*tt);
            }
        }
        for (gt, init) in &self.globals {
            global_sec.global(*gt, init);
        }
        for tt in &self.tags {
            tag_sec.tag(*tt);
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

        // Canonical section order: Table → Memory → Tag → Global → Element → Data
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

impl ModuleTarget<(), Infallible> for ModuleBuilder {
    fn declare_global(
        &mut self,
        _ctx: &mut (),
        ty: GlobalType,
        init: &ConstExpr,
    ) -> Result<u32, Infallible> {
        let idx = self.globals.len() as u32;
        self.globals.push((ty, init.clone()));
        Ok(idx)
    }

    fn declare_memory(&mut self, _ctx: &mut (), ty: MemoryType) -> Result<u32, Infallible> {
        let idx = self.memories.len() as u32;
        self.memories.push(ty);
        Ok(idx)
    }

    fn declare_table(
        &mut self,
        _ctx: &mut (),
        ty: TableType,
        init: Option<&ConstExpr>,
    ) -> Result<u32, Infallible> {
        let idx = self.tables.len() as u32;
        self.tables.push((ty, init.cloned()));
        Ok(idx)
    }

    fn declare_tag(&mut self, _ctx: &mut (), ty: TagType) -> Result<u32, Infallible> {
        let idx = self.tags.len() as u32;
        self.tags.push(ty);
        Ok(idx)
    }

    fn add_memory_data(
        &mut self,
        _ctx: &mut (),
        memory_index: u32,
        offset: &ConstExpr,
        data: &[u8],
    ) -> Result<(), Infallible> {
        self.active_data.push((memory_index, offset.clone(), data.to_vec()));
        Ok(())
    }

    fn add_passive_memory_data(
        &mut self,
        _ctx: &mut (),
        data: &[u8],
    ) -> Result<u32, Infallible> {
        let idx = self.passive_data.len() as u32;
        self.passive_data.push(data.to_vec());
        Ok(idx)
    }

    fn add_element_segment(
        &mut self,
        _ctx: &mut (),
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
        _ctx: &mut (),
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
    use speet_link::unit::{BinaryUnit, DataSegment};
    use wasm_encoder::{Function, Instruction, TagKind, ValType};

    fn validate(bytes: &[u8]) {
        wasmparser::validate(bytes).expect("wasm validation failed");
    }

    fn empty_fn() -> Function {
        let mut f = Function::new([]);
        f.instruction(&Instruction::End);
        f
    }

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

        // Empty passive element segment (no functions to reference in a declaration-only module).
        use alloc::borrow::Cow;
        let es = builder
            .add_passive_element_segment(&mut (), Elements::Functions(Cow::Borrowed(&[])))
            .unwrap();
        assert_eq!(es, 0);

        let bytes = builder.finish().finish();
        validate(&bytes);
    }

    #[test]
    fn megabinary_module_target_with_linker_plugin() {
        let mut target: MegabinaryModuleTarget<Function> = MegabinaryModuleTarget::new();

        // Declare module-level entities via the ModuleTarget trait.
        target
            .declare_global(
                &mut (),
                GlobalType { val_type: ValType::I64, mutable: false, shared: false },
                &ConstExpr::i64_const(0),
            )
            .unwrap();

        target
            .declare_memory(
                &mut (),
                MemoryType { minimum: 2, maximum: None, memory64: false, shared: false, page_size_log2: None },
            )
            .unwrap();

        // Feed a BinaryUnit.
        let ft = speet_link::unit::FuncType::from_val_types(&[], &[]);
        target.on_unit(BinaryUnit {
            fns: alloc::vec![empty_fn()],
            base_func_offset: 0,
            entry_points: alloc::vec![("main".into(), 0)],
            func_types: alloc::vec![ft],
            data_segments: alloc::vec![],
            data_init_fn: None,
        });

        let output = target.finish();
        assert_eq!(output.fns.len(), 1);
        assert_eq!(output.globals.len(), 1);
        assert_eq!(output.memories.len(), 1);

        let bytes = assemble(output).finish();
        validate(&bytes);
    }

    #[test]
    fn assemble_with_data_init_fn() {
        let mut target: MegabinaryModuleTarget<Function> = MegabinaryModuleTarget::new();

        target.0.declare_memory(
            MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None },
        );

        let body_ft = speet_link::unit::FuncType::from_val_types(&[], &[]);
        let mut init_body = Function::new([]);
        init_body.instruction(&Instruction::End);

        target.on_unit(BinaryUnit {
            fns: alloc::vec![empty_fn()],
            base_func_offset: 0,
            entry_points: alloc::vec![],
            func_types: alloc::vec![body_ft.clone()],
            data_segments: alloc::vec![DataSegment { data: b"test data".to_vec() }],
            data_init_fn: Some((init_body, body_ft)),
        });

        let output = target.finish();
        assert_eq!(output.passive_data.len(), 1);
        assert_eq!(output.data_init_fns.len(), 1);

        let bytes = assemble(output).finish();
        validate(&bytes);
    }
}
