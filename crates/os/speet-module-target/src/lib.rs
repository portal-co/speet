#![no_std]

use wasm_encoder::{ConstExpr, Elements, GlobalType, MemoryType, TableType, TagType};

pub trait ModuleTarget<Ctx, Err> {
    // --- Declarations — each returns the allocated section index ---

    fn declare_global(
        &mut self,
        ctx: &mut Ctx,
        ty: GlobalType,
        init: &ConstExpr,
    ) -> Result<u32, Err>;

    fn declare_memory(&mut self, ctx: &mut Ctx, ty: MemoryType) -> Result<u32, Err>;

    /// `init` is an optional initializer expression (tables-with-init-value proposal).
    fn declare_table(
        &mut self,
        ctx: &mut Ctx,
        ty: TableType,
        init: Option<&ConstExpr>,
    ) -> Result<u32, Err>;

    fn declare_tag(&mut self, ctx: &mut Ctx, ty: TagType) -> Result<u32, Err>;

    // --- Contents ---

    fn add_memory_data(
        &mut self,
        ctx: &mut Ctx,
        memory_index: u32,
        offset: &ConstExpr,
        data: &[u8],
    ) -> Result<(), Err>;

    /// Adds a passive data segment; returns its data segment index.
    fn add_passive_memory_data(&mut self, ctx: &mut Ctx, data: &[u8]) -> Result<u32, Err>;

    fn add_element_segment(
        &mut self,
        ctx: &mut Ctx,
        table_index: u32,
        offset: &ConstExpr,
        elements: Elements<'_>,
    ) -> Result<(), Err>;

    /// Adds a passive element segment; returns its element segment index.
    fn add_passive_element_segment(
        &mut self,
        ctx: &mut Ctx,
        elements: Elements<'_>,
    ) -> Result<u32, Err>;
}

pub trait ModuleTargetDeclarator<Ctx, Err> {
    fn declare_module(
        &self,
        module: &mut (dyn ModuleTarget<Ctx, Err> + '_),
    ) -> Result<(), Err>;
}
