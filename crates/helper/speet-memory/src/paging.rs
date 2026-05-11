//! Page-table code-generation helpers.
//!
//! Each function in this module is a *builder*: it takes the static
//! configuration up-front and returns a struct that implements
//! [`AddressMapper`].  The returned struct calls `declare_locals` to
//! allocate its own scratch locals from a [`LocalLayout`], removing the
//! need for callers to manage [`PageMapLocals`] manually.
//!
//! ```ignore
//! let mut mapper = standard_page_table_mapper(
//!     0x1000_0000u64,     // page_table_base
//!     0x2000_0000u64,     // security_directory_base
//!     0,                   // wasm memory index
//!     true,                // use i64 (RV64 / memory64)
//! );
//! mapper.declare_locals(&mut layout);  // allocates 4 × i32 scratch locals
//! recompiler.set_memory_access(&mut mapper);
//! ```
//!
//! # Scratch-local convention
//!
//! The mapper calls `declare_locals` to append four consecutive `i32` locals
//! to the layout and stores them internally via [`PageMapLocals::consecutive`].
//! Calling `translate` before `declare_locals` will panic.
//!
//! # Stack contract (all variants)
//! * **Before**: virtual address (`i32` when `use_i64 = false`, `i64` when
//!   `use_i64 = true`) is on the wasm value stack.
//! * **After**: physical address of the same type is on the stack.

use crate::mapper::AddressMapper;
use speet_ordering::MemorySink;
use yecta::{LocalDeclarator, LocalLayout, layout::CellIdx};
use wasm_encoder::{Instruction, MemArg, ValType};

// ── PageTableBase ──────────────────────────────────────────────────────────────

/// Specifies where the page-table base address comes from at runtime.
#[derive(Clone, Copy, Debug)]
pub enum PageTableBase {
    /// Compile-time constant physical address.
    Constant(u64),
    /// Value loaded from a wasm local variable.
    Local(u32),
    /// Value loaded from a wasm global variable.
    Global(u32),
    /// Sentinel: the address will arrive as a WASM function parameter.
    ///
    /// Mapper structs replace this with `Local(resolved_idx)` during
    /// `declare_params`.  Calling `emit_load` while this variant is still
    /// present panics — it means `declare_params` was never called.
    Param,
}

impl From<u64> for PageTableBase {
    fn from(c: u64) -> Self {
        PageTableBase::Constant(c)
    }
}

impl PageTableBase {
    /// Emit the instructions that push this value onto the wasm stack.
    pub fn emit_load<Context, E>(
        &self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        use_i64: bool,
    ) -> Result<(), E> {
        match self {
            PageTableBase::Constant(addr) => {
                if use_i64 {
                    sink.instruction(ctx, &Instruction::I64Const(*addr as i64))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Const(*addr as i32))?;
                }
            }
            PageTableBase::Local(idx) => {
                sink.instruction(ctx, &Instruction::LocalGet(*idx))?;
            }
            PageTableBase::Global(idx) => {
                sink.instruction(ctx, &Instruction::GlobalGet(*idx))?;
            }
            PageTableBase::Param => {
                panic!("PageTableBase::Param was not replaced by declare_params");
            }
        }
        Ok(())
    }
}

// ── PageMapLocals ──────────────────────────────────────────────────────────────

/// The wasm local variable indices that a page-table mapper may use as
/// scratch space.
///
/// The `vaddr` local is used to save the incoming virtual address so it can be
/// reloaded after intermediate computations.  The `scratch` locals are used for
/// intermediate page-table entry values; individual mapper variants document
/// how many of the three scratch slots they actually consume.
///
/// # Example
///
/// If a recompiler allocates locals 0–63 for guest registers, local 64 for the
/// PC, and wants to reserve locals 65–68 for the mapper:
///
/// ```ignore
/// let locals = PageMapLocals::new(65, [66, 67, 68]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PageMapLocals {
    /// Local used to save (and later reload) the virtual address.
    pub vaddr: u32,
    /// Up to three scratch locals for intermediate values.
    pub scratch: [u32; 3],
}

impl PageMapLocals {
    /// Construct a `PageMapLocals` with the given `vaddr` local and three
    /// scratch locals.
    #[inline]
    pub const fn new(vaddr: u32, scratch: [u32; 3]) -> Self {
        Self { vaddr, scratch }
    }

    /// Convenience: allocate four consecutive locals starting at `first_local`.
    ///
    /// * `first_local + 0` → `vaddr`
    /// * `first_local + 1..=3` → `scratch[0..=2]`
    #[inline]
    pub const fn consecutive(first_local: u32) -> Self {
        Self {
            vaddr: first_local,
            scratch: [first_local + 1, first_local + 2, first_local + 3],
        }
    }
}

// ── StandardPageTableMapper ────────────────────────────────────────────────────

/// A single-level 64 KiB page-table mapper.
///
/// Each page-table entry is 8 bytes (`i64`) for `use_i64 = true` or 4 bytes
/// (`i32`) for `use_i64 = false`.  Bits \[63:16\] of the virtual address
/// select the page; bits \[15:0\] are the page offset.
///
/// The top 16 bits of the physical address come from a "security directory"
/// entry and are combined with the lower 48 bits from the page-table entry.
///
/// Construct via [`standard_page_table_mapper`].  Call `declare_locals` on the
/// returned struct before using it as a mapper callback.
pub struct StandardPageTableMapper {
    page_table_base: PageTableBase,
    security_dir_base: PageTableBase,
    memory_index: u32,
    use_i64: bool,
    locals: Option<PageMapLocals>,
}

impl LocalDeclarator for StandardPageTableMapper {
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        let val_type = if self.use_i64 { ValType::I64 } else { ValType::I32 };
        if matches!(self.page_table_base, PageTableBase::Param) {
            let slot = layout.append(1, val_type);
            self.page_table_base = PageTableBase::Local(layout.base(slot));
        }
        if matches!(self.security_dir_base, PageTableBase::Param) {
            let slot = layout.append(1, val_type);
            self.security_dir_base = PageTableBase::Local(layout.base(slot));
        }
        let _ = cell;
    }

    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        let slot = layout.append(4, ValType::I32);
        self.locals = Some(PageMapLocals::consecutive(layout.base(slot)));
        let _ = cell;
    }
}

impl<Context, E> AddressMapper<Context, E>
    for StandardPageTableMapper
{
    fn chunk_size(&self) -> Option<u64> {
        Some(0x10000)
    }

    fn translate(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        let locals = self.locals.expect("declare_locals must be called before translate");
        let lv = locals.vaddr;
        let [ls0, ls1, _] = locals.scratch;

        if self.use_i64 {
            sink.instruction(ctx, &Instruction::LocalTee(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(16))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(3))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            self.page_table_base.emit_load(ctx, sink, true)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I64Load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: self.memory_index,
                }),
            )?;

            sink.instruction(ctx, &Instruction::LocalTee(ls0))?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;

            sink.instruction(ctx, &Instruction::LocalGet(ls0))?;
            sink.instruction(ctx, &Instruction::I64Const(16))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::LocalSet(ls1))?;

            sink.instruction(ctx, &Instruction::I64Const(2))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            self.security_dir_base.emit_load(ctx, sink, true)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::I64ExtendI32U)?;

            sink.instruction(ctx, &Instruction::I64Const(16))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(48))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls1))?;
            sink.instruction(ctx, &Instruction::I64Or)?;

            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
        } else {
            sink.instruction(ctx, &Instruction::LocalTee(lv))?;
            sink.instruction(ctx, &Instruction::I32Const(16))?;
            sink.instruction(ctx, &Instruction::I32ShrU)?;
            sink.instruction(ctx, &Instruction::I32Const(3))?;
            sink.instruction(ctx, &Instruction::I32Shl)?;
            self.page_table_base.emit_load(ctx, sink, false)?;
            sink.instruction(ctx, &Instruction::I32Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I32Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I32And)?;
            sink.instruction(ctx, &Instruction::I32Add)?;
        }
        Ok(())
    }
}

/// Build a single-level 64 KiB page-table mapper.
///
/// Call [`AddressMapper::declare_locals`] on the returned struct before using
/// it as a mapper callback.
pub fn standard_page_table_mapper(
    page_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> StandardPageTableMapper {
    StandardPageTableMapper {
        page_table_base: page_table_base.into(),
        security_dir_base: security_directory_base.into(),
        memory_index,
        use_i64,
        locals: None,
    }
}

// ── StandardPageTableMapper32 ─────────────────────────────────────────────────

/// A single-level page-table mapper with 32-bit physical addresses.
///
/// Construct via [`standard_page_table_mapper_32`].
pub struct StandardPageTableMapper32 {
    page_table_base: PageTableBase,
    security_dir_base: PageTableBase,
    memory_index: u32,
    use_i64: bool,
    locals: Option<PageMapLocals>,
}

impl LocalDeclarator for StandardPageTableMapper32 {
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        let val_type = if self.use_i64 { ValType::I64 } else { ValType::I32 };
        if matches!(self.page_table_base, PageTableBase::Param) {
            let slot = layout.append(1, val_type);
            self.page_table_base = PageTableBase::Local(layout.base(slot));
        }
        if matches!(self.security_dir_base, PageTableBase::Param) {
            let slot = layout.append(1, val_type);
            self.security_dir_base = PageTableBase::Local(layout.base(slot));
        }
        let _ = cell;
    }

    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        let slot = layout.append(4, ValType::I32);
        self.locals = Some(PageMapLocals::consecutive(layout.base(slot)));
        let _ = cell;
    }
}

impl<Context, E> AddressMapper<Context, E>
    for StandardPageTableMapper32
{
    fn chunk_size(&self) -> Option<u64> {
        Some(0x10000)
    }

    fn translate(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        let locals = self.locals.expect("declare_locals must be called before translate");
        let lv = locals.vaddr;
        let [ls0, ls1, ls2] = locals.scratch;

        if self.use_i64 {
            sink.instruction(ctx, &Instruction::LocalTee(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(16))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(2))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            self.page_table_base.emit_load(ctx, sink, true)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::LocalTee(ls0))?;
            sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
            sink.instruction(ctx, &Instruction::LocalSet(ls1))?;

            sink.instruction(ctx, &Instruction::LocalGet(ls1))?;
            sink.instruction(ctx, &Instruction::I64Const(0xFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls1))?;
            sink.instruction(ctx, &Instruction::I64Const(8))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::LocalSet(ls2))?;

            sink.instruction(ctx, &Instruction::I64Const(2))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            self.security_dir_base.emit_load(ctx, sink, true)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
            sink.instruction(ctx, &Instruction::I64Const(24))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(24))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls2))?;
            sink.instruction(ctx, &Instruction::I64Or)?;

            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
        } else {
            sink.instruction(ctx, &Instruction::LocalTee(lv))?;
            sink.instruction(ctx, &Instruction::I32Const(16))?;
            sink.instruction(ctx, &Instruction::I32ShrU)?;
            sink.instruction(ctx, &Instruction::I32Const(2))?;
            sink.instruction(ctx, &Instruction::I32Shl)?;
            self.page_table_base.emit_load(ctx, sink, false)?;
            sink.instruction(ctx, &Instruction::I32Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::LocalTee(ls0))?;

            sink.instruction(ctx, &Instruction::I32Const(0xFF))?;
            sink.instruction(ctx, &Instruction::I32And)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls0))?;
            sink.instruction(ctx, &Instruction::I32Const(8))?;
            sink.instruction(ctx, &Instruction::I32ShrU)?;
            sink.instruction(ctx, &Instruction::LocalSet(ls1))?;

            sink.instruction(ctx, &Instruction::I32Const(2))?;
            sink.instruction(ctx, &Instruction::I32Shl)?;
            self.security_dir_base.emit_load(ctx, sink, false)?;
            sink.instruction(ctx, &Instruction::I32Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::I32Const(24))?;
            sink.instruction(ctx, &Instruction::I32ShrU)?;
            sink.instruction(ctx, &Instruction::I32Const(24))?;
            sink.instruction(ctx, &Instruction::I32Shl)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls1))?;
            sink.instruction(ctx, &Instruction::I32Or)?;

            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I32Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I32And)?;
            sink.instruction(ctx, &Instruction::I32Add)?;
        }
        Ok(())
    }
}

/// Build a single-level page-table mapper with 32-bit physical addresses.
///
/// Call [`AddressMapper::declare_locals`] on the returned struct before using
/// it as a mapper callback.
pub fn standard_page_table_mapper_32(
    page_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> StandardPageTableMapper32 {
    StandardPageTableMapper32 {
        page_table_base: page_table_base.into(),
        security_dir_base: security_directory_base.into(),
        memory_index,
        use_i64,
        locals: None,
    }
}

// ── MultilevelPageTableMapper ──────────────────────────────────────────────────

/// A three-level page-table mapper for 64 KiB pages with 64-bit physical addresses.
///
/// Construct via [`multilevel_page_table_mapper`].
pub struct MultilevelPageTableMapper {
    l3_base: PageTableBase,
    security_dir_base: PageTableBase,
    memory_index: u32,
    use_i64: bool,
    locals: Option<PageMapLocals>,
}

impl LocalDeclarator for MultilevelPageTableMapper {
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        let val_type = if self.use_i64 { ValType::I64 } else { ValType::I32 };
        if matches!(self.l3_base, PageTableBase::Param) {
            let slot = layout.append(1, val_type);
            self.l3_base = PageTableBase::Local(layout.base(slot));
        }
        if matches!(self.security_dir_base, PageTableBase::Param) {
            let slot = layout.append(1, val_type);
            self.security_dir_base = PageTableBase::Local(layout.base(slot));
        }
        let _ = cell;
    }

    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        let slot = layout.append(4, ValType::I32);
        self.locals = Some(PageMapLocals::consecutive(layout.base(slot)));
        let _ = cell;
    }
}

impl<Context, E> AddressMapper<Context, E>
    for MultilevelPageTableMapper
{
    fn chunk_size(&self) -> Option<u64> {
        Some(0x10000)
    }

    fn translate(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        let locals = self.locals.expect("declare_locals must be called before translate");
        let lv = locals.vaddr;
        let [ls0, ls1, _] = locals.scratch;

        if self.use_i64 {
            sink.instruction(ctx, &Instruction::LocalTee(lv))?;

            // Level 3 lookup
            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(48))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Const(3))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            self.l3_base.emit_load(ctx, sink, true)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I64Load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: self.memory_index,
                }),
            )?;

            // Level 2 lookup
            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(32))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Const(3))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I64Load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: self.memory_index,
                }),
            )?;

            // Level 1 lookup
            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(16))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Const(3))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I64Load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: self.memory_index,
                }),
            )?;

            // Security + final address
            sink.instruction(ctx, &Instruction::LocalTee(ls0))?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls0))?;
            sink.instruction(ctx, &Instruction::I64Const(16))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::LocalSet(ls1))?;
            sink.instruction(ctx, &Instruction::I64Const(3))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            self.security_dir_base.emit_load(ctx, sink, true)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I64Load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::I64Const(48))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(48))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls1))?;
            sink.instruction(ctx, &Instruction::I64Or)?;

            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
        } else {
            // 32-bit vaddr/paddr: delegate to the 32-bit single-level mapper
            let mut inner = StandardPageTableMapper32 {
                page_table_base: self.l3_base,
                security_dir_base: self.security_dir_base,
                memory_index: self.memory_index,
                use_i64: false,
                locals: self.locals,
            };
            inner.translate(ctx, sink)?;
        }
        Ok(())
    }
}

/// Build a three-level page-table mapper for 64 KiB pages with 64-bit
/// physical addresses.
///
/// When `use_i64 = false` this delegates to
/// [`standard_page_table_mapper_32`] (the 32-bit multi-level fallback).
///
/// Call [`AddressMapper::declare_locals`] on the returned struct before using
/// it as a mapper callback.
pub fn multilevel_page_table_mapper(
    l3_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> MultilevelPageTableMapper {
    MultilevelPageTableMapper {
        l3_base: l3_table_base.into(),
        security_dir_base: security_directory_base.into(),
        memory_index,
        use_i64,
        locals: None,
    }
}

// ── MultilevelPageTableMapper32 ────────────────────────────────────────────────

/// A three-level page-table mapper with 32-bit physical addresses.
///
/// Construct via [`multilevel_page_table_mapper_32`].
pub struct MultilevelPageTableMapper32 {
    l3_base: PageTableBase,
    security_dir_base: PageTableBase,
    memory_index: u32,
    use_i64: bool,
    locals: Option<PageMapLocals>,
}

impl LocalDeclarator for MultilevelPageTableMapper32 {
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        let val_type = if self.use_i64 { ValType::I64 } else { ValType::I32 };
        if matches!(self.l3_base, PageTableBase::Param) {
            let slot = layout.append(1, val_type);
            self.l3_base = PageTableBase::Local(layout.base(slot));
        }
        if matches!(self.security_dir_base, PageTableBase::Param) {
            let slot = layout.append(1, val_type);
            self.security_dir_base = PageTableBase::Local(layout.base(slot));
        }
        let _ = cell;
    }

    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        let slot = layout.append(4, ValType::I32);
        self.locals = Some(PageMapLocals::consecutive(layout.base(slot)));
        let _ = cell;
    }
}

impl<Context, E> AddressMapper<Context, E>
    for MultilevelPageTableMapper32
{
    fn chunk_size(&self) -> Option<u64> {
        Some(0x10000)
    }

    fn translate(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        let locals = self.locals.expect("declare_locals must be called before translate");
        let lv = locals.vaddr;
        let [ls0, ls1, ls2] = locals.scratch;

        if self.use_i64 {
            sink.instruction(ctx, &Instruction::LocalTee(lv))?;

            // Level 3
            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(48))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Const(2))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            self.l3_base.emit_load(ctx, sink, true)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::I64ExtendI32U)?;

            // Level 2
            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(32))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Const(2))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::I64ExtendI32U)?;

            // Level 1
            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(16))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Const(2))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::LocalTee(ls0))?;
            sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
            sink.instruction(ctx, &Instruction::LocalSet(ls1))?;

            sink.instruction(ctx, &Instruction::LocalGet(ls1))?;
            sink.instruction(ctx, &Instruction::I64Const(0xFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls1))?;
            sink.instruction(ctx, &Instruction::I64Const(8))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::LocalSet(ls2))?;
            sink.instruction(ctx, &Instruction::I64Const(3))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            self.security_dir_base.emit_load(ctx, sink, true)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
            sink.instruction(
                ctx,
                &Instruction::I64Load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: self.memory_index,
                }),
            )?;
            sink.instruction(ctx, &Instruction::I64Const(56))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
            sink.instruction(ctx, &Instruction::I64Const(24))?;
            sink.instruction(ctx, &Instruction::I64Shl)?;
            sink.instruction(ctx, &Instruction::LocalGet(ls2))?;
            sink.instruction(ctx, &Instruction::I64Or)?;

            sink.instruction(ctx, &Instruction::LocalGet(lv))?;
            sink.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
            sink.instruction(ctx, &Instruction::I64And)?;
            sink.instruction(ctx, &Instruction::I64Add)?;
        } else {
            // 32-bit fallback: delegate to standard_page_table_mapper_32
            let mut inner = StandardPageTableMapper32 {
                page_table_base: self.l3_base,
                security_dir_base: self.security_dir_base,
                memory_index: self.memory_index,
                use_i64: false,
                locals: self.locals,
            };
            inner.translate(ctx, sink)?;
        }
        Ok(())
    }
}

/// Build a three-level page-table mapper with 32-bit physical addresses.
///
/// When `use_i64 = false` this delegates to
/// [`standard_page_table_mapper_32`].
///
/// Call [`AddressMapper::declare_locals`] on the returned struct before using
/// it as a mapper callback.
pub fn multilevel_page_table_mapper_32(
    l3_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> MultilevelPageTableMapper32 {
    MultilevelPageTableMapper32 {
        l3_base: l3_table_base.into(),
        security_dir_base: security_directory_base.into(),
        memory_index,
        use_i64,
        locals: None,
    }
}
