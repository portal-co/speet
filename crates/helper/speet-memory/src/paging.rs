//! Page-table code-generation helpers.
//!
//! Each function in this module is a *builder*: it takes the static
//! configuration up-front and returns a [`MapperCallback`]-compatible closure.
//! The returned closure can be handed directly to
//! `recompiler.set_mapper_callback(…)`.
//!
//! ```ignore
//! // Pick which wasm locals to use for the mapper's scratch space.
//! let locals = PageMapLocals::new(66, [67, 68, 69]);
//!
//! let mut mapper = standard_page_table_mapper(
//!     0x1000_0000u64,     // page_table_base
//!     0x2000_0000u64,     // security_directory_base
//!     0,                   // wasm memory index
//!     true,                // use i64 (RV64 / memory64)
//!     locals,
//! );
//! recompiler.set_mapper_callback(&mut mapper);
//! ```
//!
//! # Scratch-local convention
//!
//! Callers choose which wasm locals the mapper may use by supplying a
//! [`PageMapLocals`] value.  The mapper closure saves the incoming virtual
//! address into `locals.vaddr` with `LocalTee` at the start so it can reload
//! it later.  Up to three additional scratch locals (`locals.scratch`) are
//! used by the multi-level and 32-bit-physical-address variants.
//!
//! # Stack contract (all variants)
//! * **Before**: virtual address (`i32` when `use_i64 = false`, `i64` when
//!   `use_i64 = true`) is on the wasm value stack.
//! * **After**: physical address of the same type is on the stack.

use crate::mapper::{CallbackContext, MapperCallback};
use wax_core::build::InstructionSink;
use wasm_encoder::{Instruction, MemArg};

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
}

impl From<u64> for PageTableBase {
    fn from(c: u64) -> Self {
        PageTableBase::Constant(c)
    }
}

impl PageTableBase {
    /// Emit the instructions that push this value onto the wasm stack.
    pub fn emit_load<Context, E, F: InstructionSink<Context, E>>(
        &self,
        ctx: &mut Context,
        cb: &mut CallbackContext<Context, E, F>,
        use_i64: bool,
    ) -> Result<(), E> {
        match self {
            PageTableBase::Constant(addr) => {
                if use_i64 {
                    cb.emit(ctx, &Instruction::I64Const(*addr as i64))?;
                } else {
                    cb.emit(ctx, &Instruction::I32Const(*addr as i32))?;
                }
            }
            PageTableBase::Local(idx) => {
                cb.emit(ctx, &Instruction::LocalGet(*idx))?;
            }
            PageTableBase::Global(idx) => {
                cb.emit(ctx, &Instruction::GlobalGet(*idx))?;
            }
        }
        Ok(())
    }
}

// ── PageMapLocals ──────────────────────────────────────────────────────────────

/// The wasm local variable indices that a page-table mapper closure may use as
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

// ── standard_page_table_mapper ─────────────────────────────────────────────────

/// Build a single-level 64 KiB page-table mapper closure.
///
/// Each page-table entry is 8 bytes (`i64`) for `use_i64 = true` or 4 bytes
/// (`i32`) for `use_i64 = false`.  Bits \[63:16\] of the virtual address
/// select the page; bits \[15:0\] are the page offset.
///
/// The top 16 bits of the physical address come from a "security directory"
/// entry and are combined with the lower 48 bits from the page-table entry.
///
/// # Scratch locals consumed
/// `locals.vaddr`, `locals.scratch[0]`, `locals.scratch[1]`.
///
/// # Returns
/// A closure implementing [`MapperCallback`] that can be passed directly to
/// `recompiler.set_mapper_callback`.
pub fn standard_page_table_mapper<Context, E, F>(
    page_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
    locals: PageMapLocals,
) -> impl MapperCallback<Context, E, F>
where
    F: InstructionSink<Context, E>,
{
    let pt_base = page_table_base.into();
    let sec_dir_base = security_directory_base.into();

    move |ctx: &mut Context, cb: &mut CallbackContext<Context, E, F>| {
        let lv = locals.vaddr;
        let [ls0, ls1, _] = locals.scratch;

        if use_i64 {
            // Save vaddr, then compute page index
            cb.emit(ctx, &Instruction::LocalTee(lv))?;
            cb.emit(ctx, &Instruction::I64Const(16))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(3))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            pt_base.emit_load(ctx, cb, true)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;

            // page_pointer → scratch[0]; page_base_low48 → scratch[1]
            cb.emit(ctx, &Instruction::LocalTee(ls0))?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;       // security_index on stack

            cb.emit(ctx, &Instruction::LocalGet(ls0))?;
            cb.emit(ctx, &Instruction::I64Const(16))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::LocalSet(ls1))?; // page_base_low48

            cb.emit(ctx, &Instruction::I64Const(2))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            sec_dir_base.emit_load(ctx, cb, true)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::I64ExtendI32U)?;

            // page_base_top16 = sec_entry >> 16
            cb.emit(ctx, &Instruction::I64Const(16))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            // phys_page_base = (top16 << 48) | low48
            cb.emit(ctx, &Instruction::I64Const(48))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            cb.emit(ctx, &Instruction::LocalGet(ls1))?;
            cb.emit(ctx, &Instruction::I64Or)?;

            // page_offset = vaddr & 0xFFFF
            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Add)?;
        } else {
            // 32-bit fallback — simple single-level
            cb.emit(ctx, &Instruction::LocalTee(lv))?;
            cb.emit(ctx, &Instruction::I32Const(16))?;
            cb.emit(ctx, &Instruction::I32ShrU)?;
            cb.emit(ctx, &Instruction::I32Const(3))?;
            cb.emit(ctx, &Instruction::I32Shl)?;
            pt_base.emit_load(ctx, cb, false)?;
            cb.emit(ctx, &Instruction::I32Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I32Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I32And)?;
            cb.emit(ctx, &Instruction::I32Add)?;
        }
        Ok(())
    }
}

// ── standard_page_table_mapper_32 ─────────────────────────────────────────────

/// Build a single-level page-table mapper with 32-bit physical addresses.
///
/// Each page-table entry is 4 bytes (`i32`).  The physical page base occupies
/// bits \[31:8\] of the entry; bits \[7:0\] are a security index.
///
/// # Scratch locals consumed
/// `locals.vaddr`, `locals.scratch[0]`, `locals.scratch[1]`, `locals.scratch[2]`.
///
/// # Returns
/// A closure implementing [`MapperCallback`].
pub fn standard_page_table_mapper_32<Context, E, F>(
    page_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
    locals: PageMapLocals,
) -> impl MapperCallback<Context, E, F>
where
    F: InstructionSink<Context, E>,
{
    let pt_base = page_table_base.into();
    let sec_dir_base = security_directory_base.into();

    move |ctx: &mut Context, cb: &mut CallbackContext<Context, E, F>| {
        let lv = locals.vaddr;
        let [ls0, ls1, ls2] = locals.scratch;

        if use_i64 {
            cb.emit(ctx, &Instruction::LocalTee(lv))?;
            cb.emit(ctx, &Instruction::I64Const(16))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(2))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            pt_base.emit_load(ctx, cb, true)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::LocalTee(ls0))?;  // page_pointer (i32)
            cb.emit(ctx, &Instruction::I64ExtendI32U)?;
            cb.emit(ctx, &Instruction::LocalSet(ls1))?;  // page_pointer as i64

            // security_index = page_pointer & 0xFF
            cb.emit(ctx, &Instruction::LocalGet(ls1))?;
            cb.emit(ctx, &Instruction::I64Const(0xFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            // page_base_low24 = page_pointer >> 8
            cb.emit(ctx, &Instruction::LocalGet(ls1))?;
            cb.emit(ctx, &Instruction::I64Const(8))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::LocalSet(ls2))?;  // page_base_low24

            cb.emit(ctx, &Instruction::I64Const(2))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            sec_dir_base.emit_load(ctx, cb, true)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::I64ExtendI32U)?;
            cb.emit(ctx, &Instruction::I64Const(24))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(24))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            cb.emit(ctx, &Instruction::LocalGet(ls2))?;
            cb.emit(ctx, &Instruction::I64Or)?;

            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Add)?;
        } else {
            cb.emit(ctx, &Instruction::LocalTee(lv))?;
            cb.emit(ctx, &Instruction::I32Const(16))?;
            cb.emit(ctx, &Instruction::I32ShrU)?;
            cb.emit(ctx, &Instruction::I32Const(2))?;
            cb.emit(ctx, &Instruction::I32Shl)?;
            pt_base.emit_load(ctx, cb, false)?;
            cb.emit(ctx, &Instruction::I32Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::LocalTee(ls0))?;

            // security_index = page_pointer & 0xFF
            cb.emit(ctx, &Instruction::I32Const(0xFF))?;
            cb.emit(ctx, &Instruction::I32And)?;
            // page_base_low24 = page_pointer >> 8
            cb.emit(ctx, &Instruction::LocalGet(ls0))?;
            cb.emit(ctx, &Instruction::I32Const(8))?;
            cb.emit(ctx, &Instruction::I32ShrU)?;
            cb.emit(ctx, &Instruction::LocalSet(ls1))?;

            cb.emit(ctx, &Instruction::I32Const(2))?;
            cb.emit(ctx, &Instruction::I32Shl)?;
            sec_dir_base.emit_load(ctx, cb, false)?;
            cb.emit(ctx, &Instruction::I32Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::I32Const(24))?;
            cb.emit(ctx, &Instruction::I32ShrU)?;
            cb.emit(ctx, &Instruction::I32Const(24))?;
            cb.emit(ctx, &Instruction::I32Shl)?;
            cb.emit(ctx, &Instruction::LocalGet(ls1))?;
            cb.emit(ctx, &Instruction::I32Or)?;

            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I32Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I32And)?;
            cb.emit(ctx, &Instruction::I32Add)?;
        }
        Ok(())
    }
}

// ── multilevel_page_table_mapper ───────────────────────────────────────────────

/// Build a three-level page-table mapper for 64 KiB pages with 64-bit
/// physical addresses.
///
/// Level indices: bits \[63:48\], \[47:32\], \[31:16\] of the virtual address.
/// Each entry is 8 bytes (`i64`).
///
/// When `use_i64 = false` this delegates to
/// [`standard_page_table_mapper_32`] (the 32-bit multi-level fallback).
///
/// # Scratch locals consumed
/// `locals.vaddr`, `locals.scratch[0]`, `locals.scratch[1]`.
///
/// # Returns
/// A closure implementing [`MapperCallback`].
pub fn multilevel_page_table_mapper<Context, E, F>(
    l3_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
    locals: PageMapLocals,
) -> impl MapperCallback<Context, E, F>
where
    F: InstructionSink<Context, E>,
{
    let l3_base = l3_table_base.into();
    let sec_dir_base = security_directory_base.into();

    move |ctx: &mut Context, cb: &mut CallbackContext<Context, E, F>| {
        let lv = locals.vaddr;
        let [ls0, ls1, _] = locals.scratch;

        if use_i64 {
            // Save vaddr
            cb.emit(ctx, &Instruction::LocalTee(lv))?;

            // Level 3 lookup
            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(48))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Const(3))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            l3_base.emit_load(ctx, cb, true)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;

            // Level 2 lookup
            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(32))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Const(3))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;

            // Level 1 lookup
            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(16))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Const(3))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;

            // Security + final address (same as standard mapper)
            cb.emit(ctx, &Instruction::LocalTee(ls0))?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;       // security_index
            cb.emit(ctx, &Instruction::LocalGet(ls0))?;
            cb.emit(ctx, &Instruction::I64Const(16))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::LocalSet(ls1))?; // page_base_low48
            cb.emit(ctx, &Instruction::I64Const(3))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            sec_dir_base.emit_load(ctx, cb, true)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;
            cb.emit(ctx, &Instruction::I64Const(48))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(48))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            cb.emit(ctx, &Instruction::LocalGet(ls1))?;
            cb.emit(ctx, &Instruction::I64Or)?;

            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Add)?;
        } else {
            // 32-bit vaddr/paddr: delegate to the 32-bit single-level mapper
            let mut inner = standard_page_table_mapper_32(
                l3_base,
                sec_dir_base,
                memory_index,
                false,
                locals,
            );
            inner.call(ctx, cb)?;
        }
        Ok(())
    }
}

// ── multilevel_page_table_mapper_32 ───────────────────────────────────────────

/// Build a three-level page-table mapper with 32-bit physical addresses.
///
/// Each entry is 4 bytes.  The upper 8 bits of the physical address come from
/// the security directory; the lower 24 bits from the page-table entry.
///
/// When `use_i64 = false` this delegates to
/// [`standard_page_table_mapper_32`].
///
/// # Scratch locals consumed
/// `locals.vaddr`, `locals.scratch[0]`, `locals.scratch[1]`, `locals.scratch[2]`.
///
/// # Returns
/// A closure implementing [`MapperCallback`].
pub fn multilevel_page_table_mapper_32<Context, E, F>(
    l3_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
    locals: PageMapLocals,
) -> impl MapperCallback<Context, E, F>
where
    F: InstructionSink<Context, E>,
{
    let l3_base = l3_table_base.into();
    let sec_dir_base = security_directory_base.into();

    move |ctx: &mut Context, cb: &mut CallbackContext<Context, E, F>| {
        let lv = locals.vaddr;
        let [ls0, ls1, ls2] = locals.scratch;

        if use_i64 {
            // Save vaddr
            cb.emit(ctx, &Instruction::LocalTee(lv))?;

            // Level 3
            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(48))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Const(2))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            l3_base.emit_load(ctx, cb, true)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::I64ExtendI32U)?;

            // Level 2
            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(32))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Const(2))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::I64ExtendI32U)?;

            // Level 1
            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(16))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Const(2))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
            cb.emit(ctx, &Instruction::LocalTee(ls0))?;  // page_pointer (i32)
            cb.emit(ctx, &Instruction::I64ExtendI32U)?;
            cb.emit(ctx, &Instruction::LocalSet(ls1))?;  // page_pointer as i64

            cb.emit(ctx, &Instruction::LocalGet(ls1))?;
            cb.emit(ctx, &Instruction::I64Const(0xFF))?;
            cb.emit(ctx, &Instruction::I64And)?;          // security_index
            cb.emit(ctx, &Instruction::LocalGet(ls1))?;
            cb.emit(ctx, &Instruction::I64Const(8))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::LocalSet(ls2))?;  // page_base_low24
            cb.emit(ctx, &Instruction::I64Const(3))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            sec_dir_base.emit_load(ctx, cb, true)?;
            cb.emit(ctx, &Instruction::I64Add)?;
            cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;
            cb.emit(ctx, &Instruction::I64Const(56))?;
            cb.emit(ctx, &Instruction::I64ShrU)?;
            cb.emit(ctx, &Instruction::I64Const(24))?;
            cb.emit(ctx, &Instruction::I64Shl)?;
            cb.emit(ctx, &Instruction::LocalGet(ls2))?;
            cb.emit(ctx, &Instruction::I64Or)?;

            cb.emit(ctx, &Instruction::LocalGet(lv))?;
            cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
            cb.emit(ctx, &Instruction::I64And)?;
            cb.emit(ctx, &Instruction::I64Add)?;
        } else {
            // 32-bit fallback
            let mut inner = standard_page_table_mapper_32(
                l3_base,
                sec_dir_base,
                memory_index,
                false,
                locals,
            );
            inner.call(ctx, cb)?;
        }
        Ok(())
    }
}
