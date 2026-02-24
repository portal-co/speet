//! Page-table code-generation helpers.
//!
//! These free functions emit the wasm instruction sequences required to walk
//! an in-memory page table and produce a physical address from a virtual
//! address.  They were previously housed in `speet-riscv`; the move here
//! makes them usable by every architecture recompiler.
//!
//! # Scratch-local convention
//!
//! All helpers assume the caller has already saved the virtual address into
//! **local 66** (`LocalTee(66)`) immediately before calling the mapper, so
//! that the helpers can load it back when computing the page number and the
//! page offset.  Locals 67, 68 and 69 are used as additional scratch space
//! by the multi-level and 64-bit variants.

use crate::mapper::CallbackContext;
use wax_core::build::InstructionSink;
use wasm_encoder::{Instruction, MemArg};

// ── PageTableBase ──────────────────────────────────────────────────────────────

/// Specifies where the page-table base address comes from at runtime.
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

// ── standard_page_table_mapper ─────────────────────────────────────────────────

/// Single-level 64-bit page-table mapper for 64 KiB pages.
///
/// Each page-table entry is 8 bytes (i64).  Entry *n* is at
/// `pt_base + n * 8`.  Bits \[63:16\] of the virtual address select the
/// page; bits \[15:0\] are the page offset.
///
/// The top 16 bits of the physical address are stored in a "security
/// directory" and combined with the lower 48 bits from the page-table entry.
///
/// # Scratch locals used
/// `66` — virtual address (must be pre-saved by caller), `67`, `68`.
///
/// # Stack contract
/// Before: virtual address (i64 or i32 per `use_i64`).
/// After:  physical address (same type).
pub fn standard_page_table_mapper<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    cb: &mut CallbackContext<Context, E, F>,
    page_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> Result<(), E> {
    let pt_base = page_table_base.into();
    let sec_dir_base = security_directory_base.into();

    if use_i64 {
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(16))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::I64Const(3))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        pt_base.emit_load(ctx, cb, true)?;
        cb.emit(ctx, &Instruction::I64Add)?;
        cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;

        // page_pointer → scratch 67; page_base_low48 → scratch 68
        cb.emit(ctx, &Instruction::LocalTee(67))?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;          // security_index on stack

        cb.emit(ctx, &Instruction::LocalGet(67))?;
        cb.emit(ctx, &Instruction::I64Const(16))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::LocalSet(68))?;    // page_base_low48

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
        cb.emit(ctx, &Instruction::LocalGet(68))?;
        cb.emit(ctx, &Instruction::I64Or)?;

        // page_offset = vaddr & 0xFFFF
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;
        cb.emit(ctx, &Instruction::I64Add)?;
    } else {
        // 32-bit fallback — simple identity-style single-level
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I32Const(16))?;
        cb.emit(ctx, &Instruction::I32ShrU)?;
        cb.emit(ctx, &Instruction::I32Const(3))?;
        cb.emit(ctx, &Instruction::I32Shl)?;
        pt_base.emit_load(ctx, cb, false)?;
        cb.emit(ctx, &Instruction::I32Add)?;
        cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I32Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I32And)?;
        cb.emit(ctx, &Instruction::I32Add)?;
    }
    Ok(())
}

// ── standard_page_table_mapper_32 ─────────────────────────────────────────────

/// Single-level page-table mapper with 32-bit physical addresses.
///
/// Each page-table entry is 4 bytes (`i32`).  The physical page base occupies
/// bits \[31:8\] of the entry; bits \[7:0\] are a security index.
///
/// # Scratch locals used
/// `66`, `67`, `68`, `69`.
pub fn standard_page_table_mapper_32<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    cb: &mut CallbackContext<Context, E, F>,
    page_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> Result<(), E> {
    let pt_base = page_table_base.into();
    let sec_dir_base = security_directory_base.into();

    if use_i64 {
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(16))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::I64Const(2))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        pt_base.emit_load(ctx, cb, true)?;
        cb.emit(ctx, &Instruction::I64Add)?;
        cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
        cb.emit(ctx, &Instruction::LocalTee(67))?;      // page_pointer (i32)
        cb.emit(ctx, &Instruction::I64ExtendI32U)?;
        cb.emit(ctx, &Instruction::LocalSet(68))?;      // page_pointer as i64

        // security_index = page_pointer & 0xFF
        cb.emit(ctx, &Instruction::LocalGet(68))?;
        cb.emit(ctx, &Instruction::I64Const(0xFF))?;
        cb.emit(ctx, &Instruction::I64And)?;
        // page_base_low24 = page_pointer >> 8
        cb.emit(ctx, &Instruction::LocalGet(68))?;
        cb.emit(ctx, &Instruction::I64Const(8))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::LocalSet(69))?;      // page_base_low24

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
        cb.emit(ctx, &Instruction::LocalGet(69))?;
        cb.emit(ctx, &Instruction::I64Or)?;

        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;
        cb.emit(ctx, &Instruction::I64Add)?;
    } else {
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I32Const(16))?;
        cb.emit(ctx, &Instruction::I32ShrU)?;
        cb.emit(ctx, &Instruction::I32Const(2))?;
        cb.emit(ctx, &Instruction::I32Shl)?;
        pt_base.emit_load(ctx, cb, false)?;
        cb.emit(ctx, &Instruction::I32Add)?;
        cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
        cb.emit(ctx, &Instruction::LocalTee(67))?;

        // security_index = page_pointer & 0xFF
        cb.emit(ctx, &Instruction::I32Const(0xFF))?;
        cb.emit(ctx, &Instruction::I32And)?;
        // page_base_low24 = page_pointer >> 8
        cb.emit(ctx, &Instruction::LocalGet(67))?;
        cb.emit(ctx, &Instruction::I32Const(8))?;
        cb.emit(ctx, &Instruction::I32ShrU)?;
        cb.emit(ctx, &Instruction::LocalSet(68))?;

        cb.emit(ctx, &Instruction::I32Const(2))?;
        cb.emit(ctx, &Instruction::I32Shl)?;
        sec_dir_base.emit_load(ctx, cb, false)?;
        cb.emit(ctx, &Instruction::I32Add)?;
        cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
        cb.emit(ctx, &Instruction::I32Const(24))?;
        cb.emit(ctx, &Instruction::I32ShrU)?;
        cb.emit(ctx, &Instruction::I32Const(24))?;
        cb.emit(ctx, &Instruction::I32Shl)?;
        cb.emit(ctx, &Instruction::LocalGet(68))?;
        cb.emit(ctx, &Instruction::I32Or)?;

        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I32Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I32And)?;
        cb.emit(ctx, &Instruction::I32Add)?;
    }
    Ok(())
}

// ── multilevel_page_table_mapper ───────────────────────────────────────────────

/// Three-level page-table mapper for 64 KiB pages with 64-bit physical
/// addresses.
///
/// Level indices: bits \[63:48\], \[47:32\], \[31:16\] of the virtual address.
/// Each entry is 8 bytes.
///
/// # Scratch locals used
/// `66`, `67`, `68`.
pub fn multilevel_page_table_mapper<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    cb: &mut CallbackContext<Context, E, F>,
    l3_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> Result<(), E> {
    let l3_base = l3_table_base.into();
    let sec_dir_base = security_directory_base.into();

    if use_i64 {
        // Level 3 lookup
        cb.emit(ctx, &Instruction::LocalGet(66))?;
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
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(32))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;
        cb.emit(ctx, &Instruction::I64Const(3))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        cb.emit(ctx, &Instruction::I64Add)?;
        cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;

        // Level 1 lookup
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(16))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;
        cb.emit(ctx, &Instruction::I64Const(3))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        cb.emit(ctx, &Instruction::I64Add)?;
        cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;

        // Security + final address (same as standard_page_table_mapper)
        cb.emit(ctx, &Instruction::LocalTee(67))?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;           // security_index
        cb.emit(ctx, &Instruction::LocalGet(67))?;
        cb.emit(ctx, &Instruction::I64Const(16))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::LocalSet(68))?;     // page_base_low48
        cb.emit(ctx, &Instruction::I64Const(3))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        sec_dir_base.emit_load(ctx, cb, true)?;
        cb.emit(ctx, &Instruction::I64Add)?;
        cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;
        cb.emit(ctx, &Instruction::I64Const(48))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::I64Const(48))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        cb.emit(ctx, &Instruction::LocalGet(68))?;
        cb.emit(ctx, &Instruction::I64Or)?;

        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;
        cb.emit(ctx, &Instruction::I64Add)?;
    } else {
        multilevel_page_table_mapper_32(ctx, cb, l3_base, sec_dir_base, memory_index, false)?;
    }
    Ok(())
}

// ── multilevel_page_table_mapper_32 ───────────────────────────────────────────

/// Three-level page-table mapper with 32-bit physical addresses.
///
/// Each entry is 4 bytes.  The upper 8 bits come from the security directory;
/// lower 24 bits from the page-table entry.
///
/// # Scratch locals used
/// `66`, `67`, `68`, `69`.
pub fn multilevel_page_table_mapper_32<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    cb: &mut CallbackContext<Context, E, F>,
    l3_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> Result<(), E> {
    let l3_base = l3_table_base.into();
    let sec_dir_base = security_directory_base.into();

    if use_i64 {
        // Level 3
        cb.emit(ctx, &Instruction::LocalGet(66))?;
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
        cb.emit(ctx, &Instruction::LocalGet(66))?;
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
        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(16))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;
        cb.emit(ctx, &Instruction::I64Const(2))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        cb.emit(ctx, &Instruction::I64Add)?;
        cb.emit(ctx, &Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index }))?;
        cb.emit(ctx, &Instruction::LocalTee(67))?;    // page_pointer (i32)
        cb.emit(ctx, &Instruction::I64ExtendI32U)?;
        cb.emit(ctx, &Instruction::LocalSet(68))?;    // page_pointer as i64

        cb.emit(ctx, &Instruction::LocalGet(68))?;
        cb.emit(ctx, &Instruction::I64Const(0xFF))?;
        cb.emit(ctx, &Instruction::I64And)?;           // security_index
        cb.emit(ctx, &Instruction::LocalGet(68))?;
        cb.emit(ctx, &Instruction::I64Const(8))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::LocalSet(69))?;     // page_base_low24
        cb.emit(ctx, &Instruction::I64Const(3))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        sec_dir_base.emit_load(ctx, cb, true)?;
        cb.emit(ctx, &Instruction::I64Add)?;
        cb.emit(ctx, &Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index }))?;
        cb.emit(ctx, &Instruction::I64Const(56))?;
        cb.emit(ctx, &Instruction::I64ShrU)?;
        cb.emit(ctx, &Instruction::I64Const(24))?;
        cb.emit(ctx, &Instruction::I64Shl)?;
        cb.emit(ctx, &Instruction::LocalGet(69))?;
        cb.emit(ctx, &Instruction::I64Or)?;

        cb.emit(ctx, &Instruction::LocalGet(66))?;
        cb.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        cb.emit(ctx, &Instruction::I64And)?;
        cb.emit(ctx, &Instruction::I64Add)?;
    } else {
        // 32-bit vaddr/paddr fallback
        standard_page_table_mapper_32(ctx, cb, l3_base, sec_dir_base, memory_index, false)?;
    }
    Ok(())
}
