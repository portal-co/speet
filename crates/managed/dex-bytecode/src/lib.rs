//! # dex-bytecode
//!
//! A zero-dependency, `no_std`-compatible, enum-based parser for
//! [Dalvik Executable (DEX)](https://source.android.com/docs/core/runtime/dex-format)
//! bytecode.
//!
//! Every standard Dalvik opcode maps to one [`Instruction`] variant with
//! fully-typed, named operand fields.  Decoding a slice of 16-bit code units
//! is a single call to [`decode`].
//!
//! Variable-length pseudo-instructions (packed-switch data, sparse-switch data,
//! fill-array data) are handled separately by [`decode_pseudo`] / the
//! [`Pseudo`] enum, and their byte lengths are computed by [`pseudo_len`].
//!
//! ## Usage
//!
//! ```rust
//! use dex_bytecode::{decode, Instruction};
//!
//! // `move v1, v2`  (12x format, opcode 0x01)
//! let units: &[u16] = &[0x2101];
//! let (insn, consumed) = decode(units).unwrap();
//! assert_eq!(consumed, 1);
//! assert!(matches!(insn, Instruction::Move { dst: 1, src: 2 }));
//! ```
//!
//! ## References
//! - [Dalvik Bytecode](https://source.android.com/docs/core/runtime/dalvik-bytecode)
//! - [Dalvik Executable Format](https://source.android.com/docs/core/runtime/dex-format)

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
use alloc::vec::Vec;

// ── Index newtypes ─────────────────────────────────────────────────────────────

/// Index into the DEX string pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringIdx(pub u16);

/// Wide (32-bit) index into the DEX string pool (used by `const-string/jumbo`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringIdx32(pub u32);

/// Index into the DEX type descriptor pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeIdx(pub u16);

/// Index into the DEX field pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FieldIdx(pub u16);

/// Index into the DEX method pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MethodIdx(pub u16);

/// Index into the DEX prototype (method-signature) pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProtoIdx(pub u16);

/// Index into the DEX call-site pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CallSiteIdx(pub u16);

/// Index into the DEX method-handle pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MethodHandleIdx(pub u16);

// ── Up-to-5-register argument list (35c / 45cc formats) ───────────────────────

/// Up to five 4-bit register arguments used by 35c-format instructions
/// (`invoke-{virtual,super,direct,static,interface}`, `filled-new-array`,
/// `invoke-custom`, `invoke-polymorphic`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Args5 {
    /// Number of valid register arguments (0-5).
    pub count: u8,
    /// Register numbers; only `regs[..count]` are meaningful.
    pub regs: [u8; 5],
}

impl Args5 {
    /// Returns the slice of valid register numbers.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.regs[..self.count as usize]
    }
}

// ── Instruction enum ───────────────────────────────────────────────────────────

/// A fully decoded Dalvik instruction.
///
/// Field naming follows the Dalvik specification:
/// - `dst` - destination register
/// - `src` - source register
/// - `a`, `b` - additional register operands in source order
/// - `obj` - object/array reference register
/// - `array` / `index` - array base and index register
/// - `reg` - a single register operand
/// - `offset` - signed branch offset **in code units**, relative to the
///   current instruction
/// - `lit` - inline literal value
/// - `value` - inline constant (for wider literals)
/// - struct-type fields (`field`, `method`, `ty`, etc.) - pool indices
///
/// Instruction widths are available via [`Instruction::code_units`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Instruction {
    // ── Nop ────────────────────────────────────────────────────────────────
    /// `0x00` `nop` (10x)
    Nop,

    // ── Move ────────────────────────────────────────────────────────────────
    /// `0x01` `move vA, vB` (12x)
    Move { dst: u8, src: u8 },
    /// `0x02` `move/from16 vAA, vBBBB` (22x)
    MoveFrom16 { dst: u8, src: u16 },
    /// `0x03` `move/16 vAAAA, vBBBB` (32x)
    Move16 { dst: u16, src: u16 },
    /// `0x04` `move-wide vA, vB` (12x)
    MoveWide { dst: u8, src: u8 },
    /// `0x05` `move-wide/from16 vAA, vBBBB` (22x)
    MoveWideFrom16 { dst: u8, src: u16 },
    /// `0x06` `move-wide/16 vAAAA, vBBBB` (32x)
    MoveWide16 { dst: u16, src: u16 },
    /// `0x07` `move-object vA, vB` (12x)
    MoveObject { dst: u8, src: u8 },
    /// `0x08` `move-object/from16 vAA, vBBBB` (22x)
    MoveObjectFrom16 { dst: u8, src: u16 },
    /// `0x09` `move-object/16 vAAAA, vBBBB` (32x)
    MoveObject16 { dst: u16, src: u16 },
    /// `0x0a` `move-result vAA` (11x)
    MoveResult { dst: u8 },
    /// `0x0b` `move-result-wide vAA` (11x)
    MoveResultWide { dst: u8 },
    /// `0x0c` `move-result-object vAA` (11x)
    MoveResultObject { dst: u8 },
    /// `0x0d` `move-exception vAA` (11x)
    MoveException { dst: u8 },

    // ── Return ──────────────────────────────────────────────────────────────
    /// `0x0e` `return-void` (10x)
    ReturnVoid,
    /// `0x0f` `return vAA` (11x)
    Return { src: u8 },
    /// `0x10` `return-wide vAA` (11x)
    ReturnWide { src: u8 },
    /// `0x11` `return-object vAA` (11x)
    ReturnObject { src: u8 },

    // ── Constants ───────────────────────────────────────────────────────────
    /// `0x12` `const/4 vA, #+B` - load 4-bit sign-extended literal (11n)
    Const4 { dst: u8, value: i8 },
    /// `0x13` `const/16 vAA, #+BBBB` (21s)
    Const16 { dst: u8, value: i16 },
    /// `0x14` `const vAA, #+BBBBBBBB` (31i)
    Const { dst: u8, value: i32 },
    /// `0x15` `const/high16 vAA, #+BBBB0000` - literal into high 16 bits (21h)
    ConstHigh16 { dst: u8, value: i16 },
    /// `0x16` `const-wide/16 vAA, #+BBBB` (21s)
    ConstWide16 { dst: u8, value: i16 },
    /// `0x17` `const-wide/32 vAA, #+BBBBBBBB` (31i)
    ConstWide32 { dst: u8, value: i32 },
    /// `0x18` `const-wide vAA, #+BBBBBBBBBBBBBBBB` (51l)
    ConstWide { dst: u8, value: i64 },
    /// `0x19` `const-wide/high16 vAA, #+BBBB000000000000` (21h)
    ConstWideHigh16 { dst: u8, value: i16 },
    /// `0x1a` `const-string vAA, string@BBBB` (21c)
    ConstString { dst: u8, string: StringIdx },
    /// `0x1b` `const-string/jumbo vAA, string@BBBBBBBB` (31c)
    ConstStringJumbo { dst: u8, string: StringIdx32 },
    /// `0x1c` `const-class vAA, type@BBBB` (21c)
    ConstClass { dst: u8, ty: TypeIdx },

    // ── Synchronisation ─────────────────────────────────────────────────────
    /// `0x1d` `monitor-enter vAA` (11x)
    MonitorEnter { reg: u8 },
    /// `0x1e` `monitor-exit vAA` (11x)
    MonitorExit { reg: u8 },

    // ── Type checking and arrays ─────────────────────────────────────────────
    /// `0x1f` `check-cast vAA, type@BBBB` (21c)
    CheckCast { reg: u8, ty: TypeIdx },
    /// `0x20` `instance-of vA, vB, type@CCCC` (22c)
    InstanceOf { dst: u8, obj: u8, ty: TypeIdx },
    /// `0x21` `array-length vA, vB` (12x)
    ArrayLength { dst: u8, array: u8 },
    /// `0x22` `new-instance vAA, type@BBBB` (21c)
    NewInstance { dst: u8, ty: TypeIdx },
    /// `0x23` `new-array vA, vB, type@CCCC` (22c)
    NewArray { dst: u8, size: u8, ty: TypeIdx },
    /// `0x24` `filled-new-array {vC..vG}, type@BBBB` (35c)
    FilledNewArray { ty: TypeIdx, args: Args5 },
    /// `0x25` `filled-new-array/range {vCCCC..vNNNN}, type@BBBB` (3rc)
    FilledNewArrayRange { ty: TypeIdx, first_reg: u16, count: u8 },
    /// `0x26` `fill-array-data vAA, +BBBBBBBB` (31t)
    FillArrayData { array: u8, table_offset: i32 },

    // ── Exceptions and unconditional branches ────────────────────────────────
    /// `0x27` `throw vAA` (11x)
    Throw { exception: u8 },
    /// `0x28` `goto +AA` (10t)
    Goto { offset: i8 },
    /// `0x29` `goto/16 +AAAA` (20t)
    Goto16 { offset: i16 },
    /// `0x2a` `goto/32 +AAAAAAAA` (30t)
    Goto32 { offset: i32 },
    /// `0x2b` `packed-switch vAA, +BBBBBBBB` (31t)
    PackedSwitch { reg: u8, table_offset: i32 },
    /// `0x2c` `sparse-switch vAA, +BBBBBBBB` (31t)
    SparseSwitch { reg: u8, table_offset: i32 },

    // ── Comparisons (23x) ───────────────────────────────────────────────────
    /// `0x2d` `cmpl-float vAA, vBB, vCC` - NaN biases toward -1
    CmplFloat { dst: u8, a: u8, b: u8 },
    /// `0x2e` `cmpg-float vAA, vBB, vCC` - NaN biases toward +1
    CmpgFloat { dst: u8, a: u8, b: u8 },
    /// `0x2f` `cmpl-double vAA, vBB, vCC`
    CmplDouble { dst: u8, a: u8, b: u8 },
    /// `0x30` `cmpg-double vAA, vBB, vCC`
    CmpgDouble { dst: u8, a: u8, b: u8 },
    /// `0x31` `cmp-long vAA, vBB, vCC`
    CmpLong { dst: u8, a: u8, b: u8 },

    // ── Conditional branches (22t) ───────────────────────────────────────────
    /// `0x32` `if-eq vA, vB, +CCCC`
    IfEq { a: u8, b: u8, offset: i16 },
    /// `0x33` `if-ne vA, vB, +CCCC`
    IfNe { a: u8, b: u8, offset: i16 },
    /// `0x34` `if-lt vA, vB, +CCCC`
    IfLt { a: u8, b: u8, offset: i16 },
    /// `0x35` `if-ge vA, vB, +CCCC`
    IfGe { a: u8, b: u8, offset: i16 },
    /// `0x36` `if-gt vA, vB, +CCCC`
    IfGt { a: u8, b: u8, offset: i16 },
    /// `0x37` `if-le vA, vB, +CCCC`
    IfLe { a: u8, b: u8, offset: i16 },
    // ── Conditional branch-if-zero (21t) ────────────────────────────────────
    /// `0x38` `if-eqz vAA, +BBBB`
    IfEqz { reg: u8, offset: i16 },
    /// `0x39` `if-nez vAA, +BBBB`
    IfNez { reg: u8, offset: i16 },
    /// `0x3a` `if-ltz vAA, +BBBB`
    IfLtz { reg: u8, offset: i16 },
    /// `0x3b` `if-gez vAA, +BBBB`
    IfGez { reg: u8, offset: i16 },
    /// `0x3c` `if-gtz vAA, +BBBB`
    IfGtz { reg: u8, offset: i16 },
    /// `0x3d` `if-lez vAA, +BBBB`
    IfLez { reg: u8, offset: i16 },

    // ── Array access (23x) ───────────────────────────────────────────────────
    /// `0x44` `aget vAA, vBB, vCC`
    Aget { dst: u8, array: u8, index: u8 },
    /// `0x45` `aget-wide vAA, vBB, vCC`
    AgetWide { dst: u8, array: u8, index: u8 },
    /// `0x46` `aget-object vAA, vBB, vCC`
    AgetObject { dst: u8, array: u8, index: u8 },
    /// `0x47` `aget-boolean vAA, vBB, vCC`
    AgetBoolean { dst: u8, array: u8, index: u8 },
    /// `0x48` `aget-byte vAA, vBB, vCC`
    AgetByte { dst: u8, array: u8, index: u8 },
    /// `0x49` `aget-char vAA, vBB, vCC`
    AgetChar { dst: u8, array: u8, index: u8 },
    /// `0x4a` `aget-short vAA, vBB, vCC`
    AgetShort { dst: u8, array: u8, index: u8 },
    /// `0x4b` `aput vAA, vBB, vCC`
    Aput { src: u8, array: u8, index: u8 },
    /// `0x4c` `aput-wide vAA, vBB, vCC`
    AputWide { src: u8, array: u8, index: u8 },
    /// `0x4d` `aput-object vAA, vBB, vCC`
    AputObject { src: u8, array: u8, index: u8 },
    /// `0x4e` `aput-boolean vAA, vBB, vCC`
    AputBoolean { src: u8, array: u8, index: u8 },
    /// `0x4f` `aput-byte vAA, vBB, vCC`
    AputByte { src: u8, array: u8, index: u8 },
    /// `0x50` `aput-char vAA, vBB, vCC`
    AputChar { src: u8, array: u8, index: u8 },
    /// `0x51` `aput-short vAA, vBB, vCC`
    AputShort { src: u8, array: u8, index: u8 },

    // ── Instance field access (22c) ──────────────────────────────────────────
    /// `0x52` `iget vA, vB, field@CCCC`
    Iget { dst: u8, obj: u8, field: FieldIdx },
    /// `0x53` `iget-wide vA, vB, field@CCCC`
    IgetWide { dst: u8, obj: u8, field: FieldIdx },
    /// `0x54` `iget-object vA, vB, field@CCCC`
    IgetObject { dst: u8, obj: u8, field: FieldIdx },
    /// `0x55` `iget-boolean vA, vB, field@CCCC`
    IgetBoolean { dst: u8, obj: u8, field: FieldIdx },
    /// `0x56` `iget-byte vA, vB, field@CCCC`
    IgetByte { dst: u8, obj: u8, field: FieldIdx },
    /// `0x57` `iget-char vA, vB, field@CCCC`
    IgetChar { dst: u8, obj: u8, field: FieldIdx },
    /// `0x58` `iget-short vA, vB, field@CCCC`
    IgetShort { dst: u8, obj: u8, field: FieldIdx },
    /// `0x59` `iput vA, vB, field@CCCC`
    Iput { src: u8, obj: u8, field: FieldIdx },
    /// `0x5a` `iput-wide vA, vB, field@CCCC`
    IputWide { src: u8, obj: u8, field: FieldIdx },
    /// `0x5b` `iput-object vA, vB, field@CCCC`
    IputObject { src: u8, obj: u8, field: FieldIdx },
    /// `0x5c` `iput-boolean vA, vB, field@CCCC`
    IputBoolean { src: u8, obj: u8, field: FieldIdx },
    /// `0x5d` `iput-byte vA, vB, field@CCCC`
    IputByte { src: u8, obj: u8, field: FieldIdx },
    /// `0x5e` `iput-char vA, vB, field@CCCC`
    IputChar { src: u8, obj: u8, field: FieldIdx },
    /// `0x5f` `iput-short vA, vB, field@CCCC`
    IputShort { src: u8, obj: u8, field: FieldIdx },

    // ── Static field access (21c) ────────────────────────────────────────────
    /// `0x60` `sget vAA, field@BBBB`
    Sget { dst: u8, field: FieldIdx },
    /// `0x61` `sget-wide vAA, field@BBBB`
    SgetWide { dst: u8, field: FieldIdx },
    /// `0x62` `sget-object vAA, field@BBBB`
    SgetObject { dst: u8, field: FieldIdx },
    /// `0x63` `sget-boolean vAA, field@BBBB`
    SgetBoolean { dst: u8, field: FieldIdx },
    /// `0x64` `sget-byte vAA, field@BBBB`
    SgetByte { dst: u8, field: FieldIdx },
    /// `0x65` `sget-char vAA, field@BBBB`
    SgetChar { dst: u8, field: FieldIdx },
    /// `0x66` `sget-short vAA, field@BBBB`
    SgetShort { dst: u8, field: FieldIdx },
    /// `0x67` `sput vAA, field@BBBB`
    Sput { src: u8, field: FieldIdx },
    /// `0x68` `sput-wide vAA, field@BBBB`
    SputWide { src: u8, field: FieldIdx },
    /// `0x69` `sput-object vAA, field@BBBB`
    SputObject { src: u8, field: FieldIdx },
    /// `0x6a` `sput-boolean vAA, field@BBBB`
    SputBoolean { src: u8, field: FieldIdx },
    /// `0x6b` `sput-byte vAA, field@BBBB`
    SputByte { src: u8, field: FieldIdx },
    /// `0x6c` `sput-char vAA, field@BBBB`
    SputChar { src: u8, field: FieldIdx },
    /// `0x6d` `sput-short vAA, field@BBBB`
    SputShort { src: u8, field: FieldIdx },

    // ── Invocations - 5-register form (35c) ─────────────────────────────────
    /// `0x6e` `invoke-virtual {vC..vG}, meth@BBBB`
    InvokeVirtual { method: MethodIdx, args: Args5 },
    /// `0x6f` `invoke-super {vC..vG}, meth@BBBB`
    InvokeSuper { method: MethodIdx, args: Args5 },
    /// `0x70` `invoke-direct {vC..vG}, meth@BBBB`
    InvokeDirect { method: MethodIdx, args: Args5 },
    /// `0x71` `invoke-static {vC..vG}, meth@BBBB`
    InvokeStatic { method: MethodIdx, args: Args5 },
    /// `0x72` `invoke-interface {vC..vG}, meth@BBBB`
    InvokeInterface { method: MethodIdx, args: Args5 },

    // ── Invocations - register-range form (3rc) ──────────────────────────────
    /// `0x74` `invoke-virtual/range {vCCCC..vNNNN}, meth@BBBB`
    InvokeVirtualRange { method: MethodIdx, first_reg: u16, count: u8 },
    /// `0x75` `invoke-super/range {vCCCC..vNNNN}, meth@BBBB`
    InvokeSuperRange { method: MethodIdx, first_reg: u16, count: u8 },
    /// `0x76` `invoke-direct/range {vCCCC..vNNNN}, meth@BBBB`
    InvokeDirectRange { method: MethodIdx, first_reg: u16, count: u8 },
    /// `0x77` `invoke-static/range {vCCCC..vNNNN}, meth@BBBB`
    InvokeStaticRange { method: MethodIdx, first_reg: u16, count: u8 },
    /// `0x78` `invoke-interface/range {vCCCC..vNNNN}, meth@BBBB`
    InvokeInterfaceRange { method: MethodIdx, first_reg: u16, count: u8 },

    // ── Unary operations (12x) ───────────────────────────────────────────────
    /// `0x7b` `neg-int vA, vB`
    NegInt { dst: u8, src: u8 },
    /// `0x7c` `not-int vA, vB`
    NotInt { dst: u8, src: u8 },
    /// `0x7d` `neg-long vA, vB`
    NegLong { dst: u8, src: u8 },
    /// `0x7e` `not-long vA, vB`
    NotLong { dst: u8, src: u8 },
    /// `0x7f` `neg-float vA, vB`
    NegFloat { dst: u8, src: u8 },
    /// `0x80` `neg-double vA, vB`
    NegDouble { dst: u8, src: u8 },
    /// `0x81` `int-to-long vA, vB`
    IntToLong { dst: u8, src: u8 },
    /// `0x82` `int-to-float vA, vB`
    IntToFloat { dst: u8, src: u8 },
    /// `0x83` `int-to-double vA, vB`
    IntToDouble { dst: u8, src: u8 },
    /// `0x84` `long-to-int vA, vB`
    LongToInt { dst: u8, src: u8 },
    /// `0x85` `long-to-float vA, vB`
    LongToFloat { dst: u8, src: u8 },
    /// `0x86` `long-to-double vA, vB`
    LongToDouble { dst: u8, src: u8 },
    /// `0x87` `float-to-int vA, vB`
    FloatToInt { dst: u8, src: u8 },
    /// `0x88` `float-to-long vA, vB`
    FloatToLong { dst: u8, src: u8 },
    /// `0x89` `float-to-double vA, vB`
    FloatToDouble { dst: u8, src: u8 },
    /// `0x8a` `double-to-int vA, vB`
    DoubleToInt { dst: u8, src: u8 },
    /// `0x8b` `double-to-long vA, vB`
    DoubleToLong { dst: u8, src: u8 },
    /// `0x8c` `double-to-float vA, vB`
    DoubleToFloat { dst: u8, src: u8 },
    /// `0x8d` `int-to-byte vA, vB`
    IntToByte { dst: u8, src: u8 },
    /// `0x8e` `int-to-char vA, vB`
    IntToChar { dst: u8, src: u8 },
    /// `0x8f` `int-to-short vA, vB`
    IntToShort { dst: u8, src: u8 },

    // ── Binary operations (23x) ──────────────────────────────────────────────
    /// `0x90` `add-int vAA, vBB, vCC`
    AddInt { dst: u8, a: u8, b: u8 },
    /// `0x91` `sub-int vAA, vBB, vCC`
    SubInt { dst: u8, a: u8, b: u8 },
    /// `0x92` `mul-int vAA, vBB, vCC`
    MulInt { dst: u8, a: u8, b: u8 },
    /// `0x93` `div-int vAA, vBB, vCC`
    DivInt { dst: u8, a: u8, b: u8 },
    /// `0x94` `rem-int vAA, vBB, vCC`
    RemInt { dst: u8, a: u8, b: u8 },
    /// `0x95` `and-int vAA, vBB, vCC`
    AndInt { dst: u8, a: u8, b: u8 },
    /// `0x96` `or-int vAA, vBB, vCC`
    OrInt { dst: u8, a: u8, b: u8 },
    /// `0x97` `xor-int vAA, vBB, vCC`
    XorInt { dst: u8, a: u8, b: u8 },
    /// `0x98` `shl-int vAA, vBB, vCC`
    ShlInt { dst: u8, a: u8, b: u8 },
    /// `0x99` `shr-int vAA, vBB, vCC`
    ShrInt { dst: u8, a: u8, b: u8 },
    /// `0x9a` `ushr-int vAA, vBB, vCC`
    UshrInt { dst: u8, a: u8, b: u8 },
    /// `0x9b` `add-long vAA, vBB, vCC`
    AddLong { dst: u8, a: u8, b: u8 },
    /// `0x9c` `sub-long vAA, vBB, vCC`
    SubLong { dst: u8, a: u8, b: u8 },
    /// `0x9d` `mul-long vAA, vBB, vCC`
    MulLong { dst: u8, a: u8, b: u8 },
    /// `0x9e` `div-long vAA, vBB, vCC`
    DivLong { dst: u8, a: u8, b: u8 },
    /// `0x9f` `rem-long vAA, vBB, vCC`
    RemLong { dst: u8, a: u8, b: u8 },
    /// `0xa0` `and-long vAA, vBB, vCC`
    AndLong { dst: u8, a: u8, b: u8 },
    /// `0xa1` `or-long vAA, vBB, vCC`
    OrLong { dst: u8, a: u8, b: u8 },
    /// `0xa2` `xor-long vAA, vBB, vCC`
    XorLong { dst: u8, a: u8, b: u8 },
    /// `0xa3` `shl-long vAA, vBB, vCC`
    ShlLong { dst: u8, a: u8, b: u8 },
    /// `0xa4` `shr-long vAA, vBB, vCC`
    ShrLong { dst: u8, a: u8, b: u8 },
    /// `0xa5` `ushr-long vAA, vBB, vCC`
    UshrLong { dst: u8, a: u8, b: u8 },
    /// `0xa6` `add-float vAA, vBB, vCC`
    AddFloat { dst: u8, a: u8, b: u8 },
    /// `0xa7` `sub-float vAA, vBB, vCC`
    SubFloat { dst: u8, a: u8, b: u8 },
    /// `0xa8` `mul-float vAA, vBB, vCC`
    MulFloat { dst: u8, a: u8, b: u8 },
    /// `0xa9` `div-float vAA, vBB, vCC`
    DivFloat { dst: u8, a: u8, b: u8 },
    /// `0xaa` `rem-float vAA, vBB, vCC`
    RemFloat { dst: u8, a: u8, b: u8 },
    /// `0xab` `add-double vAA, vBB, vCC`
    AddDouble { dst: u8, a: u8, b: u8 },
    /// `0xac` `sub-double vAA, vBB, vCC`
    SubDouble { dst: u8, a: u8, b: u8 },
    /// `0xad` `mul-double vAA, vBB, vCC`
    MulDouble { dst: u8, a: u8, b: u8 },
    /// `0xae` `div-double vAA, vBB, vCC`
    DivDouble { dst: u8, a: u8, b: u8 },
    /// `0xaf` `rem-double vAA, vBB, vCC`
    RemDouble { dst: u8, a: u8, b: u8 },

    // ── Binary 2addr (12x) - dst is also the first source ────────────────────
    /// `0xb0` `add-int/2addr vA, vB`
    AddInt2addr { dst: u8, src: u8 },
    /// `0xb1` `sub-int/2addr vA, vB`
    SubInt2addr { dst: u8, src: u8 },
    /// `0xb2` `mul-int/2addr vA, vB`
    MulInt2addr { dst: u8, src: u8 },
    /// `0xb3` `div-int/2addr vA, vB`
    DivInt2addr { dst: u8, src: u8 },
    /// `0xb4` `rem-int/2addr vA, vB`
    RemInt2addr { dst: u8, src: u8 },
    /// `0xb5` `and-int/2addr vA, vB`
    AndInt2addr { dst: u8, src: u8 },
    /// `0xb6` `or-int/2addr vA, vB`
    OrInt2addr { dst: u8, src: u8 },
    /// `0xb7` `xor-int/2addr vA, vB`
    XorInt2addr { dst: u8, src: u8 },
    /// `0xb8` `shl-int/2addr vA, vB`
    ShlInt2addr { dst: u8, src: u8 },
    /// `0xb9` `shr-int/2addr vA, vB`
    ShrInt2addr { dst: u8, src: u8 },
    /// `0xba` `ushr-int/2addr vA, vB`
    UshrInt2addr { dst: u8, src: u8 },
    /// `0xbb` `add-long/2addr vA, vB`
    AddLong2addr { dst: u8, src: u8 },
    /// `0xbc` `sub-long/2addr vA, vB`
    SubLong2addr { dst: u8, src: u8 },
    /// `0xbd` `mul-long/2addr vA, vB`
    MulLong2addr { dst: u8, src: u8 },
    /// `0xbe` `div-long/2addr vA, vB`
    DivLong2addr { dst: u8, src: u8 },
    /// `0xbf` `rem-long/2addr vA, vB`
    RemLong2addr { dst: u8, src: u8 },
    /// `0xc0` `and-long/2addr vA, vB`
    AndLong2addr { dst: u8, src: u8 },
    /// `0xc1` `or-long/2addr vA, vB`
    OrLong2addr { dst: u8, src: u8 },
    /// `0xc2` `xor-long/2addr vA, vB`
    XorLong2addr { dst: u8, src: u8 },
    /// `0xc3` `shl-long/2addr vA, vB`
    ShlLong2addr { dst: u8, src: u8 },
    /// `0xc4` `shr-long/2addr vA, vB`
    ShrLong2addr { dst: u8, src: u8 },
    /// `0xc5` `ushr-long/2addr vA, vB`
    UshrLong2addr { dst: u8, src: u8 },
    /// `0xc6` `add-float/2addr vA, vB`
    AddFloat2addr { dst: u8, src: u8 },
    /// `0xc7` `sub-float/2addr vA, vB`
    SubFloat2addr { dst: u8, src: u8 },
    /// `0xc8` `mul-float/2addr vA, vB`
    MulFloat2addr { dst: u8, src: u8 },
    /// `0xc9` `div-float/2addr vA, vB`
    DivFloat2addr { dst: u8, src: u8 },
    /// `0xca` `rem-float/2addr vA, vB`
    RemFloat2addr { dst: u8, src: u8 },
    /// `0xcb` `add-double/2addr vA, vB`
    AddDouble2addr { dst: u8, src: u8 },
    /// `0xcc` `sub-double/2addr vA, vB`
    SubDouble2addr { dst: u8, src: u8 },
    /// `0xcd` `mul-double/2addr vA, vB`
    MulDouble2addr { dst: u8, src: u8 },
    /// `0xce` `div-double/2addr vA, vB`
    DivDouble2addr { dst: u8, src: u8 },
    /// `0xcf` `rem-double/2addr vA, vB`
    RemDouble2addr { dst: u8, src: u8 },

    // ── Binary lit16 (22s) ───────────────────────────────────────────────────
    /// `0xd0` `add-int/lit16 vA, vB, #+CCCC`
    AddIntLit16 { dst: u8, src: u8, lit: i16 },
    /// `0xd1` `rsub-int vA, vB, #+CCCC` - reverse subtract
    RsubInt { dst: u8, src: u8, lit: i16 },
    /// `0xd2` `mul-int/lit16 vA, vB, #+CCCC`
    MulIntLit16 { dst: u8, src: u8, lit: i16 },
    /// `0xd3` `div-int/lit16 vA, vB, #+CCCC`
    DivIntLit16 { dst: u8, src: u8, lit: i16 },
    /// `0xd4` `rem-int/lit16 vA, vB, #+CCCC`
    RemIntLit16 { dst: u8, src: u8, lit: i16 },
    /// `0xd5` `and-int/lit16 vA, vB, #+CCCC`
    AndIntLit16 { dst: u8, src: u8, lit: i16 },
    /// `0xd6` `or-int/lit16 vA, vB, #+CCCC`
    OrIntLit16 { dst: u8, src: u8, lit: i16 },
    /// `0xd7` `xor-int/lit16 vA, vB, #+CCCC`
    XorIntLit16 { dst: u8, src: u8, lit: i16 },

    // ── Binary lit8 (22b) ────────────────────────────────────────────────────
    /// `0xd8` `add-int/lit8 vAA, vBB, #+CC`
    AddIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xd9` `rsub-int/lit8 vAA, vBB, #+CC`
    RsubIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xda` `mul-int/lit8 vAA, vBB, #+CC`
    MulIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xdb` `div-int/lit8 vAA, vBB, #+CC`
    DivIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xdc` `rem-int/lit8 vAA, vBB, #+CC`
    RemIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xdd` `and-int/lit8 vAA, vBB, #+CC`
    AndIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xde` `or-int/lit8 vAA, vBB, #+CC`
    OrIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xdf` `xor-int/lit8 vAA, vBB, #+CC`
    XorIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xe0` `shl-int/lit8 vAA, vBB, #+CC`
    ShlIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xe1` `shr-int/lit8 vAA, vBB, #+CC`
    ShrIntLit8 { dst: u8, src: u8, lit: i8 },
    /// `0xe2` `ushr-int/lit8 vAA, vBB, #+CC`
    UshrIntLit8 { dst: u8, src: u8, lit: i8 },

    // ── Polymorphic, custom, handles ─────────────────────────────────────────
    /// `0xfa` `invoke-polymorphic {vC..vG}, meth@BBBB, proto@HHHH` (45cc)
    InvokePolymorphic { method: MethodIdx, proto: ProtoIdx, args: Args5 },
    /// `0xfb` `invoke-polymorphic/range {vCCCC..vNNNN}, meth@BBBB, proto@HHHH` (4rcc)
    InvokePolymorphicRange { method: MethodIdx, proto: ProtoIdx, first_reg: u16, count: u8 },
    /// `0xfc` `invoke-custom {vC..vG}, call_site@BBBB` (35c)
    InvokeCustom { call_site: CallSiteIdx, args: Args5 },
    /// `0xfd` `invoke-custom/range {vCCCC..vNNNN}, call_site@BBBB` (3rc)
    InvokeCustomRange { call_site: CallSiteIdx, first_reg: u16, count: u8 },
    /// `0xfe` `const-method-handle vAA, method_handle@BBBB` (21c)
    ConstMethodHandle { dst: u8, handle: MethodHandleIdx },
    /// `0xff` `const-method-type vAA, proto@BBBB` (21c)
    ConstMethodType { dst: u8, proto: ProtoIdx },
}

impl Instruction {
    /// Number of 16-bit code units this instruction occupies.
    pub fn code_units(&self) -> usize {
        match self {
            // 1-unit (10x, 12x, 11n, 11x, 10t)
            Self::Nop
            | Self::ReturnVoid
            | Self::Return { .. }
            | Self::ReturnWide { .. }
            | Self::ReturnObject { .. }
            | Self::Move { .. }
            | Self::MoveWide { .. }
            | Self::MoveObject { .. }
            | Self::MoveResult { .. }
            | Self::MoveResultWide { .. }
            | Self::MoveResultObject { .. }
            | Self::MoveException { .. }
            | Self::Const4 { .. }
            | Self::MonitorEnter { .. }
            | Self::MonitorExit { .. }
            | Self::Throw { .. }
            | Self::Goto { .. }
            | Self::ArrayLength { .. }
            | Self::NegInt { .. }
            | Self::NotInt { .. }
            | Self::NegLong { .. }
            | Self::NotLong { .. }
            | Self::NegFloat { .. }
            | Self::NegDouble { .. }
            | Self::IntToLong { .. }
            | Self::IntToFloat { .. }
            | Self::IntToDouble { .. }
            | Self::LongToInt { .. }
            | Self::LongToFloat { .. }
            | Self::LongToDouble { .. }
            | Self::FloatToInt { .. }
            | Self::FloatToLong { .. }
            | Self::FloatToDouble { .. }
            | Self::DoubleToInt { .. }
            | Self::DoubleToLong { .. }
            | Self::DoubleToFloat { .. }
            | Self::IntToByte { .. }
            | Self::IntToChar { .. }
            | Self::IntToShort { .. }
            | Self::AddInt2addr { .. }
            | Self::SubInt2addr { .. }
            | Self::MulInt2addr { .. }
            | Self::DivInt2addr { .. }
            | Self::RemInt2addr { .. }
            | Self::AndInt2addr { .. }
            | Self::OrInt2addr { .. }
            | Self::XorInt2addr { .. }
            | Self::ShlInt2addr { .. }
            | Self::ShrInt2addr { .. }
            | Self::UshrInt2addr { .. }
            | Self::AddLong2addr { .. }
            | Self::SubLong2addr { .. }
            | Self::MulLong2addr { .. }
            | Self::DivLong2addr { .. }
            | Self::RemLong2addr { .. }
            | Self::AndLong2addr { .. }
            | Self::OrLong2addr { .. }
            | Self::XorLong2addr { .. }
            | Self::ShlLong2addr { .. }
            | Self::ShrLong2addr { .. }
            | Self::UshrLong2addr { .. }
            | Self::AddFloat2addr { .. }
            | Self::SubFloat2addr { .. }
            | Self::MulFloat2addr { .. }
            | Self::DivFloat2addr { .. }
            | Self::RemFloat2addr { .. }
            | Self::AddDouble2addr { .. }
            | Self::SubDouble2addr { .. }
            | Self::MulDouble2addr { .. }
            | Self::DivDouble2addr { .. }
            | Self::RemDouble2addr { .. } => 1,

            // 2-unit (22x, 21t, 21s, 21h, 21c, 23x, 22b, 22t, 22s, 22c, 20t)
            Self::MoveFrom16 { .. }
            | Self::MoveWideFrom16 { .. }
            | Self::MoveObjectFrom16 { .. }
            | Self::Const16 { .. }
            | Self::ConstHigh16 { .. }
            | Self::ConstWide16 { .. }
            | Self::ConstWideHigh16 { .. }
            | Self::ConstString { .. }
            | Self::ConstClass { .. }
            | Self::CheckCast { .. }
            | Self::InstanceOf { .. }
            | Self::NewInstance { .. }
            | Self::NewArray { .. }
            | Self::Goto16 { .. }
            | Self::CmplFloat { .. }
            | Self::CmpgFloat { .. }
            | Self::CmplDouble { .. }
            | Self::CmpgDouble { .. }
            | Self::CmpLong { .. }
            | Self::IfEq { .. }
            | Self::IfNe { .. }
            | Self::IfLt { .. }
            | Self::IfGe { .. }
            | Self::IfGt { .. }
            | Self::IfLe { .. }
            | Self::IfEqz { .. }
            | Self::IfNez { .. }
            | Self::IfLtz { .. }
            | Self::IfGez { .. }
            | Self::IfGtz { .. }
            | Self::IfLez { .. }
            | Self::Aget { .. }
            | Self::AgetWide { .. }
            | Self::AgetObject { .. }
            | Self::AgetBoolean { .. }
            | Self::AgetByte { .. }
            | Self::AgetChar { .. }
            | Self::AgetShort { .. }
            | Self::Aput { .. }
            | Self::AputWide { .. }
            | Self::AputObject { .. }
            | Self::AputBoolean { .. }
            | Self::AputByte { .. }
            | Self::AputChar { .. }
            | Self::AputShort { .. }
            | Self::Iget { .. }
            | Self::IgetWide { .. }
            | Self::IgetObject { .. }
            | Self::IgetBoolean { .. }
            | Self::IgetByte { .. }
            | Self::IgetChar { .. }
            | Self::IgetShort { .. }
            | Self::Iput { .. }
            | Self::IputWide { .. }
            | Self::IputObject { .. }
            | Self::IputBoolean { .. }
            | Self::IputByte { .. }
            | Self::IputChar { .. }
            | Self::IputShort { .. }
            | Self::Sget { .. }
            | Self::SgetWide { .. }
            | Self::SgetObject { .. }
            | Self::SgetBoolean { .. }
            | Self::SgetByte { .. }
            | Self::SgetChar { .. }
            | Self::SgetShort { .. }
            | Self::Sput { .. }
            | Self::SputWide { .. }
            | Self::SputObject { .. }
            | Self::SputBoolean { .. }
            | Self::SputByte { .. }
            | Self::SputChar { .. }
            | Self::SputShort { .. }
            | Self::AddInt { .. }
            | Self::SubInt { .. }
            | Self::MulInt { .. }
            | Self::DivInt { .. }
            | Self::RemInt { .. }
            | Self::AndInt { .. }
            | Self::OrInt { .. }
            | Self::XorInt { .. }
            | Self::ShlInt { .. }
            | Self::ShrInt { .. }
            | Self::UshrInt { .. }
            | Self::AddLong { .. }
            | Self::SubLong { .. }
            | Self::MulLong { .. }
            | Self::DivLong { .. }
            | Self::RemLong { .. }
            | Self::AndLong { .. }
            | Self::OrLong { .. }
            | Self::XorLong { .. }
            | Self::ShlLong { .. }
            | Self::ShrLong { .. }
            | Self::UshrLong { .. }
            | Self::AddFloat { .. }
            | Self::SubFloat { .. }
            | Self::MulFloat { .. }
            | Self::DivFloat { .. }
            | Self::RemFloat { .. }
            | Self::AddDouble { .. }
            | Self::SubDouble { .. }
            | Self::MulDouble { .. }
            | Self::DivDouble { .. }
            | Self::RemDouble { .. }
            | Self::AddIntLit16 { .. }
            | Self::RsubInt { .. }
            | Self::MulIntLit16 { .. }
            | Self::DivIntLit16 { .. }
            | Self::RemIntLit16 { .. }
            | Self::AndIntLit16 { .. }
            | Self::OrIntLit16 { .. }
            | Self::XorIntLit16 { .. }
            | Self::AddIntLit8 { .. }
            | Self::RsubIntLit8 { .. }
            | Self::MulIntLit8 { .. }
            | Self::DivIntLit8 { .. }
            | Self::RemIntLit8 { .. }
            | Self::AndIntLit8 { .. }
            | Self::OrIntLit8 { .. }
            | Self::XorIntLit8 { .. }
            | Self::ShlIntLit8 { .. }
            | Self::ShrIntLit8 { .. }
            | Self::UshrIntLit8 { .. }
            | Self::ConstMethodHandle { .. }
            | Self::ConstMethodType { .. } => 2,

            // 3-unit (32x, 31t, 31i, 31c, 30t, 35c, 3rc)
            Self::Move16 { .. }
            | Self::MoveWide16 { .. }
            | Self::MoveObject16 { .. }
            | Self::Const { .. }
            | Self::ConstWide32 { .. }
            | Self::ConstStringJumbo { .. }
            | Self::Goto32 { .. }
            | Self::PackedSwitch { .. }
            | Self::SparseSwitch { .. }
            | Self::FillArrayData { .. }
            | Self::FilledNewArray { .. }
            | Self::FilledNewArrayRange { .. }
            | Self::InvokeVirtual { .. }
            | Self::InvokeSuper { .. }
            | Self::InvokeDirect { .. }
            | Self::InvokeStatic { .. }
            | Self::InvokeInterface { .. }
            | Self::InvokeVirtualRange { .. }
            | Self::InvokeSuperRange { .. }
            | Self::InvokeDirectRange { .. }
            | Self::InvokeStaticRange { .. }
            | Self::InvokeInterfaceRange { .. }
            | Self::InvokeCustom { .. }
            | Self::InvokeCustomRange { .. } => 3,

            // 4-unit (45cc, 4rcc)
            Self::InvokePolymorphic { .. } | Self::InvokePolymorphicRange { .. } => 4,

            // 5-unit (51l)
            Self::ConstWide { .. } => 5,
        }
    }

    /// Returns `true` if this instruction transfers control flow.
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            Self::ReturnVoid
                | Self::Return { .. }
                | Self::ReturnWide { .. }
                | Self::ReturnObject { .. }
                | Self::Throw { .. }
                | Self::Goto { .. }
                | Self::Goto16 { .. }
                | Self::Goto32 { .. }
                | Self::PackedSwitch { .. }
                | Self::SparseSwitch { .. }
                | Self::IfEq { .. }
                | Self::IfNe { .. }
                | Self::IfLt { .. }
                | Self::IfGe { .. }
                | Self::IfGt { .. }
                | Self::IfLe { .. }
                | Self::IfEqz { .. }
                | Self::IfNez { .. }
                | Self::IfLtz { .. }
                | Self::IfGez { .. }
                | Self::IfGtz { .. }
                | Self::IfLez { .. }
                | Self::InvokeVirtual { .. }
                | Self::InvokeSuper { .. }
                | Self::InvokeDirect { .. }
                | Self::InvokeStatic { .. }
                | Self::InvokeInterface { .. }
                | Self::InvokeVirtualRange { .. }
                | Self::InvokeSuperRange { .. }
                | Self::InvokeDirectRange { .. }
                | Self::InvokeStaticRange { .. }
                | Self::InvokeInterfaceRange { .. }
                | Self::InvokePolymorphic { .. }
                | Self::InvokePolymorphicRange { .. }
                | Self::InvokeCustom { .. }
                | Self::InvokeCustomRange { .. }
        )
    }
}

// ── Low-level format helpers ───────────────────────────────────────────────────

#[inline]
fn high_byte(u: u16) -> u8 { (u >> 8) as u8 }

#[inline]
fn nibble_lo(u: u16) -> u8 { ((u >> 8) & 0xf) as u8 }

#[inline]
fn nibble_hi(u: u16) -> u8 { ((u >> 12) & 0xf) as u8 }

/// Sign-extend a 4-bit value (stored in the low nibble) to `i8`.
#[inline]
fn sign4(n: u8) -> i8 {
    // Shift into the high nibble of an i8, then arithmetic-right-shift back.
    (n << 4) as i8 >> 4
}

#[inline]
fn get(units: &[u16], i: usize) -> Option<u16> { units.get(i).copied() }

/// Combine two consecutive u16 units into a u32 (little-endian: lo first).
#[inline]
fn u32_at(units: &[u16], lo: usize, hi: usize) -> Option<u32> {
    Some((get(units, lo)? as u32) | ((get(units, hi)? as u32) << 16))
}

/// Decode a 35c-format argument list.
/// Returns `(index_raw, Args5)` on success.
fn decode_35c_args(units: &[u16]) -> Option<(u16, Args5)> {
    let count = nibble_hi(units[0]);
    let g     = nibble_lo(units[0]);
    let index = get(units, 1)?;
    let third = get(units, 2)?;
    Some((index, Args5 {
        count: count.min(5),
        regs: [
            (third & 0xf) as u8,
            ((third >> 4) & 0xf) as u8,
            ((third >> 8) & 0xf) as u8,
            ((third >> 12) & 0xf) as u8,
            g,
        ],
    }))
}

/// Decode a 3rc-format range.
/// Returns `(count, index_raw, first_reg)` on success.
fn decode_3rc(units: &[u16]) -> Option<(u8, u16, u16)> {
    Some((high_byte(units[0]), get(units, 1)?, get(units, 2)?))
}

// ── Main decode function ───────────────────────────────────────────────────────

/// Decode the first instruction from a DEX code unit slice.
///
/// Returns `Some((instruction, units_consumed))` on success.
/// Returns `None` for:
/// - empty slice
/// - reserved / unused opcode ranges
/// - opcode `0x00` with non-zero high byte (pseudo-instruction; use [`decode_pseudo`])
/// - slice too short for the instruction format
pub fn decode(units: &[u16]) -> Option<(Instruction, usize)> {
    let u0 = *units.first()?;
    let op = (u0 & 0xff) as u8;

    let insn: Instruction = match op {
        0x00 => {
            if (u0 >> 8) != 0 { return None; }
            Instruction::Nop
        }

        0x01 => Instruction::Move { dst: nibble_lo(u0), src: nibble_hi(u0) },
        0x02 => Instruction::MoveFrom16 { dst: high_byte(u0), src: get(units, 1)? },
        0x03 => Instruction::Move16 { dst: get(units, 1)?, src: get(units, 2)? },
        0x04 => Instruction::MoveWide { dst: nibble_lo(u0), src: nibble_hi(u0) },
        0x05 => Instruction::MoveWideFrom16 { dst: high_byte(u0), src: get(units, 1)? },
        0x06 => Instruction::MoveWide16 { dst: get(units, 1)?, src: get(units, 2)? },
        0x07 => Instruction::MoveObject { dst: nibble_lo(u0), src: nibble_hi(u0) },
        0x08 => Instruction::MoveObjectFrom16 { dst: high_byte(u0), src: get(units, 1)? },
        0x09 => Instruction::MoveObject16 { dst: get(units, 1)?, src: get(units, 2)? },
        0x0a => Instruction::MoveResult { dst: high_byte(u0) },
        0x0b => Instruction::MoveResultWide { dst: high_byte(u0) },
        0x0c => Instruction::MoveResultObject { dst: high_byte(u0) },
        0x0d => Instruction::MoveException { dst: high_byte(u0) },

        0x0e => Instruction::ReturnVoid,
        0x0f => Instruction::Return { src: high_byte(u0) },
        0x10 => Instruction::ReturnWide { src: high_byte(u0) },
        0x11 => Instruction::ReturnObject { src: high_byte(u0) },

        0x12 => Instruction::Const4 { dst: nibble_lo(u0), value: sign4(nibble_hi(u0)) },
        0x13 => Instruction::Const16 { dst: high_byte(u0), value: get(units, 1)? as i16 },
        0x14 => Instruction::Const { dst: high_byte(u0), value: u32_at(units, 1, 2)? as i32 },
        0x15 => Instruction::ConstHigh16 { dst: high_byte(u0), value: get(units, 1)? as i16 },
        0x16 => Instruction::ConstWide16 { dst: high_byte(u0), value: get(units, 1)? as i16 },
        0x17 => Instruction::ConstWide32 { dst: high_byte(u0), value: u32_at(units, 1, 2)? as i32 },
        0x18 => {
            let w0 = get(units, 1)? as u64;
            let w1 = get(units, 2)? as u64;
            let w2 = get(units, 3)? as u64;
            let w3 = get(units, 4)? as u64;
            Instruction::ConstWide {
                dst: high_byte(u0),
                value: (w0 | (w1 << 16) | (w2 << 32) | (w3 << 48)) as i64,
            }
        }
        0x19 => Instruction::ConstWideHigh16 { dst: high_byte(u0), value: get(units, 1)? as i16 },
        0x1a => Instruction::ConstString { dst: high_byte(u0), string: StringIdx(get(units, 1)?) },
        0x1b => Instruction::ConstStringJumbo {
            dst: high_byte(u0),
            string: StringIdx32(u32_at(units, 1, 2)?),
        },
        0x1c => Instruction::ConstClass { dst: high_byte(u0), ty: TypeIdx(get(units, 1)?) },

        0x1d => Instruction::MonitorEnter { reg: high_byte(u0) },
        0x1e => Instruction::MonitorExit { reg: high_byte(u0) },

        0x1f => Instruction::CheckCast { reg: high_byte(u0), ty: TypeIdx(get(units, 1)?) },
        0x20 => Instruction::InstanceOf {
            dst: nibble_lo(u0), obj: nibble_hi(u0), ty: TypeIdx(get(units, 1)?),
        },
        0x21 => Instruction::ArrayLength { dst: nibble_lo(u0), array: nibble_hi(u0) },
        0x22 => Instruction::NewInstance { dst: high_byte(u0), ty: TypeIdx(get(units, 1)?) },
        0x23 => Instruction::NewArray {
            dst: nibble_lo(u0), size: nibble_hi(u0), ty: TypeIdx(get(units, 1)?),
        },
        0x24 => {
            let (idx, args) = decode_35c_args(units)?;
            Instruction::FilledNewArray { ty: TypeIdx(idx), args }
        }
        0x25 => {
            let (count, idx, first) = decode_3rc(units)?;
            Instruction::FilledNewArrayRange { ty: TypeIdx(idx), first_reg: first, count }
        }
        0x26 => Instruction::FillArrayData {
            array: high_byte(u0), table_offset: u32_at(units, 1, 2)? as i32,
        },

        0x27 => Instruction::Throw { exception: high_byte(u0) },
        0x28 => Instruction::Goto { offset: high_byte(u0) as i8 },
        0x29 => Instruction::Goto16 { offset: get(units, 1)? as i16 },
        0x2a => Instruction::Goto32 { offset: u32_at(units, 1, 2)? as i32 },
        0x2b => Instruction::PackedSwitch {
            reg: high_byte(u0), table_offset: u32_at(units, 1, 2)? as i32,
        },
        0x2c => Instruction::SparseSwitch {
            reg: high_byte(u0), table_offset: u32_at(units, 1, 2)? as i32,
        },

        0x2d..=0x31 => {
            let u1 = get(units, 1)?;
            let (dst, a, b) = (high_byte(u0), u1 as u8, (u1 >> 8) as u8);
            match op {
                0x2d => Instruction::CmplFloat  { dst, a, b },
                0x2e => Instruction::CmpgFloat  { dst, a, b },
                0x2f => Instruction::CmplDouble { dst, a, b },
                0x30 => Instruction::CmpgDouble { dst, a, b },
                0x31 => Instruction::CmpLong    { dst, a, b },
                _ => unreachable!(),
            }
        }

        0x32 => Instruction::IfEq { a: nibble_lo(u0), b: nibble_hi(u0), offset: get(units, 1)? as i16 },
        0x33 => Instruction::IfNe { a: nibble_lo(u0), b: nibble_hi(u0), offset: get(units, 1)? as i16 },
        0x34 => Instruction::IfLt { a: nibble_lo(u0), b: nibble_hi(u0), offset: get(units, 1)? as i16 },
        0x35 => Instruction::IfGe { a: nibble_lo(u0), b: nibble_hi(u0), offset: get(units, 1)? as i16 },
        0x36 => Instruction::IfGt { a: nibble_lo(u0), b: nibble_hi(u0), offset: get(units, 1)? as i16 },
        0x37 => Instruction::IfLe { a: nibble_lo(u0), b: nibble_hi(u0), offset: get(units, 1)? as i16 },
        0x38 => Instruction::IfEqz { reg: high_byte(u0), offset: get(units, 1)? as i16 },
        0x39 => Instruction::IfNez { reg: high_byte(u0), offset: get(units, 1)? as i16 },
        0x3a => Instruction::IfLtz { reg: high_byte(u0), offset: get(units, 1)? as i16 },
        0x3b => Instruction::IfGez { reg: high_byte(u0), offset: get(units, 1)? as i16 },
        0x3c => Instruction::IfGtz { reg: high_byte(u0), offset: get(units, 1)? as i16 },
        0x3d => Instruction::IfLez { reg: high_byte(u0), offset: get(units, 1)? as i16 },

        0x3e..=0x43 => return None, // unused

        0x44..=0x51 => {
            let u1 = get(units, 1)?;
            let (r0, array, index) = (high_byte(u0), u1 as u8, (u1 >> 8) as u8);
            match op {
                0x44 => Instruction::Aget        { dst: r0, array, index },
                0x45 => Instruction::AgetWide    { dst: r0, array, index },
                0x46 => Instruction::AgetObject  { dst: r0, array, index },
                0x47 => Instruction::AgetBoolean { dst: r0, array, index },
                0x48 => Instruction::AgetByte    { dst: r0, array, index },
                0x49 => Instruction::AgetChar    { dst: r0, array, index },
                0x4a => Instruction::AgetShort   { dst: r0, array, index },
                0x4b => Instruction::Aput        { src: r0, array, index },
                0x4c => Instruction::AputWide    { src: r0, array, index },
                0x4d => Instruction::AputObject  { src: r0, array, index },
                0x4e => Instruction::AputBoolean { src: r0, array, index },
                0x4f => Instruction::AputByte    { src: r0, array, index },
                0x50 => Instruction::AputChar    { src: r0, array, index },
                0x51 => Instruction::AputShort   { src: r0, array, index },
                _ => unreachable!(),
            }
        }

        0x52..=0x5f => {
            let f = FieldIdx(get(units, 1)?);
            let (dst_src, obj) = (nibble_lo(u0), nibble_hi(u0));
            match op {
                0x52 => Instruction::Iget        { dst: dst_src, obj, field: f },
                0x53 => Instruction::IgetWide    { dst: dst_src, obj, field: f },
                0x54 => Instruction::IgetObject  { dst: dst_src, obj, field: f },
                0x55 => Instruction::IgetBoolean { dst: dst_src, obj, field: f },
                0x56 => Instruction::IgetByte    { dst: dst_src, obj, field: f },
                0x57 => Instruction::IgetChar    { dst: dst_src, obj, field: f },
                0x58 => Instruction::IgetShort   { dst: dst_src, obj, field: f },
                0x59 => Instruction::Iput        { src: dst_src, obj, field: f },
                0x5a => Instruction::IputWide    { src: dst_src, obj, field: f },
                0x5b => Instruction::IputObject  { src: dst_src, obj, field: f },
                0x5c => Instruction::IputBoolean { src: dst_src, obj, field: f },
                0x5d => Instruction::IputByte    { src: dst_src, obj, field: f },
                0x5e => Instruction::IputChar    { src: dst_src, obj, field: f },
                0x5f => Instruction::IputShort   { src: dst_src, obj, field: f },
                _ => unreachable!(),
            }
        }

        0x60..=0x6d => {
            let f = FieldIdx(get(units, 1)?);
            let r = high_byte(u0);
            match op {
                0x60 => Instruction::Sget        { dst: r, field: f },
                0x61 => Instruction::SgetWide    { dst: r, field: f },
                0x62 => Instruction::SgetObject  { dst: r, field: f },
                0x63 => Instruction::SgetBoolean { dst: r, field: f },
                0x64 => Instruction::SgetByte    { dst: r, field: f },
                0x65 => Instruction::SgetChar    { dst: r, field: f },
                0x66 => Instruction::SgetShort   { dst: r, field: f },
                0x67 => Instruction::Sput        { src: r, field: f },
                0x68 => Instruction::SputWide    { src: r, field: f },
                0x69 => Instruction::SputObject  { src: r, field: f },
                0x6a => Instruction::SputBoolean { src: r, field: f },
                0x6b => Instruction::SputByte    { src: r, field: f },
                0x6c => Instruction::SputChar    { src: r, field: f },
                0x6d => Instruction::SputShort   { src: r, field: f },
                _ => unreachable!(),
            }
        }

        0x6e..=0x72 => {
            let (idx, args) = decode_35c_args(units)?;
            match op {
                0x6e => Instruction::InvokeVirtual   { method: MethodIdx(idx), args },
                0x6f => Instruction::InvokeSuper     { method: MethodIdx(idx), args },
                0x70 => Instruction::InvokeDirect    { method: MethodIdx(idx), args },
                0x71 => Instruction::InvokeStatic    { method: MethodIdx(idx), args },
                0x72 => Instruction::InvokeInterface { method: MethodIdx(idx), args },
                _ => unreachable!(),
            }
        }

        0x73 => return None, // unused

        0x74..=0x78 => {
            let (count, idx, first) = decode_3rc(units)?;
            match op {
                0x74 => Instruction::InvokeVirtualRange   { method: MethodIdx(idx), first_reg: first, count },
                0x75 => Instruction::InvokeSuperRange     { method: MethodIdx(idx), first_reg: first, count },
                0x76 => Instruction::InvokeDirectRange    { method: MethodIdx(idx), first_reg: first, count },
                0x77 => Instruction::InvokeStaticRange    { method: MethodIdx(idx), first_reg: first, count },
                0x78 => Instruction::InvokeInterfaceRange { method: MethodIdx(idx), first_reg: first, count },
                _ => unreachable!(),
            }
        }

        0x79..=0x7a => return None, // unused

        0x7b..=0x8f => {
            let (dst, src) = (nibble_lo(u0), nibble_hi(u0));
            match op {
                0x7b => Instruction::NegInt      { dst, src },
                0x7c => Instruction::NotInt      { dst, src },
                0x7d => Instruction::NegLong     { dst, src },
                0x7e => Instruction::NotLong     { dst, src },
                0x7f => Instruction::NegFloat    { dst, src },
                0x80 => Instruction::NegDouble   { dst, src },
                0x81 => Instruction::IntToLong   { dst, src },
                0x82 => Instruction::IntToFloat  { dst, src },
                0x83 => Instruction::IntToDouble { dst, src },
                0x84 => Instruction::LongToInt   { dst, src },
                0x85 => Instruction::LongToFloat { dst, src },
                0x86 => Instruction::LongToDouble{ dst, src },
                0x87 => Instruction::FloatToInt  { dst, src },
                0x88 => Instruction::FloatToLong { dst, src },
                0x89 => Instruction::FloatToDouble{dst, src },
                0x8a => Instruction::DoubleToInt { dst, src },
                0x8b => Instruction::DoubleToLong{ dst, src },
                0x8c => Instruction::DoubleToFloat{dst, src },
                0x8d => Instruction::IntToByte   { dst, src },
                0x8e => Instruction::IntToChar   { dst, src },
                0x8f => Instruction::IntToShort  { dst, src },
                _ => unreachable!(),
            }
        }

        0x90..=0xaf => {
            let u1 = get(units, 1)?;
            let (dst, a, b) = (high_byte(u0), u1 as u8, (u1 >> 8) as u8);
            match op {
                0x90 => Instruction::AddInt    { dst, a, b },
                0x91 => Instruction::SubInt    { dst, a, b },
                0x92 => Instruction::MulInt    { dst, a, b },
                0x93 => Instruction::DivInt    { dst, a, b },
                0x94 => Instruction::RemInt    { dst, a, b },
                0x95 => Instruction::AndInt    { dst, a, b },
                0x96 => Instruction::OrInt     { dst, a, b },
                0x97 => Instruction::XorInt    { dst, a, b },
                0x98 => Instruction::ShlInt    { dst, a, b },
                0x99 => Instruction::ShrInt    { dst, a, b },
                0x9a => Instruction::UshrInt   { dst, a, b },
                0x9b => Instruction::AddLong   { dst, a, b },
                0x9c => Instruction::SubLong   { dst, a, b },
                0x9d => Instruction::MulLong   { dst, a, b },
                0x9e => Instruction::DivLong   { dst, a, b },
                0x9f => Instruction::RemLong   { dst, a, b },
                0xa0 => Instruction::AndLong   { dst, a, b },
                0xa1 => Instruction::OrLong    { dst, a, b },
                0xa2 => Instruction::XorLong   { dst, a, b },
                0xa3 => Instruction::ShlLong   { dst, a, b },
                0xa4 => Instruction::ShrLong   { dst, a, b },
                0xa5 => Instruction::UshrLong  { dst, a, b },
                0xa6 => Instruction::AddFloat  { dst, a, b },
                0xa7 => Instruction::SubFloat  { dst, a, b },
                0xa8 => Instruction::MulFloat  { dst, a, b },
                0xa9 => Instruction::DivFloat  { dst, a, b },
                0xaa => Instruction::RemFloat  { dst, a, b },
                0xab => Instruction::AddDouble { dst, a, b },
                0xac => Instruction::SubDouble { dst, a, b },
                0xad => Instruction::MulDouble { dst, a, b },
                0xae => Instruction::DivDouble { dst, a, b },
                0xaf => Instruction::RemDouble { dst, a, b },
                _ => unreachable!(),
            }
        }

        0xb0..=0xcf => {
            let (dst, src) = (nibble_lo(u0), nibble_hi(u0));
            match op {
                0xb0 => Instruction::AddInt2addr    { dst, src },
                0xb1 => Instruction::SubInt2addr    { dst, src },
                0xb2 => Instruction::MulInt2addr    { dst, src },
                0xb3 => Instruction::DivInt2addr    { dst, src },
                0xb4 => Instruction::RemInt2addr    { dst, src },
                0xb5 => Instruction::AndInt2addr    { dst, src },
                0xb6 => Instruction::OrInt2addr     { dst, src },
                0xb7 => Instruction::XorInt2addr    { dst, src },
                0xb8 => Instruction::ShlInt2addr    { dst, src },
                0xb9 => Instruction::ShrInt2addr    { dst, src },
                0xba => Instruction::UshrInt2addr   { dst, src },
                0xbb => Instruction::AddLong2addr   { dst, src },
                0xbc => Instruction::SubLong2addr   { dst, src },
                0xbd => Instruction::MulLong2addr   { dst, src },
                0xbe => Instruction::DivLong2addr   { dst, src },
                0xbf => Instruction::RemLong2addr   { dst, src },
                0xc0 => Instruction::AndLong2addr   { dst, src },
                0xc1 => Instruction::OrLong2addr    { dst, src },
                0xc2 => Instruction::XorLong2addr   { dst, src },
                0xc3 => Instruction::ShlLong2addr   { dst, src },
                0xc4 => Instruction::ShrLong2addr   { dst, src },
                0xc5 => Instruction::UshrLong2addr  { dst, src },
                0xc6 => Instruction::AddFloat2addr  { dst, src },
                0xc7 => Instruction::SubFloat2addr  { dst, src },
                0xc8 => Instruction::MulFloat2addr  { dst, src },
                0xc9 => Instruction::DivFloat2addr  { dst, src },
                0xca => Instruction::RemFloat2addr  { dst, src },
                0xcb => Instruction::AddDouble2addr { dst, src },
                0xcc => Instruction::SubDouble2addr { dst, src },
                0xcd => Instruction::MulDouble2addr { dst, src },
                0xce => Instruction::DivDouble2addr { dst, src },
                0xcf => Instruction::RemDouble2addr { dst, src },
                _ => unreachable!(),
            }
        }

        0xd0..=0xd7 => {
            let lit = get(units, 1)? as i16;
            let (dst, src) = (nibble_lo(u0), nibble_hi(u0));
            match op {
                0xd0 => Instruction::AddIntLit16 { dst, src, lit },
                0xd1 => Instruction::RsubInt     { dst, src, lit },
                0xd2 => Instruction::MulIntLit16 { dst, src, lit },
                0xd3 => Instruction::DivIntLit16 { dst, src, lit },
                0xd4 => Instruction::RemIntLit16 { dst, src, lit },
                0xd5 => Instruction::AndIntLit16 { dst, src, lit },
                0xd6 => Instruction::OrIntLit16  { dst, src, lit },
                0xd7 => Instruction::XorIntLit16 { dst, src, lit },
                _ => unreachable!(),
            }
        }

        0xd8..=0xe2 => {
            let u1 = get(units, 1)?;
            let (dst, src, lit) = (high_byte(u0), u1 as u8, (u1 >> 8) as i8);
            match op {
                0xd8 => Instruction::AddIntLit8  { dst, src, lit },
                0xd9 => Instruction::RsubIntLit8 { dst, src, lit },
                0xda => Instruction::MulIntLit8  { dst, src, lit },
                0xdb => Instruction::DivIntLit8  { dst, src, lit },
                0xdc => Instruction::RemIntLit8  { dst, src, lit },
                0xdd => Instruction::AndIntLit8  { dst, src, lit },
                0xde => Instruction::OrIntLit8   { dst, src, lit },
                0xdf => Instruction::XorIntLit8  { dst, src, lit },
                0xe0 => Instruction::ShlIntLit8  { dst, src, lit },
                0xe1 => Instruction::ShrIntLit8  { dst, src, lit },
                0xe2 => Instruction::UshrIntLit8 { dst, src, lit },
                _ => unreachable!(),
            }
        }

        0xe3..=0xf9 => return None, // unused

        0xfa => {
            let (method_idx, args) = decode_35c_args(units)?;
            let proto_idx = get(units, 3)?;
            Instruction::InvokePolymorphic {
                method: MethodIdx(method_idx),
                proto: ProtoIdx(proto_idx),
                args,
            }
        }
        0xfb => {
            let (count, method_idx, first) = decode_3rc(units)?;
            let proto_idx = get(units, 3)?;
            Instruction::InvokePolymorphicRange {
                method: MethodIdx(method_idx),
                proto: ProtoIdx(proto_idx),
                first_reg: first,
                count,
            }
        }
        0xfc => {
            let (idx, args) = decode_35c_args(units)?;
            Instruction::InvokeCustom { call_site: CallSiteIdx(idx), args }
        }
        0xfd => {
            let (count, idx, first) = decode_3rc(units)?;
            Instruction::InvokeCustomRange { call_site: CallSiteIdx(idx), first_reg: first, count }
        }
        0xfe => Instruction::ConstMethodHandle {
            dst: high_byte(u0), handle: MethodHandleIdx(get(units, 1)?),
        },
        0xff => Instruction::ConstMethodType {
            dst: high_byte(u0), proto: ProtoIdx(get(units, 1)?),
        },
    };

    let len = insn.code_units();
    Some((insn, len))
}

// ── Pseudo-instruction types ───────────────────────────────────────────────────

/// A decoded DEX pseudo-instruction (variable-length data embedded in the code
/// unit stream alongside real instructions).
///
/// Pseudo-instructions share the `0x00` opcode byte; they are distinguished by
/// the high byte of their first code unit:
/// - `0x01` — packed-switch data
/// - `0x02` — sparse-switch data
/// - `0x03` — fill-array data
///
/// Use [`pseudo_len`] to obtain the code-unit length without fully parsing, or
/// [`decode_pseudo`] to parse into a [`Pseudo`] value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pseudo {
    /// Packed-switch data — consecutive integer keys mapping to branch offsets.
    PackedSwitch(PackedSwitchData),
    /// Sparse-switch data — arbitrary sorted integer keys mapping to branch offsets.
    SparseSwitch(SparseSwitchData),
    /// Fill-array data — raw element bytes for `fill-array-data`.
    ArrayData(ArrayData),
}

/// Decoded packed-switch data table (ident `0x0100`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedSwitchData {
    /// First (lowest) key in the consecutive range.
    pub first_key: i32,
    /// Branch offsets (code units, relative to the originating `packed-switch` instruction).
    pub targets: Vec<i32>,
}

/// Decoded sparse-switch data table (ident `0x0200`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseSwitchData {
    /// Keys in ascending order.
    pub keys: Vec<i32>,
    /// Branch offsets corresponding to each key (same length as `keys`).
    pub targets: Vec<i32>,
}

/// Decoded fill-array-data payload (ident `0x0300`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayData {
    /// Width of each element in bytes (1, 2, 4, or 8).
    pub element_width: u16,
    /// Raw element bytes, `element_width * element_count` bytes in little-endian order.
    pub data: Vec<u8>,
}

/// Return the length in code units of the pseudo-instruction at `units[0]`.
///
/// `units[0]` must have `0x00` as its low byte and a non-zero high byte.
/// Returns `1` for unknown ident bytes so the instruction walker always advances.
pub fn pseudo_len(units: &[u16]) -> usize {
    if units.is_empty() { return 1; }
    match units[0] >> 8 {
        0x01 => {
            // packed-switch-data: ident(1) + size(1) + first_key(2) + 2*N targets
            let n = units.get(1).copied().unwrap_or(0) as usize;
            4 + 2 * n
        }
        0x02 => {
            // sparse-switch-data: ident(1) + size(1) + 2*N keys + 2*N targets
            let n = units.get(1).copied().unwrap_or(0) as usize;
            2 + 4 * n
        }
        0x03 => {
            // fill-array-data: ident(1) + element_width(1) + size(2) + ceil(N*W/2)
            let w = units.get(1).copied().unwrap_or(1) as usize;
            let n = if units.len() >= 4 {
                ((units[2] as u32) | ((units[3] as u32) << 16)) as usize
            } else { 0 };
            4 + (n * w + 1) / 2
        }
        _ => 1,
    }
}

/// Decode a pseudo-instruction from `units`.
///
/// `units` must start at the pseudo-instruction (opcode byte `0x00`, high byte
/// `0x01`, `0x02`, or `0x03`).  Returns `None` if the slice is too short or
/// the ident byte is not recognised.
pub fn decode_pseudo(units: &[u16]) -> Option<(Pseudo, usize)> {
    let ident = (units.first()? >> 8) as u8;
    let len = pseudo_len(units);
    if units.len() < len { return None; }

    let pseudo = match ident {
        0x01 => {
            let n = units[1] as usize;
            let first_key = u32_at(units, 2, 3)? as i32;
            let mut targets = Vec::with_capacity(n);
            for i in 0..n {
                targets.push(u32_at(units, 4 + i * 2, 5 + i * 2)? as i32);
            }
            Pseudo::PackedSwitch(PackedSwitchData { first_key, targets })
        }
        0x02 => {
            let n = units[1] as usize;
            let keys_base = 2usize;
            let tgts_base = 2 + 2 * n;
            let mut keys    = Vec::with_capacity(n);
            let mut targets = Vec::with_capacity(n);
            for i in 0..n {
                keys.push(u32_at(units, keys_base + i * 2, keys_base + i * 2 + 1)? as i32);
                targets.push(u32_at(units, tgts_base + i * 2, tgts_base + i * 2 + 1)? as i32);
            }
            Pseudo::SparseSwitch(SparseSwitchData { keys, targets })
        }
        0x03 => {
            let element_width = units[1];
            let n  = u32_at(units, 2, 3)? as usize;
            let w  = element_width as usize;
            let byte_count = n * w;
            let mut data = Vec::with_capacity(byte_count);
            let raw = &units[4..];
            for i in 0..byte_count {
                let word = raw[i / 2];
                data.push(if i % 2 == 0 { word as u8 } else { (word >> 8) as u8 });
            }
            Pseudo::ArrayData(ArrayData { element_width, data })
        }
        _ => return None,
    };
    Some((pseudo, len))
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nop() {
        let (insn, len) = decode(&[0x0000]).unwrap();
        assert_eq!(insn, Instruction::Nop);
        assert_eq!(len, 1);
    }

    #[test]
    fn pseudo_marker_returns_none() {
        assert!(decode(&[0x0100]).is_none());
    }

    #[test]
    fn move_12x() {
        // move v1, v2: opcode=0x01, nibble_lo=A=1 (bits 8-11), nibble_hi=B=2 (bits 12-15)
        // unit = 0x2101
        let (insn, len) = decode(&[0x2101]).unwrap();
        assert_eq!(insn, Instruction::Move { dst: 1, src: 2 });
        assert_eq!(len, 1);
    }

    #[test]
    fn move_from16() {
        let (insn, len) = decode(&[0x4202, 0x1234]).unwrap();
        assert_eq!(insn, Instruction::MoveFrom16 { dst: 0x42, src: 0x1234 });
        assert_eq!(len, 2);
    }

    #[test]
    fn const4_negative() {
        // const/4 v0, #-1: opcode=0x12, A=0, B=0xf
        let (insn, _) = decode(&[0xf012]).unwrap();
        assert_eq!(insn, Instruction::Const4 { dst: 0, value: -1 });
    }

    #[test]
    fn const4_positive() {
        // const/4 v3, #+7: A=3, B=7
        let (insn, _) = decode(&[0x7312]).unwrap();
        assert_eq!(insn, Instruction::Const4 { dst: 3, value: 7 });
    }

    #[test]
    fn const_wide_51l() {
        let units: [u16; 5] = [0x0018, 1, 0, 0, 0];
        let (insn, len) = decode(&units).unwrap();
        assert_eq!(insn, Instruction::ConstWide { dst: 0, value: 1 });
        assert_eq!(len, 5);
    }

    #[test]
    fn return_void() {
        let (insn, len) = decode(&[0x000e]).unwrap();
        assert_eq!(insn, Instruction::ReturnVoid);
        assert_eq!(len, 1);
    }

    #[test]
    fn goto_positive() {
        let (insn, len) = decode(&[0x0528]).unwrap();
        assert_eq!(insn, Instruction::Goto { offset: 5 });
        assert_eq!(len, 1);
    }

    #[test]
    fn goto_negative() {
        let (insn, _) = decode(&[0xff28]).unwrap();
        assert_eq!(insn, Instruction::Goto { offset: -1 });
    }

    #[test]
    fn if_eq_22t() {
        // if-eq v1, v2, +8: opcode=0x32, A=1, B=2, offset=8
        let (insn, len) = decode(&[0x2132, 0x0008]).unwrap();
        assert_eq!(insn, Instruction::IfEq { a: 1, b: 2, offset: 8 });
        assert_eq!(len, 2);
    }

    #[test]
    fn invoke_static_35c() {
        // invoke-static {v0}, meth@5: A=1, G=0, BBBB=5, third=0x0000
        let (insn, len) = decode(&[0x1071, 0x0005, 0x0000]).unwrap();
        match insn {
            Instruction::InvokeStatic { method, args } => {
                assert_eq!(method, MethodIdx(5));
                assert_eq!(args.count, 1);
                assert_eq!(args.regs[0], 0);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(len, 3);
    }

    #[test]
    fn add_int_2addr() {
        // add-int/2addr v3, v4: opcode=0xb0, A=3, B=4
        let (insn, len) = decode(&[0x43b0]).unwrap();
        assert_eq!(insn, Instruction::AddInt2addr { dst: 3, src: 4 });
        assert_eq!(len, 1);
    }

    #[test]
    fn add_int_lit8() {
        // add-int/lit8 v0, v1, #+7: opcode=0xd8, AA=0, BB=1, CC=7
        // unit[0] = 0x00d8, unit[1] = 0x0701
        let (insn, len) = decode(&[0x00d8, 0x0701]).unwrap();
        assert_eq!(insn, Instruction::AddIntLit8 { dst: 0, src: 1, lit: 7 });
        assert_eq!(len, 2);
    }

    #[test]
    fn iget_22c() {
        let (insn, len) = decode(&[0x2152, 0x0003]).unwrap();
        assert_eq!(insn, Instruction::Iget { dst: 1, obj: 2, field: FieldIdx(3) });
        assert_eq!(len, 2);
    }

    #[test]
    fn unused_returns_none() {
        assert!(decode(&[0x003e]).is_none());
        assert!(decode(&[0x0073]).is_none());
        assert!(decode(&[0x00e3]).is_none());
    }

    #[test]
    fn pseudo_len_packed() {
        // N=3 => len = 4 + 6 = 10
        let units = [0x0100u16, 3, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(pseudo_len(&units), 10);
    }

    #[test]
    fn decode_pseudo_packed() {
        let units: [u16; 10] = [0x0100, 3, 10, 0, 2, 0, 4, 0, 6, 0];
        let (pseudo, len) = decode_pseudo(&units).unwrap();
        assert_eq!(len, 10);
        match pseudo {
            Pseudo::PackedSwitch(d) => {
                assert_eq!(d.first_key, 10);
                assert_eq!(d.targets, [2, 4, 6]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn decode_pseudo_sparse() {
        let units: [u16; 10] = [0x0200, 2, 1, 0, 3, 0, 10, 0, 20, 0];
        let (pseudo, len) = decode_pseudo(&units).unwrap();
        assert_eq!(len, 10);
        match pseudo {
            Pseudo::SparseSwitch(d) => {
                assert_eq!(d.keys, [1, 3]);
                assert_eq!(d.targets, [10, 20]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn is_control_flow() {
        assert!(Instruction::ReturnVoid.is_control_flow());
        assert!(Instruction::Goto { offset: 0 }.is_control_flow());
        assert!(Instruction::IfEqz { reg: 0, offset: 0 }.is_control_flow());
        assert!(!Instruction::Nop.is_control_flow());
        assert!(!Instruction::AddInt { dst: 0, a: 1, b: 2 }.is_control_flow());
    }
}
