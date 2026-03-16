//! The [`ObjectModel`] trait and supporting types.

use wasm_encoder::{Instruction, ValType};
use wax_core::build::InstructionSink;

use crate::TypeHash;

// ── FieldValType ─────────────────────────────────────────────────────────────

/// The wasm-level value type of a field or array element.
///
/// This describes how a value is stored in linear memory (or in a WASMGC
/// struct/array), independent of how it is held in a DEX/JVM register.
///
/// ## Register ↔ memory conventions
///
/// The [`ObjectModel`] emit methods assume the following conventions between
/// the register representation (what lives in a wasm local) and the memory
/// representation (what the emit method loads or stores):
///
/// | `FieldValType` | Memory layout | Register representation |
/// |---|---|---|
/// | `I32` / `Ref` | 4-byte LE integer | `i32` wasm local |
/// | `I64` | 8-byte LE integer | `i64` wasm local |
/// | `F32` | 4-byte IEEE 754 float | `i32` bit-pattern |
/// | `F64` | 8-byte IEEE 754 double | `i64` bit-pattern |
/// | `I8S` | 1-byte, sign-extended on load | `i32` wasm local |
/// | `I8U` | 1-byte, zero-extended on load | `i32` wasm local |
/// | `I16S` | 2-byte, sign-extended on load | `i32` wasm local |
/// | `I16U` | 2-byte, zero-extended on load | `i32` wasm local |
///
/// The emit methods automatically handle reinterpretation for `F32`/`F64`
/// so the callers never need to emit `f32.reinterpret_i32` themselves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldValType {
    /// 32-bit integer (int, boolean stored as i32, etc.)
    I32,
    /// 64-bit integer (long).
    I64,
    /// 32-bit IEEE 754 float; loaded/stored as `f32` in memory, appears as
    /// `i32` bit-pattern in registers.
    F32,
    /// 64-bit IEEE 754 double; loaded/stored as `f64` in memory, appears as
    /// `i64` bit-pattern in registers.
    F64,
    /// 1-byte signed integer (byte); sign-extended to i32 on load.
    I8S,
    /// 1-byte unsigned integer (boolean); zero-extended to i32 on load.
    I8U,
    /// 2-byte signed integer (short); sign-extended to i32 on load.
    I16S,
    /// 2-byte unsigned integer (char); zero-extended to i32 on load.
    I16U,
    /// Object reference (4 bytes in linear memory).
    Ref,
}

impl FieldValType {
    /// Number of bytes this type occupies in the object/array data area.
    pub const fn size_bytes(self) -> u32 {
        match self {
            FieldValType::I8S | FieldValType::I8U => 1,
            FieldValType::I16S | FieldValType::I16U => 2,
            FieldValType::I32 | FieldValType::F32 | FieldValType::Ref => 4,
            FieldValType::I64 | FieldValType::F64 => 8,
        }
    }

    /// Natural alignment of this type as `log2(bytes)`, for use in `MemArg`.
    pub const fn align(self) -> u32 {
        match self {
            FieldValType::I8S | FieldValType::I8U => 0,
            FieldValType::I16S | FieldValType::I16U => 1,
            FieldValType::I32 | FieldValType::F32 | FieldValType::Ref => 2,
            FieldValType::I64 | FieldValType::F64 => 3,
        }
    }
}

// ── ObjectModel trait ─────────────────────────────────────────────────────────

/// Pluggable object-model for managed-runtime recompilers.
///
/// An implementation decides how heap objects and arrays are represented in
/// wasm, and emits wasm instruction sequences to allocate, access, and type-
/// check them.  Each method emits instructions directly into an
/// [`InstructionSink`], allowing implementations to write to the reactor
/// without an intermediate buffer.
///
/// ## Object layout contract
///
/// The trait does *not* mandate a specific layout, but the provided
/// [`LinearMemoryObjects`](crate::LinearMemoryObjects) implementation uses:
///
/// ```text
/// Offset  Size  Field
/// 0       32    type_hash: [u8; 32]   — SHA3-256 of class name; primitive sentinel for arrays
/// 32       4    array_dim: u32        — 0 for plain objects, n for n-dimensional arrays
/// 36      ...   data
///                   array_dim == 0 → instance fields at fixed byte offsets
///                   array_dim  > 0 → [length: u32][elements...]
/// ```
///
/// ## Stack conventions
///
/// Each method documents the wasm operand stack state before and after.
/// Object references are always the first thing pushed (below any indices or
/// values).
pub trait ObjectModel<C, E> {
    /// The wasm [`ValType`] used to represent object references.
    ///
    /// `ValType::I32` for linear-memory models; `ValType::Ref(…)` for WASMGC.
    fn ref_val_type(&self) -> ValType;

    // ── Allocation ────────────────────────────────────────────────────────

    /// Emit wasm to allocate a new plain (non-array) object.
    ///
    /// - `hash`: SHA3-256 of the class name (see [`TypeHash::of_class`]).
    /// - `data_size`: total byte size of all instance fields.
    ///
    /// Stack before: `[]`  
    /// Stack after:  `[ref]`
    fn emit_new_object(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        hash: &TypeHash,
        data_size: u32,
    ) -> Result<(), E>;

    /// Emit wasm to allocate a new array.
    ///
    /// - `elem_hash`: type hash of the *element* type (see [`TypeHash`]).
    /// - `dim`: array dimension (1 for `T[]`, 2 for `T[][]`, …).
    /// - `elem_bytes`: byte size of each element.
    ///
    /// Stack before: `[length: i32]`  
    /// Stack after:  `[ref]`
    fn emit_new_array(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        elem_hash: &TypeHash,
        dim: u32,
        elem_bytes: u32,
    ) -> Result<(), E>;

    // ── Instance fields ───────────────────────────────────────────────────

    /// Emit wasm to load an instance field.
    ///
    /// - `byte_offset`: offset of the field within the object's data area
    ///   (i.e., relative to `OBJECT_HEADER_SIZE`, not the absolute object pointer).
    /// - `ty`: memory type of the field (controls load instruction and any reinterpretation).
    ///
    /// Stack before: `[ref]`  
    /// Stack after:  `[value]` — type depends on `ty`:
    ///   - `I32`/`Ref`/`I8S`/`I8U`/`I16S`/`I16U` → `i32`
    ///   - `I64` → `i64`
    ///   - `F32` → `i32` (bit-pattern; reinterpretation is done internally)
    ///   - `F64` → `i64` (bit-pattern; reinterpretation is done internally)
    fn emit_iget(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        byte_offset: u32,
        ty: FieldValType,
    ) -> Result<(), E>;

    /// Emit wasm to store an instance field.
    ///
    /// Stack before: `[ref, value]` — `value` type depends on `ty`:
    ///   - `I32`/`Ref`/`I8S`/`I8U`/`I16S`/`I16U` → `i32`
    ///   - `I64` → `i64`
    ///   - `F32` → `i32` bit-pattern (reinterpretation is done internally)
    ///   - `F64` → `i64` bit-pattern (reinterpretation is done internally)  
    /// Stack after: `[]`
    fn emit_iput(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        byte_offset: u32,
        ty: FieldValType,
    ) -> Result<(), E>;

    // ── Array access ──────────────────────────────────────────────────────

    /// Emit wasm to load one array element.
    ///
    /// Stack before: `[ref, index: i32]`  
    /// Stack after:  `[value]` — same type convention as [`emit_iget`](Self::emit_iget).
    fn emit_aget(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        ty: FieldValType,
    ) -> Result<(), E>;

    /// Emit wasm to store one array element.
    ///
    /// `scratch_i32` and `scratch_i64` are the absolute wasm local indices
    /// (obtained via [`LocalLayout::base`](yecta::LocalLayout::base)) of
    /// `i32` and `i64` temporaries; used while computing the element address.
    ///
    /// Stack before: `[ref, index: i32, value]` — same type convention as
    /// [`emit_iput`](Self::emit_iput).  
    /// Stack after: `[]`
    fn emit_aput(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        ty: FieldValType,
        scratch_i32: u32,
        scratch_i64: u32,
    ) -> Result<(), E>;

    /// Emit wasm to read the length of an array.
    ///
    /// Stack before: `[ref]`  
    /// Stack after:  `[length: i32]`
    fn emit_array_length(&self, ctx: &mut C, sink: &mut dyn InstructionSink<C, E>)
    -> Result<(), E>;

    // ── Type checks ───────────────────────────────────────────────────────

    /// Emit wasm to test whether an object is an instance of a type.
    ///
    /// Returns 0 for `null` (a null reference is never an instance of anything).
    ///
    /// - `hash`: type hash of the *base* class (no array brackets).
    /// - `dim`: expected array dimension (0 for non-arrays).
    /// - `scratch`: absolute wasm local index of an `i32` temporary
    ///   (obtained via [`LocalLayout::base`](yecta::LocalLayout::base)).
    ///
    /// Stack before: `[ref]`  
    /// Stack after:  `[result: i32]` — 1 if the object is an instance, 0 otherwise.
    fn emit_instanceof(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        hash: &TypeHash,
        dim: u32,
        scratch: u32,
    ) -> Result<(), E>;

    /// Emit wasm to assert that an object is an instance of a type, throwing
    /// a `ClassCastException` (or equivalent) if not.
    ///
    /// A null reference silently passes (matching Java semantics).
    ///
    /// - `hash`: type hash of the *base* class.
    /// - `dim`: expected array dimension.
    /// - `scratch`: absolute wasm local index of an `i32` temporary.
    ///
    /// Stack before: `[ref]`  
    /// Stack after:  `[]`
    fn emit_check_cast(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        hash: &TypeHash,
        dim: u32,
        scratch: u32,
    ) -> Result<(), E>;
}

// ── NoObjectModel ─────────────────────────────────────────────────────────────

/// A stub [`ObjectModel`] that emits `unreachable` for every operation.
///
/// This is the default model used by [`DexRecompiler::new`].  It lets you
/// translate DEX code that does not use any object operations without
/// supplying a real runtime.  Any instruction that touches the object model
/// at runtime will trap.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoObjectModel;

impl<C, E> ObjectModel<C, E> for NoObjectModel {
    fn ref_val_type(&self) -> ValType {
        ValType::I32
    }

    fn emit_new_object(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        _hash: &TypeHash,
        _data_size: u32,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }

    fn emit_new_array(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        _elem_hash: &TypeHash,
        _dim: u32,
        _elem_bytes: u32,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }

    fn emit_iget(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        _byte_offset: u32,
        _ty: FieldValType,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }

    fn emit_iput(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        _byte_offset: u32,
        _ty: FieldValType,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }

    fn emit_aget(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        _ty: FieldValType,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }

    fn emit_aput(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        _ty: FieldValType,
        _scratch_i32: u32,
        _scratch_i64: u32,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }

    fn emit_array_length(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }

    fn emit_instanceof(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        _hash: &TypeHash,
        _dim: u32,
        _scratch: u32,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }

    fn emit_check_cast(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        _hash: &TypeHash,
        _dim: u32,
        _scratch: u32,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::Unreachable)
    }
}
