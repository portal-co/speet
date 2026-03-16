//! Linear-memory [`ObjectModel`] implementation.

use wasm_encoder::{BlockType, Instruction, MemArg, ValType};
use wax_core::build::InstructionSink;

use crate::{FieldValType, ObjectModel, TypeHash};

// ── Layout constants ──────────────────────────────────────────────────────────

/// Byte size of the object header: 32 bytes for the type hash + 4 bytes for
/// the array dimension.
pub const OBJECT_HEADER_SIZE: u32 = 36;

/// Byte offset of the array length field within a heap object.
///
/// Only meaningful when `array_dim > 0`.  The length is a 32-bit little-endian
/// unsigned integer stored immediately after the header.
pub const ARRAY_LENGTH_OFFSET: u32 = OBJECT_HEADER_SIZE; // == 36

/// Byte offset of the first array element within a heap object.
///
/// Only meaningful when `array_dim > 0`.  Elements begin 4 bytes after the
/// length field.
pub const ARRAY_DATA_OFFSET: u32 = OBJECT_HEADER_SIZE + 4; // == 40

// ── LinearMemoryObjects ───────────────────────────────────────────────────────

/// Object model for linear-memory managed runtimes.
///
/// Objects are laid out in linear memory as:
///
/// ```text
/// Offset  Size  Field
/// 0       32    type_hash: [u8; 32]
///                   SHA3-256 of the class name (no brackets) for reference types;
///                   all-zero except last byte for primitive array element types.
/// 32       4    array_dim: u32 (little-endian)
///                   0 for plain objects; n for n-dimensional arrays.
/// 36      ...   data
///                   array_dim == 0 → instance fields packed at ascending byte offsets
///                   array_dim  > 0 → [length: u32][elements: elem_size × length]
/// ```
///
/// Object references are `i32` pointers into linear memory (memory index 0).
///
/// ## Runtime functions
///
/// The allocators and exception thrower are wasm functions identified by
/// their indices.  The expected signatures are:
///
/// | Field | Wasm signature |
/// |---|---|
/// | `alloc_object_fn` | `(h0: i64, h1: i64, h2: i64, h3: i64, data_bytes: i32) → i32` |
/// | `alloc_array_fn`  | `(length: i32, h0: i64, h1: i64, h2: i64, h3: i64, dim: i32, elem_bytes: i32) → i32` |
/// | `throw_class_cast_fn` | `() → (unreachable)` |
///
/// The hash chunks `h0..h3` are the four little-endian `i64` values that
/// cover all 32 bytes of the [`TypeHash`] (see [`TypeHash::as_i64_chunks`]).
/// The runtime writes the full header (hash + array_dim) before returning.
pub struct LinearMemoryObjects {
    /// Wasm function index for the object allocator.
    ///
    /// Called by [`emit_new_object`](ObjectModel::emit_new_object).
    pub alloc_object_fn: u32,

    /// Wasm function index for the array allocator.
    ///
    /// Called by [`emit_new_array`](ObjectModel::emit_new_array).
    pub alloc_array_fn: u32,

    /// Wasm function index for the `ClassCastException` thrower.
    ///
    /// Called when a [`check-cast`](ObjectModel::emit_check_cast) fails.
    /// Must not return.
    pub throw_class_cast_fn: u32,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn emit_load<C, E>(
    sink: &mut dyn InstructionSink<C, E>,
    ctx: &mut C,
    byte_offset: u32,
    ty: FieldValType,
) -> Result<(), E> {
    let offset = (OBJECT_HEADER_SIZE + byte_offset) as u64;
    let memarg = MemArg {
        offset,
        align: 0,
        memory_index: 0,
    };
    match ty {
        FieldValType::I32 | FieldValType::Ref => {
            sink.instruction(ctx, &Instruction::I32Load(memarg))?
        }
        FieldValType::I64 => sink.instruction(ctx, &Instruction::I64Load(memarg))?,
        FieldValType::F32 => {
            sink.instruction(ctx, &Instruction::F32Load(memarg))?;
            sink.instruction(ctx, &Instruction::I32ReinterpretF32)?;
        }
        FieldValType::F64 => {
            sink.instruction(ctx, &Instruction::F64Load(memarg))?;
            sink.instruction(ctx, &Instruction::I64ReinterpretF64)?;
        }
        FieldValType::I8S => sink.instruction(ctx, &Instruction::I32Load8S(memarg))?,
        FieldValType::I8U => sink.instruction(ctx, &Instruction::I32Load8U(memarg))?,
        FieldValType::I16S => sink.instruction(ctx, &Instruction::I32Load16S(memarg))?,
        FieldValType::I16U => sink.instruction(ctx, &Instruction::I32Load16U(memarg))?,
    }
    Ok(())
}

fn emit_store<C, E>(
    sink: &mut dyn InstructionSink<C, E>,
    ctx: &mut C,
    byte_offset: u32,
    ty: FieldValType,
) -> Result<(), E> {
    let offset = (OBJECT_HEADER_SIZE + byte_offset) as u64;
    let memarg = MemArg {
        offset,
        align: 0,
        memory_index: 0,
    };
    match ty {
        FieldValType::I32 | FieldValType::Ref => {
            sink.instruction(ctx, &Instruction::I32Store(memarg))?
        }
        FieldValType::I64 => sink.instruction(ctx, &Instruction::I64Store(memarg))?,
        FieldValType::F32 => {
            sink.instruction(ctx, &Instruction::F32ReinterpretI32)?;
            sink.instruction(ctx, &Instruction::F32Store(memarg))?;
        }
        FieldValType::F64 => {
            sink.instruction(ctx, &Instruction::F64ReinterpretI64)?;
            sink.instruction(ctx, &Instruction::F64Store(memarg))?;
        }
        FieldValType::I8S | FieldValType::I8U => {
            sink.instruction(ctx, &Instruction::I32Store8(memarg))?
        }
        FieldValType::I16S | FieldValType::I16U => {
            sink.instruction(ctx, &Instruction::I32Store16(memarg))?
        }
    }
    Ok(())
}

/// Emit the type-hash comparison sequence used by instanceof / check-cast.
///
/// Assumes `scratch` has already been set to the object reference via
/// `local.set(scratch)`.  Leaves a single `i32` (0 or 1) on the stack
/// representing whether the object's stored hash matches `(hash, dim)`.
fn emit_hash_compare<C, E>(
    sink: &mut dyn InstructionSink<C, E>,
    ctx: &mut C,
    hash: &TypeHash,
    dim: u32,
    scratch: u32,
) -> Result<(), E> {
    let chunks = hash.as_i32_chunks();
    let mut first = true;
    for (i, &expected) in chunks.iter().enumerate() {
        sink.instruction(ctx, &Instruction::LocalGet(scratch))?;
        sink.instruction(
            ctx,
            &Instruction::I32Load(MemArg {
                offset: (i as u64) * 4,
                align: 0,
                memory_index: 0,
            }),
        )?;
        sink.instruction(ctx, &Instruction::I32Const(expected))?;
        sink.instruction(ctx, &Instruction::I32Eq)?;
        if !first {
            sink.instruction(ctx, &Instruction::I32And)?;
        }
        first = false;
    }
    // Compare stored array_dim against expected.
    sink.instruction(ctx, &Instruction::LocalGet(scratch))?;
    sink.instruction(
        ctx,
        &Instruction::I32Load(MemArg {
            offset: 32,
            align: 0,
            memory_index: 0,
        }),
    )?;
    sink.instruction(ctx, &Instruction::I32Const(dim as i32))?;
    sink.instruction(ctx, &Instruction::I32Eq)?;
    sink.instruction(ctx, &Instruction::I32And)?;
    Ok(())
}

// ── ObjectModel impl ──────────────────────────────────────────────────────────

impl<C, E> ObjectModel<C, E> for LinearMemoryObjects {
    fn ref_val_type(&self) -> ValType {
        ValType::I32
    }

    fn emit_new_object(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        hash: &TypeHash,
        data_size: u32,
    ) -> Result<(), E> {
        let [h0, h1, h2, h3] = hash.as_i64_chunks();
        sink.instruction(ctx, &Instruction::I64Const(h0))?;
        sink.instruction(ctx, &Instruction::I64Const(h1))?;
        sink.instruction(ctx, &Instruction::I64Const(h2))?;
        sink.instruction(ctx, &Instruction::I64Const(h3))?;
        sink.instruction(ctx, &Instruction::I32Const(data_size as i32))?;
        sink.instruction(ctx, &Instruction::Call(self.alloc_object_fn))?;
        Ok(())
    }

    fn emit_new_array(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        elem_hash: &TypeHash,
        dim: u32,
        elem_bytes: u32,
    ) -> Result<(), E> {
        // Stack before: [length: i32]
        // alloc_array_fn(length, h0, h1, h2, h3, dim, elem_bytes) → i32
        let [h0, h1, h2, h3] = elem_hash.as_i64_chunks();
        sink.instruction(ctx, &Instruction::I64Const(h0))?;
        sink.instruction(ctx, &Instruction::I64Const(h1))?;
        sink.instruction(ctx, &Instruction::I64Const(h2))?;
        sink.instruction(ctx, &Instruction::I64Const(h3))?;
        sink.instruction(ctx, &Instruction::I32Const(dim as i32))?;
        sink.instruction(ctx, &Instruction::I32Const(elem_bytes as i32))?;
        sink.instruction(ctx, &Instruction::Call(self.alloc_array_fn))?;
        Ok(())
    }

    fn emit_iget(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        byte_offset: u32,
        ty: FieldValType,
    ) -> Result<(), E> {
        emit_load(sink, ctx, byte_offset, ty)
    }

    fn emit_iput(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        byte_offset: u32,
        ty: FieldValType,
    ) -> Result<(), E> {
        emit_store(sink, ctx, byte_offset, ty)
    }

    fn emit_aget(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        ty: FieldValType,
    ) -> Result<(), E> {
        // Stack before: [arr_ref: i32, index: i32]
        // Compute addr = arr_ref + ARRAY_DATA_OFFSET + index * elem_bytes
        let elem_bytes = ty.size_bytes();
        sink.instruction(ctx, &Instruction::I32Const(elem_bytes as i32))?;
        sink.instruction(ctx, &Instruction::I32Mul)?;
        sink.instruction(ctx, &Instruction::I32Const(ARRAY_DATA_OFFSET as i32))?;
        sink.instruction(ctx, &Instruction::I32Add)?;
        sink.instruction(ctx, &Instruction::I32Add)?; // arr_ref + offset
        // Load value — offset=0 since we computed the exact address above.
        let memarg = MemArg {
            offset: 0,
            align: 0,
            memory_index: 0,
        };
        match ty {
            FieldValType::I32 | FieldValType::Ref => {
                sink.instruction(ctx, &Instruction::I32Load(memarg))?
            }
            FieldValType::I64 => sink.instruction(ctx, &Instruction::I64Load(memarg))?,
            FieldValType::F32 => {
                sink.instruction(ctx, &Instruction::F32Load(memarg))?;
                sink.instruction(ctx, &Instruction::I32ReinterpretF32)?;
            }
            FieldValType::F64 => {
                sink.instruction(ctx, &Instruction::F64Load(memarg))?;
                sink.instruction(ctx, &Instruction::I64ReinterpretF64)?;
            }
            FieldValType::I8S => sink.instruction(ctx, &Instruction::I32Load8S(memarg))?,
            FieldValType::I8U => sink.instruction(ctx, &Instruction::I32Load8U(memarg))?,
            FieldValType::I16S => sink.instruction(ctx, &Instruction::I32Load16S(memarg))?,
            FieldValType::I16U => sink.instruction(ctx, &Instruction::I32Load16U(memarg))?,
        }
        Ok(())
    }

    fn emit_aput(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        ty: FieldValType,
        scratch_i32: u32,
        scratch_i64: u32,
    ) -> Result<(), E> {
        // Stack before: [arr_ref: i32, index: i32, value]
        //
        // We need [addr, value] for the store, where addr = arr_ref + ARRAY_DATA_OFFSET + idx*ebs.
        // Since the value is on top, we save it, compute the address, then restore.
        let elem_bytes = ty.size_bytes();
        let is_wide = matches!(ty, FieldValType::I64 | FieldValType::F64);

        if is_wide {
            sink.instruction(ctx, &Instruction::LocalSet(scratch_i64))?;
        } else {
            sink.instruction(ctx, &Instruction::LocalSet(scratch_i32))?;
        }

        // Stack: [arr_ref, index]
        sink.instruction(ctx, &Instruction::I32Const(elem_bytes as i32))?;
        sink.instruction(ctx, &Instruction::I32Mul)?;
        sink.instruction(ctx, &Instruction::I32Const(ARRAY_DATA_OFFSET as i32))?;
        sink.instruction(ctx, &Instruction::I32Add)?;
        sink.instruction(ctx, &Instruction::I32Add)?; // addr

        let memarg = MemArg {
            offset: 0,
            align: 0,
            memory_index: 0,
        };
        match ty {
            FieldValType::I32 | FieldValType::Ref => {
                sink.instruction(ctx, &Instruction::LocalGet(scratch_i32))?;
                sink.instruction(ctx, &Instruction::I32Store(memarg))?;
            }
            FieldValType::I64 => {
                sink.instruction(ctx, &Instruction::LocalGet(scratch_i64))?;
                sink.instruction(ctx, &Instruction::I64Store(memarg))?;
            }
            FieldValType::F32 => {
                sink.instruction(ctx, &Instruction::LocalGet(scratch_i32))?;
                sink.instruction(ctx, &Instruction::F32ReinterpretI32)?;
                sink.instruction(ctx, &Instruction::F32Store(memarg))?;
            }
            FieldValType::F64 => {
                sink.instruction(ctx, &Instruction::LocalGet(scratch_i64))?;
                sink.instruction(ctx, &Instruction::F64ReinterpretI64)?;
                sink.instruction(ctx, &Instruction::F64Store(memarg))?;
            }
            FieldValType::I8S | FieldValType::I8U => {
                sink.instruction(ctx, &Instruction::LocalGet(scratch_i32))?;
                sink.instruction(ctx, &Instruction::I32Store8(memarg))?;
            }
            FieldValType::I16S | FieldValType::I16U => {
                sink.instruction(ctx, &Instruction::LocalGet(scratch_i32))?;
                sink.instruction(ctx, &Instruction::I32Store16(memarg))?;
            }
        }
        Ok(())
    }

    fn emit_array_length(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
    ) -> Result<(), E> {
        // Stack: [arr_ref] → [length: i32]
        sink.instruction(
            ctx,
            &Instruction::I32Load(MemArg {
                offset: ARRAY_LENGTH_OFFSET as u64,
                align: 0,
                memory_index: 0,
            }),
        )
    }

    fn emit_instanceof(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        hash: &TypeHash,
        dim: u32,
        scratch: u32,
    ) -> Result<(), E> {
        // Stack: [ref] → [i32 (0 or 1)]
        //
        // null → 0;  type match → 1;  type mismatch → 0.
        sink.instruction(ctx, &Instruction::LocalSet(scratch))?;

        sink.instruction(ctx, &Instruction::LocalGet(scratch))?;
        sink.instruction(ctx, &Instruction::I32Eqz)?;
        sink.instruction(ctx, &Instruction::If(BlockType::Result(ValType::I32)))?;
        sink.instruction(ctx, &Instruction::I32Const(0))?;
        sink.instruction(ctx, &Instruction::Else)?;
        emit_hash_compare(sink, ctx, hash, dim, scratch)?;
        sink.instruction(ctx, &Instruction::End)?;
        Ok(())
    }

    fn emit_check_cast(
        &self,
        ctx: &mut C,
        sink: &mut dyn InstructionSink<C, E>,
        hash: &TypeHash,
        dim: u32,
        scratch: u32,
    ) -> Result<(), E> {
        // Stack: [ref] → []
        //
        // null silently passes (Java semantics).  Type mismatch calls
        // throw_class_cast_fn (which must not return).
        sink.instruction(ctx, &Instruction::LocalSet(scratch))?;

        sink.instruction(ctx, &Instruction::LocalGet(scratch))?;
        sink.instruction(ctx, &Instruction::I32Eqz)?;
        sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
        sink.instruction(ctx, &Instruction::Else)?;
        emit_hash_compare(sink, ctx, hash, dim, scratch)?;
        sink.instruction(ctx, &Instruction::I32Eqz)?; // 1 → mismatch
        sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
        sink.instruction(ctx, &Instruction::Call(self.throw_class_cast_fn))?;
        sink.instruction(ctx, &Instruction::Unreachable)?;
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::End)?;
        Ok(())
    }
}
