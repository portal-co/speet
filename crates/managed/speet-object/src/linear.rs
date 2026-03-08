//! Linear-memory [`ObjectModel`] implementation.

use alloc::vec::Vec;
use wasm_encoder::{BlockType, Instruction, MemArg, ValType};
use yecta::LocalSlot;

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

/// Emit a load-from-object-field instruction sequence.
///
/// Expects `[obj_ref]` on the wasm stack; leaves `[value]` (see
/// [`FieldValType`] conventions in [`ObjectModel::emit_iget`]).
fn push_load(byte_offset: u32, ty: FieldValType, out: &mut Vec<Instruction<'static>>) {
    let offset = (OBJECT_HEADER_SIZE + byte_offset) as u64;
    let memarg = MemArg { offset, align: 0, memory_index: 0 };
    match ty {
        FieldValType::I32 | FieldValType::Ref => out.push(Instruction::I32Load(memarg)),
        FieldValType::I64 => out.push(Instruction::I64Load(memarg)),
        FieldValType::F32 => {
            out.push(Instruction::F32Load(memarg));
            out.push(Instruction::I32ReinterpretF32);
        }
        FieldValType::F64 => {
            out.push(Instruction::F64Load(memarg));
            out.push(Instruction::I64ReinterpretF64);
        }
        FieldValType::I8S  => out.push(Instruction::I32Load8S(memarg)),
        FieldValType::I8U  => out.push(Instruction::I32Load8U(memarg)),
        FieldValType::I16S => out.push(Instruction::I32Load16S(memarg)),
        FieldValType::I16U => out.push(Instruction::I32Load16U(memarg)),
    }
}

/// Emit a store-to-object-field instruction sequence.
///
/// Expects `[obj_ref, value]` on the wasm stack (see [`FieldValType`]
/// conventions in [`ObjectModel::emit_iput`]).
fn push_store(byte_offset: u32, ty: FieldValType, out: &mut Vec<Instruction<'static>>) {
    let offset = (OBJECT_HEADER_SIZE + byte_offset) as u64;
    let memarg = MemArg { offset, align: 0, memory_index: 0 };
    match ty {
        FieldValType::I32 | FieldValType::Ref => out.push(Instruction::I32Store(memarg)),
        FieldValType::I64 => out.push(Instruction::I64Store(memarg)),
        FieldValType::F32 => {
            out.push(Instruction::F32ReinterpretI32);
            out.push(Instruction::F32Store(memarg));
        }
        FieldValType::F64 => {
            out.push(Instruction::F64ReinterpretI64);
            out.push(Instruction::F64Store(memarg));
        }
        FieldValType::I8S | FieldValType::I8U   => out.push(Instruction::I32Store8(memarg)),
        FieldValType::I16S | FieldValType::I16U => out.push(Instruction::I32Store16(memarg)),
    }
}

/// Emit the type-hash comparison sequence used by instanceof / check-cast.
///
/// Assumes `scratch` has already been set to the object reference via
/// `local.set(scratch)`.  Leaves a single `i32` (0 or 1) on the stack
/// representing whether the object's stored hash matches `(hash, dim)`.
fn push_hash_compare(
    hash: &TypeHash,
    dim: u32,
    scratch: LocalSlot,
    out: &mut Vec<Instruction<'static>>,
) {
    let chunks = hash.as_i32_chunks();
    // Compare all 8 × 4-byte chunks of the stored hash against expected.
    let mut first = true;
    for (i, &expected) in chunks.iter().enumerate() {
        out.push(Instruction::LocalGet(scratch.0 as u32));
        out.push(Instruction::I32Load(MemArg {
            offset: (i as u64) * 4,
            align: 0,
            memory_index: 0,
        }));
        out.push(Instruction::I32Const(expected));
        out.push(Instruction::I32Eq);
        if !first {
            out.push(Instruction::I32And);
        }
        first = false;
    }
    // Compare stored array_dim against expected.
    out.push(Instruction::LocalGet(scratch.0 as u32));
    out.push(Instruction::I32Load(MemArg {
        offset: 32,
        align: 0,
        memory_index: 0,
    }));
    out.push(Instruction::I32Const(dim as i32));
    out.push(Instruction::I32Eq);
    out.push(Instruction::I32And);
}

// ── ObjectModel impl ──────────────────────────────────────────────────────────

impl<C, E> ObjectModel<C, E> for LinearMemoryObjects {
    fn ref_val_type(&self) -> ValType {
        ValType::I32
    }

    fn emit_new_object(
        &self,
        _ctx: &mut C,
        hash: &TypeHash,
        data_size: u32,
        out: &mut Vec<Instruction<'static>>,
    ) -> Result<(), E> {
        let [h0, h1, h2, h3] = hash.as_i64_chunks();
        out.push(Instruction::I64Const(h0));
        out.push(Instruction::I64Const(h1));
        out.push(Instruction::I64Const(h2));
        out.push(Instruction::I64Const(h3));
        out.push(Instruction::I32Const(data_size as i32));
        out.push(Instruction::Call(self.alloc_object_fn));
        Ok(())
    }

    fn emit_new_array(
        &self,
        _ctx: &mut C,
        elem_hash: &TypeHash,
        dim: u32,
        elem_bytes: u32,
        out: &mut Vec<Instruction<'static>>,
    ) -> Result<(), E> {
        // Stack before: [length: i32]
        // alloc_array_fn(length, h0, h1, h2, h3, dim, elem_bytes) → i32
        let [h0, h1, h2, h3] = elem_hash.as_i64_chunks();
        out.push(Instruction::I64Const(h0));
        out.push(Instruction::I64Const(h1));
        out.push(Instruction::I64Const(h2));
        out.push(Instruction::I64Const(h3));
        out.push(Instruction::I32Const(dim as i32));
        out.push(Instruction::I32Const(elem_bytes as i32));
        out.push(Instruction::Call(self.alloc_array_fn));
        Ok(())
    }

    fn emit_iget(
        &self,
        byte_offset: u32,
        ty: FieldValType,
        out: &mut Vec<Instruction<'static>>,
    ) {
        push_load(byte_offset, ty, out);
    }

    fn emit_iput(
        &self,
        byte_offset: u32,
        ty: FieldValType,
        out: &mut Vec<Instruction<'static>>,
    ) {
        push_store(byte_offset, ty, out);
    }

    fn emit_aget(&self, ty: FieldValType, out: &mut Vec<Instruction<'static>>) {
        // Stack before: [arr_ref: i32, index: i32]
        // Compute addr = arr_ref + ARRAY_DATA_OFFSET + index * elem_bytes
        let elem_bytes = ty.size_bytes();
        // index × elem_bytes
        out.push(Instruction::I32Const(elem_bytes as i32));
        out.push(Instruction::I32Mul);
        // + ARRAY_DATA_OFFSET
        out.push(Instruction::I32Const(ARRAY_DATA_OFFSET as i32));
        out.push(Instruction::I32Add);
        // arr_ref + offset
        out.push(Instruction::I32Add);
        // Load value — offset=0 since we computed the exact address above.
        let memarg = MemArg { offset: 0, align: 0, memory_index: 0 };
        match ty {
            FieldValType::I32 | FieldValType::Ref => out.push(Instruction::I32Load(memarg)),
            FieldValType::I64 => out.push(Instruction::I64Load(memarg)),
            FieldValType::F32 => {
                out.push(Instruction::F32Load(memarg));
                out.push(Instruction::I32ReinterpretF32);
            }
            FieldValType::F64 => {
                out.push(Instruction::F64Load(memarg));
                out.push(Instruction::I64ReinterpretF64);
            }
            FieldValType::I8S  => out.push(Instruction::I32Load8S(memarg)),
            FieldValType::I8U  => out.push(Instruction::I32Load8U(memarg)),
            FieldValType::I16S => out.push(Instruction::I32Load16S(memarg)),
            FieldValType::I16U => out.push(Instruction::I32Load16U(memarg)),
        }
    }

    fn emit_aput(
        &self,
        ty: FieldValType,
        scratch_i32: LocalSlot,
        scratch_i64: LocalSlot,
        out: &mut Vec<Instruction<'static>>,
    ) {
        // Stack before: [arr_ref: i32, index: i32, value]
        //
        // We need [addr, value] for the store, where addr = arr_ref + ARRAY_DATA_OFFSET + idx*ebs.
        // Since the value is on top, we save it, compute the address, then restore.
        let elem_bytes = ty.size_bytes();
        let is_wide = matches!(ty, FieldValType::I64 | FieldValType::F64);

        if is_wide {
            out.push(Instruction::LocalSet(scratch_i64.0 as u32));
        } else {
            out.push(Instruction::LocalSet(scratch_i32.0 as u32));
        }

        // Stack: [arr_ref, index]
        out.push(Instruction::I32Const(elem_bytes as i32));
        out.push(Instruction::I32Mul);
        out.push(Instruction::I32Const(ARRAY_DATA_OFFSET as i32));
        out.push(Instruction::I32Add);
        out.push(Instruction::I32Add); // addr

        // Restore value and store.
        let memarg = MemArg { offset: 0, align: 0, memory_index: 0 };
        match ty {
            FieldValType::I32 | FieldValType::Ref => {
                out.push(Instruction::LocalGet(scratch_i32.0 as u32));
                out.push(Instruction::I32Store(memarg));
            }
            FieldValType::I64 => {
                out.push(Instruction::LocalGet(scratch_i64.0 as u32));
                out.push(Instruction::I64Store(memarg));
            }
            FieldValType::F32 => {
                out.push(Instruction::LocalGet(scratch_i32.0 as u32));
                out.push(Instruction::F32ReinterpretI32);
                out.push(Instruction::F32Store(memarg));
            }
            FieldValType::F64 => {
                out.push(Instruction::LocalGet(scratch_i64.0 as u32));
                out.push(Instruction::F64ReinterpretI64);
                out.push(Instruction::F64Store(memarg));
            }
            FieldValType::I8S | FieldValType::I8U => {
                out.push(Instruction::LocalGet(scratch_i32.0 as u32));
                out.push(Instruction::I32Store8(memarg));
            }
            FieldValType::I16S | FieldValType::I16U => {
                out.push(Instruction::LocalGet(scratch_i32.0 as u32));
                out.push(Instruction::I32Store16(memarg));
            }
        }
    }

    fn emit_array_length(&self, out: &mut Vec<Instruction<'static>>) {
        // Stack: [arr_ref] → [length: i32]
        out.push(Instruction::I32Load(MemArg {
            offset: ARRAY_LENGTH_OFFSET as u64,
            align: 0,
            memory_index: 0,
        }));
    }

    fn emit_instanceof(
        &self,
        hash: &TypeHash,
        dim: u32,
        scratch: LocalSlot,
        out: &mut Vec<Instruction<'static>>,
    ) {
        // Stack: [ref] → [i32 (0 or 1)]
        //
        // null → 0;  type match → 1;  type mismatch → 0.
        out.push(Instruction::LocalSet(scratch.0 as u32));

        // null check: if scratch == 0, return 0.
        out.push(Instruction::LocalGet(scratch.0 as u32));
        out.push(Instruction::I32Eqz);
        out.push(Instruction::If(BlockType::Result(ValType::I32)));
        out.push(Instruction::I32Const(0));
        out.push(Instruction::Else);
        // Non-null: compare hash and dim.
        push_hash_compare(hash, dim, scratch, out);
        out.push(Instruction::End);
    }

    fn emit_check_cast(
        &self,
        hash: &TypeHash,
        dim: u32,
        scratch: LocalSlot,
        out: &mut Vec<Instruction<'static>>,
    ) {
        // Stack: [ref] → []
        //
        // null silently passes (Java semantics).  Type mismatch calls
        // throw_class_cast_fn (which must not return).
        out.push(Instruction::LocalSet(scratch.0 as u32));

        // null check: if null, skip the type test entirely.
        out.push(Instruction::LocalGet(scratch.0 as u32));
        out.push(Instruction::I32Eqz);
        out.push(Instruction::If(BlockType::Empty));
        out.push(Instruction::Else);
        // Non-null: compare hash and dim; throw if mismatch.
        push_hash_compare(hash, dim, scratch, out);
        out.push(Instruction::I32Eqz); // 1 → mismatch
        out.push(Instruction::If(BlockType::Empty));
        out.push(Instruction::Call(self.throw_class_cast_fn));
        out.push(Instruction::Unreachable);
        out.push(Instruction::End);
        out.push(Instruction::End);
    }
}
