//! [`MemoryEmitter`] — architecture-agnostic load/store code generation.
//!
//! A `MemoryEmitter` holds the few configuration bits that vary per
//! architecture (address width, memory index, optional mapper) and exposes
//! [`emit_load`][MemoryEmitter::emit_load] /
//! [`emit_store`][MemoryEmitter::emit_store] methods that handle:
//!
//! 1. Computing `base + offset` with the correct integer width.
//! 2. Optionally narrowing a 64-bit guest address to 32-bit wasm (`I32WrapI64`).
//! 3. Calling the mapper callback (if any).
//! 4. Emitting the appropriate wasm memory instruction.
//! 5. Sign- or zero-extending the loaded value to the register integer type
//!    when needed.

use crate::mapper::{CallbackContext, MapperCallback};
use wax_core::build::InstructionSink;
use wasm_encoder::{Instruction, MemArg};

// ── Width helpers ──────────────────────────────────────────────────────────────

/// Whether the guest address space is 32-bit or 64-bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressWidth {
    /// 32-bit guest registers; addresses are `i32` on the wasm stack.
    W32,
    /// 64-bit guest registers; guest addresses are `i64` and may need wrapping
    /// to `i32` when `memory64` is *not* enabled.
    W64 {
        /// When `true` the Wasm memory uses 64-bit addressing (memory64
        /// proposal); no `I32WrapI64` is emitted.  When `false` the address
        /// must be wrapped before use.
        memory64: bool,
    },
}

impl AddressWidth {
    /// `true` when the native register width is 64 bits.
    #[inline]
    pub fn is_64(&self) -> bool {
        matches!(self, Self::W64 { .. })
    }

    /// `true` when wasm memory instructions use 64-bit (`i64`) addresses.
    #[inline]
    pub fn memory64(&self) -> bool {
        matches!(self, Self::W64 { memory64: true })
    }
}

/// Width of the integer register file — used when choosing sign-/zero-extend
/// sequences after narrow loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntWidth {
    I32,
    I64,
}

// ── Load/Store descriptors ─────────────────────────────────────────────────────

/// What memory load to perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadKind {
    /// Load a signed 8-bit value, sign-extended.
    I8S,
    /// Load an unsigned 8-bit value, zero-extended.
    I8U,
    /// Load a signed 16-bit value, sign-extended.
    I16S,
    /// Load an unsigned 16-bit value, zero-extended.
    I16U,
    /// Load a 32-bit value (sign-extended when target register is 64-bit).
    I32S,
    /// Load a 32-bit value, zero-extended (RV64 `LWU`-style).
    I32U,
    /// Load a 64-bit value.
    I64,
    /// Load a 32-bit float, promoting to f64.
    F32,
    /// Load a 64-bit float.
    F64,
}

/// What memory store to perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreKind {
    I8,
    I16,
    I32,
    I64,
    /// Store f64 demoted to f32.
    F32,
    F64,
}

// ── MemoryEmitter ──────────────────────────────────────────────────────────────

/// Architecture-agnostic helper that emits complete load/store sequences.
///
/// # Address convention
///
/// All `emit_load` / `emit_store` methods expect the caller to have already
/// pushed the *raw guest address* (`base_reg + imm_offset`) onto the wasm
/// value stack, computed with `I32Add` / `I64Add` as appropriate.
/// `MemoryEmitter` then:
///
/// 1. Wraps i64 → i32 if `AddressWidth::W64 { memory64: false }`.
/// 2. Invokes the mapper callback (if set).
/// 3. Emits the wasm memory instruction.
/// 4. For loads: emits any needed sign-/zero-extend.
///
/// # Mapper scratch local
///
/// When a mapper callback is present, it typically needs to stash the address
/// in a scratch local (e.g. `LocalTee`) before emitting table-walk
/// instructions.  The convention followed by `speet-riscv`'s page-table
/// helpers is to save the address at `scratch_local` (passed to
/// [`MemoryEmitter::new`]) *before* calling the mapper.  The emitter itself
/// only calls the mapper; management of `scratch_local` is the caller's
/// responsibility.
pub struct MemoryEmitter<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>> {
    pub addr_width: AddressWidth,
    pub int_width: IntWidth,
    pub memory_index: u32,
    pub mapper: Option<&'cb mut (dyn MapperCallback<Context, E, F> + 'ctx)>,
}

impl<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>>
    MemoryEmitter<'cb, 'ctx, Context, E, F>
{
    /// Create a new `MemoryEmitter`.
    ///
    /// * `addr_width`   — address/register width for this architecture.
    /// * `int_width`    — integer register width (controls sign-extend after narrow loads).
    /// * `memory_index` — wasm memory index (almost always `0`).
    /// * `mapper`       — optional address-translation callback.
    pub fn new(
        addr_width: AddressWidth,
        int_width: IntWidth,
        memory_index: u32,
        mapper: Option<&'cb mut (dyn MapperCallback<Context, E, F> + 'ctx)>,
    ) -> Self {
        Self { addr_width, int_width, memory_index, mapper }
    }

    // ── internal helpers ───────────────────────────────────────────────────

    /// Build a [`MemArg`] with `align` and the configured `memory_index`.
    #[inline]
    fn mem_arg(&self, align: u32) -> MemArg {
        MemArg { offset: 0, align, memory_index: self.memory_index }
    }

    /// Wrap i64 address to i32 if required by the address/memory width combo.
    fn maybe_wrap_addr(
        &self,
        ctx: &mut Context,
        sink: &mut F,
    ) -> Result<(), E> {
        if self.addr_width == (AddressWidth::W64 { memory64: false }) {
            sink.instruction(ctx, &Instruction::I32WrapI64)?;
        }
        Ok(())
    }

    /// Optionally invoke the mapper callback.
    fn maybe_map(
        &mut self,
        ctx: &mut Context,
        sink: &mut F,
    ) -> Result<(), E> {
        if let Some(mapper) = self.mapper.as_mut() {
            let mut cb = CallbackContext::new(sink);
            mapper.call(ctx, &mut cb)?;
        }
        Ok(())
    }

    // ── public API ─────────────────────────────────────────────────────────

    /// Emit a complete load sequence.
    ///
    /// Stack before: `... guest_address`
    /// Stack after:  `... value` (type determined by `int_width` / `LoadKind`)
    pub fn emit_load(
        &mut self,
        ctx: &mut Context,
        sink: &mut F,
        kind: LoadKind,
    ) -> Result<(), E> {
        self.maybe_wrap_addr(ctx, sink)?;
        self.maybe_map(ctx, sink)?;

        let mem64 = self.addr_width.memory64();
        let int64 = self.int_width == IntWidth::I64;

        match kind {
            LoadKind::I8S => {
                if mem64 && int64 {
                    sink.instruction(ctx, &Instruction::I64Load8S(self.mem_arg(0)))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Load8S(self.mem_arg(0)))?;
                    if int64 {
                        sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadKind::I8U => {
                if mem64 && int64 {
                    sink.instruction(ctx, &Instruction::I64Load8U(self.mem_arg(0)))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Load8U(self.mem_arg(0)))?;
                    if int64 {
                        sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
                    }
                }
            }
            LoadKind::I16S => {
                if mem64 && int64 {
                    sink.instruction(ctx, &Instruction::I64Load16S(self.mem_arg(1)))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Load16S(self.mem_arg(1)))?;
                    if int64 {
                        sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadKind::I16U => {
                if mem64 && int64 {
                    sink.instruction(ctx, &Instruction::I64Load16U(self.mem_arg(1)))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Load16U(self.mem_arg(1)))?;
                    if int64 {
                        sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
                    }
                }
            }
            LoadKind::I32S => {
                if mem64 && int64 {
                    sink.instruction(ctx, &Instruction::I64Load32S(self.mem_arg(2)))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Load(self.mem_arg(2)))?;
                    if int64 {
                        sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadKind::I32U => {
                // Used for RV64 LWU / zero-extended 32-bit load
                if mem64 {
                    sink.instruction(ctx, &Instruction::I64Load32U(self.mem_arg(2)))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Load(self.mem_arg(2)))?;
                    sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
                }
            }
            LoadKind::I64 => {
                sink.instruction(ctx, &Instruction::I64Load(self.mem_arg(3)))?;
            }
            LoadKind::F32 => {
                sink.instruction(ctx, &Instruction::F32Load(self.mem_arg(2)))?;
                sink.instruction(ctx, &Instruction::F64PromoteF32)?;
            }
            LoadKind::F64 => {
                sink.instruction(ctx, &Instruction::F64Load(self.mem_arg(3)))?;
            }
        }
        Ok(())
    }

    /// Emit a complete store sequence.
    ///
    /// For integer stores the caller must push `[address, value]` (in that
    /// order) onto the wasm stack before calling this method.  The address
    /// wrap + mapper are applied to the address *before* the value is consumed,
    /// so the caller should structure the wasm stack as:
    ///
    /// ```text
    /// ... base_reg_val  offset_val  I32Add/I64Add    ← address on top
    ///     LocalGet(src)                              ← value to store
    /// ```
    ///
    /// However this only works if the mapper call happens *before* the value
    /// is loaded.  Therefore `emit_store` requires:
    ///
    /// **Stack before**: `... guest_address  value`
    ///
    /// The address must already be on the stack **below** the value.
    /// `emit_store` will emit wrap + mapper on a newly computed
    /// address-shaped local if needed — but this is architecture-specific.
    ///
    /// **Simplified contract** (matching the speet-riscv pattern):
    ///
    /// The caller pushes the address, then calls `emit_store_addr` to apply
    /// the wrap + mapper, then pushes the value, then calls `emit_store_value`
    /// to emit the actual `store` instruction.  See the two-step variant
    /// below.
    ///
    /// **One-shot contract**: both address and value are already on the
    /// stack; this method applies wrap + mapper + store.  The mapper sees
    /// only the address (not the value), which means the caller must *not*
    /// have pushed the value yet when the mapper runs.  Use the two-step
    /// API for stores with a mapper.
    ///
    /// Stack before: `... guest_address`  (value NOT yet pushed)
    /// Emits: wrap? + mapper?
    /// Returns: caller should then push the value and call [`emit_store_insn`].
    pub fn emit_store_addr(
        &mut self,
        ctx: &mut Context,
        sink: &mut F,
    ) -> Result<(), E> {
        self.maybe_wrap_addr(ctx, sink)?;
        self.maybe_map(ctx, sink)?;
        Ok(())
    }

    /// Emit the raw wasm store instruction (no wrap, no mapper).
    ///
    /// Call after `emit_store_addr` + pushing the value to store.
    ///
    /// Stack before: `... mapped_address  value`
    pub fn emit_store_insn(
        &self,
        ctx: &mut Context,
        sink: &mut F,
        kind: StoreKind,
    ) -> Result<(), E> {
        let mem64 = self.addr_width.memory64();
        let int64 = self.int_width == IntWidth::I64;

        match kind {
            StoreKind::I8 => {
                if mem64 && int64 {
                    sink.instruction(ctx, &Instruction::I64Store8(self.mem_arg(0)))?;
                } else {
                    // If int64 but not mem64, caller must wrap value with I32WrapI64 first
                    sink.instruction(ctx, &Instruction::I32Store8(self.mem_arg(0)))?;
                }
            }
            StoreKind::I16 => {
                if mem64 && int64 {
                    sink.instruction(ctx, &Instruction::I64Store16(self.mem_arg(1)))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Store16(self.mem_arg(1)))?;
                }
            }
            StoreKind::I32 => {
                if mem64 && int64 {
                    sink.instruction(ctx, &Instruction::I64Store32(self.mem_arg(2)))?;
                } else {
                    sink.instruction(ctx, &Instruction::I32Store(self.mem_arg(2)))?;
                }
            }
            StoreKind::I64 => {
                sink.instruction(ctx, &Instruction::I64Store(self.mem_arg(3)))?;
            }
            StoreKind::F32 => {
                sink.instruction(ctx, &Instruction::F32DemoteF64)?;
                sink.instruction(ctx, &Instruction::F32Store(self.mem_arg(2)))?;
            }
            StoreKind::F64 => {
                sink.instruction(ctx, &Instruction::F64Store(self.mem_arg(3)))?;
            }
        }
        Ok(())
    }

    /// Convenience: whether a narrow integer store needs a `I32WrapI64` before
    /// the actual store instruction when registers are 64-bit but memory is
    /// 32-bit.
    ///
    /// Returns `true` for `I8`, `I16`, `I32` stores when `int_width == I64`
    /// and `memory64 == false`.
    #[inline]
    pub fn needs_wrap_for_narrow_store(&self, kind: StoreKind) -> bool {
        self.int_width == IntWidth::I64
            && !self.addr_width.memory64()
            && matches!(kind, StoreKind::I8 | StoreKind::I16 | StoreKind::I32)
    }
}
