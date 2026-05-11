//! [`CallbackContext`] and [`AddressMapper`] — the mapper-callback
//! interface shared by all architecture recompilers.
//!
//! Also provides [`MemoryAccess`] — the higher-level trait that owns a
//! complete load/store sequence — and [`DirectMemory`], the standard
//! implementation backed by an [`AddressMapper`].

use crate::mem::{AddressWidth, IntWidth, LoadKind, StoreKind};
use speet_ordering::MemorySink;
use yecta::{LocalDeclarator, LocalLayout, layout::CellIdx};
use wasm_encoder::{Instruction, ValType};
use wax_core::build::InstructionSink;

// ── CallbackContext ────────────────────────────────────────────────────────────

/// A thin wrapper around a `dyn InstructionSink` reference that lets
/// address-mapper callbacks emit wasm instructions without depending on
/// any concrete sink type.
///
/// Recompilers construct this on the stack immediately before calling a
/// mapper and pass `&mut callback_ctx` to the mapper.
///
/// Still used by non-mapper callbacks (hint, ecall, ebreak) in RISC-V.
pub struct CallbackContext<'a, Context, E> {
    /// The underlying instruction sink.
    pub sink: &'a mut dyn InstructionSink<Context, E>,
}

impl<'a, Context, E> CallbackContext<'a, Context, E> {
    /// Construct a `CallbackContext` wrapping any `InstructionSink`.
    #[inline]
    pub fn new(sink: &'a mut dyn InstructionSink<Context, E>) -> Self {
        Self { sink }
    }

    /// Emit a single wasm instruction via the underlying sink.
    #[inline]
    pub fn emit(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        self.sink.instruction(ctx, instruction)
    }
}

// ── AddressMapper ──────────────────────────────────────────────────────────────

/// Trait for virtual-to-physical address translation callbacks.
///
/// Replaces the old `MapperCallback`.
///
/// # Stack contract
/// * **Before call**: virtual address of type `i32` or `i64` is on the stack.
/// * **After call**: physical address of the same type is on the stack.
///
/// The sink passed to `translate` is a `dyn MemorySink` so that paging
/// structures can use `sink.instruction()` for their table-walk reads and the
/// final data-access `feed_load`/`feed_store` calls are handled by
/// `DirectMemory` after translation completes.
pub trait AddressMapper<Context, E>: LocalDeclarator {
    fn translate(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E>;

    fn chunk_size(&self) -> Option<u64> {
        None
    }
}

/// Identity mapper: leaves the address unchanged.
impl<Context, E> AddressMapper<Context, E> for () {
    #[inline]
    fn translate(
        &mut self,
        _ctx: &mut Context,
        _sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        Ok(())
    }
}

// ── ChunkedMapper ──────────────────────────────────────────────────────────────

/// Wraps any [`AddressMapper`] and advertises a fixed page granularity for
/// data-segment chunking via [`AddressMapper::chunk_size`].
pub struct ChunkedMapper<M> {
    pub inner: M,
    pub page_size: u64,
}

impl<M: LocalDeclarator> LocalDeclarator for ChunkedMapper<M> {
    #[inline]
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.inner.declare_params(cell, layout);
    }

    #[inline]
    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.inner.declare_locals(cell, layout);
    }
}

impl<Context, E, M: AddressMapper<Context, E>> AddressMapper<Context, E> for ChunkedMapper<M> {
    #[inline]
    fn translate(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        self.inner.translate(ctx, sink)
    }

    #[inline]
    fn chunk_size(&self) -> Option<u64> {
        Some(self.page_size)
    }
}

// ── StackedMapper ──────────────────────────────────────────────────────────────

/// Composes two [`AddressMapper`]s in series: the outer mapper runs first,
/// then the inner mapper refines the address further.
///
/// `chunk_size` is the minimum of the two component chunk sizes (with `None`
/// treated as the identity, i.e. "no constraint").
///
/// `LocalDeclarator` delegates to both components — outer first, then inner.
pub struct StackedMapper<Outer, Inner> {
    pub outer: Outer,
    pub inner: Inner,
}

impl<Outer: LocalDeclarator, Inner: LocalDeclarator> LocalDeclarator
    for StackedMapper<Outer, Inner>
{
    #[inline]
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.outer.declare_params(cell, layout);
        self.inner.declare_params(cell, layout);
    }

    #[inline]
    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.outer.declare_locals(cell, layout);
        self.inner.declare_locals(cell, layout);
    }
}

impl<Context, E, Outer, Inner> AddressMapper<Context, E>
    for StackedMapper<Outer, Inner>
where
    Outer: AddressMapper<Context, E>,
    Inner: AddressMapper<Context, E>,
{
    #[inline]
    fn translate(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        self.outer.translate(ctx, sink)?;
        self.inner.translate(ctx, sink)
    }

    #[inline]
    fn chunk_size(&self) -> Option<u64> {
        match (self.outer.chunk_size(), self.inner.chunk_size()) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(a.min(b)),
        }
    }
}

// ── MapperCallback type alias (backwards compat) ───────────────────────────────

/// Backwards-compatibility alias: `MapperCallback` is now `AddressMapper`.
///
/// New code should use `AddressMapper` directly.
pub use AddressMapper as MapperCallback;

// ── MemoryAccess ───────────────────────────────────────────────────────────────

/// High-level trait that owns a complete load/store sequence for a single
/// memory region.
///
/// Replaces the combination of `MapperCallback` + hardcoded `memory_index: 0`
/// scattered across the individual recompilers.  Implementations hold the
/// memory index, address/integer width, and mapper internally.
///
/// `declare_locals` allocates any scratch locals needed (e.g. the physical
/// address local for alias-check alias checks in `DirectMemory`).
pub trait MemoryAccess<Context, E>: LocalDeclarator {
    // ── Load ──────────────────────────────────────────────────────────────

    /// Emit a complete load sequence.
    ///
    /// Stack before: `... guest_address`
    /// Stack after:  `... value`
    ///
    /// The implementation handles:
    ///   1. `I32WrapI64` if needed.
    ///   2. Address mapping via the inner `AddressMapper`.
    ///   3. `local.tee` into the addr scratch local.
    ///   4. Alias-check flush + load instruction via the sink.
    ///   5. Post-load sign/zero-extend.
    fn emit_load(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        kind: LoadKind,
    ) -> Result<(), E>;

    // ── Store (two-step) ──────────────────────────────────────────────────

    /// Step 1: emit address wrap + mapper for a store.
    ///
    /// Stack before: `... guest_address`
    /// Stack after:  `... physical_address`
    ///
    /// After this the caller pushes the value (with `I32WrapI64` if needed for
    /// narrow stores), then calls [`emit_store_insn`].
    fn emit_store_addr(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E>;

    /// Step 2: emit the actual store instruction.
    ///
    /// Stack before: `... physical_address  value`
    /// Stack after:  `...`
    ///
    /// Integer stores may be deferred by the underlying `MemorySink`; float
    /// stores are always emitted eagerly.
    fn emit_store_insn(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        kind: StoreKind,
    ) -> Result<(), E>;

    // ── memory.size / memory.grow ─────────────────────────────────────────

    /// Emit a `memory.size` instruction for this memory region.
    ///
    /// Default implementation panics — override in implementations that
    /// support runtime memory size queries.
    fn emit_memory_size(
        &mut self,
        _ctx: &mut Context,
        _sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        unimplemented!("emit_memory_size not supported by this MemoryAccess implementation")
    }

    /// Emit a `memory.grow` instruction for this memory region.
    ///
    /// Default implementation panics — override in implementations that
    /// support runtime memory growth.
    fn emit_memory_grow(
        &mut self,
        _ctx: &mut Context,
        _sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        unimplemented!("emit_memory_grow not supported by this MemoryAccess implementation")
    }

    // ── chunk hint ────────────────────────────────────────────────────────

    /// Optional hint: the granularity of address-space chunks this access
    /// pattern uses (e.g. page size for page-table mappers).
    ///
    /// Returns `None` if not applicable.
    fn chunk_size(&self) -> Option<u64> {
        None
    }

    /// Convenience: whether a narrow integer store needs a `I32WrapI64`
    /// before the actual store instruction when registers are 64-bit but
    /// memory is 32-bit.
    fn needs_wrap_for_narrow_store(&self, _kind: StoreKind) -> bool {
        false
    }
}

// ── DirectMemory ───────────────────────────────────────────────────────────────

/// Standard [`MemoryAccess`] implementation backed by an [`AddressMapper`].
///
/// Holds:
/// * `mapper: M`             — the address-translation callback.
/// * `data_memory_index: u32` — the wasm memory index for data accesses.
/// * `addr_width: AddressWidth`
/// * `int_width: IntWidth`
/// * `addr_local: Option<u32>` — allocated by `declare_locals`; holds the
///   physical (post-mapper) address for alias-check comparisons.
pub struct DirectMemory<M> {
    pub mapper: M,
    pub data_memory_index: u32,
    pub addr_width: AddressWidth,
    pub int_width: IntWidth,
    /// Set to `Some(local_idx)` after `declare_locals` is called.
    addr_local: Option<u32>,
}

impl<M> DirectMemory<M> {
    /// Construct a `DirectMemory`.
    ///
    /// `addr_local` starts as `None`; it is filled in by `declare_locals`.
    pub fn new(
        mapper: M,
        data_memory_index: u32,
        addr_width: AddressWidth,
        int_width: IntWidth,
    ) -> Self {
        Self {
            mapper,
            data_memory_index,
            addr_width,
            int_width,
            addr_local: None,
        }
    }

    // ── helpers ────────────────────────────────────────────────────────────

    fn mem_arg(&self, align: u32) -> wasm_encoder::MemArg {
        wasm_encoder::MemArg {
            offset: 0,
            align,
            memory_index: self.data_memory_index,
        }
    }

    fn addr_type(&self) -> ValType {
        if self.addr_width.memory64() {
            ValType::I64
        } else {
            ValType::I32
        }
    }

    /// Emit `I32WrapI64` if the guest is 64-bit but wasm memory is 32-bit.
    fn maybe_wrap_addr<Context, E>(
        &self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        if self.addr_width == (AddressWidth::W64 { memory64: false }) {
            sink.instruction(ctx, &Instruction::I32WrapI64)?;
        }
        Ok(())
    }
}

impl<M: LocalDeclarator> LocalDeclarator for DirectMemory<M> {
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.mapper.declare_params(cell, layout);
    }

    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.mapper.declare_locals(cell, layout);
        // Allocate one scratch local of addr_type for alias checks.
        let addr_vt = self.addr_type();
        let slot = layout.append(1, addr_vt);
        self.addr_local = Some(layout.base(slot));
        let _ = cell; // cell not used beyond delegation
    }
}

impl<Context, E, M: AddressMapper<Context, E>> MemoryAccess<Context, E> for DirectMemory<M> {
    fn emit_load(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        kind: LoadKind,
    ) -> Result<(), E> {
        let addr_local = self
            .addr_local
            .expect("DirectMemory::emit_load called before declare_locals");

        // 1. Optionally wrap i64 → i32.
        self.maybe_wrap_addr(ctx, sink)?;

        // 2. Address translation.
        self.mapper.translate(ctx, sink)?;

        // 3. Tee the physical address into addr_local (keep on stack).
        sink.instruction(ctx, &Instruction::LocalTee(addr_local))?;

        // 4. Emit the load instruction (with alias-check flush).
        let addr_type = self.addr_type();
        let mem64 = self.addr_width.memory64();
        let int64 = self.int_width == IntWidth::I64;

        let load_instr: Instruction<'static> = match kind {
            LoadKind::I8S => {
                if mem64 && int64 {
                    Instruction::I64Load8S(self.mem_arg(0))
                } else {
                    Instruction::I32Load8S(self.mem_arg(0))
                }
            }
            LoadKind::I8U => {
                if mem64 && int64 {
                    Instruction::I64Load8U(self.mem_arg(0))
                } else {
                    Instruction::I32Load8U(self.mem_arg(0))
                }
            }
            LoadKind::I16S => {
                if mem64 && int64 {
                    Instruction::I64Load16S(self.mem_arg(1))
                } else {
                    Instruction::I32Load16S(self.mem_arg(1))
                }
            }
            LoadKind::I16U => {
                if mem64 && int64 {
                    Instruction::I64Load16U(self.mem_arg(1))
                } else {
                    Instruction::I32Load16U(self.mem_arg(1))
                }
            }
            LoadKind::I32S => {
                if mem64 && int64 {
                    Instruction::I64Load32S(self.mem_arg(2))
                } else {
                    Instruction::I32Load(self.mem_arg(2))
                }
            }
            LoadKind::I32U => {
                if mem64 {
                    Instruction::I64Load32U(self.mem_arg(2))
                } else {
                    Instruction::I32Load(self.mem_arg(2))
                }
            }
            LoadKind::I64 => Instruction::I64Load(self.mem_arg(3)),
            LoadKind::F32 => Instruction::F32Load(self.mem_arg(2)),
            LoadKind::F64 => Instruction::F64Load(self.mem_arg(3)),
        };

        sink.feed_load(ctx, addr_local, addr_type, load_instr)?;

        // 5. Post-load sign/zero-extend (emitted eagerly via sink.instruction).
        match kind {
            LoadKind::I8S if !mem64 && int64 => {
                sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
            }
            LoadKind::I8U if !mem64 && int64 => {
                sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
            }
            LoadKind::I16S if !mem64 && int64 => {
                sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
            }
            LoadKind::I16U if !mem64 && int64 => {
                sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
            }
            LoadKind::I32S if !mem64 && int64 => {
                sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
            }
            LoadKind::I32U if !mem64 => {
                // Zero-extend: I32Load result → i64
                sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
            }
            LoadKind::F32 => {
                sink.instruction(ctx, &Instruction::F64PromoteF32)?;
            }
            _ => {}
        }

        Ok(())
    }

    fn emit_store_addr(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        self.maybe_wrap_addr(ctx, sink)?;
        self.mapper.translate(ctx, sink)
    }

    fn emit_store_insn(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        kind: StoreKind,
    ) -> Result<(), E> {
        let addr_type = self.addr_type();
        let mem64 = self.addr_width.memory64();
        let int64 = self.int_width == IntWidth::I64;

        match kind {
            StoreKind::I8 => {
                let instr = if mem64 && int64 {
                    Instruction::I64Store8(self.mem_arg(0))
                } else {
                    Instruction::I32Store8(self.mem_arg(0))
                };
                sink.feed_store(ctx, addr_type, instr)?;
            }
            StoreKind::I16 => {
                let instr = if mem64 && int64 {
                    Instruction::I64Store16(self.mem_arg(1))
                } else {
                    Instruction::I32Store16(self.mem_arg(1))
                };
                sink.feed_store(ctx, addr_type, instr)?;
            }
            StoreKind::I32 => {
                let instr = if mem64 && int64 {
                    Instruction::I64Store32(self.mem_arg(2))
                } else {
                    Instruction::I32Store(self.mem_arg(2))
                };
                sink.feed_store(ctx, addr_type, instr)?;
            }
            StoreKind::I64 => {
                let instr = Instruction::I64Store(self.mem_arg(3));
                sink.feed_store(ctx, addr_type, instr)?;
            }
            // Float stores are always eager (no feed_store — their val type
            // cannot be saved in the i32/i64 local pool).
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

    fn emit_memory_size(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::MemorySize(self.data_memory_index))
    }

    fn emit_memory_grow(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::MemoryGrow(self.data_memory_index))
    }

    fn chunk_size(&self) -> Option<u64> {
        self.mapper.chunk_size()
    }

    fn needs_wrap_for_narrow_store(&self, kind: StoreKind) -> bool {
        self.int_width == IntWidth::I64
            && !self.addr_width.memory64()
            && matches!(kind, StoreKind::I8 | StoreKind::I16 | StoreKind::I32)
    }
}
