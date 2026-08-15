//! `speet-interp` — lookup stub and interpreter WASM function generators.
//!
//! This crate generates the two special WASM functions required for OOB jump
//! dispatch:
//!
//! ## Lookup stub
//!
//! A WASM function with the same type as every compiled function
//! (arch-regs + `target_pc: i64` → arch-regs).  Given a `target_pc` it
//! binary-searches a sorted `[(pc: i64, table_slot: i32)]` array that is
//! baked into a passive data segment, then issues a `return_call_indirect`
//! to the correct compiled function.  Falls back to the interpreter when no
//! entry matches.
//!
//! ## Interpreter stub
//!
//! A WASM function with the same type.  By default emits `unreachable`; ISA-
//! specific interpreters can be plugged in via [`InterpBodyBuilder`].  The
//! Thompson-threaded model generates one dispatch function plus N per-opcode-
//! class handler functions, all chained via `return_call` tail calls.
//!
//! ## Sink-based API
//!
//! The primary emit functions take `&mut dyn InstructionSink<Context, E>` so
//! callers can use any WASM backend.  `wasm_encoder::Function` implements
//! `InstructionSink` out of the box.  The backwards-compatible [`OobInterp::generate`]
//! wrapper creates `Function` objects internally.
//!
//! ## Usage (backward-compatible)
//!
//! ```ignore
//! // Phase 1:
//! let oob = OobConfig::register(&mut entity_space);
//! let interp = OobInterp::register(&mut entity_space, &oob);
//!
//! // Phase 2:
//! let (lookup_fn, interp_fn, data_seg) = interp.generate(
//!     &entity_space, &oob, &compiled_pc_table, params, returns, type_idx,
//!     table_idx, data_seg_idx, data_mem_idx,
//! );
//! ```
//!
//! ## Usage (Thompson-threaded interpreter)
//!
//! ```ignore
//! // Phase 1:
//! let oob = OobConfig::register(&mut entity_space);
//! let mut builder = RiscVThompsonInterp::new();
//! let interp = OobInterp::register_with_builder(&mut entity_space, &mut oob, &builder);
//!
//! // Phase 2 (after layout is finalised):
//! let mut ictx = InterpBuildCtx { oob: &oob, layout: &layout, .. };
//! let (lookup_fn, interp_fns, data_seg) = interp.emit_with_builder(
//!     &entity_space, &oob, &entries, params, returns, type_idx,
//!     table_idx, data_seg_idx, data_mem_idx, ctx, &mut ictx, &mut builder,
//! )?;
//! ```

extern crate alloc;

use alloc::vec::Vec;
use speet_link_core::{EntityIndexSpace, IndexSlot, JitConfig, OobConfig};
use wasm_encoder::{BlockType, Function, Instruction, MemArg, ValType};
use wax_core::build::InstructionSink;

pub mod builder;
pub mod context;

pub use builder::{InterpBodyBuilder, NullInterpBuilder};
pub use context::{BufferedEmitSink, FlatEmitSink, FlatMemorySink, InterpBuildCtx};

// ── PcEntry ───────────────────────────────────────────────────────────────────

/// One entry in the compiled-PC dispatch table.
///
/// Entries are stored sorted by `guest_pc` in ascending order so the lookup
/// stub can binary-search them.  `table_slot` is the absolute WASM function
/// table slot index for the corresponding compiled function.
#[derive(Clone, Copy, Debug)]
pub struct PcEntry {
    /// Guest PC of the compiled function.
    pub guest_pc: u64,
    /// Absolute WASM function table slot (index into the dispatch table).
    pub table_slot: u32,
}

// ── OobInterp ─────────────────────────────────────────────────────────────────

/// Registers and generates the lookup stub + interpreter for OOB dispatch.
pub struct OobInterp {
    /// Index slot for the generated functions (lookup stub + interpreter + handlers).
    pub func_slot: IndexSlot,
}

impl OobInterp {
    /// Register two function slots in Phase 1 (lookup stub + interpreter stub),
    /// and fill their absolute indices into `oob`.
    pub fn register(entity_space: &mut EntityIndexSpace, oob: &mut OobConfig) -> Self {
        let func_slot = entity_space.functions.append(2);
        let base = entity_space.functions.base(func_slot);
        oob.set_func_indices(base, base + 1);
        Self { func_slot }
    }

    /// Register function slots in Phase 1 for a Thompson-threaded builder.
    ///
    /// Pre-allocates `2 + builder.num_handler_fns()` slots:
    /// slot 0 = lookup stub, slot 1 = dispatch fn, slots 2..N = opcode handlers.
    pub fn register_with_builder<C, E>(
        entity_space: &mut EntityIndexSpace,
        oob: &mut OobConfig,
        builder: &dyn InterpBodyBuilder<C, E>,
    ) -> Self {
        let n = 2 + builder.num_handler_fns();
        let func_slot = entity_space.functions.append(n);
        let base = entity_space.functions.base(func_slot);
        oob.set_func_indices(base, base + 1);
        Self { func_slot }
    }

    /// Generate the lookup stub and interpreter WASM function bodies.
    ///
    /// Returns `(lookup_stub_fn, interp_fn, passive_data_bytes)`.
    /// This is the original backwards-compatible API; internally it calls
    /// [`generate_into`](Self::generate_into) with `wasm_encoder::Function` sinks.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        entity_space: &EntityIndexSpace,
        oob: &OobConfig,
        entries: &[PcEntry],
        params: &[ValType],
        returns: &[ValType],
        type_idx: u32,
        table_idx: u32,
        data_seg_idx: u32,
        data_mem_idx: u32,
    ) -> (Function, Function, Vec<u8>) {
        let mut lookup_fn = Function::new([
            (1, ValType::I32), // lo
            (1, ValType::I32), // hi
            (1, ValType::I32), // mid
            (1, ValType::I64), // mid_pc
            (1, ValType::I32), // byte_off
        ]);
        let mut interp_fn = Function::new([]);

        let data = self
            .generate_into::<(), core::convert::Infallible>(
                &mut lookup_fn,
                &mut interp_fn,
                &mut (),
                oob,
                entries,
                params,
                returns,
                type_idx,
                table_idx,
                data_seg_idx,
                data_mem_idx,
            )
            .expect("infallible");

        let _ = entity_space; // entity_space no longer needed for index resolution
        (lookup_fn, interp_fn, data)
    }

    /// Sink-based variant of [`generate`](Self::generate).
    ///
    /// Emits the lookup stub into `lookup_sink` and the interpreter stub into
    /// `interp_sink`.  Returns the passive data bytes for the PC dispatch table.
    ///
    /// Callers that use `wasm_encoder::Function` can pass `&mut my_fn` directly,
    /// since `Function` implements `InstructionSink`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_into<Context, E>(
        &self,
        lookup_sink: &mut dyn InstructionSink<Context, E>,
        interp_sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        oob: &OobConfig,
        entries: &[PcEntry],
        params: &[ValType],
        returns: &[ValType],
        type_idx: u32,
        table_idx: u32,
        data_seg_idx: u32,
        data_mem_idx: u32,
    ) -> Result<Vec<u8>, E> {
        let data = Self::build_pc_table(entries);
        emit_lookup_stub(
            lookup_sink,
            ctx,
            oob,
            params,
            type_idx,
            table_idx,
            data_mem_idx,
            entries.len() as u32,
        )?;
        emit_interp_stub(interp_sink, ctx, params)?;
        let _ = (data_seg_idx, returns);
        Ok(data)
    }

    /// Generate all interpreter functions through caller-provided sinks.
    ///
    /// The caller owns three sinks plus a slice of handler sinks — each may be
    /// any `InstructionSink<C, E>` (typically `wasm_encoder::Function`, but any
    /// `wax-core` sink works). This keeps the emitter fully backend-agnostic
    /// and avoids any internal allocation of `Function` objects.
    ///
    /// `handler_sinks.len()` must equal `builder.num_handler_fns()`.
    ///
    /// Use [`register_with_builder`](Self::register_with_builder) in Phase 1 to
    /// pre-allocate the correct number of function slots, then create one sink
    /// per slot (lookup stub, dispatch fn, N handlers) before calling this.
    ///
    /// Returns the passive data bytes for the PC dispatch table.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_with_builder<C, E, S, D, H>(
        &self,
        entity_space: &EntityIndexSpace,
        oob: &OobConfig,
        entries: &[PcEntry],
        params: &[ValType],
        _returns: &[ValType],
        type_idx: u32,
        table_idx: u32,
        _data_seg_idx: u32,
        data_mem_idx: u32,
        lookup_sink: &mut S,
        dispatch_sink: &mut D,
        handler_sinks: &mut [H],
        ctx: &mut C,
        ictx: &mut InterpBuildCtx<'_, C, E>,
        builder: &mut dyn InterpBodyBuilder<C, E>,
    ) -> Result<Vec<u8>, E>
    where
        S: InstructionSink<C, E>,
        D: InstructionSink<C, E>,
        H: InstructionSink<C, E>,
    {
        let n = builder.num_handler_fns() as usize;
        assert_eq!(
            handler_sinks.len(),
            n,
            "handler_sinks length must match builder.num_handler_fns()",
        );

        // Build PC table data.
        let data = Self::build_pc_table(entries);

        // Emit lookup stub into caller's sink.
        emit_lookup_stub(
            lookup_sink,
            ctx,
            oob,
            params,
            type_idx,
            table_idx,
            data_mem_idx,
            entries.len() as u32,
        )?;

        // Resolve absolute function indices for dispatch + handler fns.
        let base = entity_space.functions.base(self.func_slot);
        let dispatch_func_idx = base + 1;
        let handler_func_indices: Vec<u32> = (0..n as u32).map(|i| base + 2 + i).collect();

        ictx.dispatch_func_idx = dispatch_func_idx;
        ictx.handler_func_indices = handler_func_indices;

        // Erase each handler sink to a trait object so `build_interp` can take
        // a uniform slice without further generics.
        let mut handler_sink_refs: Vec<&mut (dyn InstructionSink<C, E>)> =
            handler_sinks.iter_mut().map(|h| h as &mut dyn InstructionSink<C, E>).collect();
        builder.build_interp(dispatch_sink, handler_sink_refs.as_mut_slice(), ctx, ictx)?;

        Ok(data)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn build_pc_table(entries: &[PcEntry]) -> Vec<u8> {
        let mut data = Vec::with_capacity(entries.len() * 12);
        for e in entries {
            data.extend_from_slice(&e.guest_pc.to_le_bytes());
            data.extend_from_slice(&e.table_slot.to_le_bytes());
        }
        data
    }
}

// ── emit_lookup_stub ──────────────────────────────────────────────────────────

/// Emit the lookup stub body into `sink`.
///
/// The stub binary-searches the PC dispatch table in `data_mem_idx` and
/// tail-calls the matching compiled function via `return_call_indirect`, or
/// falls through to the interpreter (`oob.interp_func_idx`) when no entry
/// matches.
#[allow(clippy::too_many_arguments)]
pub fn emit_lookup_stub<Context, E>(
    sink: &mut dyn InstructionSink<Context, E>,
    ctx: &mut Context,
    oob: &OobConfig,
    params: &[ValType],
    type_idx: u32,
    table_idx: u32,
    data_mem_idx: u32,
    n_entries: u32,
) -> Result<(), E> {
    let n_params = params.len() as u32;
    let tpc_local = n_params - 1;
    let lo = n_params;
    let hi = n_params + 1;
    let mid = n_params + 2;
    let mid_pc = n_params + 3;
    let byte_off = n_params + 4;

    // lo = 0; hi = n_entries
    sink.instruction(ctx, &Instruction::I32Const(0))?;
    sink.instruction(ctx, &Instruction::LocalSet(lo))?;
    sink.instruction(ctx, &Instruction::I32Const(n_entries as i32))?;
    sink.instruction(ctx, &Instruction::LocalSet(hi))?;

    // loop { ... }
    sink.instruction(ctx, &Instruction::Loop(BlockType::Empty))?;

    // if lo >= hi → break out of loop
    sink.instruction(ctx, &Instruction::LocalGet(lo))?;
    sink.instruction(ctx, &Instruction::LocalGet(hi))?;
    sink.instruction(ctx, &Instruction::I32GeU)?;
    sink.instruction(ctx, &Instruction::BrIf(1))?;

    // mid = (lo + hi) >> 1
    sink.instruction(ctx, &Instruction::LocalGet(lo))?;
    sink.instruction(ctx, &Instruction::LocalGet(hi))?;
    sink.instruction(ctx, &Instruction::I32Add)?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::I32ShrU)?;
    sink.instruction(ctx, &Instruction::LocalSet(mid))?;

    // byte_off = mid * 12
    sink.instruction(ctx, &Instruction::LocalGet(mid))?;
    sink.instruction(ctx, &Instruction::I32Const(12))?;
    sink.instruction(ctx, &Instruction::I32Mul)?;
    sink.instruction(ctx, &Instruction::LocalSet(byte_off))?;

    // mid_pc = load i64 from mem[byte_off]
    sink.instruction(ctx, &Instruction::LocalGet(byte_off))?;
    sink.instruction(ctx, &Instruction::I64Load(MemArg {
        offset: 0,
        align: 0,
        memory_index: data_mem_idx,
    }))?;
    sink.instruction(ctx, &Instruction::LocalSet(mid_pc))?;

    // if mid_pc == target_pc { dispatch }
    sink.instruction(ctx, &Instruction::LocalGet(mid_pc))?;
    sink.instruction(ctx, &Instruction::LocalGet(tpc_local))?;
    sink.instruction(ctx, &Instruction::I64Eq)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    for p in 0..n_params {
        sink.instruction(ctx, &Instruction::LocalGet(p))?;
    }
    sink.instruction(ctx, &Instruction::LocalGet(byte_off))?;
    sink.instruction(ctx, &Instruction::I32Load(MemArg {
        offset: 8,
        align: 0,
        memory_index: data_mem_idx,
    }))?;
    sink.instruction(ctx, &Instruction::ReturnCallIndirect {
        type_index: type_idx,
        table_index: table_idx,
    })?;
    sink.instruction(ctx, &Instruction::End)?; // end if

    // elif mid_pc < target_pc: lo = mid + 1
    sink.instruction(ctx, &Instruction::LocalGet(mid_pc))?;
    sink.instruction(ctx, &Instruction::LocalGet(tpc_local))?;
    sink.instruction(ctx, &Instruction::I64LtU)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    sink.instruction(ctx, &Instruction::LocalGet(mid))?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::I32Add)?;
    sink.instruction(ctx, &Instruction::LocalSet(lo))?;
    sink.instruction(ctx, &Instruction::Else)?;
    // else: hi = mid
    sink.instruction(ctx, &Instruction::LocalGet(mid))?;
    sink.instruction(ctx, &Instruction::LocalSet(hi))?;
    sink.instruction(ctx, &Instruction::End)?; // end if/else

    sink.instruction(ctx, &Instruction::Br(0))?; // continue loop
    sink.instruction(ctx, &Instruction::End)?; // end loop

    // Not found: tail-call interpreter
    for p in 0..n_params {
        sink.instruction(ctx, &Instruction::LocalGet(p))?;
    }
    sink.instruction(ctx, &Instruction::ReturnCall(oob.interp_func_idx))?;

    sink.instruction(ctx, &Instruction::End) // end function
}

// ── emit_jit_lookup_stub ──────────────────────────────────────────────────────

/// Extra WASM local declarations (beyond `params`) required by
/// [`emit_jit_lookup_stub`], in the order it expects them.
///
/// Callers building a `wasm_encoder::Function` for the JIT-aware lookup stub
/// must append these after `params.len()` — i.e. the function's own locals
/// start at index `params.len()` exactly as with [`emit_lookup_stub`], just
/// with six extra slots at the end (indices `params.len()+5 ..
/// params.len()+11`) beyond the five [`emit_lookup_stub`] already needs.
pub fn jit_lookup_stub_extra_locals() -> [(u32, ValType); 6] {
    [
        (1, ValType::I32), // scan_i
        (1, ValType::I32), // scan_off
        (1, ValType::I64), // scan_pc
        (1, ValType::I32), // scan_val (table_slot / count)
        (1, ValType::I32), // hit_found_flag
        (1, ValType::I32), // hit_empty_off
    ]
}

/// Emit the JIT-aware lookup stub body into `sink`.
///
/// Extends [`emit_lookup_stub`]'s static binary search with two additional,
/// dynamically-populated tiers (see [`JitConfig`] for the on-disk/in-memory
/// table layouts), tried in order on a static-table miss:
///
/// 1. **Dynamic dispatch table** — a bounded linear scan of
///    `jit.dyn_table_capacity` `(pc, table_slot)` entries. A hit
///    `return_call_indirect`s into `dyn_table_idx` (the runtime-mutable
///    sibling of the static `table_idx`).
/// 2. **Hit-count table** — a bounded linear scan of `jit.hit_table_capacity`
///    `(pc, count)` entries. The current PC's count is incremented (or
///    inserted with count 1, if room remains) on every miss of both tables
///    above. Crossing `jit.hit_threshold` fires a fire-and-forget call to
///    `jit.jit_request_func_idx`.
///
/// In every case the stub falls through to `oob.interp_func_idx` — a slow or
/// absent JIT backend can never stall or break execution.
///
/// Requires the caller to have declared [`jit_lookup_stub_extra_locals`] in
/// addition to [`emit_lookup_stub`]'s usual five locals.
#[allow(clippy::too_many_arguments)]
pub fn emit_jit_lookup_stub<Context, E>(
    sink: &mut dyn InstructionSink<Context, E>,
    ctx: &mut Context,
    oob: &OobConfig,
    jit: &JitConfig,
    params: &[ValType],
    type_idx: u32,
    table_idx: u32,
    dyn_table_idx: u32,
    data_mem_idx: u32,
    n_entries: u32,
) -> Result<(), E> {
    let n_params = params.len() as u32;
    let tpc_local = n_params - 1;
    let lo = n_params;
    let hi = n_params + 1;
    let mid = n_params + 2;
    let mid_pc = n_params + 3;
    let byte_off = n_params + 4;
    let scan_i = n_params + 5;
    let scan_off = n_params + 6;
    let scan_pc = n_params + 7;
    let scan_val = n_params + 8;
    let hit_found_flag = n_params + 9;
    let hit_empty_off = n_params + 10;

    // ── Phase A: static binary search (identical logic to emit_lookup_stub,
    // except the "not found" path falls into Phase B instead of the
    // interpreter directly). ──
    sink.instruction(ctx, &Instruction::I32Const(0))?;
    sink.instruction(ctx, &Instruction::LocalSet(lo))?;
    sink.instruction(ctx, &Instruction::I32Const(n_entries as i32))?;
    sink.instruction(ctx, &Instruction::LocalSet(hi))?;

    // Wrapped in an explicit `Block` (unlike `emit_lookup_stub`'s bare loop) so
    // the `BrIf(1)` break below targets this block's `Empty` end rather than
    // the function's own implicit result-typed frame — branching to the
    // latter would require the function's non-empty result types on the
    // stack, which a plain loop-break does not provide. Without this wrapper
    // the module fails validation with e.g. "type mismatch in br_if, expected
    // [i64] but got []".
    sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?;
    sink.instruction(ctx, &Instruction::Loop(BlockType::Empty))?;

    sink.instruction(ctx, &Instruction::LocalGet(lo))?;
    sink.instruction(ctx, &Instruction::LocalGet(hi))?;
    sink.instruction(ctx, &Instruction::I32GeU)?;
    sink.instruction(ctx, &Instruction::BrIf(1))?;

    sink.instruction(ctx, &Instruction::LocalGet(lo))?;
    sink.instruction(ctx, &Instruction::LocalGet(hi))?;
    sink.instruction(ctx, &Instruction::I32Add)?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::I32ShrU)?;
    sink.instruction(ctx, &Instruction::LocalSet(mid))?;

    sink.instruction(ctx, &Instruction::LocalGet(mid))?;
    sink.instruction(ctx, &Instruction::I32Const(12))?;
    sink.instruction(ctx, &Instruction::I32Mul)?;
    sink.instruction(ctx, &Instruction::LocalSet(byte_off))?;

    sink.instruction(ctx, &Instruction::LocalGet(byte_off))?;
    sink.instruction(
        ctx,
        &Instruction::I64Load(MemArg { offset: 0, align: 0, memory_index: data_mem_idx }),
    )?;
    sink.instruction(ctx, &Instruction::LocalSet(mid_pc))?;

    sink.instruction(ctx, &Instruction::LocalGet(mid_pc))?;
    sink.instruction(ctx, &Instruction::LocalGet(tpc_local))?;
    sink.instruction(ctx, &Instruction::I64Eq)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    for p in 0..n_params {
        sink.instruction(ctx, &Instruction::LocalGet(p))?;
    }
    sink.instruction(ctx, &Instruction::LocalGet(byte_off))?;
    sink.instruction(
        ctx,
        &Instruction::I32Load(MemArg { offset: 8, align: 0, memory_index: data_mem_idx }),
    )?;
    sink.instruction(
        ctx,
        &Instruction::ReturnCallIndirect { type_index: type_idx, table_index: table_idx },
    )?;
    sink.instruction(ctx, &Instruction::End)?;

    sink.instruction(ctx, &Instruction::LocalGet(mid_pc))?;
    sink.instruction(ctx, &Instruction::LocalGet(tpc_local))?;
    sink.instruction(ctx, &Instruction::I64LtU)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    sink.instruction(ctx, &Instruction::LocalGet(mid))?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::I32Add)?;
    sink.instruction(ctx, &Instruction::LocalSet(lo))?;
    sink.instruction(ctx, &Instruction::Else)?;
    sink.instruction(ctx, &Instruction::LocalGet(mid))?;
    sink.instruction(ctx, &Instruction::LocalSet(hi))?;
    sink.instruction(ctx, &Instruction::End)?;

    sink.instruction(ctx, &Instruction::Br(0))?;
    sink.instruction(ctx, &Instruction::End)?; // end static-search loop
    sink.instruction(ctx, &Instruction::End)?; // end static-search block

    // ── Phase B: dynamic dispatch table (bounded linear scan). ──
    sink.instruction(ctx, &Instruction::I32Const(0))?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_i))?;

    sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?; // see Phase A's comment on why this wraps the loop
    sink.instruction(ctx, &Instruction::Loop(BlockType::Empty))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_i))?;
    sink.instruction(ctx, &Instruction::I32Const(jit.dyn_table_capacity as i32))?;
    sink.instruction(ctx, &Instruction::I32GeU)?;
    sink.instruction(ctx, &Instruction::BrIf(1))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_i))?;
    sink.instruction(ctx, &Instruction::I32Const(12))?;
    sink.instruction(ctx, &Instruction::I32Mul)?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_off))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_off))?;
    sink.instruction(
        ctx,
        &Instruction::I32Load(MemArg {
            offset: jit.dyn_table_mem_offset + 8,
            align: 0,
            memory_index: jit.dyn_table_mem_idx,
        }),
    )?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_val))?;

    // if table_slot != -1 (occupied)
    sink.instruction(ctx, &Instruction::LocalGet(scan_val))?;
    sink.instruction(ctx, &Instruction::I32Const(-1))?;
    sink.instruction(ctx, &Instruction::I32Ne)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_off))?;
    sink.instruction(
        ctx,
        &Instruction::I64Load(MemArg {
            offset: jit.dyn_table_mem_offset,
            align: 0,
            memory_index: jit.dyn_table_mem_idx,
        }),
    )?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_pc))?;

    // if scan_pc == target_pc: dispatch
    sink.instruction(ctx, &Instruction::LocalGet(scan_pc))?;
    sink.instruction(ctx, &Instruction::LocalGet(tpc_local))?;
    sink.instruction(ctx, &Instruction::I64Eq)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    for p in 0..n_params {
        sink.instruction(ctx, &Instruction::LocalGet(p))?;
    }
    sink.instruction(ctx, &Instruction::LocalGet(scan_val))?;
    sink.instruction(
        ctx,
        &Instruction::ReturnCallIndirect { type_index: type_idx, table_index: dyn_table_idx },
    )?;
    sink.instruction(ctx, &Instruction::End)?; // end pc-match if
    sink.instruction(ctx, &Instruction::End)?; // end occupied if

    sink.instruction(ctx, &Instruction::LocalGet(scan_i))?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::I32Add)?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_i))?;
    sink.instruction(ctx, &Instruction::Br(0))?;
    sink.instruction(ctx, &Instruction::End)?; // end dynamic-scan loop
    sink.instruction(ctx, &Instruction::End)?; // end dynamic-scan block

    // ── Phase C: hit-count table (bounded linear scan; insert-or-increment). ──
    sink.instruction(ctx, &Instruction::I32Const(0))?;
    sink.instruction(ctx, &Instruction::LocalSet(hit_found_flag))?;
    sink.instruction(ctx, &Instruction::I32Const(-1))?;
    sink.instruction(ctx, &Instruction::LocalSet(hit_empty_off))?;
    sink.instruction(ctx, &Instruction::I32Const(0))?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_i))?;

    sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?; // see Phase A's comment on why this wraps the loop
    sink.instruction(ctx, &Instruction::Loop(BlockType::Empty))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_i))?;
    sink.instruction(ctx, &Instruction::I32Const(jit.hit_table_capacity as i32))?;
    sink.instruction(ctx, &Instruction::I32GeU)?;
    sink.instruction(ctx, &Instruction::BrIf(1))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_i))?;
    sink.instruction(ctx, &Instruction::I32Const(12))?;
    sink.instruction(ctx, &Instruction::I32Mul)?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_off))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_off))?;
    sink.instruction(
        ctx,
        &Instruction::I32Load(MemArg {
            offset: jit.hit_table_mem_offset + 8,
            align: 0,
            memory_index: jit.hit_table_mem_idx,
        }),
    )?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_val))?;

    // if count == 0 (empty slot) { remember first empty offset } else { check for pc match }
    sink.instruction(ctx, &Instruction::LocalGet(scan_val))?;
    sink.instruction(ctx, &Instruction::I32Eqz)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;

    sink.instruction(ctx, &Instruction::LocalGet(hit_empty_off))?;
    sink.instruction(ctx, &Instruction::I32Const(-1))?;
    sink.instruction(ctx, &Instruction::I32Eq)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    sink.instruction(ctx, &Instruction::LocalGet(scan_off))?;
    sink.instruction(ctx, &Instruction::LocalSet(hit_empty_off))?;
    sink.instruction(ctx, &Instruction::End)?; // end "remember first empty"

    sink.instruction(ctx, &Instruction::Else)?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_off))?;
    sink.instruction(
        ctx,
        &Instruction::I64Load(MemArg {
            offset: jit.hit_table_mem_offset,
            align: 0,
            memory_index: jit.hit_table_mem_idx,
        }),
    )?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_pc))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_pc))?;
    sink.instruction(ctx, &Instruction::LocalGet(tpc_local))?;
    sink.instruction(ctx, &Instruction::I64Eq)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    // count += 1; store back
    sink.instruction(ctx, &Instruction::LocalGet(scan_off))?;
    sink.instruction(ctx, &Instruction::LocalGet(scan_val))?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::I32Add)?;
    sink.instruction(ctx, &Instruction::LocalTee(scan_val))?;
    sink.instruction(
        ctx,
        &Instruction::I32Store(MemArg {
            offset: jit.hit_table_mem_offset + 8,
            align: 0,
            memory_index: jit.hit_table_mem_idx,
        }),
    )?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::LocalSet(hit_found_flag))?;
    emit_maybe_request_jit(sink, ctx, jit, scan_val, tpc_local)?;
    sink.instruction(ctx, &Instruction::End)?; // end pc-match if
    sink.instruction(ctx, &Instruction::End)?; // end empty/occupied if-else

    // if hit_found_flag: break out of the scan loop now. At this point the
    // only enclosing structures (both ifs above already closed) are the Loop
    // and its wrapping Block, so br 1 targets the Block — same as the
    // capacity check above.
    sink.instruction(ctx, &Instruction::LocalGet(hit_found_flag))?;
    sink.instruction(ctx, &Instruction::BrIf(1))?;

    sink.instruction(ctx, &Instruction::LocalGet(scan_i))?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::I32Add)?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_i))?;
    sink.instruction(ctx, &Instruction::Br(0))?;
    sink.instruction(ctx, &Instruction::End)?; // end hit-scan loop
    sink.instruction(ctx, &Instruction::End)?; // end hit-scan block

    // Not found and not already handled: insert at first empty slot, if any.
    sink.instruction(ctx, &Instruction::LocalGet(hit_found_flag))?;
    sink.instruction(ctx, &Instruction::I32Eqz)?;
    sink.instruction(ctx, &Instruction::LocalGet(hit_empty_off))?;
    sink.instruction(ctx, &Instruction::I32Const(-1))?;
    sink.instruction(ctx, &Instruction::I32Ne)?;
    sink.instruction(ctx, &Instruction::I32And)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    sink.instruction(ctx, &Instruction::LocalGet(hit_empty_off))?;
    sink.instruction(ctx, &Instruction::LocalGet(tpc_local))?;
    sink.instruction(
        ctx,
        &Instruction::I64Store(MemArg {
            offset: jit.hit_table_mem_offset,
            align: 0,
            memory_index: jit.hit_table_mem_idx,
        }),
    )?;
    sink.instruction(ctx, &Instruction::LocalGet(hit_empty_off))?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(
        ctx,
        &Instruction::I32Store(MemArg {
            offset: jit.hit_table_mem_offset + 8,
            align: 0,
            memory_index: jit.hit_table_mem_idx,
        }),
    )?;
    sink.instruction(ctx, &Instruction::I32Const(1))?;
    sink.instruction(ctx, &Instruction::LocalSet(scan_val))?; // count is exactly 1 on fresh insert
    emit_maybe_request_jit(sink, ctx, jit, scan_val, tpc_local)?;
    sink.instruction(ctx, &Instruction::End)?; // end "insert" if
    // (else: both tables full — degrade gracefully, never request JIT for this PC.)

    // ── Phase D: tail-call interpreter (same as emit_lookup_stub's not-found
    // path). ──
    for p in 0..n_params {
        sink.instruction(ctx, &Instruction::LocalGet(p))?;
    }
    sink.instruction(ctx, &Instruction::ReturnCall(oob.interp_func_idx))?;

    sink.instruction(ctx, &Instruction::End) // end function
}

/// Emit `if count_local >= jit.hit_threshold { call
/// jit.jit_request_func_idx(local.get tpc_local) }`, a no-op if
/// `jit.jit_request_func_idx` is `None` or `jit.hit_threshold` is `0`.
///
/// `count_local` must already hold the just-updated count as an i32;
/// `tpc_local` must be the local index holding `target_pc: i64`.
fn emit_maybe_request_jit<Context, E>(
    sink: &mut dyn InstructionSink<Context, E>,
    ctx: &mut Context,
    jit: &JitConfig,
    count_local: u32,
    tpc_local: u32,
) -> Result<(), E> {
    let (Some(request_func_idx), true) = (jit.jit_request_func_idx, jit.hit_threshold > 0) else {
        return Ok(());
    };
    sink.instruction(ctx, &Instruction::LocalGet(count_local))?;
    sink.instruction(ctx, &Instruction::I32Const(jit.hit_threshold as i32))?;
    sink.instruction(ctx, &Instruction::I32GeU)?;
    sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
    sink.instruction(ctx, &Instruction::LocalGet(tpc_local))?;
    sink.instruction(ctx, &Instruction::Call(request_func_idx))?;
    sink.instruction(ctx, &Instruction::End)
}

/// Emit the lookup stub, dispatching to [`emit_jit_lookup_stub`] when
/// `oob.jit` is `Some`, or to [`emit_lookup_stub`] (unmodified) when `None`.
///
/// This is the single call site that guarantees `oob.jit: None` reproduces
/// today's interpreter-only OOB behaviour byte-for-byte: the `None` arm calls
/// [`emit_lookup_stub`] directly, unchanged from before the JIT seam existed.
#[allow(clippy::too_many_arguments)]
pub fn emit_lookup_stub_dispatch<Context, E>(
    sink: &mut dyn InstructionSink<Context, E>,
    ctx: &mut Context,
    oob: &OobConfig,
    params: &[ValType],
    type_idx: u32,
    table_idx: u32,
    dyn_table_idx: u32,
    data_mem_idx: u32,
    n_entries: u32,
) -> Result<(), E> {
    match &oob.jit {
        None => emit_lookup_stub(sink, ctx, oob, params, type_idx, table_idx, data_mem_idx, n_entries),
        Some(jit) => emit_jit_lookup_stub(
            sink, ctx, oob, jit, params, type_idx, table_idx, dyn_table_idx, data_mem_idx, n_entries,
        ),
    }
}

// ── emit_interp_stub ──────────────────────────────────────────────────────────

/// Emit a placeholder interpreter function body that traps (`unreachable`).
///
/// Used by [`OobInterp::generate`] (the backwards-compatible path) and by
/// [`NullInterpBuilder`].
pub fn emit_interp_stub<Context, E>(
    sink: &mut dyn InstructionSink<Context, E>,
    ctx: &mut Context,
    params: &[ValType],
) -> Result<(), E> {
    let n_params = params.len() as u32;
    for p in 0..n_params {
        sink.instruction(ctx, &Instruction::LocalGet(p))?;
        sink.instruction(ctx, &Instruction::Drop)?;
    }
    sink.instruction(ctx, &Instruction::Unreachable)?;
    sink.instruction(ctx, &Instruction::End)
}
