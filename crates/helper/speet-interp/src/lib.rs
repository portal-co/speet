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

use alloc::boxed::Box;
use alloc::vec::Vec;
use speet_link_core::{EntityIndexSpace, IndexSlot, OobConfig};
use wasm_encoder::{BlockType, Function, Instruction, MemArg, ValType};
use wax_core::build::InstructionSink;

pub mod builder;
pub mod context;

pub use builder::{InterpBodyBuilder, NullInterpBuilder};
pub use context::{FlatEmitSink, FlatMemorySink, InterpBuildCtx};

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

    /// Generate all interpreter functions using a [`InterpBodyBuilder`].
    ///
    /// Returns `(lookup_fn, Vec<interp_fns>, passive_data_bytes)` where
    /// `interp_fns` contains the dispatch fn at index 0 followed by N handler fns.
    ///
    /// Use [`register_with_builder`](Self::register_with_builder) in Phase 1 to
    /// pre-allocate the correct number of function slots.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_with_builder<C, E>(
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
        ctx: &mut C,
        ictx: &mut InterpBuildCtx<'_, C, E>,
        builder: &mut dyn InterpBodyBuilder<C, E>,
    ) -> Result<(Function, Vec<Function>, Vec<u8>), E> {
        // Build PC table data.
        let data = Self::build_pc_table(entries);

        // Emit lookup stub into a fresh Function.
        let mut lookup_fn = Function::new([
            (1, ValType::I32),
            (1, ValType::I32),
            (1, ValType::I32),
            (1, ValType::I64),
            (1, ValType::I32),
        ]);
        emit_lookup_stub(
            &mut lookup_fn,
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
        let n = builder.num_handler_fns();
        let handler_func_indices: Vec<u32> = (0..n).map(|i| base + 2 + i).collect();

        ictx.dispatch_func_idx = dispatch_func_idx;
        ictx.handler_func_indices = handler_func_indices;

        // Create sinks for dispatch fn + handler fns using declared locals.
        let mut dispatch_fn = Function::new(builder.dispatch_fn_locals());
        let mut handler_fns: Vec<Function> = (0..n)
            .map(|i| Function::new(builder.handler_fn_locals(i)))
            .collect();

        // SAFETY: handler_fns outlives handler_sinks; we drop handler_sinks
        // (and thus all borrows) before accessing handler_fns below.
        let mut handler_sinks: Vec<Box<dyn InstructionSink<C, E>>> = handler_fns
            .iter_mut()
            .map(|f| -> Box<dyn InstructionSink<C, E>> {
                let f: &'static mut Function = unsafe { &mut *(f as *mut Function) };
                Box::new(BorrowedFnSink(f))
            })
            .collect();

        builder.build_interp(&mut dispatch_fn, &mut handler_sinks, ctx, ictx)?;
        drop(handler_sinks); // release raw borrows before moving handler_fns

        let _ = (data_seg_idx, returns);
        let mut interp_fns = Vec::with_capacity(1 + n as usize);
        interp_fns.push(dispatch_fn);
        interp_fns.extend(handler_fns);

        Ok((lookup_fn, interp_fns, data))
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

// ── BorrowedFnSink ────────────────────────────────────────────────────────────

/// Newtype that lets a `&mut Function` act as a `Box<dyn InstructionSink>`.
struct BorrowedFnSink<'a>(&'a mut Function);

impl<Context, E> InstructionSink<Context, E> for BorrowedFnSink<'_> {
    fn instruction(&mut self, _ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        self.0.instruction(instruction);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), E> {
        self.0.instruction(&Instruction::End);
        Ok(())
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
