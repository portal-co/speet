//! # speet-wasm — WASM-to-WASM transforming frontend
//!
//! Parses a WebAssembly module and re-emits each function with optional
//! address-space translation, global-index remapping, and call-index offsetting.
//!
//! Unlike the native-arch recompilers, this frontend bypasses `yecta::Reactor`
//! entirely: WASM functions are already self-contained, so no CFG reconstruction
//! or tail-call splitting is needed.  Each input function maps 1:1 to one output `F`.
//!
//! ## Usage
//!
//! ```ignore
//! use speet_wasm::{WasmFrontend, MemoryMapConfig, IndexOffsets};
//! use speet_memory::AddressWidth;
//! use wasm_encoder::Function;
//!
//! let mut frontend = WasmFrontend::new(
//!     MemoryMapConfig { addr_width: AddressWidth::W32, memory_index: 0 },
//!     IndexOffsets { func: 10, global: 0, table: 0 },
//!     Box::new(|locals| Function::new(locals)),
//!     None,
//! );
//! frontend.translate_module(&mut ctx, &wasm_bytes)?;
//! let unit = frontend.drain_unit(&mut linker, entry_points);
//! ```

extern crate alloc;

use alloc::{boxed::Box, string::String, vec::Vec};
use speet_link::{BinaryUnit, Recompile, context::ReactorContext, unit::FuncType};
use speet_memory::{AddressWidth, IntWidth, LoadKind, MapperCallback, MemoryEmitter, PageMapLocals, StoreKind};
use wasm_encoder::{Function, Instruction, MemArg, ValType};
use wax_core::build::InstructionSink;
use wasmparser::{CompositeInnerType, FunctionBody, Operator, Parser, Payload};

// ── Configuration ─────────────────────────────────────────────────────────────

/// How the memory address-space mapper is configured for guest loads/stores.
#[derive(Clone, Copy, Debug)]
pub struct MemoryMapConfig {
    /// Width of guest addresses pushed onto the WASM stack.
    pub addr_width: AddressWidth,
    /// Index of the host linear memory to use (typically 0).
    pub memory_index: u32,
}

/// Additive offsets applied when re-emitting call and global instructions.
#[derive(Clone, Copy, Default, Debug)]
pub struct IndexOffsets {
    /// Added to every `call N` function index in the guest.
    pub func: u32,
    /// Added (as `i64`, then cast to `u32`) to every `global.get/set N` index.
    pub global: i64,
    /// Added to `call_indirect` table indices.
    pub table: u32,
}

// ── WasmFrontend ──────────────────────────────────────────────────────────────

/// WASM-to-WASM transforming frontend.
///
/// Generic over the output function type `F` (defaults to `wasm_encoder::Function`).
/// Implements [`Recompile`] so it can participate in the `speet-link`
/// multi-binary linking flow without going through `yecta::Reactor`.
pub struct WasmFrontend<Context, E, F = Function> {
    /// Accumulated (function, type) pairs; drained by `drain_unit`.
    compiled: Vec<(F, FuncType)>,
    /// Memory-access configuration.
    memory_cfg: MemoryMapConfig,
    /// Index offsets for calls, globals, and tables.
    offsets: IndexOffsets,
    /// Factory that constructs an output function from its local-variable list.
    ///
    /// Called once per translated function; receives the full local list
    /// (original locals + injected scratch slots).
    fn_creator: Box<dyn Fn(Vec<(u32, ValType)>) -> F>,
    /// Optional mapper factory.  Called once per function to produce a
    /// [`MapperCallback`] for that function's scratch locals.
    mapper_factory: Option<Box<dyn Fn(PageMapLocals) -> Box<dyn MapperCallback<Context, E, F>>>>,
}

impl<Context, E, F> WasmFrontend<Context, E, F> {
    /// Create a new `WasmFrontend`.
    ///
    /// * `memory_cfg`     — address width and target memory index.
    /// * `offsets`        — additive index offsets for calls/globals/tables.
    /// * `fn_creator`     — closure that builds an `F` from a local-variable list.
    /// * `mapper_factory` — optional factory producing a per-function mapper.
    pub fn new(
        memory_cfg: MemoryMapConfig,
        offsets: IndexOffsets,
        fn_creator: Box<dyn Fn(Vec<(u32, ValType)>) -> F>,
        mapper_factory: Option<Box<dyn Fn(PageMapLocals) -> Box<dyn MapperCallback<Context, E, F>>>>,
    ) -> Self {
        Self {
            compiled: Vec::new(),
            memory_cfg,
            offsets,
            fn_creator,
            mapper_factory,
        }
    }
}

impl<Context, E> WasmFrontend<Context, E, Function> {
    /// Convenience constructor that uses `wasm_encoder::Function` as the
    /// output type.  Equivalent to calling [`new`](Self::new) with
    /// `Box::new(|locals| Function::new(locals))`.
    pub fn with_wasm_encoder_fn(
        memory_cfg: MemoryMapConfig,
        offsets: IndexOffsets,
        mapper_factory: Option<
            Box<dyn Fn(PageMapLocals) -> Box<dyn MapperCallback<Context, E, Function>>>,
        >,
    ) -> Self {
        Self::new(
            memory_cfg,
            offsets,
            Box::new(|locals| Function::new(locals)),
            mapper_factory,
        )
    }
}

impl<Context, E, F> WasmFrontend<Context, E, F>
where
    F: InstructionSink<Context, E>,
    E: From<wasmparser::BinaryReaderError>,
{
    // ── Public API ────────────────────────────────────────────────────────

    /// Parse a WASM module's type + code sections and transform every function.
    ///
    /// Translated functions are accumulated in `self.compiled`.  Call
    /// `Recompile::drain_unit` to collect them into a [`BinaryUnit`].
    pub fn translate_module(&mut self, ctx: &mut Context, bytes: &[u8]) -> Result<(), E> {
        let mut types: Vec<FuncType> = Vec::new();
        let mut fn_type_indices: Vec<u32> = Vec::new();
        let mut code_fn_idx: usize = 0;

        for payload in Parser::new(0).parse_all(bytes) {
            let payload = payload?;
            match payload {
                Payload::TypeSection(reader) => {
                    for rec_group in reader {
                        let rec = rec_group?;
                        for sub_ty in rec.types() {
                            match &sub_ty.composite_type.inner {
                                CompositeInnerType::Func(ft) => {
                                    types.push(func_type_from_wasmparser(ft));
                                }
                                _ => {
                                    types.push(FuncType {
                                        params: Vec::new(),
                                        results: Vec::new(),
                                    });
                                }
                            }
                        }
                    }
                }

                Payload::FunctionSection(reader) => {
                    for type_idx in reader {
                        fn_type_indices.push(type_idx?);
                    }
                }

                Payload::CodeSectionEntry(body) => {
                    let type_idx = fn_type_indices[code_fn_idx] as usize;
                    let func_type = types
                        .get(type_idx)
                        .cloned()
                        .unwrap_or_else(|| FuncType { params: Vec::new(), results: Vec::new() });
                    self.translate_fn(ctx, func_type, body)?;
                    code_fn_idx += 1;
                }

                _ => {}
            }
        }
        Ok(())
    }

    // ── Per-function translation ───────────────────────────────────────────

    fn translate_fn(
        &mut self,
        ctx: &mut Context,
        func_type: FuncType,
        body: FunctionBody<'_>,
    ) -> Result<(), E> {
        // ── Collect locals ────────────────────────────────────────────────
        let mut out_locals: Vec<(u32, ValType)> = Vec::new();
        let mut original_local_count: u32 = 0;

        let locals_reader = body.get_locals_reader()?;
        for local in locals_reader {
            let (count, wasm_ty) = local?;
            let vt = val_type_from_wasmparser(wasm_ty);
            out_locals.push((count, vt));
            original_local_count += count;
        }

        // Params occupy indices 0..param_count; explicit locals follow.
        let param_count = func_type.params_val_types().count() as u32;
        let base_scratch_idx = param_count + original_local_count;

        // Append 4 type-specific scratch locals for store value saving:
        //   base + 0 → i32  (I32Store, I32Store8, I32Store16)
        //   base + 1 → i64  (I64Store, I64Store8, I64Store16, I64Store32)
        //   base + 2 → f32  (F32Store)
        //   base + 3 → f64  (F64Store)
        out_locals.push((1, ValType::I32));
        out_locals.push((1, ValType::I64));
        out_locals.push((1, ValType::F32));
        out_locals.push((1, ValType::F64));

        // If a mapper is configured, append 4 more i32 locals for PageMapLocals
        // (vaddr + scratch[0..2]), starting immediately after the value-scratch block.
        let mut mapper_box: Option<Box<dyn MapperCallback<Context, E, F>>> =
            if let Some(factory) = &self.mapper_factory {
                let map_locals = PageMapLocals::consecutive(base_scratch_idx + 4);
                out_locals.push((4, ValType::I32));
                Some(factory(map_locals))
            } else {
                None
            };

        let mut out = (self.fn_creator)(out_locals);

        // ── Instruction loop ──────────────────────────────────────────────
        // `mapper_box` owns the mapper; we reborrow it as
        // `Option<&mut dyn MapperCallback + '_>` on every iteration so that
        // each call to `translate_op` receives an independent, lifetime-safe
        // borrow that matches `MemoryEmitter`'s mapper field directly.
        let ops_reader = body.get_operators_reader()?;
        for op_result in ops_reader {
            let op = op_result?;
            let mapper_ref: Option<&mut (dyn MapperCallback<Context, E, F> + '_)> =
                mapper_box.as_mut().map(|b| b.as_mut() as _);
            self.translate_op(ctx, &op, &mut out, base_scratch_idx, mapper_ref)?;
        }

        self.compiled.push((out, func_type));
        Ok(())
    }

    fn translate_op(
        &mut self,
        ctx: &mut Context,
        op: &Operator<'_>,
        out: &mut F,
        base_scratch_idx: u32,
        // Lifetime-erased borrow of the per-function mapper box.
        // Matches `MemoryEmitter::mapper` exactly — no extra indirection.
        mapper: Option<&mut (dyn MapperCallback<Context, E, F> + '_)>,
    ) -> Result<(), E> {
        let addr_width = self.memory_cfg.addr_width;
        let memory_index = self.memory_cfg.memory_index;
        let offsets = self.offsets;

        match op {
            // ── Memory loads ──────────────────────────────────────────────
            // emit_offset incorporates memarg.offset into the address, then
            // emit_load handles wrap + mapper + instruction + any extension.
            // int_width is set per-instruction so that e.g. I64Load8S gets
            // `I32Load8S + I64ExtendI32S` rather than a bare `I32Load8S`.
            Operator::I32Load { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I32, memory_index, mapper, LoadKind::I32S)?;
            }
            Operator::I64Load { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I64, memory_index, mapper, LoadKind::I64)?;
            }
            Operator::F32Load { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I32, memory_index, mapper, LoadKind::F32)?;
            }
            Operator::F64Load { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I64, memory_index, mapper, LoadKind::F64)?;
            }
            Operator::I32Load8S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I32, memory_index, mapper, LoadKind::I8S)?;
            }
            Operator::I32Load8U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I32, memory_index, mapper, LoadKind::I8U)?;
            }
            Operator::I32Load16S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I32, memory_index, mapper, LoadKind::I16S)?;
            }
            Operator::I32Load16U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I32, memory_index, mapper, LoadKind::I16U)?;
            }
            Operator::I64Load8S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I64, memory_index, mapper, LoadKind::I8S)?;
            }
            Operator::I64Load8U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I64, memory_index, mapper, LoadKind::I8U)?;
            }
            Operator::I64Load16S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I64, memory_index, mapper, LoadKind::I16S)?;
            }
            Operator::I64Load16U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I64, memory_index, mapper, LoadKind::I16U)?;
            }
            Operator::I64Load32S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I64, memory_index, mapper, LoadKind::I32S)?;
            }
            Operator::I64Load32U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(ctx, out, addr_width, IntWidth::I64, memory_index, mapper, LoadKind::I32U)?;
            }

            // ── Memory stores ─────────────────────────────────────────────
            //
            // Stack at store op: [..., addr, value]  (value on top)
            // Strategy:
            //   1. local.set type_scratch   (save value; stack: [..., addr])
            //   2. emit_offset              (stack: [..., addr + offset])
            //   3. emit_store_addr          (wrap + mapper; stack: [..., mapped])
            //   4. local.get type_scratch   (stack: [..., mapped, value])
            //   5. emit store instruction   (correct type, offset=0)
            //
            // The type-specific scratch slot ensures that f32/f64 values are
            // saved and restored correctly (a single i32 scratch was wrong).
            Operator::I32Store { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::I32, IntWidth::I32, base_scratch_idx)?;
            }
            Operator::I64Store { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::I64, IntWidth::I64, base_scratch_idx)?;
            }
            Operator::F32Store { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::F32, IntWidth::I32, base_scratch_idx)?;
            }
            Operator::F64Store { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::F64, IntWidth::I64, base_scratch_idx)?;
            }
            Operator::I32Store8 { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::I8, IntWidth::I32, base_scratch_idx)?;
            }
            Operator::I32Store16 { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::I16, IntWidth::I32, base_scratch_idx)?;
            }
            Operator::I64Store8 { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::I8, IntWidth::I64, base_scratch_idx)?;
            }
            Operator::I64Store16 { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::I16, IntWidth::I64, base_scratch_idx)?;
            }
            Operator::I64Store32 { memarg } => {
                emit_store(ctx, out, memarg, addr_width, memory_index, mapper, StoreKind::I32, IntWidth::I64, base_scratch_idx)?;
            }

            // ── Global accesses ───────────────────────────────────────────
            Operator::GlobalGet { global_index } => {
                let host_idx = (*global_index as i64 + offsets.global) as u32;
                out.instruction(ctx, &Instruction::GlobalGet(host_idx))?;
            }
            Operator::GlobalSet { global_index } => {
                let host_idx = (*global_index as i64 + offsets.global) as u32;
                out.instruction(ctx, &Instruction::GlobalSet(host_idx))?;
            }

            // ── Function calls ────────────────────────────────────────────
            Operator::Call { function_index } => {
                out.instruction(ctx, &Instruction::Call(function_index + offsets.func))?;
            }
            Operator::CallIndirect { type_index, table_index } => {
                out.instruction(ctx, &Instruction::CallIndirect {
                    type_index: *type_index,
                    table_index: table_index + offsets.table,
                })?;
            }

            // ── Pass-through ──────────────────────────────────────────────
            other => {
                let insn = Instruction::try_from(other.clone())
                    .unwrap_or(Instruction::Unreachable);
                out.instruction(ctx, &insn)?;
            }
        }
        Ok(())
    }
}

// ── Recompile impl ────────────────────────────────────────────────────────────

impl<Context, E, F> Recompile<Context, E, F> for WasmFrontend<Context, E, F>
where
    F: InstructionSink<Context, E>,
    E: From<wasmparser::BinaryReaderError>,
{
    type BinaryArgs = ();

    fn reset_for_next_binary(
        &mut self,
        _ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        _args: (),
    ) {
        // `compiled` was already drained by `drain_unit`; no per-binary
        // state other than that accumulator needs resetting.
    }

    fn drain_unit(
        &mut self,
        ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        entry_points: Vec<(String, u32)>,
    ) -> BinaryUnit<F> {
        let base = ctx.base_func_offset();
        let collected: Vec<(F, FuncType)> = core::mem::take(&mut self.compiled);
        let n = collected.len() as u32;
        let (fns, func_types): (Vec<F>, Vec<FuncType>) = collected.into_iter().unzip();
        ctx.advance_base_func_offset(n);
        BinaryUnit { fns, base_func_offset: base, entry_points, func_types }
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Emit the offset addition for a memory instruction.
///
/// Stack before: `[..., addr]`
/// Stack after:  `[..., addr + offset]`  (unchanged if `offset == 0`)
fn emit_offset<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    out: &mut F,
    offset: u64,
    addr_width: AddressWidth,
) -> Result<(), E> {
    if offset == 0 {
        return Ok(());
    }
    if addr_width.is_64() {
        out.instruction(ctx, &Instruction::I64Const(offset as i64))?;
        out.instruction(ctx, &Instruction::I64Add)?;
    } else {
        out.instruction(ctx, &Instruction::I32Const(offset as i32))?;
        out.instruction(ctx, &Instruction::I32Add)?;
    }
    Ok(())
}

/// Emit a complete load sequence via [`MemoryEmitter`].
///
/// The mapper reference is passed directly — no extra `.map()` call needed —
/// because [`MemoryEmitter`]'s `mapper` field has the same type.
fn emit_load<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    out: &mut F,
    addr_width: AddressWidth,
    int_width: IntWidth,
    memory_index: u32,
    mapper: Option<&mut (dyn MapperCallback<Context, E, F> + '_)>,
    kind: LoadKind,
) -> Result<(), E> {
    let mut emitter = MemoryEmitter::new(addr_width, int_width, memory_index, mapper);
    emitter.emit_load(ctx, out, kind)
}

/// Emit a complete store sequence.
///
/// Uses [`MemoryEmitter::emit_store_addr`] for the address-translation phase
/// (wrap + mapper), then emits the correct WASM store instruction directly so
/// that f32/f64 stores and i64 narrow stores (e.g. `I64Store8`) round-trip
/// with the right value type — [`MemoryEmitter::emit_store_insn`] is designed
/// for arch-recompiler pipelines and would emit the wrong instruction for those.
///
/// The four scratch locals allocated at `base_scratch_idx + 0..3` match value
/// types: `i32`, `i64`, `f32`, `f64` respectively.
#[allow(clippy::too_many_arguments)]
fn emit_store<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    out: &mut F,
    memarg: &wasmparser::MemArg,
    addr_width: AddressWidth,
    memory_index: u32,
    mapper: Option<&mut (dyn MapperCallback<Context, E, F> + '_)>,
    kind: StoreKind,
    int_width: IntWidth,
    base_scratch_idx: u32,
) -> Result<(), E> {
    // Select the type-matched scratch local.
    let scratch = base_scratch_idx + match kind {
        StoreKind::F32 => 2,
        StoreKind::F64 => 3,
        _ => match int_width {
            IntWidth::I32 => 0,
            IntWidth::I64 => 1,
        },
    };

    // 1. Save value; stack: [..., addr]
    out.instruction(ctx, &Instruction::LocalSet(scratch))?;

    // 2. Incorporate offset; stack: [..., addr + offset]
    emit_offset(ctx, out, memarg.offset, addr_width)?;

    // 3. Address wrap + mapper via MemoryEmitter; stack: [..., mapped_addr]
    {
        let mut emitter = MemoryEmitter::new(addr_width, int_width, memory_index, mapper);
        emitter.emit_store_addr(ctx, out)?;
    }

    // 4. Restore value; stack: [..., mapped_addr, value]
    out.instruction(ctx, &Instruction::LocalGet(scratch))?;

    // 5. Emit the store instruction with offset=0 and the remapped memory index.
    //    Reconstructed directly from (kind, int_width) to preserve the correct
    //    WASM type (e.g. I64Store8 stores an i64, not an i32).
    let ma = MemArg { offset: 0, align: memarg.align as u32, memory_index };
    match kind {
        StoreKind::I8 => match int_width {
            IntWidth::I32 => out.instruction(ctx, &Instruction::I32Store8(ma))?,
            IntWidth::I64 => out.instruction(ctx, &Instruction::I64Store8(ma))?,
        },
        StoreKind::I16 => match int_width {
            IntWidth::I32 => out.instruction(ctx, &Instruction::I32Store16(ma))?,
            IntWidth::I64 => out.instruction(ctx, &Instruction::I64Store16(ma))?,
        },
        StoreKind::I32 => match int_width {
            IntWidth::I32 => out.instruction(ctx, &Instruction::I32Store(ma))?,
            IntWidth::I64 => out.instruction(ctx, &Instruction::I64Store32(ma))?,
        },
        StoreKind::I64 => out.instruction(ctx, &Instruction::I64Store(ma))?,
        StoreKind::F32 => out.instruction(ctx, &Instruction::F32Store(ma))?,
        StoreKind::F64 => out.instruction(ctx, &Instruction::F64Store(ma))?,
    }

    Ok(())
}

// ── wasmparser type conversions ───────────────────────────────────────────────

fn val_type_from_wasmparser(ty: wasmparser::ValType) -> ValType {
    match ty {
        wasmparser::ValType::I32 => ValType::I32,
        wasmparser::ValType::I64 => ValType::I64,
        wasmparser::ValType::F32 => ValType::F32,
        wasmparser::ValType::F64 => ValType::F64,
        wasmparser::ValType::V128 => ValType::V128,
        wasmparser::ValType::Ref(rt) => {
            use wasm_encoder::{HeapType, RefType};
            let heap_type = match rt.heap_type() {
                wasmparser::HeapType::Abstract { shared: false, ty } => match ty {
                    wasmparser::AbstractHeapType::Func => HeapType::Abstract {
                        shared: false,
                        ty: wasm_encoder::AbstractHeapType::Func,
                    },
                    wasmparser::AbstractHeapType::Extern => HeapType::Abstract {
                        shared: false,
                        ty: wasm_encoder::AbstractHeapType::Extern,
                    },
                    _ => HeapType::Abstract {
                        shared: false,
                        ty: wasm_encoder::AbstractHeapType::Func,
                    },
                },
                _ => HeapType::Abstract {
                    shared: false,
                    ty: wasm_encoder::AbstractHeapType::Func,
                },
            };
            ValType::Ref(RefType { nullable: rt.is_nullable(), heap_type })
        }
    }
}

fn func_type_from_wasmparser(ft: &wasmparser::FuncType) -> FuncType {
    let params: Vec<ValType> =
        ft.params().iter().copied().map(val_type_from_wasmparser).collect();
    let results: Vec<ValType> =
        ft.results().iter().copied().map(val_type_from_wasmparser).collect();
    FuncType::from_val_types(&params, &results)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use wasmparser::BinaryReaderError;

    type TestError = BinaryReaderError;

    /// Build a minimal WASM module with one function that exercises
    /// loads, stores (with non-zero offsets), globals, and a call.
    fn build_test_wasm() -> Vec<u8> {
        use wasm_encoder::*;

        let mut module = Module::new();

        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I32]);
        module.section(&types);

        let mut funcs = FunctionSection::new();
        funcs.function(0);
        module.section(&funcs);

        let mut mems = MemorySection::new();
        mems.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mems);

        let mut globals = GlobalSection::new();
        globals.global(
            GlobalType { val_type: ValType::I32, mutable: true, shared: false },
            &ConstExpr::i32_const(0),
        );
        module.section(&globals);

        let mut codes = CodeSection::new();
        let mut f = Function::new([]);
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Load(MemArg { offset: 4, align: 2, memory_index: 0 }));
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Const(100));
        f.instruction(&Instruction::I32Store(MemArg { offset: 8, align: 2, memory_index: 0 }));
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::Call(0));
        f.instruction(&Instruction::End);
        codes.function(&f);
        module.section(&codes);

        module.finish()
    }

    #[test]
    fn translate_one_function() {
        let bytes = build_test_wasm();

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            MemoryMapConfig { addr_width: AddressWidth::W32, memory_index: 0 },
            IndexOffsets::default(),
            None,
        );

        frontend.translate_module(&mut (), &bytes).unwrap();
        assert_eq!(frontend.compiled.len(), 1, "expected 1 compiled function");
    }

    #[test]
    fn drain_unit_advances_offset() {
        use speet_link::Linker;

        let bytes = build_test_wasm();

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            MemoryMapConfig { addr_width: AddressWidth::W32, memory_index: 0 },
            IndexOffsets::default(),
            None,
        );

        frontend.translate_module(&mut (), &bytes).unwrap();

        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();

        assert_eq!(linker.base_func_offset(), 0);
        let unit = frontend.drain_unit(&mut linker, alloc::vec![]);
        assert_eq!(unit.base_func_offset, 0);
        assert_eq!(unit.fns.len(), 1);
        assert_eq!(linker.base_func_offset(), 1, "offset should advance by 1");
    }

    #[test]
    fn call_index_remapped() {
        let bytes = build_test_wasm();

        // Custom fn_creator: same as Function::new but via the generic path.
        let mut frontend: WasmFrontend<(), TestError, Function> = WasmFrontend::new(
            MemoryMapConfig { addr_width: AddressWidth::W32, memory_index: 0 },
            IndexOffsets { func: 5, global: 0, table: 0 },
            Box::new(|locals| Function::new(locals)),
            None,
        );

        frontend.translate_module(&mut (), &bytes).unwrap();
        assert_eq!(frontend.compiled.len(), 1);
    }
}
