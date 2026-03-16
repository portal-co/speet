//! # speet-wasm — WASM-to-WASM transforming frontend
//!
//! Parses a WebAssembly module and re-emits each function with optional
//! address-space translation, global-index remapping, and call-index offsetting.
//!
//! Unlike the native-arch recompilers, this frontend bypasses `yecta::Reactor`
//! entirely: WASM functions are already self-contained, so no CFG reconstruction
//! or tail-call splitting is needed.  Each input function maps 1:1 to one output
//! `wasm_encoder::Function`.
//!
//! ## Usage
//!
//! ```ignore
//! use speet_wasm::{WasmFrontend, MemoryMapConfig, IndexOffsets};
//! use speet_memory::AddressWidth;
//!
//! let mut frontend = WasmFrontend::new(
//!     MemoryMapConfig { addr_width: AddressWidth::W32, memory_index: 0 },
//!     IndexOffsets { func: 10, global: 0, table: 0 },
//!     None,
//! );
//! frontend.translate_module(&mut ctx, &wasm_bytes)?;
//! let unit = frontend.drain_unit(&mut adapter, entry_points);
//! ```

extern crate alloc;

use alloc::{boxed::Box, string::String, vec::Vec};
use speet_link::{BinaryUnit, Recompile, context::ReactorContext, unit::FuncType};
use speet_memory::{AddressWidth, IntWidth, LoadKind, MapperCallback, MemoryEmitter, PageMapLocals, StoreKind};
use wasm_encoder::{Function, Instruction, ValType};
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
/// Implements [`Recompile`] so it can participate in the `speet-link`
/// multi-binary linking flow without going through `yecta::Reactor`.
pub struct WasmFrontend<Context, E> {
    /// Accumulated (function, type) pairs; drained by `drain_unit`.
    compiled: Vec<(Function, FuncType)>,
    /// Memory-access configuration.
    memory_cfg: MemoryMapConfig,
    /// Index offsets for calls, globals, and tables.
    offsets: IndexOffsets,
    /// Optional mapper factory.  Called once per function to create the
    /// `MapperCallback` for that function's scratch locals.
    mapper_factory: Option<
        Box<
            dyn Fn(
                PageMapLocals,
            ) -> Box<dyn MapperCallback<Context, E, Function>>,
        >,
    >,
}

impl<Context, E> WasmFrontend<Context, E> {
    /// Create a new `WasmFrontend`.
    ///
    /// * `memory_cfg`     — address width and target memory index.
    /// * `offsets`        — additive index offsets for calls/globals/tables.
    /// * `mapper_factory` — optional factory producing a per-function mapper.
    pub fn new(
        memory_cfg: MemoryMapConfig,
        offsets: IndexOffsets,
        mapper_factory: Option<
            Box<
                dyn Fn(
                    PageMapLocals,
                ) -> Box<dyn MapperCallback<Context, E, Function>>,
            >,
        >,
    ) -> Self {
        Self {
            compiled: Vec::new(),
            memory_cfg,
            offsets,
            mapper_factory,
        }
    }
}

impl<Context, E> WasmFrontend<Context, E>
where
    E: From<wasmparser::BinaryReaderError>,
{
    // ── Public API ────────────────────────────────────────────────────────

    /// Parse a WASM module's type + code sections and transform every function.
    ///
    /// Translated functions are accumulated in `self.compiled`.  Call
    /// [`drain_unit`](Self::drain_unit_standalone) (or the `Recompile::drain_unit`
    /// method) to collect them into a [`BinaryUnit`].
    pub fn translate_module(&mut self, ctx: &mut Context, bytes: &[u8]) -> Result<(), E> {
        // Collected types for the whole module.
        let mut types: Vec<FuncType> = Vec::new();
        // Per-function type indices (from the Function section).
        let mut fn_type_indices: Vec<u32> = Vec::new();
        // Index into fn_type_indices for the next CodeSection entry.
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
                                // Non-function types: insert a placeholder so
                                // indices remain aligned.
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

                // All other sections (imports, exports, memory, globals, …)
                // are not re-emitted here; the caller manages module assembly.
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

        // Params count toward the original local count for index purposes,
        // but `get_locals_reader` only gives the explicit (non-param) locals.
        let locals_reader = body.get_locals_reader()?;
        for local in locals_reader {
            let (count, wasm_ty) = local?;
            let vt = val_type_from_wasmparser(wasm_ty);
            out_locals.push((count, vt));
            original_local_count += count;
        }

        // The param locals are before the explicit locals in the index space.
        // We need the total param count to compute scratch indices.
        let param_count = func_type.params_val_types().count() as u32;
        let base_scratch_idx = param_count + original_local_count;

        // Always append 1 scratch local for saving the store value.
        out_locals.push((1, ValType::I32));
        let store_scratch = base_scratch_idx; // local index for saving store value

        // If a mapper is configured, append 4 more scratch locals for the
        // PageMapLocals (vaddr + scratch[0..2]).
        let mapper: Option<Box<dyn MapperCallback<Context, E, Function>>> =
            if let Some(factory) = &self.mapper_factory {
                let map_locals = PageMapLocals::consecutive(base_scratch_idx + 1);
                out_locals.push((4, ValType::I32));
                Some(factory(map_locals))
            } else {
                None
            };

        let mut out = Function::new(out_locals);

        // ── Instruction loop ──────────────────────────────────────────────
        let ops_reader = body.get_operators_reader()?;
        // We need to own the mapper mutably; use a local Option<Box<_>>.
        let mut mapper_opt = mapper;

        for op_result in ops_reader {
            let op = op_result?;
            self.translate_op(ctx, &op, &mut out, store_scratch, &mut mapper_opt)?;
        }

        self.compiled.push((out, func_type));
        Ok(())
    }

    fn translate_op(
        &mut self,
        ctx: &mut Context,
        op: &Operator<'_>,
        out: &mut Function,
        store_scratch: u32,
        mapper: &mut Option<Box<dyn MapperCallback<Context, E, Function>>>,
    ) -> Result<(), E> {
        let addr_width = self.memory_cfg.addr_width;
        let memory_index = self.memory_cfg.memory_index;
        let offsets = self.offsets;

        match op {
            // ── Memory loads ──────────────────────────────────────────────
            Operator::I32Load { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I32S)?;
            }
            Operator::I64Load { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I64)?;
            }
            Operator::F32Load { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::F32)?;
            }
            Operator::F64Load { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::F64)?;
            }
            Operator::I32Load8S { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I8S)?;
            }
            Operator::I32Load8U { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I8U)?;
            }
            Operator::I32Load16S { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I16S)?;
            }
            Operator::I32Load16U { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I16U)?;
            }
            Operator::I64Load8S { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I8S)?;
            }
            Operator::I64Load8U { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I8U)?;
            }
            Operator::I64Load16S { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I16S)?;
            }
            Operator::I64Load16U { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I16U)?;
            }
            Operator::I64Load32S { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I32S)?;
            }
            Operator::I64Load32U { memarg } => {
                emit_offset(out, memarg.offset, addr_width);
                emit_load(ctx, out, addr_width, memory_index, mapper, LoadKind::I32U)?;
            }

            // ── Memory stores ─────────────────────────────────────────────
            //
            // Stack at store op: [..., addr, value]  (value on top)
            // Strategy:
            //   1. local.set store_scratch  (save value; stack: [..., addr])
            //   2. emit_offset              (stack: [..., addr + off])
            //   3. emit_store_addr          (wrap + mapper; stack: [..., mapped])
            //   4. local.get store_scratch  (stack: [..., mapped, value])
            //   5. emit_store_insn
            Operator::I32Store { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::I32, IntWidth::I32)?;
            }
            Operator::I64Store { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::I64, IntWidth::I64)?;
            }
            Operator::F32Store { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::F32, IntWidth::I32)?;
            }
            Operator::F64Store { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::F64, IntWidth::I64)?;
            }
            Operator::I32Store8 { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::I8, IntWidth::I32)?;
            }
            Operator::I32Store16 { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::I16, IntWidth::I32)?;
            }
            Operator::I64Store8 { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::I8, IntWidth::I64)?;
            }
            Operator::I64Store16 { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::I16, IntWidth::I64)?;
            }
            Operator::I64Store32 { memarg } => {
                emit_store(ctx, out, memarg.offset, addr_width, memory_index, mapper, store_scratch, StoreKind::I32, IntWidth::I64)?;
            }

            // ── Global accesses ───────────────────────────────────────────
            Operator::GlobalGet { global_index } => {
                let host_idx = (*global_index as i64 + offsets.global) as u32;
                out.instruction(&Instruction::GlobalGet(host_idx));
            }
            Operator::GlobalSet { global_index } => {
                let host_idx = (*global_index as i64 + offsets.global) as u32;
                out.instruction(&Instruction::GlobalSet(host_idx));
            }

            // ── Function calls ────────────────────────────────────────────
            Operator::Call { function_index } => {
                out.instruction(&Instruction::Call(function_index + offsets.func));
            }
            Operator::CallIndirect {
                type_index,
                table_index,
            } => {
                out.instruction(&Instruction::CallIndirect {
                    type_index: *type_index,
                    table_index: table_index + offsets.table,
                });
            }

            // ── Pass-through ──────────────────────────────────────────────
            other => {
                let insn = Instruction::try_from(other.clone())
                    .unwrap_or(Instruction::Unreachable);
                out.instruction(&insn);
            }
        }
        Ok(())
    }
}

// ── Recompile impl ────────────────────────────────────────────────────────────

impl<Context, E> Recompile<Context, E, Function> for WasmFrontend<Context, E>
where
    E: From<wasmparser::BinaryReaderError>,
{
    type BinaryArgs = ();

    fn reset_for_next_binary(
        &mut self,
        _ctx: &mut (dyn ReactorContext<Context, E, FnType = Function> + '_),
        _args: (),
    ) {
        // `compiled` was already drained by `drain_unit`; no other per-binary
        // state needs resetting.
    }

    fn drain_unit(
        &mut self,
        ctx: &mut (dyn ReactorContext<Context, E, FnType = Function> + '_),
        entry_points: Vec<(String, u32)>,
    ) -> BinaryUnit<Function> {
        let base = ctx.base_func_offset();
        let collected: Vec<(Function, FuncType)> = core::mem::take(&mut self.compiled);
        let n = collected.len() as u32;
        let (fns, func_types): (Vec<Function>, Vec<FuncType>) =
            collected.into_iter().unzip();
        ctx.advance_base_func_offset(n);
        BinaryUnit {
            fns,
            base_func_offset: base,
            entry_points,
            func_types,
        }
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Emit the offset addition for a load instruction.
///
/// Stack before: `[..., addr]`
/// Stack after:  `[..., addr + offset]`  (or unchanged if `offset == 0`)
fn emit_offset(out: &mut Function, offset: u64, addr_width: AddressWidth) {
    if offset == 0 {
        return;
    }
    if addr_width.is_64() {
        out.instruction(&Instruction::I64Const(offset as i64));
        out.instruction(&Instruction::I64Add);
    } else {
        out.instruction(&Instruction::I32Const(offset as i32));
        out.instruction(&Instruction::I32Add);
    }
}

/// Emit a complete load sequence using `MemoryEmitter`.
fn emit_load<Context, E>(
    ctx: &mut Context,
    out: &mut Function,
    addr_width: AddressWidth,
    memory_index: u32,
    mapper: &mut Option<Box<dyn MapperCallback<Context, E, Function>>>,
    kind: LoadKind,
) -> Result<(), E> {
    let int_width = if addr_width.is_64() { IntWidth::I64 } else { IntWidth::I32 };
    let mapper_ref: Option<&mut (dyn MapperCallback<Context, E, Function> + '_)> =
        mapper.as_mut().map(|b| b.as_mut() as &mut dyn MapperCallback<Context, E, Function>);
    let mut emitter = MemoryEmitter::new(addr_width, int_width, memory_index, mapper_ref);
    emitter.emit_load(ctx, out, kind)
}

/// Emit a complete store sequence using the two-step MemoryEmitter API.
///
/// Stack before: `[..., addr, value]`
/// Stack after:  `[...]`
#[allow(clippy::too_many_arguments)]
fn emit_store<Context, E>(
    ctx: &mut Context,
    out: &mut Function,
    offset: u64,
    addr_width: AddressWidth,
    memory_index: u32,
    mapper: &mut Option<Box<dyn MapperCallback<Context, E, Function>>>,
    store_scratch: u32,
    kind: StoreKind,
    int_width: IntWidth,
) -> Result<(), E> {
    // Save the value (currently on top of the stack) to a scratch local.
    // After this, stack is: [..., addr]
    out.instruction(&Instruction::LocalSet(store_scratch));

    // Incorporate the memarg offset into the address.
    emit_offset(out, offset, addr_width);

    // Apply address wrap + mapper (addr → mapped_addr).
    {
        let mapper_ref: Option<&mut (dyn MapperCallback<Context, E, Function> + '_)> =
            mapper.as_mut().map(|b| b.as_mut() as &mut dyn MapperCallback<Context, E, Function>);
        let mut emitter = MemoryEmitter::new(addr_width, int_width, memory_index, mapper_ref);
        emitter.emit_store_addr(ctx, out)?;
    }

    // Restore the value from the scratch local.
    // Stack is now: [..., mapped_addr, value]
    out.instruction(&Instruction::LocalGet(store_scratch));

    // Emit the actual store instruction.
    {
        let mapper_ref: Option<&mut (dyn MapperCallback<Context, E, Function> + '_)> = None;
        let emitter = MemoryEmitter::new(addr_width, int_width, memory_index, mapper_ref);
        emitter.emit_store_insn(ctx, out, kind)?;
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
            // Convert ref types; fall back to funcref on unknown heap types.
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
            ValType::Ref(RefType {
                nullable: rt.is_nullable(),
                heap_type,
            })
        }
    }
}

fn func_type_from_wasmparser(ft: &wasmparser::FuncType) -> FuncType {
    let params: Vec<ValType> = ft.params().iter().copied().map(val_type_from_wasmparser).collect();
    let results: Vec<ValType> = ft.results().iter().copied().map(val_type_from_wasmparser).collect();
    FuncType::from_val_types(&params, &results)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use wasmparser::BinaryReaderError;

    /// Error type for tests: wraps `BinaryReaderError`.
    type TestError = BinaryReaderError;

    /// Build a minimal WASM module with one function:
    ///
    /// ```wat
    /// (module
    ///   (memory 1)
    ///   (global (mut i32) (i32.const 0))
    ///   (func (param i32) (result i32)
    ///     local.get 0
    ///     i32.load offset=4
    ///     local.get 0
    ///     i32.const 100
    ///     i32.store offset=8
    ///     global.get 0
    ///     i32.add
    ///     call 0
    ///   )
    /// )
    /// ```
    fn build_test_wasm() -> Vec<u8> {
        use wasm_encoder::*;

        let mut module = Module::new();

        // Type section: one func type (i32) -> (i32)
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I32]);
        module.section(&types);

        // Function section: function 0 has type 0
        let mut funcs = FunctionSection::new();
        funcs.function(0);
        module.section(&funcs);

        // Memory section
        let mut mems = MemorySection::new();
        mems.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mems);

        // Global section: one mutable i32
        let mut globals = GlobalSection::new();
        globals.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: true,
                shared: false,
            },
            &ConstExpr::i32_const(0),
        );
        module.section(&globals);

        // Code section
        let mut codes = CodeSection::new();
        let mut f = Function::new([]);
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Load(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));
        // store: addr=local0, value=100
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Const(100));
        f.instruction(&Instruction::I32Store(MemArg {
            offset: 8,
            align: 2,
            memory_index: 0,
        }));
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

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::new(
            MemoryMapConfig {
                addr_width: AddressWidth::W32,
                memory_index: 0,
            },
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

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::new(
            MemoryMapConfig {
                addr_width: AddressWidth::W32,
                memory_index: 0,
            },
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

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::new(
            MemoryMapConfig {
                addr_width: AddressWidth::W32,
                memory_index: 0,
            },
            IndexOffsets {
                func: 5,
                global: 0,
                table: 0,
            },
            None,
        );

        frontend.translate_module(&mut (), &bytes).unwrap();
        assert_eq!(frontend.compiled.len(), 1);
        // The function compiled successfully with func offset applied.
    }
}
