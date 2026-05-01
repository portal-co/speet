//! # speet-wasm — WASM-to-WASM transforming frontend
//!
//! Parses a WebAssembly module and re-emits each function with optional
//! address-space translation, global-index remapping, and call-index offsetting.
//! Data segments are emitted as passive segments whose physical placement is
//! performed at WASM runtime by a generated data-init function.
//!
//! Unlike the native-arch recompilers, this frontend bypasses `yecta::Reactor`
//! entirely: WASM functions are already self-contained, so no CFG reconstruction
//! or tail-call splitting is needed.  Each input function maps 1:1 to one output `F`.
//!
//! ## Usage
//!
//! ```ignore
//! use speet_wasm::{WasmFrontend, GuestMemoryConfig, IndexOffsets};
//! use speet_memory::{AddressWidth, standard_page_table_mapper, PageTableBase};
//! use speet_linker::Linker;
//! use wasm_encoder::Function;
//!
//! let mapper: Box<dyn Fn() -> Box<dyn MapperCallback<(), BinaryReaderError, Function>>> =
//!     Box::new(|| {
//!         let mut m = standard_page_table_mapper(
//!             PageTableBase::Constant(0x1000_0000),
//!             PageTableBase::Constant(0x2000_0000),
//!             0, false,
//!         );
//!         Box::new(m)
//!     });
//!
//! let mut frontend = WasmFrontend::new(
//!     vec![GuestMemoryConfig { addr_width: AddressWidth::W32, mapper: Some(mapper) }],
//!     0, // host_memory
//!     IndexOffsets { func: 10, global: 0, table: 0 },
//!     Box::new(|locals| Function::new(locals)),
//! );
//! let mut linker: Linker<_, _, Function> = Linker::new();
//! frontend.translate_module(&mut ctx, &mut linker, &wasm_bytes)?;
//!
//! let unit = frontend.drain_unit(&mut linker, entry_points);
//! // unit.data_segments are passive blobs; unit.data_init_fn loads them.
//! ```

extern crate alloc;

use alloc::{boxed::Box, collections::BTreeMap, string::String, vec::Vec};
use speet_link_core::{
    BaseContext, BinaryUnit, DataSegment, Recompile, context::ReactorContext, unit::FuncType,
};
use speet_memory::{
    AddressWidth, CallbackContext, IntWidth, LoadKind, MapperCallback, MemoryEmitter, StoreKind,
};
use speet_traps::cond::{ConditionInfo, ConditionTrap};
use wasm_encoder::{Function, Instruction, MemArg, ValType};
use wasm_layout::{CellIdx, LocalLayout, LocalSlot, Mark};
use wasmparser::{CompositeInnerType, DataKind, FunctionBody, Operator, Parser, Payload};
use wax_core::build::InstructionSink;

// ── GuestMemoryConfig ──────────────────────────────────────────────────────────

/// Per-guest-memory configuration for [`WasmFrontend`].
///
/// Supply one entry per guest memory (indexed by guest memory index).
/// Entries beyond the number of parsed memories fall back to identity
/// (no mapper, 32-bit addresses).
pub struct GuestMemoryConfig<Context, E> {
    /// Width of guest addresses for this memory.
    pub addr_width: AddressWidth,
    /// Mapper that translates virtual → physical addresses.
    /// The mapper declares its own scratch locals via `declare_locals`.
    pub mapper: Option<Box<dyn MapperCallback<Context, E>>>,
}

// ── IndexOffsets ──────────────────────────────────────────────────────────────

/// Additive offsets applied when re-emitting call, global, and table instructions.
#[derive(Clone, Copy, Default, Debug)]
pub struct IndexOffsets {
    /// Added to every `call N` function index in the guest.
    pub func: u32,
    /// Added (as `i64`, then cast to `u32`) to every `global.get/set N` index.
    pub global: i64,
    /// Added to `call_indirect` table indices.
    pub table: u32,
}

// ── MemoryInfo ────────────────────────────────────────────────────────────────

/// Information about one linear memory parsed from the guest WASM memory section.
///
/// Inspect this after [`WasmFrontend::translate_module`] to select the appropriate
/// mapper variant (standard vs. multi-level, 32-bit vs. 64-bit physical addresses).
#[derive(Clone, Debug, Default)]
pub struct MemoryInfo {
    /// Minimum size in 64 KiB pages.
    pub min_pages: u64,
    /// Maximum size in 64 KiB pages, or `None` if unbounded.
    pub max_pages: Option<u64>,
    /// `true` when the memory uses 64-bit addressing (memory64 proposal).
    pub memory64: bool,
    /// `true` when the memory is shared (threading proposal).
    pub shared: bool,
}

// ── DataInitOp ────────────────────────────────────────────────────────────────

/// A recorded instruction for the data-init function emitted in [`WasmFrontend::drain_unit`].
enum DataInitOp {
    /// Push `guest_va`, call the mapper for `memory_idx`, then emit
    /// `memory.init seg_idx host_memory; data.drop seg_idx`.
    InitChunk {
        guest_va: u64,
        seg_idx: u32,
        byte_len: u32,
        memory_idx: usize,
    },
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
    /// Accumulated passive data segments (raw bytes only).
    data_segs: Vec<DataSegment>,
    /// Recorded init operations used to build the data-init function.
    data_init_ops: Vec<DataInitOp>,
    /// Completed data-init function (built at end of `translate_module`).
    data_init_fn_result: Option<(F, FuncType)>,
    /// Memory information parsed from the most recent `translate_module` call.
    memory_infos: Vec<MemoryInfo>,
    /// Per-guest-memory mapper + address-width configurations.
    /// Index `N` applies to guest memory `N`.
    pub per_memory: Vec<GuestMemoryConfig<Context, E>>,
    /// Host linear memory index (always 0 in single-memory lowering).
    pub host_memory: u32,
    /// Index offsets for calls, globals, and tables.
    offsets: IndexOffsets,
    /// Factory that constructs an output function from its local-variable list.
    fn_creator: Box<dyn Fn(Vec<(u32, ValType)>) -> F>,
    /// Cache of (params-layout snapshot, params_mark) per unique guest function type.
    ///
    /// Populated lazily in `translate_fn`; cleared at the start of each
    /// `translate_module` call (traps may change between calls).
    fn_type_param_layouts: BTreeMap<FuncType, (LocalLayout, Mark)>,
    /// Extra parameters injected into every translated function.
    ///
    /// Each type here is appended to both the parameter list **and** the result
    /// list of every translated function.  They are threaded transparently through
    /// all `call`, `call_indirect`, `return_call`, `return_call_indirect`,
    /// `return`, and function-level `end` instructions so that the caller can
    /// pass opaque values through an arbitrary call chain without modifying
    /// individual functions.
    ///
    /// Declared locals in the guest are shifted to make room for the injected
    /// parameters immediately after the original parameters.
    pub injected_params: Vec<ValType>,
    /// Optional condition trap. When installed, it fires after the condition
    /// `i32` is on the WASM stack but before every `if` and `br_if` instruction
    /// is emitted.  The trap may emit instructions to transform the condition.
    cond_trap: Option<Box<dyn ConditionTrap<Context, E>>>,
}

impl<Context, E, F> WasmFrontend<Context, E, F> {
    /// Create a new `WasmFrontend`.
    ///
    /// * `per_memory`   — one [`GuestMemoryConfig`] per guest memory (index 0 = first memory).
    /// * `host_memory`  — host linear memory index (typically 0).
    /// * `offsets`      — additive index offsets for calls/globals/tables.
    /// * `fn_creator`   — closure that builds an `F` from a local-variable list.
    pub fn new(
        per_memory: Vec<GuestMemoryConfig<Context, E>>,
        host_memory: u32,
        offsets: IndexOffsets,
        fn_creator: Box<dyn Fn(Vec<(u32, ValType)>) -> F>,
    ) -> Self {
        Self {
            compiled: Vec::new(),
            data_segs: Vec::new(),
            data_init_ops: Vec::new(),
            data_init_fn_result: None,
            memory_infos: Vec::new(),
            per_memory,
            host_memory,
            offsets,
            fn_creator,
            injected_params: Vec::new(),
            fn_type_param_layouts: BTreeMap::new(),
            cond_trap: None,
        }
    }

    /// Install a condition trap.
    ///
    /// The trap fires after the condition `i32` is on the WASM stack but before
    /// every `if` and `br_if` instruction is emitted.  It may emit instructions
    /// to transform the condition value (e.g. `i32.eqz` to flip the branch, or
    /// a `call $import` for runtime backtracking).
    pub fn set_condition_trap(&mut self, trap: Box<dyn ConditionTrap<Context, E>>) {
        self.cond_trap = Some(trap);
    }

    /// Remove the condition trap.
    pub fn clear_condition_trap(&mut self) {
        self.cond_trap = None;
    }

    /// Take all compiled `(function, type)` pairs, leaving the internal buffer empty.
    ///
    /// Useful in tests and one-shot translation flows where the full
    /// [`drain_unit`](Self::drain_unit) pipeline is not needed.
    pub fn take_compiled(&mut self) -> Vec<(F, FuncType)> {
        core::mem::take(&mut self.compiled)
    }

    /// Return memory information parsed during the last [`translate_module`](Self::translate_module) call.
    pub fn memory_infos(&self) -> &[MemoryInfo] {
        &self.memory_infos
    }

    /// Convenience: infer the appropriate [`AddressWidth`] from the first parsed memory.
    ///
    /// Returns `None` if no memory section was seen.
    pub fn inferred_addr_width(&self) -> Option<AddressWidth> {
        self.memory_infos.first().map(|m| {
            if m.memory64 {
                AddressWidth::W64 { memory64: true }
            } else {
                AddressWidth::W32
            }
        })
    }

    /// Return the number of functions declared in the WASM Function section of
    /// `bytes` without performing full translation.
    ///
    /// `O(1)` — reads the LEB128 count at the start of the Function section.
    /// Returns `0` for modules with no Function section.
    pub fn parse_fn_count(bytes: &[u8]) -> Result<u32, E>
    where
        E: From<wasmparser::BinaryReaderError>,
    {
        for payload in Parser::new(0).parse_all(bytes) {
            if let Payload::FunctionSection(reader) = payload? {
                return Ok(reader.count());
            }
        }
        Ok(0)
    }

    /// Return the address width for guest memory `mem_idx`.
    fn addr_width_for_memory(&self, mem_idx: usize) -> AddressWidth {
        self.per_memory
            .get(mem_idx)
            .map(|c| c.addr_width)
            .unwrap_or(AddressWidth::W32)
    }
}

impl<Context, E> WasmFrontend<Context, E, Function> {
    /// Convenience constructor that uses `wasm_encoder::Function` as the
    /// output type.
    pub fn with_wasm_encoder_fn(
        per_memory: Vec<GuestMemoryConfig<Context, E>>,
        host_memory: u32,
        offsets: IndexOffsets,
    ) -> Self {
        Self::new(
            per_memory,
            host_memory,
            offsets,
            Box::new(|locals| Function::new(locals)),
        )
    }
}

impl<Context, E, F> WasmFrontend<Context, E, F>
where
    F: InstructionSink<Context, E>,
    E: From<wasmparser::BinaryReaderError>,
{
    // ── Helpers ───────────────────────────────────────────────────────────

    /// Return the chunk size for guest memory `mem_idx`, derived from the
    /// mapper's `chunk_size()` method.  Returns `None` when there is no
    /// mapper or the mapper does not specify a chunk size.
    fn chunk_size_for_memory(&self, mem_idx: usize) -> Option<u64> {
        let cfg = self.per_memory.get(mem_idx)?;
        cfg.mapper.as_ref()?.chunk_size()
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Parse a WASM module's type, memory, data, and code sections and transform
    /// every function.
    ///
    /// `base_ctx` provides layout management and trap coordination.  Pass a
    /// [`Linker`](speet_linker::Linker) (which implements [`BaseContext`]) to
    /// wire up installed traps.
    ///
    /// After this call:
    /// - `self.memory_infos()` reflects the parsed memory section.
    /// - Translated functions and data segments are accumulated internally.
    /// - Call `Recompile::drain_unit` to collect everything into a [`BinaryUnit`].
    pub fn translate_module(
        &mut self,
        ctx: &mut Context,
        base_ctx: &mut (dyn BaseContext<Context, E> + '_),
        bytes: &[u8],
    ) -> Result<(), E> {
        let mut types: Vec<FuncType> = Vec::new();
        let mut fn_type_indices: Vec<u32> = Vec::new();
        let mut code_fn_idx: usize = 0;

        // Clear the per-type layout cache: traps may have changed since last call.
        self.fn_type_param_layouts.clear();

        self.memory_infos.clear();
        self.data_segs.clear();
        self.data_init_ops.clear();
        self.data_init_fn_result = None;

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

                Payload::MemorySection(reader) => {
                    for mem_ty in reader {
                        let mt = mem_ty?;
                        self.memory_infos.push(MemoryInfo {
                            min_pages: mt.initial,
                            max_pages: mt.maximum,
                            memory64: mt.memory64,
                            shared: mt.shared,
                        });
                    }
                }

                Payload::DataSection(reader) => {
                    for data_result in reader {
                        let data = data_result?;
                        if let DataKind::Active {
                            memory_index,
                            ref offset_expr,
                        } = data.kind
                        {
                            let mem_idx = memory_index as usize;
                            if let Some(guest_addr) = eval_const_expr(offset_expr) {
                                let raw = data.data;
                                let chunk_size = self.chunk_size_for_memory(mem_idx);
                                match chunk_size {
                                    Some(page) => {
                                        // Split at page boundaries; record init ops.
                                        let mut offset: u64 = 0;
                                        let total = raw.len() as u64;
                                        while offset < total {
                                            let vaddr = guest_addr + offset;
                                            let space_in_page = page - (vaddr % page);
                                            let chunk_len =
                                                ((total - offset).min(space_in_page)) as usize;
                                            let seg_idx = self.data_segs.len() as u32;
                                            self.data_segs.push(DataSegment {
                                                data: raw
                                                    [offset as usize..offset as usize + chunk_len]
                                                    .to_vec(),
                                            });
                                            self.data_init_ops.push(DataInitOp::InitChunk {
                                                guest_va: vaddr,
                                                seg_idx,
                                                byte_len: chunk_len as u32,
                                                memory_idx: mem_idx,
                                            });
                                            offset += chunk_len as u64;
                                        }
                                    }
                                    None => {
                                        // No chunking: single segment, single init op.
                                        let seg_idx = self.data_segs.len() as u32;
                                        let byte_len = raw.len() as u32;
                                        self.data_segs.push(DataSegment { data: raw.to_vec() });
                                        self.data_init_ops.push(DataInitOp::InitChunk {
                                            guest_va: guest_addr,
                                            seg_idx,
                                            byte_len,
                                            memory_idx: mem_idx,
                                        });
                                    }
                                }
                            }
                            // Segments with non-const offset expressions are skipped.
                        }
                        // Passive segments: no address, not lowered here.
                    }
                }

                Payload::CodeSectionEntry(body) => {
                    let type_idx = fn_type_indices[code_fn_idx] as usize;
                    let func_type = types.get(type_idx).cloned().unwrap_or_else(|| FuncType {
                        params: Vec::new(),
                        results: Vec::new(),
                    });
                    self.translate_fn(ctx, base_ctx, func_type, body)?;
                    code_fn_idx += 1;
                }

                _ => {}
            }
        }

        // Build the data-init function after all sections are parsed.
        if !self.data_init_ops.is_empty() {
            self.build_data_init_fn(ctx, base_ctx)?;
        }

        Ok(())
    }

    /// Build the data-init function from accumulated [`DataInitOp`]s.
    ///
    /// Emits `i[32|64].const <va>; [mapper]; i32.const 0; i32.const <len>;
    /// memory.init <seg> <host>; data.drop <seg>` for each chunk, then `End`.
    fn build_data_init_fn(
        &mut self,
        ctx: &mut Context,
        base_ctx: &mut (dyn BaseContext<Context, E> + '_),
    ) -> Result<(), E> {
        // The data-init function has type (inj...) -> (inj...).
        // We cache its params layout snapshot just like regular functions.
        let n_injected = self.injected_params.len() as u32;
        let inj_clone = self.injected_params.clone();
        let init_fn_type = FuncType::from_val_types(&inj_clone, &inj_clone);
        // Collect guest type info before the cache insert may move init_fn_type.
        let init_guest_params: Vec<ValType> = init_fn_type.params_val_types().collect();
        let init_guest_results: Vec<ValType> = init_fn_type.results_val_types().collect();

        let (params_snap, params_mark) = match self.fn_type_param_layouts.get(&init_fn_type) {
            Some(entry) => entry.clone(),
            None => {
                *base_ctx.layout_mut() = LocalLayout::empty();
                if n_injected > 0 {
                    base_ctx.layout_mut().append(n_injected, ValType::I32);
                }
                base_ctx.declare_trap_params();
                let mark = base_ctx.layout().mark();
                base_ctx.set_locals_mark(mark);
                let snap = base_ctx.layout().clone();
                self.fn_type_param_layouts
                    .insert(init_fn_type, (snap.clone(), mark));
                (snap, mark)
            }
        };

        *base_ctx.layout_mut() = params_snap;
        base_ctx.set_locals_mark(params_mark);

        // Pre-allocate a cell for the data-init function.  It has no guest
        // locals (it is a synthetic function), and its type is (inj...) -> (inj...).
        let cell = base_ctx.alloc_cell_for_guest(&init_guest_params, &init_guest_results, &[]);

        // // Create one mapper instance, call declare_locals to allocate its scratch.
        for p in self.per_memory.iter_mut() {
            if let Some(m) = p.mapper.as_deref_mut() {
                m.declare_locals(cell, base_ctx.layout_mut());
            }
        }

        base_ctx.declare_trap_locals_with_cell(cell);

        let out_locals: Vec<(u32, ValType)> = base_ctx.layout().iter_since(&params_mark).collect();
        let mut out = (self.fn_creator)(out_locals);

        let host_memory = self.host_memory;

        // We need to borrow `self.data_init_ops` but also call `self.per_memory`.
        // Collect the ops into a local vec to avoid the borrow conflict.
        let ops: Vec<(u64, u32, u32, usize)> = self
            .data_init_ops
            .iter()
            .map(|op| match op {
                DataInitOp::InitChunk {
                    guest_va,
                    seg_idx,
                    byte_len,
                    memory_idx,
                } => (*guest_va, *seg_idx, *byte_len, *memory_idx),
            })
            .collect();

        for (guest_va, seg_idx, byte_len, memory_idx) in ops {
            let use_i64 = self.addr_width_for_memory(memory_idx).is_64();

            // Push guest VA.
            if use_i64 {
                out.instruction(ctx, &Instruction::I64Const(guest_va as i64))?;
            } else {
                out.instruction(ctx, &Instruction::I32Const(guest_va as i32))?;
            }

            // Call mapper if configured.
            if let Some(m) = self
                .per_memory
                .get_mut(memory_idx)
                .and_then(|cfg| cfg.mapper.as_deref_mut())
            {
                let mut cb = CallbackContext::new(&mut out);
                m.call(ctx, &mut cb)?;
            }

            // memory.init seg_idx host_memory
            out.instruction(
                ctx,
                &Instruction::MemoryInit {
                    data_index: seg_idx,
                    mem: host_memory,
                },
            )?;
            // data.drop seg_idx
            out.instruction(ctx, &Instruction::DataDrop(seg_idx))?;
        }

        // Push injected params back as return values before End.
        for i in 0..n_injected {
            out.instruction(ctx, &Instruction::LocalGet(i))?;
        }
        out.instruction(ctx, &Instruction::End)?;

        // Function type: (inj...) -> (inj...)
        let inj = &self.injected_params.clone();
        let func_type = FuncType::from_val_types(inj, inj);
        self.data_init_fn_result = Some((out, func_type));
        Ok(())
    }

    // ── Per-function translation ───────────────────────────────────────────

    fn translate_fn(
        &mut self,
        ctx: &mut Context,
        base_ctx: &mut (dyn BaseContext<Context, E> + '_),
        func_type: FuncType,
        body: FunctionBody<'_>,
    ) -> Result<(), E> {
        // -- Extend function type with injected params ---------------------
        let orig_param_count = func_type.params_val_types().count() as u32;
        let n_injected = self.injected_params.len() as u32;
        // Injected params live at local indices orig_param_count..orig_param_count+n_injected.
        let injected_base = orig_param_count;

        let func_type = if n_injected > 0 {
            let orig_params: Vec<ValType> = func_type.params_val_types().collect();
            let orig_results: Vec<ValType> = func_type.results_val_types().collect();
            let mut new_params = orig_params;
            new_params.extend_from_slice(&self.injected_params);
            let mut new_results = orig_results;
            new_results.extend_from_slice(&self.injected_params);
            FuncType::from_val_types(&new_params, &new_results)
        } else {
            func_type
        };

        // -- Parse guest locals early (needed for cell pre-allocation) -----
        // get_locals_reader() takes &self and does not consume the operators
        // reader, so this is safe to call before any layout manipulation.
        let locals_reader = body.get_locals_reader()?;
        let mut guest_locals: Vec<(u32, ValType)> = Vec::new();
        for local in locals_reader {
            let (count, wasm_ty) = local?;
            guest_locals.push((count, val_type_from_wasmparser(wasm_ty)));
        }

        // -- Pre-allocate cell keyed on (extended func_type, guest_locals) -
        // This happens before declare_locals so the real CellIdx can be
        // forwarded to every declare_locals call for this function.
        let guest_params: Vec<ValType> = func_type.params_val_types().collect();
        let guest_results: Vec<ValType> = func_type.results_val_types().collect();
        let cell = base_ctx.alloc_cell_for_guest(&guest_params, &guest_results, &guest_locals);

        // -- Params layout snapshot (cached per function type) -------------
        // Params are implicit in WASM (not in Function::new locals list).
        // We cache the (layout snapshot, mark) per unique extended function type
        // so that trap params are only declared once per type.
        let param_count = func_type.params_val_types().count() as u32; // extended count
        let (params_snap, params_mark) = match self.fn_type_param_layouts.get(&func_type) {
            Some(entry) => entry.clone(),
            None => {
                *base_ctx.layout_mut() = LocalLayout::empty();
                if param_count > 0 {
                    base_ctx.layout_mut().append(param_count, ValType::I32);
                }
                base_ctx.declare_trap_params();
                let mark = base_ctx.layout().mark();
                base_ctx.set_locals_mark(mark);
                let snap = base_ctx.layout().clone();
                self.fn_type_param_layouts
                    .insert(func_type.clone(), (snap.clone(), mark));
                (snap, mark)
            }
        };

        *base_ctx.layout_mut() = params_snap;
        base_ctx.set_locals_mark(params_mark);

        // Append guest declared locals to the layout.
        for &(count, ty) in &guest_locals {
            base_ctx.layout_mut().append(count, ty);
        }

        // 4 type-specific scratch locals for store value saving:
        //   slot_i32 -> i32 scratch  (offset +0 from base)
        //   slot_i64 -> i64 scratch  (offset +0 from base)
        //   slot_f32 -> f32 scratch  (offset +0 from base)
        //   slot_f64 -> f64 scratch  (offset +0 from base)
        let slot_i32 = base_ctx.layout_mut().append(1, ValType::I32);
        let slot_i64 = base_ctx.layout_mut().append(1, ValType::I64);
        let slot_f32 = base_ctx.layout_mut().append(1, ValType::F32);
        let slot_f64 = base_ctx.layout_mut().append(1, ValType::F64);
        // base_scratch_idx is the absolute index of slot_i32 (the first scratch).
        // emit_store uses base_scratch_idx + {0,1,2,3} for the four scratch types.
        let base_scratch_idx = base_ctx.layout().base(slot_i32);
        // Verify the four scratch slots are contiguous (they always are since
        // each is exactly 1 local, but make it explicit via debug assert).
        debug_assert_eq!(base_ctx.layout().base(slot_i64), base_scratch_idx + 1);
        debug_assert_eq!(base_ctx.layout().base(slot_f32), base_scratch_idx + 2);
        debug_assert_eq!(base_ctx.layout().base(slot_f64), base_scratch_idx + 3);

        // Use the primary memory's (index 0) mapper and addr_width for this function.
        let addr_width = self.addr_width_for_memory(0);

        // Mapper and loop scratch locals (only when a mapper is configured).
        // loop_slot is appended first; mapper declares its own locals via declare_locals.
        let mut p = self.per_memory.iter_mut();
        let mut loop_slot = None;
        let loop_slot = loop {
            let Some(p) = p.next() else {
                break loop_slot; // unused
            };
            if let Some(m) = p.mapper.as_deref_mut() {
                m.declare_locals(cell, base_ctx.layout_mut());

                if let None = loop_slot {
                    loop_slot = Some(base_ctx.layout_mut().append(3, ValType::I32));
                };
            }
        };
        let loop_base = loop_slot
            .map(|loop_slot| base_ctx.layout().base(loop_slot))
            .unwrap_or(0);
        let (loop_dst_local, loop_src_local, loop_len_local) =
            (loop_base, loop_base + 1, loop_base + 2);

        // Let traps declare their per-function locals, forwarding the
        // pre-allocated cell so traps receive a meaningful CellIdx.
        base_ctx.declare_trap_locals_with_cell(cell);
        // Extra scratch for call_indirect: saves the table index while we push
        // injected params onto the stack.  Only allocated when n_injected > 0.
        let call_indirect_scratch = if n_injected > 0 {
            let slot: LocalSlot = base_ctx.layout_mut().append(1, ValType::I32);
            base_ctx.layout().base(slot)
        } else {
            0 // unused
        };

        let out_locals: Vec<(u32, ValType)> = base_ctx.layout().iter_since(&params_mark).collect();
        let mut out = (self.fn_creator)(out_locals);

        let ops_reader = body.get_operators_reader()?;
        let mut depth: u32 = 0;
        for op_result in ops_reader {
            let op = op_result?;

            // Track block depth so we can identify the function-level `End`.
            // Also intercept `Return` and function-level `End` to push injected
            // params as extra return values.
            match &op {
                Operator::Block { .. }
                | Operator::Loop { .. }
                | Operator::If { .. }
                | Operator::Try { .. }
                | Operator::TryTable { .. } => {
                    depth += 1;
                }
                Operator::End => {
                    if depth == 0 {
                        // Function-level end: push injected locals as extra returns.
                        for i in 0..n_injected {
                            out.instruction(ctx, &Instruction::LocalGet(injected_base + i))?;
                        }
                        out.instruction(ctx, &Instruction::End)?;
                        continue;
                    } else {
                        depth -= 1;
                    }
                }
                Operator::Return => {
                    for i in 0..n_injected {
                        out.instruction(ctx, &Instruction::LocalGet(injected_base + i))?;
                    }
                    out.instruction(ctx, &Instruction::Return)?;
                    continue;
                }
                _ => {}
            }
            self.translate_op(
                ctx,
                &op,
                &mut out,
                base_scratch_idx,
                addr_width,
                loop_dst_local,
                loop_src_local,
                loop_len_local,
                orig_param_count,
                n_injected,
                injected_base,
                call_indirect_scratch,
            )?;
        }

        self.compiled.push((out, func_type));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_op(
        &mut self,
        ctx: &mut Context,
        op: &Operator<'_>,
        out: &mut F,
        base_scratch_idx: u32,
        addr_width: AddressWidth,
        loop_dst_local: u32,
        loop_src_local: u32,
        loop_len_local: u32,
        orig_param_count: u32,
        n_injected: u32,
        injected_base: u32,
        call_indirect_scratch: u32,
    ) -> Result<(), E> {
        let memory_index = self.host_memory;
        let offsets = self.offsets;

        match op {
            // ── Memory loads ──────────────────────────────────────────────
            Operator::I32Load { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I32,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I32S,
                )?;
            }
            Operator::I64Load { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I64,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I64,
                )?;
            }
            Operator::F32Load { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I32,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::F32,
                )?;
            }
            Operator::F64Load { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I64,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::F64,
                )?;
            }
            Operator::I32Load8S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I32,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I8S,
                )?;
            }
            Operator::I32Load8U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I32,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I8U,
                )?;
            }
            Operator::I32Load16S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I32,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I16S,
                )?;
            }
            Operator::I32Load16U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I32,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I16U,
                )?;
            }
            Operator::I64Load8S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I64,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I8S,
                )?;
            }
            Operator::I64Load8U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I64,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I8U,
                )?;
            }
            Operator::I64Load16S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I64,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I16S,
                )?;
            }
            Operator::I64Load16U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I64,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I16U,
                )?;
            }
            Operator::I64Load32S { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I64,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I32S,
                )?;
            }
            Operator::I64Load32U { memarg } => {
                emit_offset(ctx, out, memarg.offset, addr_width)?;
                emit_load(
                    ctx,
                    out,
                    addr_width,
                    IntWidth::I64,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    LoadKind::I32U,
                )?;
            }

            // ── Memory stores ─────────────────────────────────────────────
            Operator::I32Store { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::I32,
                    IntWidth::I32,
                    base_scratch_idx,
                )?;
            }
            Operator::I64Store { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::I64,
                    IntWidth::I64,
                    base_scratch_idx,
                )?;
            }
            Operator::F32Store { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::F32,
                    IntWidth::I32,
                    base_scratch_idx,
                )?;
            }
            Operator::F64Store { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::F64,
                    IntWidth::I64,
                    base_scratch_idx,
                )?;
            }
            Operator::I32Store8 { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::I8,
                    IntWidth::I32,
                    base_scratch_idx,
                )?;
            }
            Operator::I32Store16 { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::I16,
                    IntWidth::I32,
                    base_scratch_idx,
                )?;
            }
            Operator::I64Store8 { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::I8,
                    IntWidth::I64,
                    base_scratch_idx,
                )?;
            }
            Operator::I64Store16 { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::I16,
                    IntWidth::I64,
                    base_scratch_idx,
                )?;
            }
            Operator::I64Store32 { memarg } => {
                emit_store(
                    ctx,
                    out,
                    memarg,
                    addr_width,
                    memory_index,
                    self.per_memory
                        .get_mut(memarg.memory as usize)
                        .and_then(|cfg| cfg.mapper.as_deref_mut()),
                    StoreKind::I32,
                    IntWidth::I64,
                    base_scratch_idx,
                )?;
            }

            // ── Local variable accesses ───────────────────────────────────
            // Guest local indices >= orig_param_count must be shifted by
            // n_injected because the injected params are inserted between the
            // original params and the original declared locals.
            Operator::LocalGet { local_index } => {
                let idx = if *local_index < orig_param_count {
                    *local_index
                } else {
                    local_index + n_injected
                };
                out.instruction(ctx, &Instruction::LocalGet(idx))?;
            }
            Operator::LocalSet { local_index } => {
                let idx = if *local_index < orig_param_count {
                    *local_index
                } else {
                    local_index + n_injected
                };
                out.instruction(ctx, &Instruction::LocalSet(idx))?;
            }
            Operator::LocalTee { local_index } => {
                let idx = if *local_index < orig_param_count {
                    *local_index
                } else {
                    local_index + n_injected
                };
                out.instruction(ctx, &Instruction::LocalTee(idx))?;
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
                // Push injected params as extra call arguments.
                for i in 0..n_injected {
                    out.instruction(ctx, &Instruction::LocalGet(injected_base + i))?;
                }
                out.instruction(ctx, &Instruction::Call(function_index + offsets.func))?;
                // Callee returns injected params; save them back in reverse order.
                for i in (0..n_injected).rev() {
                    out.instruction(ctx, &Instruction::LocalSet(injected_base + i))?;
                }
            }
            Operator::CallIndirect {
                type_index,
                table_index,
            } => {
                if n_injected > 0 {
                    // Stack top is the table index; save it so we can push
                    // injected params between the args and the table index.
                    out.instruction(ctx, &Instruction::LocalSet(call_indirect_scratch))?;
                    for i in 0..n_injected {
                        out.instruction(ctx, &Instruction::LocalGet(injected_base + i))?;
                    }
                    out.instruction(ctx, &Instruction::LocalGet(call_indirect_scratch))?;
                }
                out.instruction(
                    ctx,
                    &Instruction::CallIndirect {
                        type_index: *type_index,
                        table_index: table_index + offsets.table,
                    },
                )?;
                for i in (0..n_injected).rev() {
                    out.instruction(ctx, &Instruction::LocalSet(injected_base + i))?;
                }
            }
            Operator::ReturnCall { function_index } => {
                // Tail call: push injected params, then tail-call.  No save needed.
                for i in 0..n_injected {
                    out.instruction(ctx, &Instruction::LocalGet(injected_base + i))?;
                }
                out.instruction(ctx, &Instruction::ReturnCall(function_index + offsets.func))?;
            }
            Operator::ReturnCallIndirect {
                type_index,
                table_index,
            } => {
                if n_injected > 0 {
                    out.instruction(ctx, &Instruction::LocalSet(call_indirect_scratch))?;
                    for i in 0..n_injected {
                        out.instruction(ctx, &Instruction::LocalGet(injected_base + i))?;
                    }
                    out.instruction(ctx, &Instruction::LocalGet(call_indirect_scratch))?;
                }
                out.instruction(
                    ctx,
                    &Instruction::ReturnCallIndirect {
                        type_index: *type_index,
                        table_index: table_index + offsets.table,
                    },
                )?;
            }

            // ── memory.size → static declared-page count ──────────────────
            Operator::MemorySize { mem } => {
                let pages = self
                    .memory_infos
                    .get(*mem as usize)
                    .map(|m| m.min_pages)
                    .unwrap_or(0);
                out.instruction(ctx, &Instruction::I32Const(pages as i32))?;
            }

            // ── memory.grow → unsupported (-1) ────────────────────────────
            Operator::MemoryGrow { .. } => {
                // Drop the requested page count; return -1 (spec-compliant failure).
                out.instruction(ctx, &Instruction::Drop)?;
                out.instruction(ctx, &Instruction::I32Const(-1))?;
            }

            // ── memory.copy → per-chunk loop (mapper required) ────────────
            Operator::MemoryCopy { src_mem, dst_mem } => {
                if self.per_memory.iter().any(|a| a.mapper.is_some()) {
                    // Stack: [dst_va, src_va, len]
                    let chunk = self.chunk_size_for_memory(0).unwrap_or(0x10000) as i32;
                    out.instruction(ctx, &Instruction::LocalSet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::LocalSet(loop_src_local))?;
                    out.instruction(ctx, &Instruction::LocalSet(loop_dst_local))?;
                    out.instruction(ctx, &Instruction::Block(wasm_encoder::BlockType::Empty))?;
                    out.instruction(ctx, &Instruction::Loop(wasm_encoder::BlockType::Empty))?;
                    // if len == 0 break
                    out.instruction(ctx, &Instruction::LocalGet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::I32Eqz)?;
                    out.instruction(ctx, &Instruction::BrIf(1))?;
                    // physical dst
                    out.instruction(ctx, &Instruction::LocalGet(loop_dst_local))?;
                    if let Some(m) = self.per_memory[*dst_mem as usize].mapper.as_deref_mut() {
                        let mut cb = CallbackContext::new(out);
                        m.call(ctx, &mut cb)?;
                    }
                    // physical src
                    out.instruction(ctx, &Instruction::LocalGet(loop_src_local))?;
                    if let Some(m) = self.per_memory[*src_mem as usize].mapper.as_deref_mut() {
                        let mut cb = CallbackContext::new(out);
                        m.call(ctx, &mut cb)?;
                    }
                    // len = min(chunk, remaining_len)
                    out.instruction(ctx, &Instruction::LocalGet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::LocalGet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::I32LtU)?;
                    out.instruction(ctx, &Instruction::Select)?;
                    out.instruction(
                        ctx,
                        &Instruction::MemoryCopy {
                            src_mem: memory_index,
                            dst_mem: memory_index,
                        },
                    )?;
                    // dst_va += chunk
                    out.instruction(ctx, &Instruction::LocalGet(loop_dst_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::I32Add)?;
                    out.instruction(ctx, &Instruction::LocalSet(loop_dst_local))?;
                    // src_va += chunk
                    out.instruction(ctx, &Instruction::LocalGet(loop_src_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::I32Add)?;
                    out.instruction(ctx, &Instruction::LocalSet(loop_src_local))?;
                    // len -= chunk (saturating: already min'd above)
                    out.instruction(ctx, &Instruction::LocalGet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::I32Sub)?;
                    out.instruction(ctx, &Instruction::LocalSet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::Br(0))?;
                    out.instruction(ctx, &Instruction::End)?; // loop
                    out.instruction(ctx, &Instruction::End)?; // block
                } else {
                    // No mapper: pass through unchanged.
                    out.instruction(
                        ctx,
                        &Instruction::MemoryCopy {
                            src_mem: memory_index,
                            dst_mem: memory_index,
                        },
                    )?;
                }
            }

            // ── memory.fill → per-chunk loop (mapper required) ────────────
            Operator::MemoryFill { mem } => {
                if self.per_memory.iter().any(|a| a.mapper.is_some()) {
                    // Stack: [dst_va, val, len]
                    let chunk = self.chunk_size_for_memory(*mem as usize).unwrap_or(0x10000) as i32;
                    out.instruction(ctx, &Instruction::LocalSet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::LocalSet(loop_src_local))?; // val (i32)
                    out.instruction(ctx, &Instruction::LocalSet(loop_dst_local))?;
                    out.instruction(ctx, &Instruction::Block(wasm_encoder::BlockType::Empty))?;
                    out.instruction(ctx, &Instruction::Loop(wasm_encoder::BlockType::Empty))?;
                    out.instruction(ctx, &Instruction::LocalGet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::I32Eqz)?;
                    out.instruction(ctx, &Instruction::BrIf(1))?;
                    // physical dst
                    out.instruction(ctx, &Instruction::LocalGet(loop_dst_local))?;
                    if let Some(m) = self.per_memory[*mem as usize].mapper.as_deref_mut() {
                        let mut cb = CallbackContext::new(out);
                        m.call(ctx, &mut cb)?;
                    }
                    out.instruction(ctx, &Instruction::LocalGet(loop_src_local))?; // val
                    // len = min(chunk, remaining)
                    out.instruction(ctx, &Instruction::LocalGet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::LocalGet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::I32LtU)?;
                    out.instruction(ctx, &Instruction::Select)?;
                    out.instruction(ctx, &Instruction::MemoryFill(memory_index))?;
                    out.instruction(ctx, &Instruction::LocalGet(loop_dst_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::I32Add)?;
                    out.instruction(ctx, &Instruction::LocalSet(loop_dst_local))?;
                    out.instruction(ctx, &Instruction::LocalGet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::I32Const(chunk))?;
                    out.instruction(ctx, &Instruction::I32Sub)?;
                    out.instruction(ctx, &Instruction::LocalSet(loop_len_local))?;
                    out.instruction(ctx, &Instruction::Br(0))?;
                    out.instruction(ctx, &Instruction::End)?; // loop
                    out.instruction(ctx, &Instruction::End)?; // block
                } else {
                    out.instruction(ctx, &Instruction::MemoryFill(memory_index))?;
                }
            }

            // ── Condition-trap interception ───────────────────────────────
            //
            // For `if` and `br_if` the condition i32 is already on the stack
            // (pushed by the preceding instruction).  If a condition trap is
            // installed we fire it here so it can observe or transform the
            // value before the branching instruction consumes it.
            Operator::If { blockty } => {
                if let Some(ref trap) = self.cond_trap {
                    let info = ConditionInfo { source_pc: 0, target_pc: None };
                    trap.on_condition(&info, ctx, &mut |c, instr| out.instruction(c, instr))?;
                }
                let insn = Instruction::try_from(Operator::If { blockty: *blockty })
                    .unwrap_or(Instruction::Unreachable);
                out.instruction(ctx, &insn)?;
            }
            Operator::BrIf { relative_depth } => {
                if let Some(ref trap) = self.cond_trap {
                    let info = ConditionInfo { source_pc: 0, target_pc: None };
                    trap.on_condition(&info, ctx, &mut |c, instr| out.instruction(c, instr))?;
                }
                out.instruction(ctx, &Instruction::BrIf(*relative_depth))?;
            }

            // ── Pass-through ──────────────────────────────────────────────
            other => {
                let insn = Instruction::try_from(other.clone()).unwrap_or(Instruction::Unreachable);
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

    fn count_fns(&self, bytes: &[u8]) -> u32 {
        for payload in Parser::new(0).parse_all(bytes) {
            if let Ok(Payload::FunctionSection(reader)) = payload {
                return reader.count();
            }
        }
        0
    }

    fn reset_for_next_binary(
        &mut self,
        _ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        _args: (),
    ) {
        // `compiled`, `data_segs`, `data_init_ops`, and `data_init_fn_result`
        // were already drained by `drain_unit`; no extra reset needed.
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
        let data_segments = core::mem::take(&mut self.data_segs);
        let data_init_fn = self.data_init_fn_result.take();
        self.data_init_ops.clear();
        ctx.advance_base_func_offset(n);
        BinaryUnit {
            fns,
            base_func_offset: base,
            entry_points,
            func_types,
            data_segments,
            data_init_fn,
        }
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Evaluate a WASM constant expression that is an `i32.const` or `i64.const`.
///
/// Returns `None` for expressions that cannot be statically resolved (e.g.
/// `global.get`, which is valid in active data segments but requires runtime
/// evaluation).
fn eval_const_expr(expr: &wasmparser::ConstExpr<'_>) -> Option<u64> {
    let mut reader = expr.get_binary_reader();
    match reader.read_u8().ok()? {
        0x41 => Some(reader.read_var_i32().ok()? as u32 as u64),
        0x42 => Some(reader.read_var_i64().ok()? as u64),
        _ => None,
    }
}

/// Emit the offset addition for a memory instruction.
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

fn emit_load<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    out: &mut F,
    addr_width: AddressWidth,
    int_width: IntWidth,
    memory_index: u32,
    mapper: Option<&mut (dyn MapperCallback<Context, E> + '_)>,
    kind: LoadKind,
) -> Result<(), E> {
    let mut emitter = MemoryEmitter::new(addr_width, int_width, memory_index, mapper);
    emitter.emit_load(ctx, out, kind)
}

#[allow(clippy::too_many_arguments)]
fn emit_store<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    out: &mut F,
    memarg: &wasmparser::MemArg,
    addr_width: AddressWidth,
    memory_index: u32,
    mapper: Option<&mut (dyn MapperCallback<Context, E> + '_)>,
    kind: StoreKind,
    int_width: IntWidth,
    base_scratch_idx: u32,
) -> Result<(), E> {
    let scratch = base_scratch_idx
        + match kind {
            StoreKind::F32 => 2,
            StoreKind::F64 => 3,
            _ => match int_width {
                IntWidth::I32 => 0,
                IntWidth::I64 => 1,
            },
        };

    out.instruction(ctx, &Instruction::LocalSet(scratch))?;
    emit_offset(ctx, out, memarg.offset, addr_width)?;
    {
        let mut emitter = MemoryEmitter::new(addr_width, int_width, memory_index, mapper);
        emitter.emit_store_addr(ctx, out)?;
    }
    out.instruction(ctx, &Instruction::LocalGet(scratch))?;

    let ma = MemArg {
        offset: 0,
        align: memarg.align as u32,
        memory_index,
    };
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
            ValType::Ref(RefType {
                nullable: rt.is_nullable(),
                heap_type,
            })
        }
    }
}

fn func_type_from_wasmparser(ft: &wasmparser::FuncType) -> FuncType {
    let params: Vec<ValType> = ft
        .params()
        .iter()
        .copied()
        .map(val_type_from_wasmparser)
        .collect();
    let results: Vec<ValType> = ft
        .results()
        .iter()
        .copied()
        .map(val_type_from_wasmparser)
        .collect();
    FuncType::from_val_types(&params, &results)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use speet_linker::Linker;
    use wasmparser::BinaryReaderError;

    type TestError = BinaryReaderError;

    /// Build a minimal WASM module with one function, a memory section, and a
    /// data segment.
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
            minimum: 4,
            maximum: Some(16),
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mems);

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

        let mut codes = CodeSection::new();
        let mut f = Function::new([]);
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Load(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));
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

        // Data segment: 12 bytes at guest address 0x1000.
        let mut data = DataSection::new();
        data.active(
            0,
            &ConstExpr::i32_const(0x1000),
            b"hello world!".iter().copied(),
        );
        module.section(&data);

        module.finish()
    }

    #[test]
    fn translate_one_function() {
        let bytes = build_test_wasm();

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            alloc::vec![GuestMemoryConfig {
                addr_width: AddressWidth::W32,
                mapper: None
            }],
            0,
            IndexOffsets::default(),
        );

        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();
        frontend
            .translate_module(&mut (), &mut linker, &bytes)
            .unwrap();
        assert_eq!(frontend.compiled.len(), 1, "expected 1 compiled function");
    }

    #[test]
    fn memory_info_parsed() {
        let bytes = build_test_wasm();

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            alloc::vec![GuestMemoryConfig {
                addr_width: AddressWidth::W32,
                mapper: None
            }],
            0,
            IndexOffsets::default(),
        );

        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();
        frontend
            .translate_module(&mut (), &mut linker, &bytes)
            .unwrap();

        let infos = frontend.memory_infos();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].min_pages, 4);
        assert_eq!(infos[0].max_pages, Some(16));
        assert!(!infos[0].memory64);
        assert_eq!(frontend.inferred_addr_width(), Some(AddressWidth::W32));
    }

    #[test]
    fn data_segment_no_mapper() {
        // Without a mapper, active data segment becomes one passive segment
        // with no chunking, and a data_init_fn is generated.
        let bytes = build_test_wasm();

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            alloc::vec![GuestMemoryConfig {
                addr_width: AddressWidth::W32,
                mapper: None
            }],
            0,
            IndexOffsets::default(),
        );

        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();
        frontend
            .translate_module(&mut (), &mut linker, &bytes)
            .unwrap();
        // One passive segment (no chunking without mapper).
        assert_eq!(frontend.data_segs.len(), 1);
        assert_eq!(&frontend.data_segs[0].data, b"hello world!");
        // Init fn generated because there is at least one op.
        assert!(frontend.data_init_fn_result.is_some());
    }

    #[test]
    fn data_segment_chunked_with_mapper() {
        // 3 × 64 KiB at 0x10000 with a ChunkedMapper → 3 passive segments + init fn.
        use speet_memory::ChunkedMapper;
        use wasm_encoder::*;
        use wasm_layout::LocalDeclarator;
        // No-op mapper closure wrapper that satisfies LocalDeclarator.
        struct NoopMapper;
        impl LocalDeclarator for NoopMapper {}
        impl MapperCallback<(), TestError> for NoopMapper {
            fn call(&mut self, _ctx: &mut (), _cb: &mut CallbackContext<(), TestError>) -> Result<(), TestError> { Ok(()) }
        }

        let mut module = Module::new();
        let mut types = TypeSection::new();
        types.ty().function([], []);
        module.section(&types);
        let mut funcs = FunctionSection::new();
        funcs.function(0);
        module.section(&funcs);
        let mut mems = MemorySection::new();
        mems.memory(MemoryType {
            minimum: 8,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mems);
        let mut codes = CodeSection::new();
        let mut f = Function::new([]);
        f.instruction(&Instruction::End);
        codes.function(&f);
        module.section(&codes);

        let big_data = alloc::vec![0xABu8; 3 * 65536];
        let mut data_sec = DataSection::new();
        data_sec.active(0, &ConstExpr::i32_const(0x10000), big_data);
        module.section(&data_sec);

        let bytes = module.finish();

        // Mapper factory that returns a ChunkedMapper with 64 KiB pages.
        let mapper: Box<dyn MapperCallback<(), TestError>> = Box::new(ChunkedMapper {
            page_size: 0x10000,
            inner: NoopMapper,
        });

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            alloc::vec![GuestMemoryConfig {
                addr_width: AddressWidth::W32,
                mapper: Some(mapper),
            }],
            0,
            IndexOffsets::default(),
        );

        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();
        frontend
            .translate_module(&mut (), &mut linker, &bytes)
            .unwrap();

        assert_eq!(
            frontend.data_segs.len(),
            3,
            "should split into 3 page-sized chunks"
        );
        assert_eq!(frontend.data_segs[0].data.len(), 65536);
        assert_eq!(frontend.data_segs[1].data.len(), 65536);
        assert_eq!(frontend.data_segs[2].data.len(), 65536);
        assert!(
            frontend.data_init_fn_result.is_some(),
            "init fn should be generated"
        );
    }

    #[test]
    fn drain_unit_produces_data_init_fn() {
        use speet_linker::Linker;

        let bytes = build_test_wasm();

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            alloc::vec![GuestMemoryConfig {
                addr_width: AddressWidth::W32,
                mapper: None
            }],
            0,
            IndexOffsets::default(),
        );

        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();
        frontend
            .translate_module(&mut (), &mut linker, &bytes)
            .unwrap();

        assert_eq!(linker.base_func_offset(), 0);
        let unit = frontend.drain_unit(&mut linker, alloc::vec![]);
        assert_eq!(unit.base_func_offset, 0);
        assert_eq!(unit.fns.len(), 1);
        assert_eq!(unit.data_segments.len(), 1);
        // Init fn is present since data segment was parsed.
        assert!(unit.data_init_fn.is_some());
        assert_eq!(linker.base_func_offset(), 1, "offset should advance by 1");
    }

    #[test]
    fn call_index_remapped() {
        let bytes = build_test_wasm();

        let mut frontend: WasmFrontend<(), TestError, Function> = WasmFrontend::new(
            alloc::vec![GuestMemoryConfig {
                addr_width: AddressWidth::W32,
                mapper: None
            }],
            0,
            IndexOffsets {
                func: 5,
                global: 0,
                table: 0,
            },
            Box::new(|locals| Function::new(locals)),
        );

        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();
        frontend
            .translate_module(&mut (), &mut linker, &bytes)
            .unwrap();
        assert_eq!(frontend.compiled.len(), 1);
    }

    #[test]
    fn memory_size_returns_constant() {
        // memory.size N should be lowered to i32.const <declared_pages>.
        use wasm_encoder::*;

        let mut module = Module::new();
        let mut types = TypeSection::new();
        types.ty().function([], [ValType::I32]);
        module.section(&types);
        let mut funcs = FunctionSection::new();
        funcs.function(0);
        module.section(&funcs);
        let mut mems = MemorySection::new();
        mems.memory(MemoryType {
            minimum: 7,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mems);
        let mut codes = CodeSection::new();
        let mut f = Function::new([]);
        f.instruction(&Instruction::MemorySize(0));
        f.instruction(&Instruction::End);
        codes.function(&f);
        module.section(&codes);
        let bytes = module.finish();

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            alloc::vec![GuestMemoryConfig {
                addr_width: AddressWidth::W32,
                mapper: None
            }],
            0,
            IndexOffsets::default(),
        );
        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();
        frontend
            .translate_module(&mut (), &mut linker, &bytes)
            .unwrap();
        // Just check it compiled successfully (instruction lowering does not panic).
        assert_eq!(frontend.compiled.len(), 1);
    }

    #[test]
    fn memory_copy_lowered_to_loop_with_mapper() {
        // memory.copy with a mapper should emit a block/loop, not a single instruction.
        use speet_memory::ChunkedMapper;
        use wasm_encoder::*;
        use wasm_layout::LocalDeclarator;
        // No-op mapper closure wrapper that satisfies LocalDeclarator.
        struct NoopMapper;
        impl LocalDeclarator for NoopMapper {}
        impl MapperCallback<(), TestError> for NoopMapper {
            fn call(&mut self, _ctx: &mut (), _cb: &mut CallbackContext<(), TestError>) -> Result<(), TestError> { Ok(()) }
        }

        let mut module = Module::new();
        let mut types = TypeSection::new();
        types.ty().function([], []);
        module.section(&types);
        let mut funcs = FunctionSection::new();
        funcs.function(0);
        module.section(&funcs);
        let mut mems = MemorySection::new();
        mems.memory(MemoryType {
            minimum: 4,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mems);
        let mut codes = CodeSection::new();
        let mut f = Function::new([]);
        // Push dst, src, len then memory.copy.
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::I32Const(0x10000));
        f.instruction(&Instruction::I32Const(0x1000));
        f.instruction(&Instruction::MemoryCopy {
            dst_mem: 0,
            src_mem: 0,
        });
        f.instruction(&Instruction::End);
        codes.function(&f);
        module.section(&codes);
        let bytes = module.finish();

        let mapper: Box<dyn MapperCallback<(), TestError>> = Box::new(ChunkedMapper {
            page_size: 0x10000,
            inner: NoopMapper,
        });

        let mut frontend: WasmFrontend<(), TestError> = WasmFrontend::with_wasm_encoder_fn(
            alloc::vec![GuestMemoryConfig {
                addr_width: AddressWidth::W32,
                mapper: Some(mapper),
            }],
            0,
            IndexOffsets::default(),
        );
        let mut linker: Linker<'_, '_, (), TestError, Function> = Linker::new();
        frontend
            .translate_module(&mut (), &mut linker, &bytes)
            .unwrap();
        // Verify the function was produced (the loop lowering did not error).
        assert_eq!(frontend.compiled.len(), 1);
    }

    #[test]
    fn parse_fn_count_correct() {
        use wasm_encoder::*;

        // Build a module with 3 functions.
        let mut module = Module::new();
        let mut types = TypeSection::new();
        types.ty().function([], []);
        module.section(&types);

        let mut funcs = FunctionSection::new();
        funcs.function(0);
        funcs.function(0);
        funcs.function(0);
        module.section(&funcs);

        let mut code = CodeSection::new();
        for _ in 0..3 {
            let mut f = Function::new(vec![]);
            f.instruction(&Instruction::End);
            code.function(&f);
        }
        module.section(&code);

        let bytes = module.finish();
        let count =
            WasmFrontend::<(), TestError, Function>::parse_fn_count(&bytes).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn parse_fn_count_empty_module() {
        let bytes = wasm_encoder::Module::new().finish();
        let count =
            WasmFrontend::<(), TestError, Function>::parse_fn_count(&bytes).unwrap();
        assert_eq!(count, 0);
    }
}
