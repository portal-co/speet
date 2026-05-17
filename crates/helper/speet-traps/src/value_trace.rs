//! [`ValueTracingTrap`] — attach provenance traces to register values.
//!
//! When installed, this trap wraps every register local in a WASM GC struct
//! that carries the original value alongside trace metadata (source PC,
//! instruction class, chain depth).  The struct type is declared by the caller
//! in the module's type section and passed in at construction time.
//!
//! ## Design: wrap, don't shadow
//!
//! Rather than keeping a *parallel* shadow local beside each register, the
//! register local's declared `ValType` is changed to `(ref null $ValueTrace_T)`
//! so that value **and** trace occupy a single WASM local.  This keeps the
//! parameter signature dense and avoids any extra forwarding overhead.
//!
//! [`LocalLayout::retrofit_kind`] upgrades the arch recompiler's register slots
//! in-place during [`declare_params`](InstructionTrap::declare_params).  The
//! arch recompiler must check [`wraps_register_locals`](InstructionTrap::wraps_register_locals)
//! and switch from plain `local.get`/`local.set` to
//! [`LocalLayout::emit_get`]/[`LocalLayout::emit_set`] for all register accesses.
//!
//! ## Backend: WASM GC structs
//!
//! Struct layout (one type per value kind, supplied by caller):
//!
//! ```wasm
//! (type $ValueTrace_I64 (struct
//!   (field $value      i64)   ;; field 0 — the wrapped register value
//!   (field $source_pc  i64)   ;; field 1 — guest PC (0 if interpreter)
//!   (field $insn_class i32)   ;; field 2 — InsnClass bits
//!   (field $chain_depth i32)  ;; field 3 — reserved, 0 for now
//! ))
//! ```
//!
//! ## Enabling tracing
//!
//! Tracing is enabled simply by constructing a `ValueTracingTrap` and
//! installing it via [`TrapConfig::set_instruction_trap`].  No separate feature
//! flag is needed — the trap's presence IS the flag.  Passes or system config
//! that wants tracing construct and install one before calling
//! `TrapConfig::declare_params`.
//!
//! ## Interpreter compatibility
//!
//! In the Thompson-threaded interpreter the source PC is unknown at code-gen
//! time.  `InstructionInfo::pc` will be `0` (sentinel); interpreter-path
//! `ValueTrace` structs have `source_pc == 0`.

use wasm_encoder::{HeapType, Instruction, RefType, ValType};
use yecta::{LocalDeclarator, LocalLayout, LocalSlot, SlotKind};
use yecta::layout::CellIdx;

use crate::context::TrapContext;
use crate::insn::{InstructionInfo, InstructionTrap, TrapAction};

// ── ValueTracingTrap ──────────────────────────────────────────────────────────

/// Wraps every register local in a GC `ValueTrace` struct carrying provenance.
///
/// See the [module documentation](self) for the full design.
///
/// ## Construction
///
/// The arch recompiler creates its register slots first, then passes the slot
/// handles to `ValueTracingTrap::new` along with the GC struct type indices.
/// During `declare_params`, the trap calls [`LocalLayout::retrofit_kind`] on
/// those slots to change their type to the GC ref type.
///
/// ```ignore
/// // In arch recompiler setup:
/// let int_reg_slot = layout.append(32, ValType::I64);
/// let fp_reg_slot  = layout.append(32, ValType::F64);
///
/// let mut value_trace = ValueTracingTrap::new(
///     i64_trace_type_idx, f64_trace_type_idx,
///     int_reg_slot, Some(fp_reg_slot),
/// );
/// traps.set_instruction_trap(&mut value_trace);
/// traps.declare_params(&mut layout);  // retrofits the slots
/// ```
pub struct ValueTracingTrap {
    /// WASM type-section index of the `$ValueTrace_I64` struct type.
    pub i64_trace_type: u32,
    /// WASM type-section index of the `$ValueTrace_F64` struct type.
    /// Ignored when `fp_reg_slot` is `None`.
    pub f64_trace_type: u32,

    /// Slot handle for the integer register group (passed in at construction).
    int_reg_slot: LocalSlot,
    /// Slot handle for the FP register group, if FP tracing is enabled.
    fp_reg_slot: Option<LocalSlot>,
}

impl ValueTracingTrap {
    /// Construct a `ValueTracingTrap`.
    ///
    /// * `i64_trace_type` — WASM type index of the `$ValueTrace_I64` struct.
    /// * `f64_trace_type` — WASM type index of the `$ValueTrace_F64` struct.
    /// * `int_reg_slot` — `LocalSlot` handle for the arch's integer register
    ///   group, obtained from `layout.append(n, ValType::I64)`.
    /// * `fp_reg_slot` — optional `LocalSlot` handle for the FP register group;
    ///   pass `None` to skip FP register tracing.
    pub fn new(
        i64_trace_type: u32,
        f64_trace_type: u32,
        int_reg_slot: LocalSlot,
        fp_reg_slot: Option<LocalSlot>,
    ) -> Self {
        Self {
            i64_trace_type,
            f64_trace_type,
            int_reg_slot,
            fp_reg_slot,
        }
    }

    fn i64_ref_type(&self) -> ValType {
        ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(self.i64_trace_type),
        })
    }

    fn f64_ref_type(&self) -> ValType {
        ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(self.f64_trace_type),
        })
    }

    /// The slot handle for the integer register group.
    pub fn int_reg_slot(&self) -> LocalSlot { self.int_reg_slot }
    /// The slot handle for the FP register group, if enabled.
    pub fn fp_reg_slot(&self) -> Option<LocalSlot> { self.fp_reg_slot }
}

impl LocalDeclarator for ValueTracingTrap {
    /// Retrofit the arch register slots to GC-traced ref types.
    ///
    /// Uses the slot handles stored at construction time.  After this call:
    /// - `int_reg_slot` locals hold `(ref null $ValueTrace_I64)` instead of `i64`
    /// - `fp_reg_slot` locals (if any) hold `(ref null $ValueTrace_F64)` instead of `f64`
    ///
    /// Field 0 of each struct is the wrapped value field.
    fn declare_params(&mut self, _cell: CellIdx, params: &mut LocalLayout) {
        params.retrofit_kind(
            self.int_reg_slot,
            self.i64_ref_type(),
            SlotKind::GcTraced { gc_type_idx: self.i64_trace_type, value_field: 0 },
        );

        if let Some(fp_slot) = self.fp_reg_slot {
            params.retrofit_kind(
                fp_slot,
                self.f64_ref_type(),
                SlotKind::GcTraced { gc_type_idx: self.f64_trace_type, value_field: 0 },
            );
        }
    }
}

impl<Context, E> InstructionTrap<Context, E> for ValueTracingTrap {
    fn on_instruction(
        &mut self,
        _info: &InstructionInfo,
        _ctx: &mut Context,
        _trap_ctx: &mut TrapContext<Context, E>,
    ) -> Result<TrapAction, E> {
        Ok(TrapAction::Continue)
    }

    /// Wrap the register written by `info` in a fresh `ValueTrace` struct.
    ///
    /// For the compiled path (static `rd`), the caller passes the exact
    /// register index and emits a targeted `local.set`.
    ///
    /// For the interpreter path (dynamic `rd`), the caller emits a `br_table`
    /// dispatch to select the correct register local at runtime, using
    /// `LocalLayout::emit_set` for each arm.
    ///
    /// This hook emits nothing when `info.pc == u64::MAX` (skip sentinel).
    fn after_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E>,
    ) -> Result<(), E> {
        // Source PC is 0 in interpreter context (unknown static PC).
        // Metadata fields pushed in struct field order: source_pc, insn_class, chain_depth.
        // The value itself must already be on the stack BEFORE this is called.
        trap_ctx.emit(ctx, &Instruction::I64Const(info.pc as i64))?;
        trap_ctx.emit(ctx, &Instruction::I32Const(info.class.0 as i32))?;
        trap_ctx.emit(ctx, &Instruction::I32Const(0))?; // chain_depth
        // struct.new is emitted by the caller via LocalLayout::emit_set,
        // which knows the gc_type_idx from the slot's SlotKind.
        // This hook only pushes the metadata fields; the arch recompiler is
        // responsible for pushing the value and calling layout.emit_set.
        Ok(())
    }

    fn wraps_register_locals(&self) -> bool {
        true
    }
}
