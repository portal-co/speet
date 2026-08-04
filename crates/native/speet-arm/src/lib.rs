//! # AArch32 (ARM/Thumb-2) to WebAssembly Recompiler
//!
//! Thin frontend translating a **smoke-level** subset of AArch32 machine code
//! to WebAssembly via the [`yecta`] reactor.
//!
//! ## Scope (Phase 4)
//! - **A32 primary**: data-processing (MOV/ADD/SUB/AND/ORR/EOR), LDR/STR imm,
//!   B/BL, BX, and BX lr as return.
//! - **Thumb-2**: stub path — halfwords are reported unsupported (A32 slots
//!   remain 4-byte). Minimal Thumb decode can land later without API churn.
//! - Register-file ABI `(regs) -> (regs)` with 16 GPRs (r0–r15) as `i64` slots
//!   (32-bit values zero-/sign-extended into the wide WASM locals).
//! - Speculative `CallEscape::Flag` (trailing `i32`) on BL / BX lr.
//!
//! ## Local layout
//! - Params 0–15: r0–r15 (`i64`; r13=SP, r14=LR, r15=PC architecturally)
//! - Param 16: expected return address for speculative BL / BX lr (`i64`)
//! - Param 17: CPSR stub (`i32`, currently unused)

#![no_std]
extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::string::String;

use speet_link_core::ReactorContext;
use speet_memory::MemoryAccess;
use speet_traps::{
    InstructionInfo, JumpInfo, JumpKind, TrapAction,
    insn::{ArchTag, InsnClass},
    LocalDeclarator,
};
use wasm_encoder::{Instruction, MemArg, ValType};
use yecta::{EscapeTag, FuncIdx, LocalLayout, LocalSlot, SlotAssigner};

pub mod cfg;
pub mod direct;

pub use speet_memory::CallbackContext;

/// AArch32 to WebAssembly recompiler (thin).
pub struct ArmRecompiler<Context, E> {
    pub(crate) base_pc: u64,
    unsupported_insns: BTreeSet<String>,
    pub(crate) memory_access: Option<alloc::boxed::Box<dyn MemoryAccess<Context, E>>>,
    pub(crate) gpr_slot: LocalSlot,
    pub(crate) expected_ra_slot: LocalSlot,
    pub(crate) cpsr_slot: LocalSlot,
    stub_for_pc_import_idx: Option<u32>,
    enable_speculative_calls: bool,
    slot_assigner: Option<alloc::boxed::Box<dyn SlotAssigner + Send + Sync>>,
}

impl<Context, E> ArmRecompiler<Context, E> {
    /// 16 GPRs + expected_ra + cpsr stub.
    pub const BASE_PARAMS: u32 = 18;

    /// WASM param index of SP (r13).
    pub const SP_PARAM_INDEX: u32 = 13;

    /// WASM param index of LR (r14).
    pub const LR_PARAM_INDEX: u32 = 14;

    pub fn new_with_base_pc(base_pc: u64) -> Self {
        Self {
            base_pc,
            unsupported_insns: BTreeSet::new(),
            memory_access: None,
            gpr_slot: LocalSlot::default(),
            expected_ra_slot: LocalSlot::default(),
            cpsr_slot: LocalSlot::default(),
            stub_for_pc_import_idx: None,
            enable_speculative_calls: false,
            slot_assigner: None,
        }
    }

    pub fn set_speculative_calls(&mut self, enable: bool) {
        self.enable_speculative_calls = enable;
    }

    pub fn is_speculative_calls_enabled(&self) -> bool {
        self.enable_speculative_calls
    }

    pub fn set_escape<F>(
        &mut self,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        escape: yecta::CallEscape,
    ) {
        rctx.set_escape(escape);
    }

    pub fn set_escape_tag<F>(
        &mut self,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tag: Option<EscapeTag>,
    ) {
        rctx.set_escape_tag(tag);
    }

    pub fn get_escape_tag<F>(
        &self,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
    ) -> Option<EscapeTag> {
        rctx.escape_tag()
    }

    pub fn set_slot_assigner(&mut self, gate: impl SlotAssigner + Send + Sync + 'static) {
        self.slot_assigner = Some(alloc::boxed::Box::new(gate));
    }

    pub fn count_fns(&self) -> u32 {
        self.slot_assigner
            .as_ref()
            .expect("set_slot_assigner must be called before count_fns")
            .total_slots()
    }

    pub fn set_memory_access(&mut self, ma: alloc::boxed::Box<dyn MemoryAccess<Context, E>>) {
        self.memory_access = Some(ma);
    }

    pub fn bind_memory_layout<F>(&mut self, rctx: &dyn ReactorContext<Context, E, FnType = F>) {
        if let (Some(ma), Some(params)) = (
            self.memory_access.as_deref_mut(),
            rctx.runtime_layout_params(),
        ) {
            ma.bind_layout_slots(rctx.layout(), &params.slots);
        }
    }

    pub fn set_stub_for_pc_import_idx(&mut self, idx: u32) {
        self.stub_for_pc_import_idx = Some(idx);
    }

    pub fn unsupported_insns(&self) -> &BTreeSet<String> {
        &self.unsupported_insns
    }

    pub fn clear_unsupported(&mut self) {
        self.unsupported_insns.clear();
    }

    /// Register trap / layout parameters. Must be called once before translation.
    pub fn setup_traps<F>(
        &mut self,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _ctx: &mut Context,
    ) -> u32 {
        *rctx.layout_mut() = LocalLayout::empty();
        self.gpr_slot = rctx.layout_mut().append(16, ValType::I64);
        self.expected_ra_slot = rctx.layout_mut().append(1, ValType::I64);
        self.cpsr_slot = rctx.layout_mut().append(1, ValType::I32);

        let mut unit = ();
        let extra: &mut dyn LocalDeclarator = match self.memory_access.as_deref_mut() {
            Some(m) => m as &mut dyn LocalDeclarator,
            None => &mut unit,
        };
        rctx.declare_trap_params(extra);
        let mark = rctx.layout().mark();
        rctx.set_locals_mark(mark);
        rctx.declare_trap_locals(extra);
        mark.total_locals
    }

    pub(crate) fn pc_to_func_idx(&self, pc: u64) -> Option<FuncIdx> {
        if let Some(gate) = &self.slot_assigner {
            gate.slot_for_pc(pc).map(FuncIdx)
        } else {
            let offset = pc.checked_sub(self.base_pc)?;
            Some(FuncIdx((offset / 4) as u32))
        }
    }

    pub(crate) fn emit_gpr_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        for instr in rctx.layout().emit_get(self.gpr_slot, reg & 15) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    pub(crate) fn emit_gpr_set<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        for instr in rctx.layout().emit_set(self.gpr_slot, reg & 15) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    pub(crate) fn emit_expected_ra_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
    ) -> Result<(), E> {
        for instr in rctx.layout().emit_get(self.expected_ra_slot, 0) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    pub(crate) fn note_unsupported(&mut self, name: &str) {
        self.unsupported_insns.insert(String::from(name));
    }
}

/// Expand an A32 data-processing 12-bit modified immediate.
pub fn expand_imm12(imm12: u32) -> u32 {
    let imm8 = imm12 & 0xff;
    let rot = (imm12 >> 8) & 0xf;
    imm8.rotate_right(rot * 2)
}

/// Calling convention for a compile-time PLT redirect of `symbol` on AArch32.
pub fn plt_calling_convention(symbol: &str) -> speet_plugin_api::external_target::CallingConvention {
    use binary_io::BinArch;
    use speet_host_api::ImportManifest;
    speet_abi_stubs::plt_calling_convention(&ImportManifest::integrated_native(), BinArch::Arm, symbol)
}
