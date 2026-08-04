//! # i686 (IA-32) to WebAssembly Recompiler
//!
//! Thin frontend translating a **smoke-level** subset of 32-bit x86 machine
//! code to WebAssembly via the [`yecta`] reactor. Deliberately separate from
//! [`speet_x86_64`] — no REX, 8 GPRs, 32-bit addresses.
//!
//! ## Scope (Phase 4)
//! - Decode via `iced-x86` at bitness 32
//! - Integer: MOV/ADD/SUB, PUSH/POP
//! - Control: JMP/Jcc/CALL/RET
//! - Load/store: simple `[reg+disp]` forms
//! - Speculative `CallEscape::Flag` on CALL/RET
//!
//! ## Local layout
//! - Params 0–7: EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI (`i64`)
//! - Param 8: EIP (`i64`)
//! - Params 9–12: ZF, SF, CF, OF (`i32`)
//! - Param 13: expected return address (`i64`)
//! - Params 14–15: scratch temps (`i64`)

#![no_std]
extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::string::String;

use speet_link_core::ReactorContext;
use speet_memory::MemoryAccess;
use speet_traps::{
    insn::{ArchTag, InsnClass},
    LocalDeclarator,
};
use wasm_encoder::ValType;
use yecta::{EscapeTag, FuncIdx, LocalLayout, LocalSlot, SlotAssigner};

pub mod cfg;
pub mod direct;

pub use speet_memory::CallbackContext;

/// i686 to WebAssembly recompiler (thin). Named `X86_32Recompiler` to avoid
/// colliding with `speet_x86_64::X86Recompiler`.
pub struct X86_32Recompiler<Context, E> {
    pub(crate) base_eip: u64,
    pub(crate) text_len: u32,
    unsupported_insns: BTreeSet<String>,
    pub(crate) memory_access: Option<alloc::boxed::Box<dyn MemoryAccess<Context, E>>>,
    pub(crate) gpr_slot: LocalSlot,
    pub(crate) eip_slot: LocalSlot,
    pub(crate) flags_slot: LocalSlot,
    pub(crate) expected_ra_slot: LocalSlot,
    pub(crate) tmp_slot: LocalSlot,
    stub_for_pc_import_idx: Option<u32>,
    enable_speculative_calls: bool,
    slot_assigner: Option<alloc::boxed::Box<dyn SlotAssigner + Send + Sync>>,
}

impl<Context, E> X86_32Recompiler<Context, E> {
    pub const BASE_PARAMS: u32 = 16;
    pub const SP_PARAM_INDEX: u32 = 4;

    pub fn new_with_base_eip(base_eip: u64) -> Self {
        Self {
            base_eip,
            text_len: 0,
            unsupported_insns: BTreeSet::new(),
            memory_access: None,
            gpr_slot: LocalSlot::default(),
            eip_slot: LocalSlot::default(),
            flags_slot: LocalSlot::default(),
            expected_ra_slot: LocalSlot::default(),
            tmp_slot: LocalSlot::default(),
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

    pub fn setup_traps<F>(
        &mut self,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _ctx: &mut Context,
    ) -> u32 {
        *rctx.layout_mut() = LocalLayout::empty();
        self.gpr_slot = rctx.layout_mut().append(8, ValType::I64);
        self.eip_slot = rctx.layout_mut().append(1, ValType::I64);
        self.flags_slot = rctx.layout_mut().append(4, ValType::I32);
        self.expected_ra_slot = rctx.layout_mut().append(1, ValType::I64);
        self.tmp_slot = rctx.layout_mut().append(2, ValType::I64);

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

    pub(crate) fn eip_to_func_idx(&self, eip: u64) -> Option<FuncIdx> {
        if let Some(gate) = &self.slot_assigner {
            gate.slot_for_pc(eip).map(FuncIdx)
        } else {
            let offset = eip.checked_sub(self.base_eip)?;
            if offset >= self.text_len as u64 && self.text_len != 0 {
                return None;
            }
            Some(FuncIdx(offset as u32))
        }
    }

    pub(crate) fn gpr_index(reg: iced_x86::Register) -> Option<u32> {
        use iced_x86::Register::*;
        match reg {
            EAX | AX | AL | AH => Some(0),
            ECX | CX | CL | CH => Some(1),
            EDX | DX | DL | DH => Some(2),
            EBX | BX | BL | BH => Some(3),
            ESP | SP => Some(4),
            EBP | BP => Some(5),
            ESI | SI => Some(6),
            EDI | DI => Some(7),
            _ => Option::None,
        }
    }

    pub(crate) fn emit_gpr_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        idx: u32,
    ) -> Result<(), E> {
        for instr in rctx.layout().emit_get(self.gpr_slot, idx) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    pub(crate) fn emit_gpr_set<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        idx: u32,
    ) -> Result<(), E> {
        for instr in rctx.layout().emit_set(self.gpr_slot, idx) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    pub(crate) fn note_unsupported(&mut self, name: &str) {
        self.unsupported_insns.insert(String::from(name));
    }

    pub(crate) fn classify_mnemonic(mnemonic: iced_x86::Mnemonic) -> InsnClass {
        use iced_x86::Mnemonic::*;
        match mnemonic {
            Mov | Movzx | Movsx | Lea | Push | Pop | Add | Sub | And | Or | Xor | Cmp | Test
            | Inc | Dec | Neg | Not | Imul | Mul | Idiv | Div | Shl | Shr | Sar | Rol | Ror => {
                InsnClass::OTHER
            }
            Jmp => InsnClass::BRANCH,
            Je | Jne | Jb | Jae | Jbe | Ja | Jl | Jge | Jle | Jg | Js | Jns | Jo | Jno | Jp
            | Jnp => InsnClass::BRANCH,
            Call => InsnClass::CALL,
            Ret => InsnClass::RETURN,
            _ => InsnClass::OTHER,
        }
    }
}

/// Calling convention for a compile-time PLT redirect of `symbol` on i686.
pub fn plt_calling_convention(symbol: &str) -> speet_plugin_api::external_target::CallingConvention {
    use binary_io::BinArch;
    use speet_host_api::ImportManifest;
    speet_abi_stubs::plt_calling_convention(&ImportManifest::integrated_native(), BinArch::X86, symbol)
}
