//! # AArch64 to WebAssembly Recompiler
//!
//! Translates a subset of AArch64 machine code to WebAssembly using the
//! [`yecta`] reactor for control-flow management.
//!
//! ## Supported instructions
//! Integer ALU with immediate (`ADD`, `SUB`, `ADDS`, `SUBS`, `CMP`),
//! integer ALU register (`ADD`, `SUB` shifted), logical (`AND`, `ORR`,
//! `EOR`, `BIC` shifted), move-wide (`MOVZ`, `MOVK`, `MOVN`), control
//! flow (`B`, `BL`, `BR`, `BLR`, `RET`, `CBZ`, `CBNZ`, `B.cond`),
//! and scalar loads/stores (`LDR`, `LDRB`, `LDRH`, `LDRSW`,
//! `STR`, `STRB`, `STRH`).
//!
//! ## Local-variable layout (one WASM function per instruction)
//! - Params  0–30: x0–x30 general-purpose registers (each `i64`).
//! - Param  31:    program counter / PC (`i64`).
//! - Params 32–35: condition flags N, Z, C, V (each `i32`).
//! - Params 36–37: scratch temporaries (`i64`).
//!
//! x31 is XZR in most contexts: reads emit `i64.const 0`, writes are dropped.

#![no_std]
extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::string::String;

use disarm64::decoder_full::Mnemonic;
use speet_link_core::ReactorContext;
use speet_memory::MemoryAccess;
use speet_traps::{
    InstructionInfo, JumpInfo, JumpKind, TrapAction,
    insn::{ArchTag, InsnClass},
    LocalDeclarator,
};
use wasm_encoder::{Instruction, ValType};
use yecta::{FuncIdx, LocalLayout, LocalSlot, Mark};

pub mod direct;

/// AArch64 to WebAssembly recompiler.
pub struct AArch64Recompiler<Context, E> {
    pub(crate) base_pc: u64,
    unsupported_insns: BTreeSet<String>,
    pub(crate) memory_access: Option<alloc::boxed::Box<dyn MemoryAccess<Context, E>>>,
    /// x0–x30 (31 × i64).
    pub(crate) gpr_slot: LocalSlot,
    /// PC (1 × i64).
    pub(crate) pc_slot: LocalSlot,
    /// N, Z, C, V (4 × i32).
    pub(crate) nzcv_slot: LocalSlot,
    /// Two scratch temporaries (2 × i64).
    pub(crate) tmp_slot: LocalSlot,
}

impl<Context, E> AArch64Recompiler<Context, E> {
    pub fn new_with_base_pc(base_pc: u64) -> Self {
        Self {
            base_pc,
            unsupported_insns: BTreeSet::new(),
            memory_access: None,
            gpr_slot: LocalSlot::default(),
            pc_slot: LocalSlot::default(),
            nzcv_slot: LocalSlot::default(),
            tmp_slot: LocalSlot::default(),
        }
    }

    pub fn new() -> Self {
        Self::new_with_base_pc(0)
    }

    pub fn unsupported_insns(&self) -> &BTreeSet<String> {
        &self.unsupported_insns
    }

    pub fn clear_unsupported(&mut self) {
        self.unsupported_insns.clear();
    }

    pub fn set_memory_access(&mut self, ma: alloc::boxed::Box<dyn MemoryAccess<Context, E>>) {
        self.memory_access = Some(ma);
    }

    /// Register trap parameters and compute the total WASM local count.
    /// Must be called once before translation.
    pub fn setup_traps<F>(
        &mut self,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _ctx: &mut Context,
    ) -> u32 {
        self.gpr_slot  = rctx.layout_mut().append(31, ValType::I64); // x0–x30
        self.pc_slot   = rctx.layout_mut().append(1,  ValType::I64); // PC
        self.nzcv_slot = rctx.layout_mut().append(4,  ValType::I32); // N, Z, C, V
        self.tmp_slot  = rctx.layout_mut().append(2,  ValType::I64); // scratch

        let mut unit = ();
        let extra: &mut dyn LocalDeclarator = match self.memory_access.as_deref_mut() {
            Some(m) => m as &mut dyn LocalDeclarator,
            None => &mut unit,
        };
        rctx.declare_trap_params(extra);
        let mark = rctx.layout().mark();
        rctx.set_locals_mark(mark);
        mark.total_locals
    }

    /// Classify an AArch64 mnemonic into [`InsnClass`] flags.
    pub(crate) fn classify_mnemonic(mnemonic: Mnemonic) -> InsnClass {
        match mnemonic {
            Mnemonic::r#ldr
            | Mnemonic::r#ldrb
            | Mnemonic::r#ldrh
            | Mnemonic::r#ldrsb
            | Mnemonic::r#ldrsh
            | Mnemonic::r#ldrsw
            | Mnemonic::r#str
            | Mnemonic::r#strb
            | Mnemonic::r#strh
            | Mnemonic::r#ldp
            | Mnemonic::r#stp => InsnClass::MEMORY,

            Mnemonic::r#b | Mnemonic::r#cbz | Mnemonic::r#cbnz => InsnClass::BRANCH,
            Mnemonic::r#br => InsnClass::BRANCH | InsnClass::INDIRECT,

            Mnemonic::r#bl => InsnClass::CALL,
            Mnemonic::r#blr => InsnClass::CALL | InsnClass::INDIRECT,

            Mnemonic::r#ret | Mnemonic::r#retaa | Mnemonic::r#retab => {
                InsnClass::RETURN | InsnClass::INDIRECT
            }

            Mnemonic::r#svc | Mnemonic::r#hvc | Mnemonic::r#smc
            | Mnemonic::r#mrs | Mnemonic::r#msr => InsnClass::PRIVILEGED,

            Mnemonic::r#fadd | Mnemonic::r#fsub | Mnemonic::r#fmul | Mnemonic::r#fdiv
            | Mnemonic::r#fcmp | Mnemonic::r#fcvt | Mnemonic::r#fmov | Mnemonic::r#fsqrt
            | Mnemonic::r#fmadd | Mnemonic::r#fmsub => InsnClass::FLOAT,

            Mnemonic::r#ldadd | Mnemonic::r#stlr | Mnemonic::r#ldaxr
            | Mnemonic::r#stlxr | Mnemonic::r#cas => InsnClass::ATOMIC | InsnClass::MEMORY,

            _ => InsnClass::OTHER,
        }
    }

    /// Convert a guest PC to a WASM function index relative to base_pc.
    pub(crate) fn pc_to_func_idx(&self, pc: u64) -> Option<FuncIdx> {
        let offset = pc.checked_sub(self.base_pc)?;
        Some(FuncIdx((offset / 4) as u32))
    }

    // ── GPR emit helpers ──────────────────────────────────────────────────────

    /// Push GPR `reg` (0–30) onto the WASM stack.  Register 31 → `i64.const 0`.
    pub(crate) fn emit_gpr_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg >= 31 {
            return rctx.feed(ctx, tail_idx, &Instruction::I64Const(0));
        }
        for instr in rctx.layout().emit_get(self.gpr_slot, reg) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// Pop the stack top into GPR `reg` (0–30).  Register 31 → `drop`.
    pub(crate) fn emit_gpr_set<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg >= 31 {
            return rctx.feed(ctx, tail_idx, &Instruction::Drop);
        }
        for instr in rctx.layout().emit_set(self.gpr_slot, reg) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// WASM local index for NZCV flag `n` (0=N, 1=Z, 2=C, 3=V).
    pub(crate) fn nzcv_local<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        rctx: &RC,
        n: u32,
    ) -> u32 {
        rctx.layout().local(self.nzcv_slot, n)
    }
}
