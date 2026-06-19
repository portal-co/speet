//! # AArch64 to WebAssembly Recompiler
//!
//! Translates a subset of AArch64 machine code to WebAssembly using the
//! [`yecta`] reactor for control-flow management.
//!
//! ## Supported instructions
//! Integer ALU with immediate (`ADD`, `SUB`, `ADDS`, `SUBS`, `CMP`),
//! integer ALU register (`ADD`, `SUB` shifted/extended), logical shifted
//! (`AND`, `ORR`, `EOR`, `BIC`, `ORN`, `EON`), logical immediate
//! (`AND`, `ORR`, `EOR`), move-wide (`MOVZ`, `MOVK`, `MOVN`),
//! bitfield (`UBFM`/`SBFM`/`BFM` — shifts, sign/zero-extends),
//! multiply (`MADD`, `MSUB`, `SMADDL`, `UMADDL`, `SMULH`, `UMULH`),
//! division (`UDIV`, `SDIV`), register shifts (`LSLV`, `LSRV`, `ASRV`, `RORV`),
//! conditional select (`CSEL`, `CSINC`, `CSINV`, `CSNEG`),
//! PC-relative address (`ADR`, `ADRP`),
//! control flow (`B`, `BL`, `BR`, `BLR`, `RET`, `CBZ`, `CBNZ`, `B.cond`),
//! system (`MRS`/`MSR` NZCV), exception (`BRK`),
//! scalar loads/stores unsigned-offset (`LDR`, `LDRB`, `LDRH`, `LDRSW`,
//! `STR`, `STRB`, `STRH`), pre/post-index loads/stores, register-offset
//! loads/stores, load/store pairs (`LDP`, `STP`, `LDPSW`),
//! floating-point arithmetic (`FADD`, `FSUB`, `FMUL`, `FDIV`, `FNMUL`, `FMIN`,
//! `FMAX`, `FMINNM`, `FMAXNM`, `FMADD`, `FMSUB`, `FNMADD`, `FNMSUB`),
//! FP unary (`FABS`, `FNEG`, `FSQRT`, `FMOV`), FP conversions
//! (`SCVTF`, `UCVTF`, `FCVTZS`, `FCVTZU`), FP compare (`FCMP`), FP select (`FCSEL`).
//!
//! ## Local-variable layout (one WASM function per instruction)
//! - Params  0–30: x0–x30 general-purpose registers (each `i64`).
//! - Param  31:    program counter / PC (`i64`).
//! - Params 32–35: condition flags N, Z, C, V (each `i32`).
//! - Params 36–38: scratch temporaries (`i64`).
//! - Params 39–70: V0–V31 floating-point registers (`f64`).
//!
//! x31 is XZR in most contexts: reads emit `i64.const 0`, writes are dropped.

#![no_std]
extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::string::String;

use disarm64::decoder_full::Mnemonic;
use speet_link_core::{FedContext, ReactorContext};
use speet_memory::MemoryAccess;
use speet_traps::{
    InstructionInfo, JumpInfo, JumpKind, TrapAction,
    insn::{ArchTag, InsnClass},
    LocalDeclarator,
};
use wasm_encoder::{Ieee64, Instruction, ValType};
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
    /// Three scratch temporaries (3 × i64): tmp0=result, tmp1=operand-a, tmp2=addr-scratch.
    pub(crate) tmp_slot: LocalSlot,
    /// V0–V31 floating-point registers (32 × f64).
    pub(crate) fp_slot: LocalSlot,
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
            fp_slot: LocalSlot::default(),
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
        self.tmp_slot  = rctx.layout_mut().append(3,  ValType::I64); // 3 scratch i64
        self.fp_slot   = rctx.layout_mut().append(32, ValType::F64); // V0–V31

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
            | Mnemonic::r#ldpsw
            | Mnemonic::r#stp => InsnClass::MEMORY,

            Mnemonic::r#b | Mnemonic::r#cbz | Mnemonic::r#cbnz => InsnClass::BRANCH,
            Mnemonic::r#br => InsnClass::BRANCH | InsnClass::INDIRECT,

            Mnemonic::r#bl => InsnClass::CALL,
            Mnemonic::r#blr => InsnClass::CALL | InsnClass::INDIRECT,

            Mnemonic::r#ret | Mnemonic::r#retaa | Mnemonic::r#retab => {
                InsnClass::RETURN | InsnClass::INDIRECT
            }

            Mnemonic::r#svc | Mnemonic::r#hvc | Mnemonic::r#smc => InsnClass::PRIVILEGED,

            Mnemonic::r#fadd | Mnemonic::r#fsub | Mnemonic::r#fmul | Mnemonic::r#fdiv
            | Mnemonic::r#fcmp | Mnemonic::r#fcmpe | Mnemonic::r#fcvt | Mnemonic::r#fmov
            | Mnemonic::r#fsqrt | Mnemonic::r#fmadd | Mnemonic::r#fmsub
            | Mnemonic::r#fabs | Mnemonic::r#fneg | Mnemonic::r#fcsel
            | Mnemonic::r#scvtf | Mnemonic::r#ucvtf
            | Mnemonic::r#fcvtzs | Mnemonic::r#fcvtzu => InsnClass::FLOAT,

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

    // ── FP register emit helpers ──────────────────────────────────────────────

    /// Push FP register `reg` (0–31) onto the WASM stack as f64.
    pub(crate) fn emit_fp_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg >= 32 {
            return rctx.feed(ctx, tail_idx, &Instruction::F64Const(Ieee64::from(0.0)));
        }
        for instr in rctx.layout().emit_get(self.fp_slot, reg) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// Pop the stack top (f64) into FP register `reg` (0–31).  reg >= 32 → `drop`.
    pub(crate) fn emit_fp_set<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg >= 32 {
            return rctx.feed(ctx, tail_idx, &Instruction::Drop);
        }
        for instr in rctx.layout().emit_set(self.fp_slot, reg) {
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

    // ── NZCV flag computation ─────────────────────────────────────────────────

    /// Set N/Z from result on stack (consumed); C=0, V=0.  Used after logical ops.
    pub(crate) fn set_nzcv_logical<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        dest: u32,
    ) -> Result<(), E> {
        let tmp0  = rctx.layout().local(self.tmp_slot, 0);
        let n_loc = rctx.layout().local(self.nzcv_slot, 0);
        let z_loc = rctx.layout().local(self.nzcv_slot, 1);
        let c_loc = rctx.layout().local(self.nzcv_slot, 2);
        let v_loc = rctx.layout().local(self.nzcv_slot, 3);

        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = 0, V = 0
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    /// Set N/Z/C/V after an ADD; result on stack.
    /// Caller must have stored operand `a` (first addend) into tmp1 before the add.
    /// `b_imm`: if Some, the immediate second addend (avoids needing tmp2 for V).
    pub(crate) fn set_nzcv_add<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        dest: u32,
        b_imm: Option<i64>,
    ) -> Result<(), E> {
        let tmp0  = rctx.layout().local(self.tmp_slot, 0);
        let tmp1  = rctx.layout().local(self.tmp_slot, 1);
        let tmp2  = rctx.layout().local(self.tmp_slot, 2);
        let n_loc = rctx.layout().local(self.nzcv_slot, 0);
        let z_loc = rctx.layout().local(self.nzcv_slot, 1);
        let c_loc = rctx.layout().local(self.nzcv_slot, 2);
        let v_loc = rctx.layout().local(self.nzcv_slot, 3);

        // result → tmp0 (keep on stack)
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = result_u64 < a_u64  (unsigned overflow / carry out)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtU)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;

        // V = signed overflow: same signs in, different sign out
        //   = ~((a ^ b) & SIGN) & ((result ^ a) & SIGN)  → i32 1/0
        // Get b: either constant or tmp2
        if let Some(b) = b_imm {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(b))?;
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
        }
        // (a ^ b) < 0  →  sign(a) != sign(b)  →  (a^b) has MSB set
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?; // i32: 1 if signs differ
        rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?; // i32: 1 if signs same
        // (result ^ a) < 0  →  sign(result) != sign(a)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    /// Set N/Z/C/V after a SUB (result = a - b); result on stack.
    /// Caller must store `a` in tmp1 and `b` in tmp2 before the sub
    /// (or pass `b_imm` to skip storing b in tmp2).
    pub(crate) fn set_nzcv_sub<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        dest: u32,
        b_imm: Option<i64>,
    ) -> Result<(), E> {
        let tmp0  = rctx.layout().local(self.tmp_slot, 0);
        let tmp1  = rctx.layout().local(self.tmp_slot, 1);
        let tmp2  = rctx.layout().local(self.tmp_slot, 2);
        let n_loc = rctx.layout().local(self.nzcv_slot, 0);
        let z_loc = rctx.layout().local(self.nzcv_slot, 1);
        let c_loc = rctx.layout().local(self.nzcv_slot, 2);
        let v_loc = rctx.layout().local(self.nzcv_slot, 3);

        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = a_u64 >= b_u64  (no borrow = carry out in ARM SUB convention)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
        if let Some(b) = b_imm {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(b))?;
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::I64GeU)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;

        // V = signed overflow: different signs in, different sign out
        //   = ((a ^ b) & SIGN) & ((result ^ a) & SIGN)
        if let Some(b) = b_imm {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(b))?;
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?; // 1 if signs differ
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    /// Legacy alias kept for callers that don't need proper C/V (conservative C=V=0).
    /// Prefer `set_nzcv_add` / `set_nzcv_sub` for flag-setting instructions.
    pub(crate) fn set_nzcv_and_store<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        dest: u32,
    ) -> Result<(), E> {
        let tmp0   = rctx.layout().local(self.tmp_slot, 0);
        let n_loc  = rctx.layout().local(self.nzcv_slot, 0);
        let z_loc  = rctx.layout().local(self.nzcv_slot, 1);
        let c_loc  = rctx.layout().local(self.nzcv_slot, 2);
        let v_loc  = rctx.layout().local(self.nzcv_slot, 3);

        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = 0 (conservative)
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;

        // V = 0 (conservative)
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }
}
