//! Integer ALU operation handlers.

use crate::*;
use super::helpers::*;
use disarm64::decoder_full::{
    ADDSUB_EXT, ADDSUB_IMM, ADDSUB_SHIFT, BITFIELD, CONDSEL, DP_2SRC, DP_3SRC,
    EXCEPTION, IC_SYSTEM, LOG_IMM, LOG_SHIFT, MOVEWIDE, PCRELADDR,
};
use disarm64::decoder_full::Mnemonic;
use speet_wasm_helpers::{MulhTemps, mulh_signed, mulh_unsigned};

impl<'cb, 'ctx, Context, E> AArch64Recompiler<'cb, 'ctx, Context, E> {
    /// High 64 bits of a 64×64→128 multiply into `rd` (SMULH / UMULH).
    fn emit_mulh<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        rn: u32,
        rm: u32,
        rd: u32,
        signed: bool,
    ) -> Result<(), E> {
        // XZR × anything → 0 in the high half.
        if rn >= 31 || rm >= 31 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
            return self.emit_gpr_set(ctx, rctx, tail_idx, rd);
        }
        let (src1, src2, temps) = {
            let layout = rctx.layout();
            (
                layout.local(self.gpr_slot, rn),
                layout.local(self.gpr_slot, rm),
                MulhTemps::new(layout.local(self.tmp_slot, 0)),
            )
        };
        let instrs = if signed {
            mulh_signed(src1, src2, temps)
        } else {
            mulh_unsigned(src1, src2, temps)
        };
        for instr in instrs {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        self.emit_gpr_set(ctx, rctx, tail_idx, rd)
    }


    // ── ADDSUB_IMM ────────────────────────────────────────────────────────────

    pub(super) fn translate_addsub_imm<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &ADDSUB_IMM,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, is_sub, set_flags, src_uses_sp, dest_uses_sp) = match inner {
            ADDSUB_IMM::ADD_Rd_SP_Rn_SP_AIMM(x)  => (x.0, false, false, true,  false),
            ADDSUB_IMM::ADDS_Rd_Rn_SP_AIMM(x)    => (x.0, false, true,  false, false),
            ADDSUB_IMM::SUB_Rd_SP_Rn_SP_AIMM(x)  => (x.0, true,  false, true,  true),
            ADDSUB_IMM::SUBS_Rd_Rn_SP_AIMM(x)    => (x.0, true,  true,  false, false),
            _ => unsup!(),
        };
        let dest = rd(w);
        let src  = rn(w);
        let imm  = addsub_actual_imm(w);

        if set_flags {
            let tmp1 = rctx.layout().local(self.tmp_slot, 1);
            if src_uses_sp {
                self.emit_addr_reg_get(ctx, rctx, tail_idx, src)?;
            } else {
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
            }
            rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm))?;
            if is_sub {
                rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                self.set_nzcv_sub(ctx, rctx, tail_idx, dest, Some(imm))?;
            } else {
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.set_nzcv_add(ctx, rctx, tail_idx, dest, Some(imm))?;
            }
        } else {
            if src_uses_sp {
                self.emit_addr_reg_get(ctx, rctx, tail_idx, src)?;
            } else {
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
            }
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm))?;
            let op = if is_sub { &Instruction::I64Sub } else { &Instruction::I64Add };
            rctx.feed(ctx, tail_idx, op)?;
            if dest_uses_sp {
                self.emit_addr_reg_set(ctx, rctx, tail_idx, dest)?;
            } else {
                self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
            }
        }
        Ok(())
    }

    // ── ADDSUB_SHIFT ──────────────────────────────────────────────────────────

    pub(super) fn translate_addsub_shift<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &ADDSUB_SHIFT,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, is_sub, set_flags) = match inner {
            ADDSUB_SHIFT::ADD_Rd_Rn_Rm_SFT(x)  => (x.0, false, false),
            ADDSUB_SHIFT::ADDS_Rd_Rn_Rm_SFT(x) => (x.0, false, true),
            ADDSUB_SHIFT::SUB_Rd_Rn_Rm_SFT(x)  => (x.0, true,  false),
            ADDSUB_SHIFT::SUBS_Rd_Rn_Rm_SFT(x) => (x.0, true,  true),
            _ => unsup!(),
        };
        let dest  = rd(w);
        let src1  = rn(w);
        let src2  = rm(w);
        let shamt = field(w, 10, 15);
        let shtyp = field(w, 22, 23);

        // Emit shifted Rm, and if set_flags store both operands in tmp1/tmp2.
        if set_flags {
            let tmp1 = rctx.layout().local(self.tmp_slot, 1);
            let tmp2 = rctx.layout().local(self.tmp_slot, 2);
            // Save a = Rn into tmp1 (LocalSet pops, leaving stack empty)
            self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(tmp1))?;
            // Compute shifted Rm and save into tmp2 (LocalSet pops, leaving stack empty)
            self.emit_gpr_get(ctx, rctx, tail_idx, src2)?;
            if shamt != 0 {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(shamt as i64))?;
                let si = match shtyp { 0 => &Instruction::I64Shl, 1 => &Instruction::I64ShrU, _ => &Instruction::I64ShrS };
                rctx.feed(ctx, tail_idx, si)?;
            }
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(tmp2))?;
            // Stack is empty; reload both operands to compute result
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?; // a
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?; // b
            if is_sub {
                rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                self.set_nzcv_sub(ctx, rctx, tail_idx, dest, None)?;
            } else {
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.set_nzcv_add(ctx, rctx, tail_idx, dest, None)?;
            }
        } else {
            self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
            self.emit_gpr_get(ctx, rctx, tail_idx, src2)?;
            if shamt != 0 {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(shamt as i64))?;
                let si = match shtyp { 0 => &Instruction::I64Shl, 1 => &Instruction::I64ShrU, _ => &Instruction::I64ShrS };
                rctx.feed(ctx, tail_idx, si)?;
            }
            let op = if is_sub { &Instruction::I64Sub } else { &Instruction::I64Add };
            rctx.feed(ctx, tail_idx, op)?;
            self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        }
        Ok(())
    }

    // ── LOG_SHIFT ─────────────────────────────────────────────────────────────

    pub(super) fn translate_log_shift<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &LOG_SHIFT,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, op, negate_rm, set_flags) = match inner {
            LOG_SHIFT::AND_Rd_Rn_Rm_SFT(x)  => (x.0, 0u8, false, false),
            LOG_SHIFT::ANDS_Rd_Rn_Rm_SFT(x) => (x.0, 0u8, false, true),
            LOG_SHIFT::BIC_Rd_Rn_Rm_SFT(x)  => (x.0, 0u8, true,  false),
            LOG_SHIFT::BICS_Rd_Rn_Rm_SFT(x) => (x.0, 0u8, true,  true),
            LOG_SHIFT::ORR_Rd_Rn_Rm_SFT(x)  => (x.0, 1u8, false, false),
            LOG_SHIFT::ORN_Rd_Rn_Rm_SFT(x)  => (x.0, 1u8, true,  false),
            LOG_SHIFT::EOR_Rd_Rn_Rm_SFT(x)  => (x.0, 2u8, false, false),
            LOG_SHIFT::EON_Rd_Rn_Rm_SFT(x)  => (x.0, 2u8, true,  false),
            _ => unsup!(),
        };
        let dest  = rd(w);
        let src1  = rn(w);
        let src2  = rm(w);
        let shamt = field(w, 10, 15);
        let shtyp = field(w, 22, 23);
        self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
        self.emit_gpr_get(ctx, rctx, tail_idx, src2)?;
        if shamt != 0 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(shamt as i64))?;
            let si = match shtyp { 0 => &Instruction::I64Shl, 1 => &Instruction::I64ShrU, _ => &Instruction::I64ShrS };
            rctx.feed(ctx, tail_idx, si)?;
        }
        if negate_rm {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        }
        let li = match op { 0 => &Instruction::I64And, 1 => &Instruction::I64Or, _ => &Instruction::I64Xor };
        rctx.feed(ctx, tail_idx, li)?;
        if set_flags { self.set_nzcv_logical(ctx, rctx, tail_idx, dest)? }
        else { self.emit_gpr_set(ctx, rctx, tail_idx, dest)?; }
        Ok(())
    }

    // ── MOVEWIDE ──────────────────────────────────────────────────────────────

    pub(super) fn translate_movewide<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &MOVEWIDE,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, op) = match inner {
            MOVEWIDE::MOVZ_Rd_HALF(x) => (x.0, 0u8),
            MOVEWIDE::MOVN_Rd_HALF(x) => (x.0, 1u8),
            MOVEWIDE::MOVK_Rd_HALF(x) => (x.0, 2u8),
            _ => unsup!(),
        };
        let dest  = rd(w);
        let imm   = imm16(w) as u64;
        let shift = hw(w) * 16;
        let val   = imm << shift;
        match op {
            0 => { // MOVZ
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(val as i64))?;
                self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
            }
            1 => { // MOVN
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(!(val as i64)))?;
                self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
            }
            _ => { // MOVK: keep other bits, insert imm16
                let mask = !(0xFFFFu64 << shift) as i64;
                self.emit_gpr_get(ctx, rctx, tail_idx, dest)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(mask))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(val as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
            }
        }
        Ok(())
    }

    // ── BITFIELD (UBFM, SBFM, BFM) ───────────────────────────────────────────

    pub(super) fn translate_bitfield<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &BITFIELD,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, signed, is_bfm) = match inner {
            BITFIELD::UBFM_Rd_Rn_IMMR_IMMS(x) => (x.0, false, false),
            BITFIELD::SBFM_Rd_Rn_IMMR_IMMS(x) => (x.0, true,  false),
            BITFIELD::BFM_Rd_Rn_IMMR_IMMS(x)  => (x.0, false, true),
            _ => unsup!(),
        };
        let dest = rd(w);
        let src  = rn(w);
        let immr = field(w, 16, 21) as u8; // 6-bit rotate
        let imms = field(w, 10, 15) as u8; // 6-bit width-1

        if is_bfm {
            // BFM Rd, Rn, #immr, #imms — insert field from Rn into Rd
            // imms >= immr: insert [imms-immr:0] of Rn into [imms:immr] of Rd
            if imms >= immr {
                let width = (imms - immr + 1) as u32;
                let mask  = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
                let dest_mask = !(mask << immr as u32);
                let tmp0 = rctx.layout().local(self.tmp_slot, 0);
                // Rd_new = (Rd & ~dest_mask) | ((Rn & mask) << immr)
                self.emit_gpr_get(ctx, rctx, tail_idx, dest)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(dest_mask as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(tmp0))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(mask as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                if immr > 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(immr as i64))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
            } else {
                // Rotation case: less common, emit unsup for now
                unsup!()
            }
            return Ok(());
        }

        // UBFM / SBFM
        if imms >= immr {
            // Extract bits [imms:immr] from Rn, zero/sign-extend
            let width = (imms - immr + 1) as u32;
            self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
            if immr > 0 {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(immr as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
            }
            if signed {
                // Sign-extend from bit (width-1) — shift left then arithmetic right
                let sh = (64 - width) as i64;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(sh))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(sh))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
            } else {
                // Zero-extend: mask to `width` bits
                let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(mask as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
            }
            self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        } else {
            // imms < immr: rotate right by immr, then extract/sign-extend [imms:0]
            // ROR(Rn, immr) = (Rn >> immr) | (Rn << (64-immr))
            let tmp0 = rctx.layout().local(self.tmp_slot, 0);
            self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(immr as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const((64 - immr as u32) as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
            // Now extract [imms:0]: width = imms+1
            let width = (imms + 1) as u32;
            if signed {
                let sh = (64 - width) as i64;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(sh))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(sh))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
            } else {
                let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(mask as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
            }
            self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        }
        Ok(())
    }

    // ── DP_2SRC (UDIV, SDIV, LSLV, LSRV, ASRV, RORV) ────────────────────────

    pub(super) fn translate_dp2src<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &DP_2SRC,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        // op: 0=UDIV 1=SDIV 2=LSLV 3=LSRV 4=ASRV 5=RORV
        let (w, op) = match inner {
            DP_2SRC::UDIV_Rd_Rn_Rm(x) => (x.0, 0u8),
            DP_2SRC::SDIV_Rd_Rn_Rm(x) => (x.0, 1u8),
            DP_2SRC::LSLV_Rd_Rn_Rm(x) => (x.0, 2u8),
            DP_2SRC::LSRV_Rd_Rn_Rm(x) => (x.0, 3u8),
            DP_2SRC::ASRV_Rd_Rn_Rm(x) => (x.0, 4u8),
            DP_2SRC::RORV_Rd_Rn_Rm(x) => (x.0, 5u8),
            _ => unsup!(),
        };
        let dest = rd(w);
        let src1 = rn(w);
        let src2 = rm(w);

        if op == 5 {
            // RORV: Rd = ROR(Rn, Rm & 63)
            // = (Rn >> r) | (Rn << (64-r))  where r = Rm & 63
            let tmp2 = rctx.layout().local(self.tmp_slot, 2);
            // Compute r = Rm & 63, store in tmp2
            self.emit_gpr_get(ctx, rctx, tail_idx, src2)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(tmp2))?;
            // Rn >> r
            self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
            // Rn << (64-r): compute 64-r = 64 - tmp2
            self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(64))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
            self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        } else {
            self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
            self.emit_gpr_get(ctx, rctx, tail_idx, src2)?;
            let insn: &Instruction = match op {
                0 => &Instruction::I64DivU,
                1 => &Instruction::I64DivS,
                2 => &Instruction::I64Shl,
                3 => &Instruction::I64ShrU,
                _ => &Instruction::I64ShrS,  // ASRV
            };
            // WASM shifts are modular by 64 already, matching AArch64 semantics.
            rctx.feed(ctx, tail_idx, insn)?;
            self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        }
        Ok(())
    }

    // ── DP_3SRC (MADD, MSUB, SMADDL, UMADDL, SMULH, UMULH) ──────────────────

    pub(super) fn translate_dp3src<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &DP_3SRC,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let w = match inner {
            DP_3SRC::MADD_Rd_Rn_Rm_Ra(x)   => {
                let w = x.0;
                // Rd = Ra + Rn*Rm  (if Ra=XZR → MUL)
                self.emit_gpr_get(ctx, rctx, tail_idx, ra(w))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rm(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                return Ok(());
            }
            DP_3SRC::MSUB_Rd_Rn_Rm_Ra(x)   => {
                let w = x.0;
                // Rd = Ra - Rn*Rm  (if Ra=XZR → MNEG)
                self.emit_gpr_get(ctx, rctx, tail_idx, ra(w))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rm(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                return Ok(());
            }
            DP_3SRC::SMADDL_Rd_Rn_Rm_Ra(x) => {
                let w = x.0;
                // Rd = Ra + sign_ext32(Wn) * sign_ext32(Wm)
                self.emit_gpr_get(ctx, rctx, tail_idx, ra(w))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Extend32S)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rm(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Extend32S)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                return Ok(());
            }
            DP_3SRC::UMADDL_Rd_Rn_Rm_Ra(x) => {
                let w = x.0;
                // Rd = Ra + zero_ext32(Wn) * zero_ext32(Wm)
                self.emit_gpr_get(ctx, rctx, tail_idx, ra(w))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF_FFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rm(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF_FFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                return Ok(());
            }
            DP_3SRC::SMSUBL_Rd_Rn_Rm_Ra(x) => {
                let w = x.0;
                // Rd = Ra - sign_ext32(Wn) * sign_ext32(Wm)
                self.emit_gpr_get(ctx, rctx, tail_idx, ra(w))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Extend32S)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rm(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Extend32S)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                return Ok(());
            }
            DP_3SRC::UMSUBL_Rd_Rn_Rm_Ra(x) => {
                let w = x.0;
                // Rd = Ra - zero_ext32(Wn) * zero_ext32(Wm)
                self.emit_gpr_get(ctx, rctx, tail_idx, ra(w))?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF_FFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rm(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF_FFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                return Ok(());
            }
            DP_3SRC::SMULH_Rd_Rn_Rm(x) => {
                let w = x.0;
                // Rd = high 64 bits of signed Rn * Rm (128-bit product).
                self.emit_mulh(ctx, rctx, tail_idx, rn(w), rm(w), rd(w), true)?;
                return Ok(());
            }
            DP_3SRC::UMULH_Rd_Rn_Rm(x) => {
                let w = x.0;
                // Rd = high 64 bits of unsigned Rn * Rm (128-bit product).
                self.emit_mulh(ctx, rctx, tail_idx, rn(w), rm(w), rd(w), false)?;
                return Ok(());
            }
            _ => unsup!(),
        };
        let _ = w;
        Ok(())
    }

    // ── ADDSUB_EXT ────────────────────────────────────────────────────────────

    pub(super) fn translate_addsub_ext<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &ADDSUB_EXT,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, is_sub, set_flags) = match inner {
            ADDSUB_EXT::ADD_Rd_SP_Rn_SP_Rm_EXT(x)  => (x.0, false, false),
            ADDSUB_EXT::ADDS_Rd_Rn_SP_Rm_EXT(x)    => (x.0, false, true),
            ADDSUB_EXT::SUB_Rd_SP_Rn_SP_Rm_EXT(x)  => (x.0, true,  false),
            ADDSUB_EXT::SUBS_Rd_Rn_SP_Rm_EXT(x)    => (x.0, true,  true),
            _ => unsup!(),
        };
        let dest   = rd(w);
        let src1   = rn(w);
        let src2   = rm(w);
        let option = ext_option(w); // 0=UXTB 1=UXTH 2=UXTW 3=LSL 4=SXTB 5=SXTH 6=SXTW 7=SXTX
        let shift  = ext_shift(w);  // 0..4

        let emit_extended = |this: &mut Self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize| -> Result<(), E> {
            this.emit_gpr_get(ctx, rctx, tail_idx, src2)?;
            match option {
                0 => { // UXTB
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                }
                1 => { // UXTH
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                }
                2 => { // UXTW
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF_FFFF))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                }
                3 | 7 => {} // LSL / UXTX / SXTX — no extension
                4 => { // SXTB — sign extend from 8
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(56))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(56))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
                }
                5 => { // SXTH — sign extend from 16
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(48))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(48))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
                }
                6 => { // SXTW — sign extend from 32
                    rctx.feed(ctx, tail_idx, &Instruction::I64Extend32S)?;
                }
                _ => {}
            }
            if shift > 0 {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(shift as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
            }
            Ok(())
        };

        if set_flags {
            let tmp1 = rctx.layout().local(self.tmp_slot, 1);
            let tmp2 = rctx.layout().local(self.tmp_slot, 2);
            // LocalSet pops: stack stays empty between saves
            self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(tmp1))?;
            emit_extended(self, ctx, rctx, tail_idx)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(tmp2))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
            if is_sub {
                rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                self.set_nzcv_sub(ctx, rctx, tail_idx, dest, None)?;
            } else {
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.set_nzcv_add(ctx, rctx, tail_idx, dest, None)?;
            }
        } else {
            self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
            emit_extended(self, ctx, rctx, tail_idx)?;
            let op = if is_sub { &Instruction::I64Sub } else { &Instruction::I64Add };
            rctx.feed(ctx, tail_idx, op)?;
            self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        }
        Ok(())
    }

    // ── LOG_IMM (AND/ORR/EOR/ANDS with bitmask immediate) ────────────────────

    pub(super) fn translate_log_imm<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &LOG_IMM,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, op, set_flags) = match inner {
            LOG_IMM::AND_Rd_SP_Rn_LIMM(x)  => (x.0, 0u8, false),
            LOG_IMM::ANDS_Rd_Rn_LIMM(x)    => (x.0, 0u8, true),
            LOG_IMM::ORR_Rd_SP_Rn_LIMM(x)  => (x.0, 1u8, false),
            LOG_IMM::EOR_Rd_SP_Rn_LIMM(x)  => (x.0, 2u8, false),
            _ => unsup!(),
        };
        let dest = rd(w);
        let imm  = decode_bitmask_imm(w);
        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm as i64))?;
        let li = match op { 0 => &Instruction::I64And, 1 => &Instruction::I64Or, _ => &Instruction::I64Xor };
        rctx.feed(ctx, tail_idx, li)?;
        if set_flags { self.set_nzcv_logical(ctx, rctx, tail_idx, dest)? }
        else { self.emit_gpr_set(ctx, rctx, tail_idx, dest)?; }
        Ok(())
    }

    // ── CONDSEL (CSEL, CSINC, CSINV, CSNEG) ──────────────────────────────────

    pub(super) fn translate_condsel<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &CONDSEL,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, op) = match inner {
            CONDSEL::CSEL_Rd_Rn_Rm_COND(x)  => (x.0, 0u8),
            CONDSEL::CSINC_Rd_Rn_Rm_COND(x) => (x.0, 1u8),
            CONDSEL::CSINV_Rd_Rn_Rm_COND(x) => (x.0, 2u8),
            CONDSEL::CSNEG_Rd_Rn_Rm_COND(x) => (x.0, 3u8),
            _ => unsup!(),
        };
        let dest = rd(w);
        let cond = field(w, 12, 15); // 4-bit condition code

        // WASM `select`: [val_true val_false condition] → val_true if cond≠0, val_false if 0.
        // value_if_true = Rn
        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
        // value_if_false = Rm (possibly modified)
        self.emit_gpr_get(ctx, rctx, tail_idx, rm(w))?;
        match op {
            1 => { // CSINC: Rm+1
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            }
            2 => { // CSINV: ~Rm
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
            }
            3 => { // CSNEG: -Rm  (0 - Rm)
                // Stack: Rm. We want 0-Rm.
                // Approach: negate = I64Const(0); Rm; I64Sub — but stack already has Rm on top.
                // Use: Rm * -1 = Rm ^ -1 + 1 (two's complement), or just:
                // tmp = Rm; I64Const(0); LocalGet(tmp); I64Sub
                let tmp2 = rctx.layout().local(self.tmp_slot, 2);
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(tmp2))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
            }
            _ => {} // CSEL: Rm as-is
        }
        // Emit condition (i32: 1=take Rn, 0=take modified-Rm)
        let (n, z, c, v) = (
            self.nzcv_local(rctx, 0),
            self.nzcv_local(rctx, 1),
            self.nzcv_local(rctx, 2),
            self.nzcv_local(rctx, 3),
        );
        // FedContext routes InstructionSink::instruction to rctx.feed(ctx, tail_idx, …)
        // instead of reactor.tail() which would emit to the wrong function.
        emit_cond(ctx, &mut FedContext::new(rctx, tail_idx), cond, n, z, c, v)?;
        rctx.feed(ctx, tail_idx, &Instruction::Select)?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    // ── PCRELADDR (ADR, ADRP) ─────────────────────────────────────────────────

    pub(super) fn translate_pcreladdr<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        pc: u64,
        inner: &PCRELADDR,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, is_adrp) = match inner {
            PCRELADDR::ADR_Rd_ADDR_PCREL21(x)  => (x.0, false),
            PCRELADDR::ADRP_Rd_ADDR_ADRP(x)   => (x.0, true),
            _ => unsup!(),
        };
        let offset = imm21_adr(w);
        let addr = if is_adrp {
            (pc & !0xFFFu64).wrapping_add_signed(offset << 12)
        } else {
            pc.wrapping_add_signed(offset)
        };
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(addr as i64))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
        Ok(())
    }

    // ── EXCEPTION (BRK / SVC) ─────────────────────────────────────────────────

    pub(super) fn translate_exception<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        pc: u64,
        inner: &EXCEPTION,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        match inner {
            EXCEPTION::BRK_EXCEPTION(_) => {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
            }
            EXCEPTION::SVC_EXCEPTION(svc) => {
                // SVC encoding: imm16 in bits [20:5] of the instruction word.
                let info = crate::SvcInfo {
                    pc,
                    imm: imm16(svc.0) as u16,
                };
                if let Some(ref mut callback) = self.svc_callback {
                    let mut fed = FedContext::new(rctx, tail_idx);
                    let mut callback_ctx = CallbackContext::new(&mut fed);
                    callback.call(&info, ctx, &mut callback_ctx);
                } else {
                    // No callback: treat as privileged / unsupported.
                    unsup!();
                }
            }
            _ => unsup!(),
        }
        Ok(())
    }

    // ── IC_SYSTEM (MRS/MSR NZCV only) ────────────────────────────────────────

    pub(super) fn translate_ic_system<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &IC_SYSTEM,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        // NZCV system register encoded as op0:op1:CRn:CRm:op2 = 11:011:0100:0010:000
        // In MRS/MSR encoding bits [20:5] = 0b1_011_0100_0010_000 = 0x5A10
        const NZCV: u32 = 0x5A10;

        match inner {
            IC_SYSTEM::MRS_Rt_SYSREG(x) => {
                let w = x.0;
                if field(w, 5, 20) != NZCV { unsup!() }
                // MRS Xt, NZCV: Xt = (N<<31)|(Z<<30)|(C<<29)|(V<<28)
                let (n, z, c, v) = (
                    self.nzcv_local(rctx, 0),
                    self.nzcv_local(rctx, 1),
                    self.nzcv_local(rctx, 2),
                    self.nzcv_local(rctx, 3),
                );
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(n))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(31))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;

                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(z))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(30))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;

                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(c))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(29))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;

                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(v))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(28))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;

                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            IC_SYSTEM::MSR_SYSREG_Rt(x) => {
                let w = x.0;
                if field(w, 5, 20) != NZCV { unsup!() }
                // MSR NZCV, Xt: unpack bits from Xt into N/Z/C/V
                let (n, z, c, v) = (
                    self.nzcv_local(rctx, 0),
                    self.nzcv_local(rctx, 1),
                    self.nzcv_local(rctx, 2),
                    self.nzcv_local(rctx, 3),
                );
                let src = rn(w); // for MSR the source register is Rt at [4:0]
                // N = (Xt >> 31) & 1
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(31))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n))?;
                // Z = (Xt >> 30) & 1
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(30))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z))?;
                // C = (Xt >> 29) & 1
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(29))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c))?;
                // V = (Xt >> 28) & 1
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(28))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v))?;
            }
            _ => unsup!(),
        }
        Ok(())
    }
}
