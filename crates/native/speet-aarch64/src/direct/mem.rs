//! Load/store operation handlers.

use crate::*;
use super::helpers::*;
use disarm64::decoder_full::{
    LDST_IMM9, LDST_POS, LDST_REGOFF, LDST_UNSCALED, LDSTPAIR_INDEXED, LDSTPAIR_OFF,
};
use disarm64::decoder_full::Mnemonic;

impl<Context, E> AArch64Recompiler<Context, E> {

    // ── LDST_POS (unsigned immediate offset) ──────────────────────────────────

    pub(super) fn translate_ldst_pos<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &LDST_POS,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        match inner {
            LDST_POS::LDR_Rt_ADDR_UIMM12(x) => {
                let w = x.0;
                let is64 = is_64bit_ldst(w);
                let scale = if is64 { 8i64 } else { 4i64 };
                self.emit_addr_reg_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * scale))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.emit_ldst_access(ctx, rctx, tail_idx, true, if is64 { 8 } else { 4 }, rd(w))?;
            }
            LDST_POS::LDRB_Rt_ADDR_UIMM12(x) => {
                let w = x.0;
                self.emit_addr_reg_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Load8U(memarg(0)))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            LDST_POS::LDRH_Rt_ADDR_UIMM12(x) => {
                let w = x.0;
                self.emit_addr_reg_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * 2))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Load16U(memarg(1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            LDST_POS::LDRSW_Rt_ADDR_UIMM12(x) => {
                let w = x.0;
                self.emit_addr_reg_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * 4))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Load32S(memarg(2)))?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            LDST_POS::STR_Rt_ADDR_UIMM12(x) => {
                let w = x.0;
                let is64 = is_64bit_ldst(w);
                let scale = if is64 { 8i64 } else { 4i64 };
                self.emit_addr_reg_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * scale))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.emit_ldst_access(ctx, rctx, tail_idx, false, if is64 { 8 } else { 4 }, rd(w))?;
            }
            LDST_POS::STRB_Rt_ADDR_UIMM12(x) => {
                let w = x.0;
                self.emit_addr_reg_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rd(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Store8(memarg(0)))?;
            }
            LDST_POS::STRH_Rt_ADDR_UIMM12(x) => {
                let w = x.0;
                self.emit_addr_reg_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * 2))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rd(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Store16(memarg(1)))?;
            }
            _ => unsup!(),
        }
        Ok(())
    }

    // ── LDST_IMM9 (pre / post-index) ──────────────────────────────────────────
    //
    // Bit [11] of the instruction word: 1 = pre-index, 0 = post-index.
    // imm9 (bits [20:12]) is the signed offset.

    pub(super) fn translate_ldst_imm9<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &LDST_IMM9,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        // Each variant carries the raw word; we dispatch on load/store and
        // width. The bare `Rt` (no B/H/S-suffix) forms cover both 32- and
        // 64-bit GPRs in one disarm64 variant — see `is_64bit_ldst`'s doc.
        let (w, is_load, load_width) = match inner {
            LDST_IMM9::LDR_Rt_ADDR_SIMM9(x)   => (x.0, true,  if is_64bit_ldst(x.0) { 8u8 } else { 4u8 }),
            LDST_IMM9::LDRB_Rt_ADDR_SIMM9(x)  => (x.0, true,  1u8),
            LDST_IMM9::LDRH_Rt_ADDR_SIMM9(x)  => (x.0, true,  2u8),
            LDST_IMM9::LDRSB_Rt_ADDR_SIMM9(x) => (x.0, true,  0xB1), // 1-byte signed
            LDST_IMM9::LDRSH_Rt_ADDR_SIMM9(x) => (x.0, true,  0xB2), // 2-byte signed
            LDST_IMM9::LDRSW_Rt_ADDR_SIMM9(x) => (x.0, true,  0xB4), // 4-byte signed
            LDST_IMM9::STR_Rt_ADDR_SIMM9(x)   => (x.0, false, if is_64bit_ldst(x.0) { 8u8 } else { 4u8 }),
            LDST_IMM9::STRB_Rt_ADDR_SIMM9(x)  => (x.0, false, 1u8),
            LDST_IMM9::STRH_Rt_ADDR_SIMM9(x)  => (x.0, false, 2u8),
            _ => unsup!(),
        };

        let src_reg = rn(w);
        let data_reg = rd(w);
        let off = imm9_signed(w);
        let pre_index = field(w, 11, 11) != 0; // bit 11: 1=pre, 0=post
        let addr_tmp = rctx.layout().local(self.tmp_slot, 2);

        if pre_index {
            // pre-index: addr = Rn + imm9; Rn ← addr; then access [addr]
            self.emit_addr_reg_get(ctx, rctx, tail_idx, src_reg)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(off))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalTee(addr_tmp))?;
            // Write updated base back to Rn
            // (addr_tmp is on stack after tee, consume it)
            rctx.feed(ctx, tail_idx, &Instruction::Drop)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_tmp))?;
            self.emit_addr_reg_set(ctx, rctx, tail_idx, src_reg)?;
            // Load/store from addr_tmp
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_tmp))?;
            self.emit_ldst_access(ctx, rctx, tail_idx, is_load, load_width, data_reg)?;
        } else {
            // post-index: access [Rn], then Rn ← Rn + imm9
            self.emit_addr_reg_get(ctx, rctx, tail_idx, src_reg)?;
            self.emit_ldst_access(ctx, rctx, tail_idx, is_load, load_width, data_reg)?;
            // Update Rn
            self.emit_addr_reg_get(ctx, rctx, tail_idx, src_reg)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(off))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            self.emit_addr_reg_set(ctx, rctx, tail_idx, src_reg)?;
        }
        Ok(())
    }

    // ── LDST_UNSCALED (LDUR/STUR — unscaled immediate, no writeback) ──────────
    //
    // Bit-identical operand layout to `LDST_IMM9` (Rn [9:5], imm9 [20:12],
    // Rt [4:0]) but semantically simpler: always `addr = Rn + imm9`, access
    // `[addr]`, and — unlike LDR/STR's `LDST_IMM9` encoding — *never* writes
    // back to Rn (there is no pre/post-index form of LDUR/STUR).

    pub(super) fn translate_ldst_unscaled<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &LDST_UNSCALED,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, is_load, load_width) = match inner {
            LDST_UNSCALED::LDUR_Rt_ADDR_SIMM9(x)   => (x.0, true,  if is_64bit_ldst(x.0) { 8u8 } else { 4u8 }),
            LDST_UNSCALED::LDURB_Rt_ADDR_SIMM9(x)  => (x.0, true,  1u8),
            LDST_UNSCALED::LDURH_Rt_ADDR_SIMM9(x)  => (x.0, true,  2u8),
            LDST_UNSCALED::LDURSB_Rt_ADDR_SIMM9(x) => (x.0, true,  0xB1), // 1-byte signed
            LDST_UNSCALED::LDURSH_Rt_ADDR_SIMM9(x) => (x.0, true,  0xB2), // 2-byte signed
            LDST_UNSCALED::LDURSW_Rt_ADDR_SIMM9(x) => (x.0, true,  0xB4), // 4-byte signed
            LDST_UNSCALED::STUR_Rt_ADDR_SIMM9(x)   => (x.0, false, if is_64bit_ldst(x.0) { 8u8 } else { 4u8 }),
            LDST_UNSCALED::STURB_Rt_ADDR_SIMM9(x)  => (x.0, false, 1u8),
            LDST_UNSCALED::STURH_Rt_ADDR_SIMM9(x)  => (x.0, false, 2u8),
            _ => unsup!(),
        };

        let base_reg = rn(w);
        let data_reg = rd(w);
        let off      = imm9_signed(w);

        self.emit_addr_reg_get(ctx, rctx, tail_idx, base_reg)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(off))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
        self.emit_ldst_access(ctx, rctx, tail_idx, is_load, load_width, data_reg)?;
        Ok(())
    }

    /// Emit the actual load or store once the address is on the stack (for loads)
    /// or we've set up address + data (for stores).
    /// `load_width`: 1=byte, 2=half, 4=word, 8=dword; 0xB1/0xB2/0xB4=signed variants.
    fn emit_ldst_access<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        is_load: bool,
        load_width: u8,
        data_reg: u32,
    ) -> Result<(), E> {
        if is_load {
            match load_width {
                8    => { rctx.feed(ctx, tail_idx, &Instruction::I64Load(memarg(3)))?; }
                4    => { rctx.feed(ctx, tail_idx, &Instruction::I64Load32U(memarg(2)))?; }
                2    => {
                    rctx.feed(ctx, tail_idx, &Instruction::I32Load16U(memarg(1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                }
                1    => {
                    rctx.feed(ctx, tail_idx, &Instruction::I32Load8U(memarg(0)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                }
                0xB4 => { rctx.feed(ctx, tail_idx, &Instruction::I64Load32S(memarg(2)))?; } // LDRSW
                0xB2 => { // LDRSH
                    rctx.feed(ctx, tail_idx, &Instruction::I32Load16S(memarg(1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                }
                0xB1 => { // LDRSB
                    rctx.feed(ctx, tail_idx, &Instruction::I32Load8S(memarg(0)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                }
                _ => { rctx.feed(ctx, tail_idx, &Instruction::I64Load(memarg(3)))?; }
            }
            self.emit_gpr_set(ctx, rctx, tail_idx, data_reg)?;
        } else {
            // Store: address already on stack; push data
            self.emit_gpr_get(ctx, rctx, tail_idx, data_reg)?;
            match load_width {
                8 => { rctx.feed(ctx, tail_idx, &Instruction::I64Store(memarg(3)))?; }
                4 => { rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                       rctx.feed(ctx, tail_idx, &Instruction::I32Store(memarg(2)))?; }
                2 => { rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                       rctx.feed(ctx, tail_idx, &Instruction::I32Store16(memarg(1)))?; }
                1 => { rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                       rctx.feed(ctx, tail_idx, &Instruction::I32Store8(memarg(0)))?; }
                _ => { rctx.feed(ctx, tail_idx, &Instruction::I64Store(memarg(3)))?; }
            }
        }
        Ok(())
    }

    // ── LDST_REGOFF (register offset) ────────────────────────────────────────

    pub(super) fn translate_ldst_regoff<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &LDST_REGOFF,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, is_load, load_width) = match inner {
            LDST_REGOFF::LDR_Rt_ADDR_REGOFF(x)   => (x.0, true,  if is_64bit_ldst(x.0) { 8u8 } else { 4u8 }),
            LDST_REGOFF::LDRB_Rt_ADDR_REGOFF(x)  => (x.0, true,  1u8),
            LDST_REGOFF::LDRH_Rt_ADDR_REGOFF(x)  => (x.0, true,  2u8),
            LDST_REGOFF::LDRSB_Rt_ADDR_REGOFF(x) => (x.0, true,  0xB1u8),
            LDST_REGOFF::LDRSH_Rt_ADDR_REGOFF(x) => (x.0, true,  0xB2u8),
            LDST_REGOFF::LDRSW_Rt_ADDR_REGOFF(x) => (x.0, true,  0xB4u8),
            LDST_REGOFF::STR_Rt_ADDR_REGOFF(x)   => (x.0, false, if is_64bit_ldst(x.0) { 8u8 } else { 4u8 }),
            LDST_REGOFF::STRB_Rt_ADDR_REGOFF(x)  => (x.0, false, 1u8),
            LDST_REGOFF::STRH_Rt_ADDR_REGOFF(x)  => (x.0, false, 2u8),
            _ => unsup!(),
        };

        let data_reg  = rd(w);
        let base_reg  = rn(w);
        let off_reg   = rm(w);
        let option    = ext_option(w); // 010=UXTW 011=LSL 110=SXTW 111=SXTX
        let s         = regoff_s(w);   // shift enable
        // scale = log2(element_bytes): 0=byte 1=half 2=word 3=dword
        let scale = match load_width { 1 | 0xB1 => 0u32, 2 | 0xB2 => 1, 4 | 0xB4 => 2, _ => 3 };

        // Emit: base + extend(off_reg) [<< scale if S=1]
        self.emit_addr_reg_get(ctx, rctx, tail_idx, base_reg)?;
        self.emit_gpr_get(ctx, rctx, tail_idx, off_reg)?;
        match option {
            0b010 => { // UXTW: zero-extend 32 bits
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF_FFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
            }
            0b110 => { // SXTW: sign-extend 32 bits
                rctx.feed(ctx, tail_idx, &Instruction::I64Extend32S)?;
            }
            // 011=LSL, 111=SXTX — value used as-is
            _ => {}
        }
        if s != 0 && scale > 0 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(scale as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
        self.emit_ldst_access(ctx, rctx, tail_idx, is_load, load_width, data_reg)?;
        Ok(())
    }

    // ── LDSTPAIR_OFF (signed-offset load/store pair) ──────────────────────────

    pub(super) fn translate_ldstpair_off<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &LDSTPAIR_OFF,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        // opc (bits[31:30]): 10=64-bit, 00=32-bit; L (bit[22]): 1=load, 0=store
        let (w, is_load, scale) = match inner {
            LDSTPAIR_OFF::LDP_Rt_Rt2_ADDR_SIMM7(x)   => (x.0, true,  8i64),
            LDSTPAIR_OFF::STP_Rt_Rt2_ADDR_SIMM7(x)   => (x.0, false, 8i64),
            LDSTPAIR_OFF::LDPSW_Rt_Rt2_ADDR_SIMM7(x) => (x.0, true, -4i64), // negative = 32-bit signed
            _ => unsup!(),
        };
        self.emit_ldstpair(ctx, rctx, tail_idx, w, is_load, scale, false, false)
    }

    // ── LDSTPAIR_INDEXED (pre/post-index pair) ────────────────────────────────

    pub(super) fn translate_ldstpair_indexed<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &LDSTPAIR_INDEXED,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        // bit[24]: 0=post-index, 1=pre-index
        let (w, is_load, scale) = match inner {
            LDSTPAIR_INDEXED::LDP_Rt_W_Rt2_W_ADDR_SIMM7_S_S(x)   => (x.0, true,   8i64),
            LDSTPAIR_INDEXED::STP_Rt_W_Rt2_W_ADDR_SIMM7_S_S(x)   => (x.0, false,  8i64),
            LDSTPAIR_INDEXED::LDPSW_Rt_X_Rt2_X_ADDR_SIMM7_S_S(x) => (x.0, true,  -4i64),
            _ => unsup!(),
        };
        let pre_index = field(w, 24, 24) != 0;
        self.emit_ldstpair(ctx, rctx, tail_idx, w, is_load, scale, true, pre_index)
    }

    /// Common pair load/store emitter.
    /// `scale`: positive = unsigned scale bytes, negative = signed (LDPSW: 4-byte signed extend).
    /// `writeback`: if true, update Rn.
    /// `pre_index`: for writeback, update before (true) or after (false) access.
    fn emit_ldstpair<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        w: u32,
        is_load: bool,
        scale: i64,
        writeback: bool,
        pre_index: bool,
    ) -> Result<(), E> {
        let base_reg = rn(w);
        let reg1     = rd(w);   // Rt  (bits [4:0])
        let reg2     = rt2(w);  // Rt2 (bits [14:10])
        let off      = imm7_signed(w) * scale.abs();
        let is_ldpsw = scale < 0; // negative scale signals LDPSW (32-bit signed pair)
        let elem_sz  = if is_ldpsw { 4i64 } else { scale.abs() };
        let addr_tmp = rctx.layout().local(self.tmp_slot, 2);

        // Compute base address (before or at access)
        let base_addr = if writeback && pre_index {
            // pre: addr = Rn + off, then Rn ← addr
            self.emit_addr_reg_get(ctx, rctx, tail_idx, base_reg)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(off))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalTee(addr_tmp))?;
            rctx.feed(ctx, tail_idx, &Instruction::Drop)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_tmp))?;
            self.emit_addr_reg_set(ctx, rctx, tail_idx, base_reg)?;
            addr_tmp
        } else {
            // post or no writeback: addr = Rn + off
            self.emit_addr_reg_get(ctx, rctx, tail_idx, base_reg)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(off))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalTee(addr_tmp))?;
            rctx.feed(ctx, tail_idx, &Instruction::Drop)?;
            addr_tmp
        };

        if is_load {
            // Load Rt from [addr], Rt2 from [addr + elem_sz]
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(base_addr))?;
            if is_ldpsw {
                rctx.feed(ctx, tail_idx, &Instruction::I64Load32S(memarg(2)))?;
            } else {
                rctx.feed(ctx, tail_idx, &Instruction::I64Load(memarg(3)))?;
            }
            self.emit_gpr_set(ctx, rctx, tail_idx, reg1)?;

            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(base_addr))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(elem_sz))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            if is_ldpsw {
                rctx.feed(ctx, tail_idx, &Instruction::I64Load32S(memarg(2)))?;
            } else {
                rctx.feed(ctx, tail_idx, &Instruction::I64Load(memarg(3)))?;
            }
            self.emit_gpr_set(ctx, rctx, tail_idx, reg2)?;
        } else {
            // Store Rt to [addr], Rt2 to [addr + elem_sz]
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(base_addr))?;
            self.emit_gpr_get(ctx, rctx, tail_idx, reg1)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Store(memarg(3)))?;

            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(base_addr))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(elem_sz))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            self.emit_gpr_get(ctx, rctx, tail_idx, reg2)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Store(memarg(3)))?;
        }

        // Post-index writeback: Rn ← Rn + off
        if writeback && !pre_index {
            self.emit_addr_reg_get(ctx, rctx, tail_idx, base_reg)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(off))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            self.emit_addr_reg_set(ctx, rctx, tail_idx, base_reg)?;
        }
        Ok(())
    }
}
