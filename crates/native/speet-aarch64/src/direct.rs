use crate::*;
use alloc::collections::BTreeMap;
use disarm64::decoder_full::{
    ADDSUB_IMM, ADDSUB_SHIFT, BRANCH_IMM, BRANCH_REG, COMPBRANCH, CONDBRANCH,
    LDST_POS, LOG_SHIFT, MOVEWIDE, Operation,
};
use wax_core::build::{InstructionOperatorSink, InstructionOperatorSource, InstructionSource};
use yecta::{FuncIdx, JumpCallParams, LocalDeclarator, Snippet, Target};

// ── AArch64 bit-field extraction helpers ─────────────────────────────────────
//
// All instructions are fixed 4-byte little-endian words.  Field positions
// follow the ARM Architecture Reference Manual (A64 instruction format).

/// Extract bits `lo..=hi` from a 32-bit word.
#[inline(always)]
fn field(word: u32, lo: u8, hi: u8) -> u32 {
    (word >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

/// Rd / Rt (bits 4:0).
#[inline(always)] fn rd(w: u32) -> u32 { w & 0x1F }
/// Rn (bits 9:5).
#[inline(always)] fn rn(w: u32) -> u32 { (w >> 5) & 0x1F }
/// Rm (bits 20:16) — shifted-register operand.
#[inline(always)] fn rm(w: u32) -> u32 { (w >> 16) & 0x1F }
/// imm12 (bits 21:10) — unsigned 12-bit immediate.
#[inline(always)] fn imm12(w: u32) -> u32 { (w >> 10) & 0xFFF }
/// shift field (bits 23:22) — ADDSUB_IMM: 0=no shift, 1=LSL#12.
#[inline(always)] fn addsub_imm_shift(w: u32) -> u32 { (w >> 22) & 0x3 }
/// imm16 (bits 20:5) — MOVEWIDE immediate.
#[inline(always)] fn imm16(w: u32) -> u32 { (w >> 5) & 0xFFFF }
/// hw (bits 22:21) — MOVEWIDE half-word selector (0..3 → shift 0/16/32/48).
#[inline(always)] fn hw(w: u32) -> u32 { (w >> 21) & 0x3 }
/// imm26 (bits 25:0) — B/BL PC-relative offset (in 4-byte units).
#[inline(always)] fn imm26(w: u32) -> u32 { w & 0x3FF_FFFF }
/// imm19 (bits 23:5) — B.cond / CBZ / CBNZ PC-relative offset.
#[inline(always)] fn imm19(w: u32) -> u32 { (w >> 5) & 0x7_FFFF }
/// cond (bits 3:0) — B.cond condition code.
#[inline(always)] fn cond_field(w: u32) -> u32 { w & 0xF }
/// Compute the actual immediate for ADDSUB_IMM, applying the optional LSL#12.
#[inline(always)]
fn addsub_actual_imm(w: u32) -> i64 {
    let imm = imm12(w) as i64;
    if addsub_imm_shift(w) == 1 { imm << 12 } else { imm }
}
/// Sign-extend a 26-bit value to i64 (B/BL offsets).
#[inline(always)]
fn sign_ext26(v: u32) -> i64 {
    let v = v & 0x3FF_FFFF;
    if v & (1 << 25) != 0 { (v | 0xFC00_0000) as i32 as i64 } else { v as i64 }
}
/// Sign-extend a 19-bit value to i64 (B.cond / CBZ / CBNZ offsets).
#[inline(always)]
fn sign_ext19(v: u32) -> i64 {
    let v = v & 0x7_FFFF;
    if v & (1 << 18) != 0 { (v | 0xFFF8_0000) as i32 as i64 } else { v as i64 }
}

// ── Condition snippet for B.cond ───────────────────────────────────────────────

/// Emits an `i32` (1 = take branch, 0 = skip) for an AArch64 condition code.
struct A64Condition {
    cond: u32,
    n: u32, // local index for N flag
    z: u32, // local index for Z flag
    c: u32, // local index for C flag
    v: u32, // local index for V flag
}

impl<Context, E> InstructionSource<Context, E> for A64Condition {
    fn emit_instruction(&self, ctx: &mut Context, sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_)) -> Result<(), E> {
        emit_cond(ctx, sink, self.cond, self.n, self.z, self.c, self.v)
    }
}
impl<Context, E> InstructionOperatorSource<Context, E> for A64Condition {
    fn emit(&self, ctx: &mut Context, sink: &mut (dyn InstructionOperatorSink<Context, E> + '_)) -> Result<(), E> {
        emit_cond(ctx, sink, self.cond, self.n, self.z, self.c, self.v)
    }
}

fn emit_cond<Context, E, S: wax_core::build::InstructionSink<Context, E> + ?Sized>(
    ctx: &mut Context,
    sink: &mut S,
    cond: u32,
    n: u32, z: u32, c: u32, v: u32,
) -> Result<(), E> {
    match cond & !1u32 {
        0 => { // EQ(0) / NE(1)
            sink.instruction(ctx, &Instruction::LocalGet(z))?;
            if cond & 1 == 1 { sink.instruction(ctx, &Instruction::I32Eqz)?; }
        }
        2 => { // CS(2) / CC(3)
            sink.instruction(ctx, &Instruction::LocalGet(c))?;
            if cond & 1 == 1 { sink.instruction(ctx, &Instruction::I32Eqz)?; }
        }
        4 => { // MI(4) / PL(5)
            sink.instruction(ctx, &Instruction::LocalGet(n))?;
            if cond & 1 == 1 { sink.instruction(ctx, &Instruction::I32Eqz)?; }
        }
        6 => { // VS(6) / VC(7)
            sink.instruction(ctx, &Instruction::LocalGet(v))?;
            if cond & 1 == 1 { sink.instruction(ctx, &Instruction::I32Eqz)?; }
        }
        8 => { // HI(8)=C&&!Z / LS(9)
            sink.instruction(ctx, &Instruction::LocalGet(c))?;
            sink.instruction(ctx, &Instruction::LocalGet(z))?;
            sink.instruction(ctx, &Instruction::I32Eqz)?;
            sink.instruction(ctx, &Instruction::I32And)?;
            if cond & 1 == 1 { sink.instruction(ctx, &Instruction::I32Eqz)?; }
        }
        10 => { // GE(10)=N==V / LT(11)=N!=V
            sink.instruction(ctx, &Instruction::LocalGet(n))?;
            sink.instruction(ctx, &Instruction::LocalGet(v))?;
            sink.instruction(ctx, &Instruction::I32Xor)?;
            sink.instruction(ctx, &Instruction::I32Eqz)?;
            if cond & 1 == 1 { sink.instruction(ctx, &Instruction::I32Eqz)?; }
        }
        12 => { // GT(12)=!Z&&N==V / LE(13)
            sink.instruction(ctx, &Instruction::LocalGet(z))?;
            sink.instruction(ctx, &Instruction::I32Eqz)?;
            sink.instruction(ctx, &Instruction::LocalGet(n))?;
            sink.instruction(ctx, &Instruction::LocalGet(v))?;
            sink.instruction(ctx, &Instruction::I32Xor)?;
            sink.instruction(ctx, &Instruction::I32Eqz)?;
            sink.instruction(ctx, &Instruction::I32And)?;
            if cond & 1 == 1 { sink.instruction(ctx, &Instruction::I32Eqz)?; }
        }
        _ => { // AL(14) / NV(15) — always
            sink.instruction(ctx, &Instruction::I32Const(1))?;
        }
    }
    Ok(())
}

// ── Indirect branch target snippet ────────────────────────────────────────────

/// Emits a WASM function index from an AArch64 register:
///   func_idx = (gpr_value - base_pc) / 4
struct A64IndirectTarget {
    gpr_local: u32,
    base_pc: u64,
}

impl<Context, E> InstructionSource<Context, E> for A64IndirectTarget {
    fn emit_instruction(&self, ctx: &mut Context, sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_)) -> Result<(), E> {
        emit_indirect_target(ctx, sink, self.gpr_local, self.base_pc)
    }
}
impl<Context, E> InstructionOperatorSource<Context, E> for A64IndirectTarget {
    fn emit(&self, ctx: &mut Context, sink: &mut (dyn InstructionOperatorSink<Context, E> + '_)) -> Result<(), E> {
        emit_indirect_target(ctx, sink, self.gpr_local, self.base_pc)
    }
}

fn emit_indirect_target<Context, E, S: wax_core::build::InstructionSink<Context, E> + ?Sized>(
    ctx: &mut Context,
    sink: &mut S,
    gpr_local: u32,
    base_pc: u64,
) -> Result<(), E> {
    // (gpr - base_pc) >> 2  →  i32 function index
    sink.instruction(ctx, &Instruction::LocalGet(gpr_local))?;
    sink.instruction(ctx, &Instruction::I64Const(base_pc as i64))?;
    sink.instruction(ctx, &Instruction::I64Sub)?;
    sink.instruction(ctx, &Instruction::I64Const(2))?;
    sink.instruction(ctx, &Instruction::I64ShrU)?;
    sink.instruction(ctx, &Instruction::I32WrapI64)?;
    Ok(())
}

// ── Compare-zero condition for CBZ / CBNZ ─────────────────────────────────────

struct CmpZeroCond { rt_local: u32, is_cbnz: bool }

impl<Context, E> InstructionSource<Context, E> for CmpZeroCond {
    fn emit_instruction(&self, ctx: &mut Context, sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_)) -> Result<(), E> {
        emit_cmp_zero(ctx, sink, self.rt_local, self.is_cbnz)
    }
}
impl<Context, E> InstructionOperatorSource<Context, E> for CmpZeroCond {
    fn emit(&self, ctx: &mut Context, sink: &mut (dyn InstructionOperatorSink<Context, E> + '_)) -> Result<(), E> {
        emit_cmp_zero(ctx, sink, self.rt_local, self.is_cbnz)
    }
}

fn emit_cmp_zero<Context, E, S: wax_core::build::InstructionSink<Context, E> + ?Sized>(
    ctx: &mut Context, sink: &mut S, rt_local: u32, is_cbnz: bool,
) -> Result<(), E> {
    sink.instruction(ctx, &Instruction::LocalGet(rt_local))?;
    sink.instruction(ctx, &Instruction::I64Eqz)?;
    if is_cbnz { sink.instruction(ctx, &Instruction::I32Eqz)?; }
    Ok(())
}

// ── MemArg helpers ────────────────────────────────────────────────────────────

fn memarg(align: u32) -> wasm_encoder::MemArg {
    wasm_encoder::MemArg { memory_index: 0, align, offset: 0 }
}

// ── AArch64Recompiler translation methods ─────────────────────────────────────

impl<Context, E> AArch64Recompiler<Context, E> {
    /// Open a new WASM function for the instruction at `pc`.
    fn init_function<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _pc: u64,
        inst_len: u32,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<usize, E> {
        let mark = rctx.locals_mark();
        rctx.layout_mut().rewind(&mark);
        let mut unit = ();
        let extra: &mut dyn LocalDeclarator = match self.memory_access.as_deref_mut() {
            Some(m) => m as &mut dyn LocalDeclarator,
            None => &mut unit,
        };
        rctx.declare_trap_locals(extra);
        let _cell = rctx.alloc_cell();
        let fn_type = f(
            &mut rctx
                .layout()
                .iter_since(&mark)
                .collect::<alloc::vec::Vec<_>>()
                .into_iter(),
        );
        rctx.next_with(ctx, fn_type, inst_len)
    }

    /// Translate a block of AArch64 bytes starting at `start_pc`.
    pub fn translate_bytes<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        bytes: &[u8],
        start_pc: u64,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<usize, ()> {
        let mut offset = 0usize;
        while offset + 4 <= bytes.len() {
            let word = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let pc = start_pc + offset as u64;

            let opcode = disarm64::decoder::decode(word);
            let class = opcode
                .as_ref()
                .map(|op| Self::classify_mnemonic(op.mnemonic))
                .unwrap_or(InsnClass::OTHER);

            let tail_idx = self.init_function(ctx, rctx, pc, 4, f).map_err(|_| ())?;

            // Write PC local.
            let pc_local = rctx.layout().local(self.pc_slot, 0);
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(pc as i64)).map_err(|_| ())?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(pc_local)).map_err(|_| ())?;

            // Fire instruction trap.
            {
                let insn_info = InstructionInfo { pc, len: 4, arch: ArchTag::AArch64, class };
                if rctx.on_instruction(&insn_info, ctx).map_err(|_| ())? == TrapAction::Skip {
                    offset += 4;
                    continue;
                }
            }

            match opcode {
                None => {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable).map_err(|_| ())?;
                    self.unsupported_insns.insert(alloc::format!("undef:{:08x}", word));
                }
                Some(ref op) => {
                    self.translate_one(ctx, rctx, tail_idx, pc, op).map_err(|_| ())?;
                }
            }
            offset += 4;
        }
        Ok(offset)
    }

    /// Translate one decoded AArch64 instruction.
    fn translate_one<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        pc: u64,
        opcode: &disarm64::decoder_full::Opcode,
    ) -> Result<(), E> {
        let total = rctx.locals_mark().total_locals;

        macro_rules! unsup {
            () => {{
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                self.unsupported_insns.insert(alloc::format!("{:?}", opcode.mnemonic));
                return Ok(());
            }};
        }

        match opcode.operation {
            // ── Integer ALU — register + scaled immediate ──────────────────────
            Operation::ADDSUB_IMM(ref inner) => {
                let (w, is_sub, set_flags) = match inner {
                    ADDSUB_IMM::ADD_Rd_SP_Rn_SP_AIMM(x)  => (x.0, false, false),
                    ADDSUB_IMM::ADDS_Rd_Rn_SP_AIMM(x)    => (x.0, false, true),
                    ADDSUB_IMM::SUB_Rd_SP_Rn_SP_AIMM(x)  => (x.0, true,  false),
                    ADDSUB_IMM::SUBS_Rd_Rn_SP_AIMM(x)    => (x.0, true,  true),
                    _ => { unsup!() }
                };
                let dest = rd(w);
                let src  = rn(w);
                let imm  = addsub_actual_imm(w);
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm))?;
                rctx.feed(ctx, tail_idx, if is_sub { &Instruction::I64Sub } else { &Instruction::I64Add })?;
                if set_flags { self.set_nzcv_and_store(ctx, rctx, tail_idx, dest)?; }
                else { self.emit_gpr_set(ctx, rctx, tail_idx, dest)?; }
            }

            // ── Integer ALU — register + shifted register ──────────────────────
            Operation::ADDSUB_SHIFT(ref inner) => {
                let (w, is_sub, set_flags) = match inner {
                    ADDSUB_SHIFT::ADD_Rd_Rn_Rm_SFT(x)  => (x.0, false, false),
                    ADDSUB_SHIFT::ADDS_Rd_Rn_Rm_SFT(x) => (x.0, false, true),
                    ADDSUB_SHIFT::SUB_Rd_Rn_Rm_SFT(x)  => (x.0, true,  false),
                    ADDSUB_SHIFT::SUBS_Rd_Rn_Rm_SFT(x) => (x.0, true,  true),
                    _ => { unsup!() }
                };
                let dest = rd(w);
                let src1 = rn(w);
                let src2 = rm(w);
                let shamt = field(w, 10, 15);
                let shtyp = field(w, 22, 23);
                self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, src2)?;
                // Apply shift to second operand if non-zero.
                if shamt != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(shamt as i64))?;
                    let shift_insn = match shtyp {
                        0 => &Instruction::I64Shl,
                        1 => &Instruction::I64ShrU,
                        _ => &Instruction::I64ShrS,
                    };
                    rctx.feed(ctx, tail_idx, shift_insn)?;
                }
                rctx.feed(ctx, tail_idx, if is_sub { &Instruction::I64Sub } else { &Instruction::I64Add })?;
                if set_flags { self.set_nzcv_and_store(ctx, rctx, tail_idx, dest)?; }
                else { self.emit_gpr_set(ctx, rctx, tail_idx, dest)?; }
            }

            // ── Logical — shifted register ─────────────────────────────────────
            Operation::LOG_SHIFT(ref inner) => {
                let (w, op, negate_rm, set_flags) = match inner {
                    LOG_SHIFT::AND_Rd_Rn_Rm_SFT(x)  => (x.0, 0u8, false, false),
                    LOG_SHIFT::ANDS_Rd_Rn_Rm_SFT(x) => (x.0, 0u8, false, true),
                    LOG_SHIFT::BIC_Rd_Rn_Rm_SFT(x)  => (x.0, 0u8, true,  false),
                    LOG_SHIFT::BICS_Rd_Rn_Rm_SFT(x) => (x.0, 0u8, true,  true),
                    LOG_SHIFT::ORR_Rd_Rn_Rm_SFT(x)  => (x.0, 1u8, false, false),
                    LOG_SHIFT::ORN_Rd_Rn_Rm_SFT(x)  => (x.0, 1u8, true,  false),
                    LOG_SHIFT::EOR_Rd_Rn_Rm_SFT(x)  => (x.0, 2u8, false, false),
                    LOG_SHIFT::EON_Rd_Rn_Rm_SFT(x)  => (x.0, 2u8, true,  false),
                    _ => { unsup!() }
                };
                let dest = rd(w);
                let src1 = rn(w);
                let src2 = rm(w);
                let shamt = field(w, 10, 15);
                let shtyp = field(w, 22, 23);
                self.emit_gpr_get(ctx, rctx, tail_idx, src1)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, src2)?;
                if shamt != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(shamt as i64))?;
                    let shift_insn = match shtyp {
                        0 => &Instruction::I64Shl,
                        1 => &Instruction::I64ShrU,
                        _ => &Instruction::I64ShrS,
                    };
                    rctx.feed(ctx, tail_idx, shift_insn)?;
                }
                if negate_rm {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1i64))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
                }
                let logical_insn = match op {
                    0 => &Instruction::I64And,
                    1 => &Instruction::I64Or,
                    _ => &Instruction::I64Xor,
                };
                rctx.feed(ctx, tail_idx, logical_insn)?;
                if set_flags { self.set_nzcv_and_store(ctx, rctx, tail_idx, dest)?; }
                else { self.emit_gpr_set(ctx, rctx, tail_idx, dest)?; }
            }

            // ── Move-wide ─────────────────────────────────────────────────────
            Operation::MOVEWIDE(ref inner) => {
                let (w, op) = match inner {
                    MOVEWIDE::MOVZ_Rd_HALF(x) => (x.0, 0u8),
                    MOVEWIDE::MOVN_Rd_HALF(x) => (x.0, 1u8),
                    MOVEWIDE::MOVK_Rd_HALF(x) => (x.0, 2u8),
                    _ => { unsup!() }
                };
                let dest   = rd(w);
                let imm    = imm16(w) as u64;
                let shift  = hw(w) * 16;
                let val    = imm << shift;
                match op {
                    0 => { // MOVZ
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(val as i64))?;
                        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
                    }
                    1 => { // MOVN
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(!(val as i64)))?;
                        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
                    }
                    _ => { // MOVK: keep other 48 bits, insert imm16 at hw position
                        let mask = !(0xFFFFu64 << shift) as i64;
                        self.emit_gpr_get(ctx, rctx, tail_idx, dest)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(mask))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(val as i64))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
                    }
                }
            }

            // ── Unconditional branch (B / BL) ─────────────────────────────────
            Operation::BRANCH_IMM(ref inner) => {
                let (w, is_bl) = match inner {
                    BRANCH_IMM::B_ADDR_PCREL26(x)  => (x.0, false),
                    BRANCH_IMM::BL_ADDR_PCREL26(x) => (x.0, true),
                    _ => { unsup!() }
                };
                if is_bl {
                    let ret_pc = pc + 4;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(ret_pc as i64))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, 30)?;
                }
                let off    = sign_ext26(imm26(w)) * 4;
                let target = pc.wrapping_add_signed(off);
                {
                    let kind = if is_bl { JumpKind::Call } else { JumpKind::DirectJump };
                    let info = JumpInfo::direct(pc, target, kind);
                    if rctx.on_jump(&info, ctx)? == TrapAction::Skip { return Ok(()); }
                }
                match self.pc_to_func_idx(target) {
                    Some(f_idx) => rctx.jmp(ctx, tail_idx, f_idx, total)?,
                    None => rctx.oob_jump(ctx, tail_idx, target, total)?,
                }
            }

            // ── Branch to register (BR / BLR / RET) ───────────────────────────
            Operation::BRANCH_REG(ref inner) => {
                let (w, kind) = match inner {
                    BRANCH_REG::RET_Rn(x)  => (x.0, JumpKind::Return),
                    BRANCH_REG::BR_Rn(x)   => (x.0, JumpKind::IndirectJump),
                    BRANCH_REG::BLR_Rn(x)  => (x.0, JumpKind::IndirectCall),
                    _ => { unsup!() }
                };
                let reg = rn(w);

                if matches!(inner, BRANCH_REG::BLR_Rn(_)) {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const((pc + 4) as i64))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, 30)?;
                }

                let gpr_local = rctx.layout().local(self.gpr_slot, reg);
                {
                    let info = JumpInfo::indirect(pc, gpr_local, kind);
                    if rctx.on_jump(&info, ctx)? == TrapAction::Skip { return Ok(()); }
                }

                let target_snippet = A64IndirectTarget { gpr_local, base_pc: self.base_pc };
                let params = JumpCallParams::indirect_jump(&target_snippet, total, rctx.pool());
                rctx.ji_with_params(ctx, tail_idx, params)?;
            }

            // ── Conditional branch — B.cond ────────────────────────────────────
            Operation::CONDBRANCH(ref inner) => {
                let w = match inner {
                    CONDBRANCH::B__ADDR_PCREL19(x)  => x.0,
                    CONDBRANCH::BC__ADDR_PCREL19(x) => x.0,
                    _ => { unsup!() }
                };
                let off   = sign_ext19(imm19(w)) * 4;
                let taken = pc.wrapping_add_signed(off);
                {
                    let info = JumpInfo::direct(pc, taken, JumpKind::ConditionalBranch);
                    if rctx.on_jump(&info, ctx)? == TrapAction::Skip { return Ok(()); }
                }
                let Some(taken_func) = self.pc_to_func_idx(taken) else { unsup!() };
                let condition = A64Condition {
                    cond: cond_field(w),
                    n: self.nzcv_local(rctx, 0),
                    z: self.nzcv_local(rctx, 1),
                    c: self.nzcv_local(rctx, 2),
                    v: self.nzcv_local(rctx, 3),
                };
                rctx.ji(
                    ctx, tail_idx, total, &BTreeMap::new(),
                    Target::Static { func: taken_func },
                    None,
                    rctx.pool(),
                    Some(&condition),
                )?;
            }

            // ── Compare-and-branch — CBZ / CBNZ ───────────────────────────────
            Operation::COMPBRANCH(ref inner) => {
                let (w, is_cbnz) = match inner {
                    COMPBRANCH::CBZ_Rt_ADDR_PCREL19(x)  => (x.0, false),
                    COMPBRANCH::CBNZ_Rt_ADDR_PCREL19(x) => (x.0, true),
                    _ => { unsup!() }
                };
                let off   = sign_ext19(imm19(w)) * 4;
                let taken = pc.wrapping_add_signed(off);
                {
                    let info = JumpInfo::direct(pc, taken, JumpKind::ConditionalBranch);
                    if rctx.on_jump(&info, ctx)? == TrapAction::Skip { return Ok(()); }
                }
                let Some(taken_func) = self.pc_to_func_idx(taken) else { unsup!() };
                let rt_local  = rctx.layout().local(self.gpr_slot, rd(w));
                let condition = CmpZeroCond { rt_local, is_cbnz };
                rctx.ji(
                    ctx, tail_idx, total, &BTreeMap::new(),
                    Target::Static { func: taken_func },
                    None,
                    rctx.pool(),
                    Some(&condition),
                )?;
            }

            // ── Scalar loads / stores — unsigned immediate offset ──────────────
            Operation::LDST_POS(ref inner) => {
                match inner {
                    LDST_POS::LDR_Rt_ADDR_UIMM12(x) => {
                        let (w, scale) = (x.0, 8i64);
                        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * scale))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Load(memarg(3)))?;
                        self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                    }
                    LDST_POS::LDRB_Rt_ADDR_UIMM12(x) => {
                        let w = x.0;
                        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Load8U(memarg(0)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                        self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                    }
                    LDST_POS::LDRH_Rt_ADDR_UIMM12(x) => {
                        let w = x.0;
                        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * 2))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Load16U(memarg(1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                        self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                    }
                    LDST_POS::LDRSW_Rt_ADDR_UIMM12(x) => {
                        let w = x.0;
                        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * 4))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Load32S(memarg(2)))?;
                        self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
                    }
                    LDST_POS::STR_Rt_ADDR_UIMM12(x) => {
                        let (w, scale) = (x.0, 8i64);
                        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * scale))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        self.emit_gpr_get(ctx, rctx, tail_idx, rd(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Store(memarg(3)))?;
                    }
                    LDST_POS::STRB_Rt_ADDR_UIMM12(x) => {
                        let w = x.0;
                        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        self.emit_gpr_get(ctx, rctx, tail_idx, rd(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Store8(memarg(0)))?;
                    }
                    LDST_POS::STRH_Rt_ADDR_UIMM12(x) => {
                        let w = x.0;
                        self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12(w) as i64 * 2))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        self.emit_gpr_get(ctx, rctx, tail_idx, rd(w))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Store16(memarg(1)))?;
                    }
                    _ => { unsup!() }
                }
            }

            // ── Everything else ────────────────────────────────────────────────
            _ => { unsup!() }
        }
        Ok(())
    }

    /// Tee result to tmp0, compute N/Z/C/V flags, then store result in GPR `rd`.
    fn set_nzcv_and_store<F>(
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

        // result on stack → tee to tmp0, still on stack after tee.
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = 0 (conservative — no carry model yet)
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;

        // V = 0 (conservative — no overflow model yet)
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        // Store result to destination register.
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }
}
