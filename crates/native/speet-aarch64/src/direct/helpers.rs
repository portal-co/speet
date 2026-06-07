//! Bit-field extraction helpers, bitmask-immediate decode, and condition/indirect snippets.

use crate::*;
use wax_core::build::{InstructionOperatorSink, InstructionOperatorSource, InstructionSource};

// ── AArch64 bit-field extraction helpers ─────────────────────────────────────

/// Extract bits `lo..=hi` from a 32-bit word.
#[inline(always)]
pub(super) fn field(word: u32, lo: u8, hi: u8) -> u32 {
    (word >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

/// Rd / Rt (bits 4:0).
#[inline(always)] pub(super) fn rd(w: u32) -> u32 { w & 0x1F }
/// Rn (bits 9:5).
#[inline(always)] pub(super) fn rn(w: u32) -> u32 { (w >> 5) & 0x1F }
/// Rm (bits 20:16).
#[inline(always)] pub(super) fn rm(w: u32) -> u32 { (w >> 16) & 0x1F }
/// Ra (bits 14:10) — third source for DP_3SRC.
#[inline(always)] pub(super) fn ra(w: u32) -> u32 { (w >> 10) & 0x1F }
/// Rt2 (bits 14:10) — second register for load/store pairs.
#[inline(always)] pub(super) fn rt2(w: u32) -> u32 { (w >> 10) & 0x1F }

/// imm12 (bits 21:10) — unsigned 12-bit immediate (ADDSUB_IMM).
#[inline(always)] pub(super) fn imm12(w: u32) -> u32 { (w >> 10) & 0xFFF }
/// shift field (bits 23:22) — ADDSUB_IMM: 0=no shift, 1=LSL#12.
#[inline(always)] pub(super) fn addsub_imm_shift(w: u32) -> u32 { (w >> 22) & 0x3 }
/// imm16 (bits 20:5) — MOVEWIDE immediate.
#[inline(always)] pub(super) fn imm16(w: u32) -> u32 { (w >> 5) & 0xFFFF }
/// hw (bits 22:21) — MOVEWIDE half-word selector (0..3 → shift 0/16/32/48).
#[inline(always)] pub(super) fn hw(w: u32) -> u32 { (w >> 21) & 0x3 }
/// imm26 (bits 25:0) — B/BL PC-relative offset (4-byte units).
#[inline(always)] pub(super) fn imm26(w: u32) -> u32 { w & 0x3FF_FFFF }
/// imm19 (bits 23:5) — B.cond / CBZ / CBNZ offset.
#[inline(always)] pub(super) fn imm19(w: u32) -> u32 { (w >> 5) & 0x7_FFFF }
/// cond (bits 3:0) — B.cond condition code.
#[inline(always)] pub(super) fn cond_field(w: u32) -> u32 { w & 0xF }

/// imm9 (bits 20:12), sign-extended to i64 — LDST_IMM9.
#[inline(always)] pub(super) fn imm9_signed(w: u32) -> i64 { sign_ext_n(field(w, 12, 20), 9) }
/// imm7 (bits 21:15), sign-extended to i64 — LDSTPAIR.
#[inline(always)] pub(super) fn imm7_signed(w: u32) -> i64 { sign_ext_n(field(w, 15, 21), 7) }

/// ADR/ADRP: immhi[23:5] | immlo[30:29], sign-extended to 21 bits.
#[inline(always)]
pub(super) fn imm21_adr(w: u32) -> i64 {
    let immhi = field(w, 5, 23);   // 19 bits
    let immlo = field(w, 29, 30);  // 2 bits
    sign_ext_n((immhi << 2) | immlo, 21)
}

/// Extension type for ADDSUB_EXT and LDST_REGOFF (bits 15:13).
#[inline(always)] pub(super) fn ext_option(w: u32) -> u32 { field(w, 13, 15) }
/// Shift amount for ADDSUB_EXT (bits 12:10, range 0–4).
#[inline(always)] pub(super) fn ext_shift(w: u32) -> u32 { field(w, 10, 12) }
/// S bit (bit 12) — shift-enable for LDST_REGOFF.
#[inline(always)] pub(super) fn regoff_s(w: u32) -> u32 { (w >> 12) & 1 }

/// Compute the actual immediate for ADDSUB_IMM, applying optional LSL#12.
#[inline(always)]
pub(super) fn addsub_actual_imm(w: u32) -> i64 {
    let imm = imm12(w) as i64;
    if addsub_imm_shift(w) == 1 { imm << 12 } else { imm }
}

/// Sign-extend an `n`-bit value (in the low `n` bits of `v`) to i64.
#[inline(always)]
pub(super) fn sign_ext_n(v: u32, n: u8) -> i64 {
    let shift = 32 - n as u32;
    ((v << shift) as i32 >> shift) as i64
}

/// Sign-extend a 26-bit value to i64 (B/BL).
#[inline(always)]
pub(super) fn sign_ext26(v: u32) -> i64 { sign_ext_n(v & 0x3FF_FFFF, 26) }
/// Sign-extend a 19-bit value to i64 (B.cond / CBZ / CBNZ).
#[inline(always)]
pub(super) fn sign_ext19(v: u32) -> i64 { sign_ext_n(v & 0x7_FFFF, 19) }

/// Decode the 8-bit float immediate (FMOV Fd, #imm) to f64.
/// imm8 is located at bits [20:13] of the instruction word.
pub(super) fn decode_fp8_imm(w: u32) -> f64 {
    let imm8 = field(w, 13, 20);
    let a    = (imm8 >> 7) & 1;   // sign
    let b    = (imm8 >> 6) & 1;
    let c    = (imm8 >> 5) & 1;
    let d    = (imm8 >> 4) & 1;
    let efgh = imm8 & 0xF;
    let not_b = 1 - b;
    // 11-bit biased exponent: NOT(b), b×8, c, d
    let exp = (not_b << 10) | (b << 9) | (b << 8) | (b << 7) | (b << 6)
            | (b << 5)  | (b << 4)  | (b << 3) | (b << 2) | (c << 1) | d;
    let bits: u64 = ((a as u64) << 63) | ((exp as u64) << 52) | ((efgh as u64) << 48);
    f64::from_bits(bits)
}

/// Decode AArch64 bitmask immediate N:immr:imms → u64.
/// Used by LOG_IMM (AND/ORR/EOR with immediate).
pub(super) fn decode_bitmask_imm(w: u32) -> u64 {
    let n    = (w >> 22) & 1;
    let immr = ((w >> 16) & 0x3F) as u32;
    let imms = ((w >> 10) & 0x3F) as u32;

    // Length of the element: MSB position of N:NOT(imms) in [6:0]
    let ni = ((n << 6) | ((!imms) & 0x3F)) as u32;
    // ni must be nonzero for a valid encoding; find its MSB
    let len = 31 - ni.leading_zeros(); // 0..=6
    let esize = 1u32 << len;           // element size in bits
    let levels = esize - 1;            // mask for valid immr/imms bits

    let s = imms & levels;             // number of set bits - 1
    let r = immr & levels;             // right-rotation amount

    // Unrotated element: (s+1) consecutive 1-bits
    let welem = if s == 63 { u64::MAX } else { (1u64 << (s + 1)) - 1 };

    // Rotate right by r within esize bits
    let elem = if r == 0 || esize == 64 {
        if r == 0 { welem }
        else {
            // esize == 64: 64-bit rotate
            welem.rotate_right(r)
        }
    } else {
        let mask = (1u64 << esize) - 1;
        ((welem >> r) | (welem << (esize - r))) & mask
    };

    // Replicate element to fill 64 bits
    let mut result = 0u64;
    let mut shift = 0u32;
    while shift < 64 {
        result |= elem << shift;
        shift += esize;
    }
    result
}

// ── wax_core MemArg helper ────────────────────────────────────────────────────

pub(super) fn memarg(align: u32) -> wasm_encoder::MemArg {
    wasm_encoder::MemArg { memory_index: 0, align, offset: 0 }
}

// ── Condition snippet for B.cond / CONDSEL / FLOATSEL ────────────────────────

/// Emits an `i32` (1 = take branch, 0 = skip) for an AArch64 condition code.
pub(crate) struct A64Condition {
    pub(crate) cond: u32,
    pub(crate) n: u32,
    pub(crate) z: u32,
    pub(crate) c: u32,
    pub(crate) v: u32,
}

impl<Context, E> InstructionSource<Context, E> for A64Condition {
    fn emit_instruction(
        &self, ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        emit_cond(ctx, sink, self.cond, self.n, self.z, self.c, self.v)
    }
}
impl<Context, E> InstructionOperatorSource<Context, E> for A64Condition {
    fn emit(
        &self, ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        emit_cond(ctx, sink, self.cond, self.n, self.z, self.c, self.v)
    }
}

pub(super) fn emit_cond<Context, E, S>(
    ctx: &mut Context, sink: &mut S,
    cond: u32, n: u32, z: u32, c: u32, v: u32,
) -> Result<(), E>
where S: wax_core::build::InstructionSink<Context, E> + ?Sized
{
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

// ── Indirect branch target snippet ───────────────────────────────────────────

/// Emits a WASM function index from an AArch64 register:
///   func_idx = (gpr_value - base_pc) / 4
pub(crate) struct A64IndirectTarget {
    pub(crate) gpr_local: u32,
    pub(crate) base_pc: u64,
}

impl<Context, E> InstructionSource<Context, E> for A64IndirectTarget {
    fn emit_instruction(
        &self, ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        emit_indirect_target(ctx, sink, self.gpr_local, self.base_pc)
    }
}
impl<Context, E> InstructionOperatorSource<Context, E> for A64IndirectTarget {
    fn emit(
        &self, ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        emit_indirect_target(ctx, sink, self.gpr_local, self.base_pc)
    }
}

pub(super) fn emit_indirect_target<Context, E, S>(
    ctx: &mut Context, sink: &mut S, gpr_local: u32, base_pc: u64,
) -> Result<(), E>
where S: wax_core::build::InstructionSink<Context, E> + ?Sized
{
    // func_idx = (gpr - base_pc) >> 2.  Leave the result as i64: the indirect
    // call table is a 64-bit table (`table64`), so `return_call_indirect`
    // consumes an i64 index.  (Do NOT wrap to i32.)
    sink.instruction(ctx, &Instruction::LocalGet(gpr_local))?;
    sink.instruction(ctx, &Instruction::I64Const(base_pc as i64))?;
    sink.instruction(ctx, &Instruction::I64Sub)?;
    sink.instruction(ctx, &Instruction::I64Const(2))?;
    sink.instruction(ctx, &Instruction::I64ShrU)?;
    Ok(())
}

// ── Compare-zero condition for CBZ / CBNZ ────────────────────────────────────

pub(crate) struct CmpZeroCond { pub(crate) rt_local: u32, pub(crate) is_cbnz: bool }

impl<Context, E> InstructionSource<Context, E> for CmpZeroCond {
    fn emit_instruction(
        &self, ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        emit_cmp_zero(ctx, sink, self.rt_local, self.is_cbnz)
    }
}
impl<Context, E> InstructionOperatorSource<Context, E> for CmpZeroCond {
    fn emit(
        &self, ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        emit_cmp_zero(ctx, sink, self.rt_local, self.is_cbnz)
    }
}

pub(super) fn emit_cmp_zero<Context, E, S>(
    ctx: &mut Context, sink: &mut S, rt_local: u32, is_cbnz: bool,
) -> Result<(), E>
where S: wax_core::build::InstructionSink<Context, E> + ?Sized
{
    sink.instruction(ctx, &Instruction::LocalGet(rt_local))?;
    sink.instruction(ctx, &Instruction::I64Eqz)?;
    if is_cbnz { sink.instruction(ctx, &Instruction::I32Eqz)?; }
    Ok(())
}
