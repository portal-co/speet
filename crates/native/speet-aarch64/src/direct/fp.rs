//! Floating-point operation handlers.
//!
//! FP registers V0–V31 are stored as f64 locals (`fp_slot`).
//! Single-precision (S-register) operations demote/promote through f32.

use crate::*;
use super::helpers::*;
use disarm64::decoder_full::{
    FLOAT2INT, FLOATCMP, FLOATDP1, FLOATDP2, FLOATDP3, FLOATIMM, FLOATSEL,
};
use disarm64::decoder_full::Mnemonic;
use wasm_encoder::Ieee64;

/// Helper: `Instruction::F64Const` expects `Ieee64`, not bare `f64`.
#[inline(always)]
fn f64c(v: f64) -> Instruction<'static> { Instruction::F64Const(Ieee64::from(v)) }

impl<Context, E> AArch64Recompiler<Context, E> {

    // ── FLOATDP2 (binary arithmetic: FADD, FSUB, FMUL, FDIV, FNMUL) ──────────

    pub(super) fn translate_floatdp2<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &FLOATDP2,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        // op: 0=FADD 1=FSUB 2=FMUL 3=FDIV 4=FNMUL
        let (w, op) = match inner {
            FLOATDP2::FADD_Fd_Fn_Fm(x)              => (x.0, 0u8),
            FLOATDP2::FADD_Fd_S_S_Fn_S_S_Fm_S_S(x)  => (x.0, 0u8),
            FLOATDP2::FSUB_Fd_Fn_Fm(x)              => (x.0, 1u8),
            FLOATDP2::FSUB_Fd_S_S_Fn_S_S_Fm_S_S(x)  => (x.0, 1u8),
            FLOATDP2::FMUL_Fd_Fn_Fm(x)              => (x.0, 2u8),
            FLOATDP2::FMUL_Fd_S_S_Fn_S_S_Fm_S_S(x)  => (x.0, 2u8),
            FLOATDP2::FDIV_Fd_Fn_Fm(x)              => (x.0, 3u8),
            FLOATDP2::FDIV_Fd_S_S_Fn_S_S_Fm_S_S(x)  => (x.0, 3u8),
            FLOATDP2::FNMUL_Fd_Fn_Fm(x)             => (x.0, 4u8),
            FLOATDP2::FNMUL_Fd_S_S_Fn_S_S_Fm_S_S(x) => (x.0, 4u8),
            _ => unsup!(),
        };
        let dest = rd(w);
        let src1 = rn(w);
        let src2 = rm(w);

        self.emit_fp_get(ctx, rctx, tail_idx, src1)?;
        self.emit_fp_get(ctx, rctx, tail_idx, src2)?;

        if op == 4 {
            // FNMUL: -(Fn * Fm)
            rctx.feed(ctx, tail_idx, &Instruction::F64Mul)?;
            rctx.feed(ctx, tail_idx, &Instruction::F64Neg)?;
        } else {
            let insn: &Instruction = match op {
                0 => &Instruction::F64Add,
                1 => &Instruction::F64Sub,
                2 => &Instruction::F64Mul,
                _ => &Instruction::F64Div,
            };
            rctx.feed(ctx, tail_idx, insn)?;
        }
        self.emit_fp_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    // ── FLOATDP1 (unary: FABS, FNEG, FSQRT, FMOV, FCVT) ─────────────────────

    pub(super) fn translate_floatdp1<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &FLOATDP1,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        // Extract raw word — every variant holds a tuple struct wrapping u32.
        let w: u32 = match inner {
            FLOATDP1::FABS_Fd_Fn(x)            => x.0,
            FLOATDP1::FABS_Fd_S_S_Fn_S_S(x)    => x.0,
            FLOATDP1::FNEG_Fd_Fn(x)            => x.0,
            FLOATDP1::FNEG_Fd_S_S_Fn_S_S(x)    => x.0,
            FLOATDP1::FSQRT_Fd_Fn(x)           => x.0,
            FLOATDP1::FSQRT_Fd_S_S_Fn_S_S(x)   => x.0,
            FLOATDP1::FMOV_Fd_Fn(x)            => x.0,
            FLOATDP1::FMOV_Fd_S_S_Fn_S_S(x)    => x.0,
            FLOATDP1::FCVT_Fd_Fn(x)            => x.0,
            _ => unsup!(),
        };

        let dest = rd(w);
        let src  = rn(w);

        // Dispatch by mnemonic (same w, different operations)
        match mnemonic {
            Mnemonic::r#fabs => {
                self.emit_fp_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Abs)?;
                self.emit_fp_set(ctx, rctx, tail_idx, dest)?;
            }
            Mnemonic::r#fneg => {
                self.emit_fp_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Neg)?;
                self.emit_fp_set(ctx, rctx, tail_idx, dest)?;
            }
            Mnemonic::r#fsqrt => {
                self.emit_fp_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Sqrt)?;
                self.emit_fp_set(ctx, rctx, tail_idx, dest)?;
            }
            Mnemonic::r#fmov => {
                // Register-to-register copy (both are FP)
                self.emit_fp_get(ctx, rctx, tail_idx, src)?;
                self.emit_fp_set(ctx, rctx, tail_idx, dest)?;
            }
            Mnemonic::r#fcvt => {
                // Convert between FP precisions.
                // type (bits [23:22]): 00=single, 01=double
                // opc  (bits [16:15]): destination type
                let ty  = field(w, 22, 23); // source type
                let opc = field(w, 15, 16); // dest type
                self.emit_fp_get(ctx, rctx, tail_idx, src)?;
                match (ty, opc) {
                    (1, 0) => { // double → single: demote then re-promote (exact round-trip)
                        rctx.feed(ctx, tail_idx, &Instruction::F32DemoteF64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::F64PromoteF32)?;
                    }
                    _ => {} // single → double or other: value is already f64, no-op
                }
                self.emit_fp_set(ctx, rctx, tail_idx, dest)?;
            }
            _ => unsup!(),
        }
        Ok(())
    }

    // ── FLOATDP3 (fused multiply-add: FMADD, FMSUB, FNMADD, FNMSUB) ──────────

    pub(super) fn translate_floatdp3<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &FLOATDP3,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, is_sub, negated) = match inner {
            FLOATDP3::FMADD_Fd_Fn_Fm_Fa(x)                          => (x.0, false, false),
            FLOATDP3::FMADD_Fd_S_S_Fn_S_S_Fm_S_S_Fa_S_S(x)          => (x.0, false, false),
            FLOATDP3::FMSUB_Fd_Fn_Fm_Fa(x)                          => (x.0, true,  false),
            FLOATDP3::FMSUB_Fd_S_S_Fn_S_S_Fm_S_S_Fa_S_S(x)          => (x.0, true,  false),
            FLOATDP3::FNMADD_Fd_Fn_Fm_Fa(x)                         => (x.0, false, true),
            FLOATDP3::FNMADD_Fd_S_S_Fn_S_S_Fm_S_S_Fa_S_S(x)         => (x.0, false, true),
            FLOATDP3::FNMSUB_Fd_Fn_Fm_Fa(x)                         => (x.0, true,  true),
            FLOATDP3::FNMSUB_Fd_S_S_Fn_S_S_Fm_S_S_Fa_S_S(x)         => (x.0, true,  true),
            _ => unsup!(),
        };
        let dest  = rd(w);
        let src_n = rn(w);
        let src_m = rm(w);
        let acc_a = ra(w);

        // product = Fn * Fm
        self.emit_fp_get(ctx, rctx, tail_idx, src_n)?;
        self.emit_fp_get(ctx, rctx, tail_idx, src_m)?;
        rctx.feed(ctx, tail_idx, &Instruction::F64Mul)?;
        // result = Fa ± product
        if is_sub {
            // FMSUB / FNMSUB: Fa - product  →  negate product, then add Fa
            rctx.feed(ctx, tail_idx, &Instruction::F64Neg)?;
        }
        self.emit_fp_get(ctx, rctx, tail_idx, acc_a)?;
        rctx.feed(ctx, tail_idx, &Instruction::F64Add)?;
        if negated {
            rctx.feed(ctx, tail_idx, &Instruction::F64Neg)?;
        }
        self.emit_fp_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    // ── FLOAT2INT (GPR↔FP moves, int↔float conversions) ──────────────────────

    pub(super) fn translate_float2int<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &FLOAT2INT,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        match inner {
            // FMOV Fd, Xn — GPR (i64) → FP register (f64 bitcast)
            FLOAT2INT::FMOV_Fd_Rn(x) => {
                let w = x.0;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64ReinterpretI64)?;
                self.emit_fp_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // FMOV Sd, Wn — GPR 32-bit → FP single (bitcast via f32)
            FLOAT2INT::FMOV_Fd_S_S_Rn_W(x) => {
                let w = x.0;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32ReinterpretI32)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64PromoteF32)?;
                self.emit_fp_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // FMOV Xd, Fn — FP register (f64) → GPR (bitcast)
            FLOAT2INT::FMOV_Rd_Fn(x) => {
                let w = x.0;
                self.emit_fp_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ReinterpretF64)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // FMOV Wd, Sn — FP single → GPR 32-bit (bitcast via f32)
            FLOAT2INT::FMOV_Rd_W_Fn_S_S(x) => {
                let w = x.0;
                self.emit_fp_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::F32DemoteF64)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32ReinterpretF32)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // SCVTF Fd, Xn — signed integer (i64) → double
            FLOAT2INT::SCVTF_Fd_Rn(x) => {
                let w = x.0;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64ConvertI64S)?;
                self.emit_fp_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // SCVTF Sd, Wn — signed integer (i32) → single (stored as f64)
            FLOAT2INT::SCVTF_Fd_S_D_Rn_W(x) => {
                let w = x.0;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32ConvertI32S)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64PromoteF32)?;
                self.emit_fp_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // UCVTF Fd, Xn — unsigned integer (i64) → double
            FLOAT2INT::UCVTF_Fd_Rn(x) => {
                let w = x.0;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64ConvertI64U)?;
                self.emit_fp_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // UCVTF Sd, Wn — unsigned integer (i32) → single
            FLOAT2INT::UCVTF_Fd_S_D_Rn_W(x) => {
                let w = x.0;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32ConvertI32U)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64PromoteF32)?;
                self.emit_fp_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // FCVTZS Xd, Fn — double → signed i64 (truncate toward zero)
            FLOAT2INT::FCVTZS_Rd_Fn(x) => {
                let w = x.0;
                self.emit_fp_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64TruncF64S)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // FCVTZS Wd, Fn — double → signed i32 (truncate), zero-extend to i64
            FLOAT2INT::FCVTZS_Rd_W_Fn_S_D(x) => {
                let w = x.0;
                self.emit_fp_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32TruncF64S)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // FCVTZU Xd, Fn — double → unsigned i64 (truncate)
            FLOAT2INT::FCVTZU_Rd_Fn(x) => {
                let w = x.0;
                self.emit_fp_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64TruncF64U)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            // FCVTZU Wd, Fn — double → unsigned i32 (truncate)
            FLOAT2INT::FCVTZU_Rd_W_Fn_S_D(x) => {
                let w = x.0;
                self.emit_fp_get(ctx, rctx, tail_idx, rn(w))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32TruncF64U)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd(w))?;
            }
            _ => unsup!(),
        }
        Ok(())
    }

    // ── FLOATIMM (FMOV with 8-bit float immediate) ────────────────────────────

    pub(super) fn translate_floatimm<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &FLOATIMM,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let w: u32 = match inner {
            FLOATIMM::FMOV_Fd_FPIMM(x)       => x.0,
            FLOATIMM::FMOV_Fd_S_S_FPIMM(x)   => x.0,
            _ => unsup!(),
        };
        let val = decode_fp8_imm(w);
        rctx.feed(ctx, tail_idx, &f64c(val))?;
        self.emit_fp_set(ctx, rctx, tail_idx, rd(w))?;
        Ok(())
    }

    // ── FLOATCMP (FCMP, FCMPE — sets NZCV from FP comparison) ────────────────
    //
    // ARM FCMP NZCV results:
    //   Equal:      N=0, Z=1, C=1, V=0
    //   Less than:  N=1, Z=0, C=0, V=0
    //   Greater:    N=0, Z=0, C=1, V=0
    //   Unordered:  N=0, Z=0, C=1, V=1

    pub(super) fn translate_floatcmp<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &FLOATCMP,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let (w, cmp_zero) = match inner {
            FLOATCMP::FCMP_Fn_Fm(x)              => (x.0, false),
            FLOATCMP::FCMPE_Fn_Fm(x)             => (x.0, false),
            FLOATCMP::FCMP_Fn_S_S_Fm_S_S(x)      => (x.0, false),
            FLOATCMP::FCMPE_Fn_S_S_Fm_S_S(x)     => (x.0, false),
            FLOATCMP::FCMP_Fn_FPIMM0(x)           => (x.0, true),
            FLOATCMP::FCMPE_Fn_FPIMM0(x)          => (x.0, true),
            FLOATCMP::FCMP_Fn_S_S_FPIMM0(x)       => (x.0, true),
            FLOATCMP::FCMPE_Fn_S_S_FPIMM0(x)      => (x.0, true),
            _ => unsup!(),
        };

        let (n_loc, z_loc, c_loc, v_loc) = (
            self.nzcv_local(rctx, 0),
            self.nzcv_local(rctx, 1),
            self.nzcv_local(rctx, 2),
            self.nzcv_local(rctx, 3),
        );
        // Use fp_slot[31] as scratch to hold operand a.
        let fa = rctx.layout().local(self.fp_slot, 31);
        // fb: for non-zero comparisons, save b in fp_slot[30].
        let fb = rctx.layout().local(self.fp_slot, 30);

        // Store a = Fn
        self.emit_fp_get(ctx, rctx, tail_idx, rn(w))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(fa))?;

        // Store b = Fm or 0.0
        if cmp_zero {
            rctx.feed(ctx, tail_idx, &f64c(0.0))?;
        } else {
            self.emit_fp_get(ctx, rctx, tail_idx, rm(w))?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(fb))?;

        // V = unordered: a != a  OR b != b  (NaN detection)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fa))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fa))?;
        rctx.feed(ctx, tail_idx, &Instruction::F64Ne)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fb))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fb))?;
        rctx.feed(ctx, tail_idx, &Instruction::F64Ne)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Or)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        // N = a < b  (returns 0 for NaN — correct per ARM spec)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fa))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fb))?;
        rctx.feed(ctx, tail_idx, &Instruction::F64Lt)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = a == b  (returns 0 for NaN — correct)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fa))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fb))?;
        rctx.feed(ctx, tail_idx, &Instruction::F64Eq)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = a >= b  OR unordered  (f64.ge returns 0 for NaN, so OR with V fixes it)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fa))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(fb))?;
        rctx.feed(ctx, tail_idx, &Instruction::F64Ge)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(v_loc))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Or)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;

        Ok(())
    }

    // ── FLOATSEL (FCSEL — conditional FP select) ──────────────────────────────

    pub(super) fn translate_floatsel<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inner: &FLOATSEL,
        mnemonic: Mnemonic,
    ) -> Result<(), E> {
        macro_rules! unsup {
            () => {{ rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                     self.unsupported_insns.insert(alloc::format!("{:?}", mnemonic));
                     return Ok(()); }};
        }
        let w: u32 = match inner {
            FLOATSEL::FCSEL_Fd_Fn_Fm_COND(x)              => x.0,
            FLOATSEL::FCSEL_Fd_S_S_Fn_S_S_Fm_S_S_COND(x)  => x.0,
            _ => unsup!(),
        };
        let dest = rd(w);
        let cond = field(w, 12, 15);

        // WASM select: [true_val, false_val, condition] → picks true if condition≠0
        self.emit_fp_get(ctx, rctx, tail_idx, rn(w))?;
        self.emit_fp_get(ctx, rctx, tail_idx, rm(w))?;
        let (n, z, c, v) = (
            self.nzcv_local(rctx, 0), self.nzcv_local(rctx, 1),
            self.nzcv_local(rctx, 2), self.nzcv_local(rctx, 3),
        );
        emit_cond(ctx, &mut FedContext::new(rctx, tail_idx), cond, n, z, c, v)?;
        rctx.feed(ctx, tail_idx, &Instruction::Select)?;
        self.emit_fp_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }
}
