//! AArch64 instruction translation — dispatch and control-flow handlers.
//!
//! Submodules carry the per-category operation handlers:
//! - `helpers`  — bit-field extraction, bitmask decode, condition/indirect snippets
//! - `alu`      — integer ALU (ADDSUB, LOG, MOVEWIDE, BITFIELD, DP_2SRC/3SRC, CONDSEL, …)
//! - `mem`      — loads and stores (all addressing modes, pairs)
//! - `fp`       — floating-point (FLOATDP*, FLOAT2INT, FLOATCMP, FLOATSEL)

pub(crate) mod helpers;
mod alu;
mod mem;
mod fp;

pub(crate) use helpers::{A64Condition, A64IndirectTarget, CmpZeroCond};

use crate::*;
use helpers::*;
use disarm64::decoder_full::{
    ADDSUB_EXT, ADDSUB_IMM, ADDSUB_SHIFT, BITFIELD, BRANCH_IMM, BRANCH_REG,
    COMPBRANCH, CONDSEL, CONDBRANCH, DP_2SRC, DP_3SRC, EXCEPTION, IC_SYSTEM,
    LDST_IMM9, LDST_POS, LDST_REGOFF, LDST_UNSCALED, LDSTPAIR_INDEXED, LDSTPAIR_OFF,
    LOG_IMM, LOG_SHIFT, MOVEWIDE, PCRELADDR,
    FLOAT2INT, FLOATCMP, FLOATDP1, FLOATDP2, FLOATDP3, FLOATIMM, FLOATSEL,
    Operation,
};
use alloc::collections::BTreeMap;
use wax_core::build::{InstructionOperatorSink, InstructionOperatorSource, InstructionSource};
use yecta::{FuncIdx, JumpCallParams, LocalDeclarator, Snippet, Target};

/// Sets `expected_ra` to a constant link address for speculative BL/BLR.
struct ExpectedRaSnippet {
    return_addr: u64,
}

impl<Context, E> InstructionSource<Context, E> for ExpectedRaSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::I64Const(self.return_addr as i64))
    }
}

impl<Context, E> InstructionOperatorSource<Context, E> for ExpectedRaSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        InstructionSource::emit_instruction(self, ctx, sink)
    }
}

impl<'cb, 'ctx, Context, E> AArch64Recompiler<'cb, 'ctx, Context, E> {
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
        // len=1: each AArch64 instruction is exactly one next_with step (4 bytes).
        // inst_len bytes, but the fall-through distance is 1 function slot, not 4.
        let _ = inst_len;
        rctx.next_with(ctx, fn_type, 1)
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
        #[cfg(feature = "logging")]
        log::trace!(target: "speet::aarch64", "translate_bytes pc={:#x} len={}", start_pc, bytes.len());
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

            let pc_local = rctx.layout().local(self.pc_slot, 0);
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(pc as i64)).map_err(|_| ())?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(pc_local)).map_err(|_| ())?;

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
        // Seal any functions not terminated by a branch (e.g. the last instruction
        // of a region that isn't a branch) with unreachable + End so the WASM
        // validator sees properly closed function bodies.
        let _ = rctx.seal_remaining(ctx);
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
        let mn    = opcode.mnemonic;

        macro_rules! unsup {
            () => {{
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                self.unsupported_insns.insert(alloc::format!("{:?}", mn));
                return Ok(());
            }};
        }

        match opcode.operation {
            // ── Integer ALU — immediate ────────────────────────────────────────
            Operation::ADDSUB_IMM(ref i) => self.translate_addsub_imm(ctx, rctx, tail_idx, i, mn)?,

            // ── Integer ALU — shifted register ────────────────────────────────
            Operation::ADDSUB_SHIFT(ref i) => self.translate_addsub_shift(ctx, rctx, tail_idx, i, mn)?,

            // ── Integer ALU — extended register ───────────────────────────────
            Operation::ADDSUB_EXT(ref i) => self.translate_addsub_ext(ctx, rctx, tail_idx, i, mn)?,

            // ── Logical — shifted register ────────────────────────────────────
            Operation::LOG_SHIFT(ref i) => self.translate_log_shift(ctx, rctx, tail_idx, i, mn)?,

            // ── Logical — bitmask immediate ───────────────────────────────────
            Operation::LOG_IMM(ref i) => self.translate_log_imm(ctx, rctx, tail_idx, i, mn)?,

            // ── Move-wide ─────────────────────────────────────────────────────
            Operation::MOVEWIDE(ref i) => self.translate_movewide(ctx, rctx, tail_idx, i, mn)?,

            // ── Bitfield (shifts, sign/zero-extend) ───────────────────────────
            Operation::BITFIELD(ref i) => self.translate_bitfield(ctx, rctx, tail_idx, i, mn)?,

            // ── Integer multiply/divide ────────────────────────────────────────
            Operation::DP_2SRC(ref i) => self.translate_dp2src(ctx, rctx, tail_idx, i, mn)?,
            Operation::DP_3SRC(ref i) => self.translate_dp3src(ctx, rctx, tail_idx, i, mn)?,

            // ── Conditional select ────────────────────────────────────────────
            Operation::CONDSEL(ref i) => self.translate_condsel(ctx, rctx, tail_idx, i, mn)?,

            // ── PC-relative address ───────────────────────────────────────────
            Operation::PCRELADDR(ref i) => self.translate_pcreladdr(ctx, rctx, tail_idx, pc, i, mn)?,

            // ── Exception (BRK / SVC) ─────────────────────────────────────────
            Operation::EXCEPTION(ref i) => {
                self.translate_exception(ctx, rctx, tail_idx, pc, i, mn)?
            }

            // ── System — MRS/MSR NZCV ─────────────────────────────────────────
            Operation::IC_SYSTEM(ref i) => self.translate_ic_system(ctx, rctx, tail_idx, i, mn)?,

            // ── Unconditional branch (B / BL) ─────────────────────────────────
            Operation::BRANCH_IMM(ref inner) => {
                let (w, is_bl) = match inner {
                    BRANCH_IMM::B_ADDR_PCREL26(x)  => (x.0, false),
                    BRANCH_IMM::BL_ADDR_PCREL26(x) => (x.0, true),
                    _ => unsup!(),
                };
                let off    = sign_ext26(imm26(w)) * 4;
                let target = pc.wrapping_add_signed(off);
                let return_addr = pc + 4;

                let use_speculative = is_bl
                    && self.enable_speculative_calls
                    && rctx.escape().is_native_stack();

                if use_speculative {
                    let escape = rctx.escape();
                    let Some(target_func) = self.pc_to_func_idx(target) else {
                        rctx.oob_jump(ctx, tail_idx, target, total)?;
                        return Ok(());
                    };
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, 30)?;
                    let expected_ra_snippet = ExpectedRaSnippet { return_addr };
                    let params = match escape {
                        yecta::CallEscape::Exception(tag) => JumpCallParams::call(
                            target_func, total, tag, rctx.pool(),
                        ),
                        yecta::CallEscape::Flag => JumpCallParams::call_flag(
                            target_func, total, rctx.pool(),
                        ),
                        yecta::CallEscape::Jump => unreachable!(),
                    }
                    .with_fixup(
                        rctx.layout().local(self.expected_ra_slot, 0),
                        &expected_ra_snippet,
                    );
                    rctx.ji_with_params(ctx, tail_idx, params)?;
                    return Ok(());
                }

                if is_bl {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, 30)?;
                }
                {
                    let kind = if is_bl { JumpKind::Call } else { JumpKind::DirectJump };
                    let info = JumpInfo::direct(pc, target, kind);
                    if rctx.on_jump(&info, ctx)? == TrapAction::Skip { return Ok(()); }
                }
                match self.pc_to_func_idx(target) {
                    Some(f_idx) => rctx.jmp(ctx, tail_idx, f_idx, total)?,
                    None        => rctx.oob_jump(ctx, tail_idx, target, total)?,
                }
            }

            // ── Branch to register (BR / BLR / RET) ───────────────────────────
            Operation::BRANCH_REG(ref inner) => {
                let (w, kind) = match inner {
                    BRANCH_REG::RET_Rn(x)  => (x.0, JumpKind::Return),
                    BRANCH_REG::BR_Rn(x)   => (x.0, JumpKind::IndirectJump),
                    BRANCH_REG::BLR_Rn(x)  => (x.0, JumpKind::IndirectCall),
                    _ => unsup!(),
                };
                let reg = rn(w);
                let return_addr = pc + 4;
                let is_blr = matches!(inner, BRANCH_REG::BLR_Rn(_));
                let is_ret = matches!(inner, BRANCH_REG::RET_Rn(_));

                // ABI RET (typically `RET` / `RET X30`): compare LR vs expected_ra.
                let use_speculative_ret = is_ret
                    && reg == 30
                    && self.enable_speculative_calls
                    && rctx.escape().is_native_stack();

                if use_speculative_ret {
                    let gpr_local = rctx.layout().local(self.gpr_slot, reg);
                    {
                        let info = JumpInfo::indirect(pc, gpr_local, JumpKind::Return);
                        if rctx.on_jump(&info, ctx)? == TrapAction::Skip { return Ok(()); }
                    }
                    let escape = rctx.escape();
                    self.emit_gpr_get(ctx, rctx, tail_idx, reg)?;
                    self.emit_expected_ra_get(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Eq)?;
                    rctx.feed(ctx, tail_idx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
                    match escape {
                        yecta::CallEscape::Flag => {
                            rctx.ret_flag(ctx, tail_idx, total, false)?;
                        }
                        yecta::CallEscape::Exception(_) | yecta::CallEscape::Jump => {
                            for p in 0..total {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(p))?;
                            }
                            rctx.feed(ctx, tail_idx, &Instruction::Return)?;
                        }
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::Else)?;
                    match escape {
                        yecta::CallEscape::Exception(tag) => {
                            rctx.ret(ctx, tail_idx, total, tag)?;
                        }
                        yecta::CallEscape::Flag => {
                            rctx.ret_flag(ctx, tail_idx, total, true)?;
                        }
                        yecta::CallEscape::Jump => unreachable!(),
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::End)?;
                    return Ok(());
                }

                let use_speculative_blr = is_blr
                    && self.enable_speculative_calls
                    && rctx.escape().is_native_stack();

                if use_speculative_blr {
                    let escape = rctx.escape();
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, 30)?;
                    let expected_ra_snippet = ExpectedRaSnippet { return_addr };
                    let gpr_local = rctx.layout().local(self.gpr_slot, reg);
                    let target_snippet = A64IndirectTarget::from_constant_base(
                        gpr_local,
                        self.base_pc,
                        rctx.base_func_offset(),
                    );
                    let mut fixups = BTreeMap::new();
                    fixups.insert(
                        rctx.layout().local(self.expected_ra_slot, 0),
                        &expected_ra_snippet as &(dyn yecta::Snippet<Context, E> + '_),
                    );
                    let params = JumpCallParams {
                        params: total,
                        fixups,
                        target: Target::Dynamic { idx: &target_snippet },
                        call: escape,
                        pool: rctx.pool(),
                        condition: None,
                        condition_hook: None,
                    };
                    rctx.ji_with_params(ctx, tail_idx, params)?;
                    return Ok(());
                }

                if is_blr {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, 30)?;
                }

                let gpr_local = rctx.layout().local(self.gpr_slot, reg);
                {
                    let info = JumpInfo::indirect(pc, gpr_local, kind);
                    if rctx.on_jump(&info, ctx)? == TrapAction::Skip { return Ok(()); }
                }

                let target_snippet = A64IndirectTarget::from_constant_base(
                    gpr_local,
                    self.base_pc,
                    rctx.base_func_offset(),
                );
                let params = JumpCallParams::indirect_jump(&target_snippet, total, rctx.pool());
                rctx.ji_with_params(ctx, tail_idx, params)?;
            }

            // ── Conditional branch — B.cond ────────────────────────────────────
            Operation::CONDBRANCH(ref inner) => {
                let w = match inner {
                    CONDBRANCH::B__ADDR_PCREL19(x)  => x.0,
                    CONDBRANCH::BC__ADDR_PCREL19(x) => x.0,
                    _ => unsup!(),
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
                    yecta::CallEscape::Jump,
                    rctx.pool(),
                    Some(&condition),
                )?;
            }

            // ── Compare-and-branch — CBZ / CBNZ ───────────────────────────────
            Operation::COMPBRANCH(ref inner) => {
                let (w, is_cbnz) = match inner {
                    COMPBRANCH::CBZ_Rt_ADDR_PCREL19(x)  => (x.0, false),
                    COMPBRANCH::CBNZ_Rt_ADDR_PCREL19(x) => (x.0, true),
                    _ => unsup!(),
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
                    yecta::CallEscape::Jump,
                    rctx.pool(),
                    Some(&condition),
                )?;
            }

            // ── Scalar loads/stores — unsigned immediate offset ────────────────
            Operation::LDST_POS(ref i) => self.translate_ldst_pos(ctx, rctx, tail_idx, i, mn)?,

            // ── Scalar loads/stores — pre/post-index ──────────────────────────
            Operation::LDST_IMM9(ref i) => self.translate_ldst_imm9(ctx, rctx, tail_idx, i, mn)?,

            // ── Scalar loads/stores — unscaled immediate (LDUR/STUR family) ────
            // Compiler-emitted stack-frame accesses (spills, negative-offset
            // locals/arrays relative to SP/FP) very commonly use this form
            // rather than the scaled `LDST_POS` — any C program with a
            // stack-allocated array or enough locals to spill will emit it.
            Operation::LDST_UNSCALED(ref i) => self.translate_ldst_unscaled(ctx, rctx, tail_idx, i, mn)?,

            // ── Scalar loads/stores — register offset ─────────────────────────
            Operation::LDST_REGOFF(ref i) => self.translate_ldst_regoff(ctx, rctx, tail_idx, i, mn)?,

            // ── Load/store pairs — signed offset ──────────────────────────────
            Operation::LDSTPAIR_OFF(ref i) => self.translate_ldstpair_off(ctx, rctx, tail_idx, i, mn)?,

            // ── Load/store pairs — pre/post-index ─────────────────────────────
            Operation::LDSTPAIR_INDEXED(ref i) => self.translate_ldstpair_indexed(ctx, rctx, tail_idx, i, mn)?,

            // ── Floating-point arithmetic ──────────────────────────────────────
            Operation::FLOATDP1(ref i) => self.translate_floatdp1(ctx, rctx, tail_idx, i, mn)?,
            Operation::FLOATDP2(ref i) => self.translate_floatdp2(ctx, rctx, tail_idx, i, mn)?,
            Operation::FLOATDP3(ref i) => self.translate_floatdp3(ctx, rctx, tail_idx, i, mn)?,
            Operation::FLOAT2INT(ref i) => self.translate_float2int(ctx, rctx, tail_idx, i, mn)?,
            Operation::FLOATIMM(ref i) => self.translate_floatimm(ctx, rctx, tail_idx, i, mn)?,
            Operation::FLOATCMP(ref i) => self.translate_floatcmp(ctx, rctx, tail_idx, i, mn)?,
            Operation::FLOATSEL(ref i) => self.translate_floatsel(ctx, rctx, tail_idx, i, mn)?,

            // ── Everything else ────────────────────────────────────────────────
            _ => unsup!(),
        }
        Ok(())
    }
}
