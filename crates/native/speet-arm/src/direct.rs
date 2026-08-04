//! AArch32 instruction translation — A32 decode + control-flow / ALU / LDR/STR.

use crate::*;
use wax_core::build::{InstructionOperatorSink, InstructionOperatorSource, InstructionSource};
use yecta::{JumpCallParams, LocalDeclarator};

/// Sets `expected_ra` to a constant link address for speculative BL.
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

/// `(Rm & ~1 - base_pc) >> 2 + base_func_offset` — clears Thumb bit before indexing.
struct ArmIndirectTarget {
    gpr_local: u32,
    text_base: speet_link_core::TextBaseSnippet,
    base_func_offset: u32,
}

impl ArmIndirectTarget {
    fn from_constant_base(gpr_local: u32, base_pc: u64, base_func_offset: u32) -> Self {
        Self {
            gpr_local,
            text_base: speet_link_core::TextBaseSnippet::new(
                speet_link_core::TextBaseSource::Constant(base_pc),
            ),
            base_func_offset,
        }
    }
}

impl<Context, E> InstructionSource<Context, E> for ArmIndirectTarget {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalGet(self.gpr_local))?;
        sink.instruction(ctx, &Instruction::I64Const(!1i64))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        wax_core::build::InstructionSource::emit_instruction(&self.text_base, ctx, sink)?;
        sink.instruction(ctx, &Instruction::I64Sub)?;
        sink.instruction(ctx, &Instruction::I64Const(2))?;
        sink.instruction(ctx, &Instruction::I64ShrU)?;
        sink.instruction(ctx, &Instruction::I64Const(self.base_func_offset as i64))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        Ok(())
    }
}

impl<Context, E> InstructionOperatorSource<Context, E> for ArmIndirectTarget {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        InstructionSource::emit_instruction(self, ctx, sink)
    }
}

impl<Context, E> ArmRecompiler<Context, E> {
    fn init_function<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _pc: u64,
        _inst_len: u32,
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
        // A32: one next_with step per 4-byte instruction (1 function slot).
        rctx.next_with(ctx, fn_type, 1)
    }

    /// Translate a block of A32 bytes starting at `start_pc`.
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

            let class = classify_a32(word);
            let tail_idx = self.init_function(ctx, rctx, pc, 4, f).map_err(|_| ())?;

            // Keep architectural PC (r15) in sync with the decode slot.
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(pc as i64))
                .map_err(|_| ())?;
            self.emit_gpr_set(ctx, rctx, tail_idx, 15).map_err(|_| ())?;

            {
                let insn_info = InstructionInfo {
                    pc,
                    len: 4,
                    arch: ArchTag::Other,
                    class,
                };
                if rctx.on_instruction(&insn_info, ctx).map_err(|_| ())? == TrapAction::Skip {
                    offset += 4;
                    continue;
                }
            }

            self.translate_a32(ctx, rctx, tail_idx, pc, word)
                .map_err(|_| ())?;
            offset += 4;
        }
        let _ = rctx.seal_remaining(ctx);
        Ok(offset)
    }

    /// Thumb-2 stub: Phase 4 smoke is A32-only. Returns a diagnostic name for
    /// common Thumb encodings without claiming translation support.
    pub fn thumb_unsupported_name(halfword: u16) -> &'static str {
        if (halfword & 0xf800) == 0xf000 || (halfword & 0xf800) == 0xe800 {
            "thumb32.bl/blx"
        } else if (halfword >> 13) == 0b000 {
            "thumb16.shift/add/sub"
        } else if (halfword >> 13) == 0b001 {
            "thumb16.mov/cmp/add/sub.imm"
        } else if (halfword >> 12) == 0b1101 {
            "thumb16.b.cond"
        } else {
            "thumb.unsupported"
        }
    }

    fn translate_a32<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        pc: u64,
        word: u32,
    ) -> Result<(), E> {
        let total = rctx.locals_mark().total_locals;
        let cond = (word >> 28) & 0xf;
        // Smoke path: only always-execute (AL) or unconditional (NV unused here).
        if cond != 0xe && cond != 0xf {
            rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
            self.note_unsupported("a32.cond");
            return Ok(());
        }

        // BX / BLX Rm
        if (word & 0x0FFF_FFD0) == 0x012F_FF10 {
            let rm = word & 0xf;
            let is_blx = (word & 0x20) != 0;
            return self.translate_bx(ctx, rctx, tail_idx, pc, rm, is_blx, total);
        }

        // B / BL
        if (word >> 25) & 0x7 == 0b101 {
            let is_bl = (word >> 24) & 1 != 0;
            let imm24 = word & 0x00FF_FFFF;
            let off = sign_ext24(imm24) * 4 + 8; // A32 PC is insn+8
            let target = pc.wrapping_add_signed(off);
            return self.translate_b_bl(ctx, rctx, tail_idx, pc, target, is_bl, total);
        }

        // LDR / STR immediate (offset addressing, word)
        if (word >> 26) & 0x3 == 0b01 && (word >> 25) & 1 == 0 {
            return self.translate_ldst_imm(ctx, rctx, tail_idx, word);
        }

        // Data-processing
        if (word >> 26) & 0x3 == 0b00 {
            // Exclude multiply / misc that share the 00xx space (bit4=1, bit7=1).
            if (word >> 25) & 1 == 0 && (word >> 4) & 1 == 1 && (word >> 7) & 1 == 1 {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                self.note_unsupported("a32.mul_misc");
                return Ok(());
            }
            return self.translate_dp(ctx, rctx, tail_idx, word);
        }

        rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
        self.note_unsupported(&alloc::format!("a32.{:08x}", word));
        Ok(())
    }

    fn translate_b_bl<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        pc: u64,
        target: u64,
        is_bl: bool,
        total: u32,
    ) -> Result<(), E> {
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
            self.emit_gpr_set(ctx, rctx, tail_idx, 14)?; // LR
            let expected_ra_snippet = ExpectedRaSnippet { return_addr };
            let params = match escape {
                yecta::CallEscape::Exception(tag) => {
                    JumpCallParams::call(target_func, total, tag, rctx.pool())
                }
                yecta::CallEscape::Flag => {
                    JumpCallParams::call_flag(target_func, total, rctx.pool())
                }
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
            self.emit_gpr_set(ctx, rctx, tail_idx, 14)?;
        }
        {
            let kind = if is_bl {
                JumpKind::Call
            } else {
                JumpKind::DirectJump
            };
            let info = JumpInfo::direct(pc, target, kind);
            if rctx.on_jump(&info, ctx)? == TrapAction::Skip {
                return Ok(());
            }
        }
        match self.pc_to_func_idx(target) {
            Some(f_idx) => rctx.jmp(ctx, tail_idx, f_idx, total)?,
            None => rctx.oob_jump(ctx, tail_idx, target, total)?,
        }
        Ok(())
    }

    fn translate_bx<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        pc: u64,
        rm: u32,
        is_blx: bool,
        total: u32,
    ) -> Result<(), E> {
        let return_addr = pc + 4;
        let is_ret = !is_blx && rm == 14;

        let use_speculative_ret = is_ret
            && self.enable_speculative_calls
            && rctx.escape().is_native_stack();

        if use_speculative_ret {
            let gpr_local = rctx.layout().local(self.gpr_slot, 14);
            {
                let info = JumpInfo::indirect(pc, gpr_local, JumpKind::Return);
                if rctx.on_jump(&info, ctx)? == TrapAction::Skip {
                    return Ok(());
                }
            }
            let escape = rctx.escape();
            self.emit_gpr_get(ctx, rctx, tail_idx, 14)?;
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

        if is_blx {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
            self.emit_gpr_set(ctx, rctx, tail_idx, 14)?;
        }

        let gpr_local = rctx.layout().local(self.gpr_slot, rm);
        {
            let kind = if is_blx {
                JumpKind::IndirectCall
            } else if is_ret {
                JumpKind::Return
            } else {
                JumpKind::IndirectJump
            };
            let info = JumpInfo::indirect(pc, gpr_local, kind);
            if rctx.on_jump(&info, ctx)? == TrapAction::Skip {
                return Ok(());
            }
        }
        let target_snippet =
            ArmIndirectTarget::from_constant_base(gpr_local, self.base_pc, rctx.base_func_offset());
        let params = JumpCallParams::indirect_jump(&target_snippet, total, rctx.pool());
        rctx.ji_with_params(ctx, tail_idx, params)?;
        Ok(())
    }

    fn translate_dp<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        word: u32,
    ) -> Result<(), E> {
        let opcode = (word >> 21) & 0xf;
        let i_bit = (word >> 25) & 1;
        let rn = (word >> 16) & 0xf;
        let rd = (word >> 12) & 0xf;
        let op2 = word & 0xfff;

        // Operand 2
        if i_bit == 1 {
            let imm = expand_imm12(op2) as i64;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm))?;
        } else {
            let rm = op2 & 0xf;
            // Only unshifted Rm for smoke.
            if (op2 >> 4) & 0xff != 0 {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                self.note_unsupported("a32.dp.shifted_rm");
                return Ok(());
            }
            self.emit_gpr_get(ctx, rctx, tail_idx, rm)?;
        }
        // Mask to 32-bit
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xffff_ffff))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;

        match opcode {
            0xD => {
                // MOV Rd, op2 — Rn ignored
                self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
            }
            0x4 => {
                // ADD Rd, Rn, op2 (commutative — stack has op2, push Rn)
                self.emit_gpr_get(ctx, rctx, tail_idx, rn)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xffff_ffff))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
            }
            0x2 => {
                // SUB Rd, Rn, op2 — rebuild so stack is [Rn, op2]
                rctx.feed(ctx, tail_idx, &Instruction::Drop)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, rn)?;
                if i_bit == 1 {
                    rctx.feed(
                        ctx,
                        tail_idx,
                        &Instruction::I64Const(expand_imm12(op2) as i64),
                    )?;
                } else {
                    self.emit_gpr_get(ctx, rctx, tail_idx, op2 & 0xf)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xffff_ffff))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
            }
            0x0 => {
                // AND
                self.emit_gpr_get(ctx, rctx, tail_idx, rn)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
            }
            0xC => {
                // ORR
                self.emit_gpr_get(ctx, rctx, tail_idx, rn)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
            }
            0x1 => {
                // EOR
                self.emit_gpr_get(ctx, rctx, tail_idx, rn)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
            }
            _ => {
                rctx.feed(ctx, tail_idx, &Instruction::Drop)?;
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                self.note_unsupported(&alloc::format!("a32.dp.op{opcode:x}"));
            }
        }
        Ok(())
    }

    fn translate_ldst_imm<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        word: u32,
    ) -> Result<(), E> {
        let p = (word >> 24) & 1;
        let u = (word >> 23) & 1;
        let b = (word >> 22) & 1;
        let w = (word >> 21) & 1;
        let l = (word >> 20) & 1;
        let rn = (word >> 16) & 0xf;
        let rd = (word >> 12) & 0xf;
        let imm12 = (word & 0xfff) as i64;

        // Smoke: offset addressing only (P=1, W=0).
        if p != 1 || w != 0 {
            rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
            self.note_unsupported("a32.ldst.addr_mode");
            return Ok(());
        }

        // Address stays i64 for memory64 linear memory.
        self.emit_gpr_get(ctx, rctx, tail_idx, rn)?;
        if u == 1 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(imm12))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
        }
        
        if l == 1 {
            if b != 0 {
                rctx.feed(
                    ctx,
                    tail_idx,
                    &Instruction::I64Load8U(MemArg {
                        offset: 0,
                        align: 0,
                        memory_index: 0,
                    }),
                )?;
            } else {
                rctx.feed(
                    ctx,
                    tail_idx,
                    &Instruction::I64Load32U(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }),
                )?;
            }
            self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
        } else {
            self.emit_gpr_get(ctx, rctx, tail_idx, rd)?;
            if b != 0 {
                rctx.feed(
                    ctx,
                    tail_idx,
                    &Instruction::I64Store8(MemArg {
                        offset: 0,
                        align: 0,
                        memory_index: 0,
                    }),
                )?;
            } else {
                rctx.feed(
                    ctx,
                    tail_idx,
                    &Instruction::I64Store32(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }),
                )?;
            }
        }
        Ok(())
    }
}

fn classify_a32(word: u32) -> InsnClass {
    if (word & 0x0FFF_FFD0) == 0x012F_FF10 {
        if (word & 0xf) == 14 {
            InsnClass::RETURN | InsnClass::INDIRECT
        } else {
            InsnClass::BRANCH | InsnClass::INDIRECT
        }
    } else if (word >> 25) & 0x7 == 0b101 {
        if (word >> 24) & 1 != 0 {
            InsnClass::CALL
        } else {
            InsnClass::BRANCH
        }
    } else if (word >> 26) & 0x3 == 0b01 {
        InsnClass::MEMORY
    } else {
        InsnClass::OTHER
    }
}

#[inline]
fn sign_ext24(imm24: u32) -> i64 {
    let shift = 32 - 24;
    ((imm24 << shift) as i32 >> shift) as i64
}
