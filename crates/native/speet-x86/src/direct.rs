//! i686 instruction translation — iced-x86 decode + Flag call/ret.

use crate::*;
use iced_x86::{Decoder, DecoderOptions, Instruction as IxInst, Mnemonic, OpKind, Register};
use speet_traps::{InstructionInfo, JumpInfo, JumpKind, TrapAction};
use wasm_encoder::{Instruction, MemArg, ValType};
use wax_core::build::{InstructionOperatorSink, InstructionOperatorSource, InstructionSource};
use yecta::{JumpCallParams, LocalDeclarator};

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

impl<Context, E> X86_32Recompiler<Context, E> {
    fn init_function<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _eip: u64,
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

    /// Translate a block of i686 bytes starting at `start_eip`.
    pub fn translate_bytes<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        bytes: &[u8],
        start_eip: u64,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        self.text_len = bytes.len() as u32;
        let mut dec = Decoder::with_ip(32, bytes, start_eip, DecoderOptions::NONE);
        let mut next_pos: usize = 0;
        while next_pos < bytes.len() {
            dec.set_position(next_pos).expect("next_pos in range");
            let eip = start_eip + next_pos as u64;
            dec.set_ip(eip);
            if !dec.can_decode() {
                break;
            }
            let inst = dec.decode();
            let inst_len = inst.len() as u32;

            if let Some(gate) = &self.slot_assigner {
                if gate.slot_for_pc(eip).is_none() {
                    next_pos += 1;
                    continue;
                }
            }

            let tail_idx = self.init_function(ctx, rctx, eip, inst_len, f)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(eip as i64))?;
            for instr in rctx.layout().emit_set(self.eip_slot, 0) {
                rctx.feed(ctx, tail_idx, &instr)?;
            }

            {
                let insn_info = InstructionInfo {
                    pc: eip,
                    len: inst_len,
                    arch: ArchTag::Other,
                    class: Self::classify_mnemonic(inst.mnemonic()),
                };
                if rctx.on_instruction(&insn_info, ctx)? == TrapAction::Skip {
                    next_pos += 1;
                    continue;
                }
            }

            if inst.is_invalid() {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                self.note_unsupported("invalid");
            } else {
                self.translate_one(ctx, rctx, tail_idx, &inst)?;
            }
            next_pos += 1;
        }
        let _ = rctx.seal_remaining(ctx);
        Ok(())
    }

    fn translate_one<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        match inst.mnemonic() {
            Mnemonic::Mov => self.handle_mov(ctx, rctx, tail_idx, inst),
            Mnemonic::Add => self.handle_binop(ctx, rctx, tail_idx, inst, Instruction::I64Add),
            Mnemonic::Sub => self.handle_binop(ctx, rctx, tail_idx, inst, Instruction::I64Sub),
            Mnemonic::And => self.handle_binop(ctx, rctx, tail_idx, inst, Instruction::I64And),
            Mnemonic::Or => self.handle_binop(ctx, rctx, tail_idx, inst, Instruction::I64Or),
            Mnemonic::Xor => self.handle_binop(ctx, rctx, tail_idx, inst, Instruction::I64Xor),
            Mnemonic::Push => self.handle_push(ctx, rctx, tail_idx, inst),
            Mnemonic::Pop => self.handle_pop(ctx, rctx, tail_idx, inst),
            Mnemonic::Jmp => self.handle_jmp(ctx, rctx, tail_idx, inst),
            Mnemonic::Call => self.handle_call(ctx, rctx, tail_idx, inst),
            Mnemonic::Ret => self.handle_ret(ctx, rctx, tail_idx, inst),
            Mnemonic::Je
            | Mnemonic::Jne
            | Mnemonic::Jb
            | Mnemonic::Jae
            | Mnemonic::Jbe
            | Mnemonic::Ja
            | Mnemonic::Jl
            | Mnemonic::Jge
            | Mnemonic::Jle
            | Mnemonic::Jg => self.handle_jcc(ctx, rctx, tail_idx, inst),
            _ => {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                self.note_unsupported(&alloc::format!("{:?}", inst.mnemonic()));
                Ok(())
            }
        }
    }

    fn mask32<F>(
        &self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
    ) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xffff_ffff))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)
    }

    fn handle_mov<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        match (inst.op0_kind(), inst.op1_kind()) {
            (OpKind::Register, OpKind::Immediate32 | OpKind::Immediate8to32 | OpKind::Immediate16) => {
                let Some(dst) = Self::gpr_index(inst.op0_register()) else {
                    return self.unsup(ctx, rctx, tail_idx, "mov.reg");
                };
                rctx.feed(
                    ctx,
                    tail_idx,
                    &Instruction::I64Const(inst.immediate32() as i64),
                )?;
                self.mask32(ctx, rctx, tail_idx)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, dst)
            }
            (OpKind::Register, OpKind::Register) => {
                let Some(dst) = Self::gpr_index(inst.op0_register()) else {
                    return self.unsup(ctx, rctx, tail_idx, "mov.rr");
                };
                let Some(src) = Self::gpr_index(inst.op1_register()) else {
                    return self.unsup(ctx, rctx, tail_idx, "mov.rr");
                };
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
                self.emit_gpr_set(ctx, rctx, tail_idx, dst)
            }
            (OpKind::Register, OpKind::Memory) => {
                let Some(dst) = Self::gpr_index(inst.op0_register()) else {
                    return self.unsup(ctx, rctx, tail_idx, "mov.rm");
                };
                self.emit_mem_addr(ctx, rctx, tail_idx, inst)?;
                rctx.feed(
                    ctx,
                    tail_idx,
                    &Instruction::I64Load32U(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }),
                )?;
                self.emit_gpr_set(ctx, rctx, tail_idx, dst)
            }
            (OpKind::Memory, OpKind::Register) => {
                let Some(src) = Self::gpr_index(inst.op1_register()) else {
                    return self.unsup(ctx, rctx, tail_idx, "mov.mr");
                };
                self.emit_mem_addr(ctx, rctx, tail_idx, inst)?;
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
                rctx.feed(
                    ctx,
                    tail_idx,
                    &Instruction::I64Store32(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }),
                )
            }
            _ => self.unsup(ctx, rctx, tail_idx, "mov.form"),
        }
    }

    fn handle_binop<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
        op: Instruction,
    ) -> Result<(), E> {
        if inst.op0_kind() != OpKind::Register {
            return self.unsup(ctx, rctx, tail_idx, "binop.dst");
        }
        let Some(dst) = Self::gpr_index(inst.op0_register()) else {
            return self.unsup(ctx, rctx, tail_idx, "binop.reg");
        };
        // Resolve op1 before emitting so mid-instruction garbage never leaves
        // a partial value stack (breaks FlagSpec `(regs…)->(regs…,i32)`).
        let src_reg = match inst.op1_kind() {
            OpKind::Register => match Self::gpr_index(inst.op1_register()) {
                Some(s) => Some(s),
                None => return self.unsup(ctx, rctx, tail_idx, "binop.src"),
            },
            OpKind::Immediate32
            | OpKind::Immediate8to32
            | OpKind::Immediate8
            | OpKind::Immediate16 => Option::None,
            _ => return self.unsup(ctx, rctx, tail_idx, "binop.imm"),
        };
        self.emit_gpr_get(ctx, rctx, tail_idx, dst)?;
        if let Some(src) = src_reg {
            self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
        } else {
            rctx.feed(
                ctx,
                tail_idx,
                &Instruction::I64Const(inst.immediate32() as i64),
            )?;
        }
        rctx.feed(ctx, tail_idx, &op)?;
        self.mask32(ctx, rctx, tail_idx)?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dst)
    }

    fn handle_push<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        // ESP -= 4; [ESP] = src
        self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(4))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
        self.emit_gpr_set(ctx, rctx, tail_idx, 4)?;
        self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
        match inst.op0_kind() {
            OpKind::Register => {
                let Some(src) = Self::gpr_index(inst.op0_register()) else {
                    return self.unsup(ctx, rctx, tail_idx, "push.reg");
                };
                self.emit_gpr_get(ctx, rctx, tail_idx, src)?;
            }
            OpKind::Immediate32 | OpKind::Immediate8to32 | OpKind::Immediate8 => {
                rctx.feed(
                    ctx,
                    tail_idx,
                    &Instruction::I64Const(inst.immediate32() as i64),
                )?;
            }
            _ => return self.unsup(ctx, rctx, tail_idx, "push.form"),
        }
        rctx.feed(
            ctx,
            tail_idx,
            &Instruction::I64Store32(MemArg {
                offset: 0,
                align: 2,
                memory_index: 0,
            }),
        )
    }

    fn handle_pop<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        let Some(dst) = (match inst.op0_kind() {
            OpKind::Register => Self::gpr_index(inst.op0_register()),
            _ => None,
        }) else {
            return self.unsup(ctx, rctx, tail_idx, "pop.reg");
        };
        self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
        rctx.feed(
            ctx,
            tail_idx,
            &Instruction::I64Load32U(MemArg {
                offset: 0,
                align: 2,
                memory_index: 0,
            }),
        )?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dst)?;
        self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(4))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
        self.emit_gpr_set(ctx, rctx, tail_idx, 4)
    }

    fn handle_jmp<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        let total = rctx.locals_mark().total_locals;
        match inst.op0_kind() {
            OpKind::NearBranch32 | OpKind::NearBranch16 => {
                let target = inst.near_branch32() as u64;
                {
                    let info = JumpInfo::direct(inst.ip(), target, JumpKind::DirectJump);
                    if rctx.on_jump(&info, ctx)? == TrapAction::Skip {
                        return Ok(());
                    }
                }
                match self.eip_to_func_idx(target) {
                    Some(f) => rctx.jmp(ctx, tail_idx, f, total),
                    None => rctx.oob_jump(ctx, tail_idx, target, total),
                }
            }
            _ => self.unsup(ctx, rctx, tail_idx, "jmp.form"),
        }
    }

    fn handle_jcc<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        // Smoke: fall through only (condition flags not modeled yet).
        let target = inst.near_branch32() as u64;
        {
            let info = JumpInfo::direct(inst.ip(), target, JumpKind::ConditionalBranch);
            if rctx.on_jump(&info, ctx)? == TrapAction::Skip {
                return Ok(());
            }
        }
        self.note_unsupported("jcc.flags_stub");
        Ok(())
    }

    fn handle_call<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        let total = rctx.locals_mark().total_locals;
        let return_addr = inst.next_ip();
        let target = match inst.op0_kind() {
            OpKind::NearBranch32 | OpKind::NearBranch16 => inst.near_branch32() as u64,
            _ => return self.unsup(ctx, rctx, tail_idx, "call.form"),
        };

        // Push return address (4 bytes)
        self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(4))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
        self.emit_gpr_set(ctx, rctx, tail_idx, 4)?;
        self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
        rctx.feed(
            ctx,
            tail_idx,
            &Instruction::I64Store32(MemArg {
                offset: 0,
                align: 2,
                memory_index: 0,
            }),
        )?;

        let use_speculative =
            self.enable_speculative_calls && rctx.escape().is_native_stack();

        if use_speculative {
            let escape = rctx.escape();
            let Some(target_func) = self.eip_to_func_idx(target) else {
                rctx.oob_jump(ctx, tail_idx, target, total)?;
                return Ok(());
            };
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

        {
            let info = JumpInfo::direct(inst.ip(), target, JumpKind::Call);
            if rctx.on_jump(&info, ctx)? == TrapAction::Skip {
                return Ok(());
            }
        }
        match self.eip_to_func_idx(target) {
            Some(f) => rctx.jmp(ctx, tail_idx, f, total),
            None => rctx.oob_jump(ctx, tail_idx, target, total),
        }
    }

    fn handle_ret<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        let total = rctx.locals_mark().total_locals;
        let use_speculative =
            self.enable_speculative_calls && rctx.escape().is_native_stack();

        if use_speculative {
            let escape = rctx.escape();
            // Compare [ESP] with expected_ra
            self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
            rctx.feed(
                ctx,
                tail_idx,
                &Instruction::I64Load32U(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }),
            )?;
            for instr in rctx.layout().emit_get(self.expected_ra_slot, 0) {
                rctx.feed(ctx, tail_idx, &instr)?;
            }
            rctx.feed(ctx, tail_idx, &Instruction::I64Eq)?;
            rctx.feed(ctx, tail_idx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
            // pop
            self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            self.emit_gpr_set(ctx, rctx, tail_idx, 4)?;
            match escape {
                yecta::CallEscape::Flag => rctx.ret_flag(ctx, tail_idx, total, false)?,
                yecta::CallEscape::Exception(_) | yecta::CallEscape::Jump => {
                    for p in 0..total {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(p))?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::Return)?;
                }
            }
            rctx.feed(ctx, tail_idx, &Instruction::Else)?;
            match escape {
                yecta::CallEscape::Exception(tag) => rctx.ret(ctx, tail_idx, total, tag)?,
                yecta::CallEscape::Flag => rctx.ret_flag(ctx, tail_idx, total, true)?,
                yecta::CallEscape::Jump => unreachable!(),
            }
            rctx.feed(ctx, tail_idx, &Instruction::End)?;
            return Ok(());
        }

        // Non-speculative: pop target into tmp, ESP+=4, indirect jump
        let tmp = rctx.layout().local(self.tmp_slot, 0);
        self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
        rctx.feed(
            ctx,
            tail_idx,
            &Instruction::I64Load32U(MemArg {
                offset: 0,
                align: 2,
                memory_index: 0,
            }),
        )?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(tmp))?;
        self.emit_gpr_get(ctx, rctx, tail_idx, 4)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(4))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
        self.emit_gpr_set(ctx, rctx, tail_idx, 4)?;
        {
            let info = JumpInfo::indirect(inst.ip(), tmp, JumpKind::Return);
            if rctx.on_jump(&info, ctx)? == TrapAction::Skip {
                return Ok(());
            }
        }
        let idx_snip = RetIdxSnippet {
            tmp_local: tmp,
            base_eip: self.base_eip,
            base_func_offset: rctx.base_func_offset(),
        };
        let params = JumpCallParams::indirect_jump(&idx_snip, total, rctx.pool());
        rctx.ji_with_params(ctx, tail_idx, params)
    }

    fn emit_mem_addr<F>(
        &self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        let base = inst.memory_base();
        let index = inst.memory_index();
        let scale = inst.memory_index_scale() as i64;
        let disp = inst.memory_displacement32() as i32 as i64;

        if base != Register::None {
            let Some(b) = Self::gpr_index(base) else {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
                return Ok(());
            };
            self.emit_gpr_get(ctx, rctx, tail_idx, b)?;
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        }
        if index != Register::None {
            if let Some(i) = Self::gpr_index(index) {
                self.emit_gpr_get(ctx, rctx, tail_idx, i)?;
                if scale != 1 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(scale))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            }
        }
        if disp != 0 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(disp))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
        }
        Ok(())
    }

    fn unsup<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        name: &str,
    ) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
        self.note_unsupported(name);
        Ok(())
    }
}

struct RetIdxSnippet {
    tmp_local: u32,
    base_eip: u64,
    base_func_offset: u32,
}

impl<Context, E> InstructionSource<Context, E> for RetIdxSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalGet(self.tmp_local))?;
        sink.instruction(ctx, &Instruction::I64Const(self.base_eip as i64))?;
        sink.instruction(ctx, &Instruction::I64Sub)?;
        sink.instruction(ctx, &Instruction::I64Const(self.base_func_offset as i64))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        Ok(())
    }
}

impl<Context, E> InstructionOperatorSource<Context, E> for RetIdxSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        InstructionSource::emit_instruction(self, ctx, sink)
    }
}
