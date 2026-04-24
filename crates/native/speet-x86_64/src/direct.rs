use crate::*;

use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use wax_core::build::InstructionSource;
use yecta::layout::CellIdx;
use yecta::{FuncIdx, JumpCallParams};

use iced_x86::{Decoder, DecoderOptions, Instruction as IxInst, Mnemonic, OpKind, Register};

#[derive(Clone, Copy)]
enum ConditionType {
    ZF,
    NZF,
    SF_NE_OF,
    ZF_OR_SF_NE_OF,
    NZF_AND_SF_EQ_OF,
    SF_EQ_OF,
    CF,
    CF_OR_ZF,
    NCF_AND_NZF,
    NCF,
    SF,
    NSF,
    OF,
    NOF,
    PF,
    NPF,
}

// Struct to represent a condition that can be used as a Snippet
#[derive(Clone, Copy)]
struct ConditionSnippet {
    condition_type: ConditionType,
}

/// Snippet that computes function index from return address stored in local 23
struct ReturnAddressSnippet {
    base_rip: u64,
}

/// Snippet for setting expected_ra to a constant return address in speculative calls
struct ExpectedRaSnippet {
    return_addr: u64,
}

impl<Context, E> wax_core::build::InstructionSource<Context, E> for ExpectedRaSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        // x86_64 always uses 64-bit addresses
        sink.instruction(ctx, &Instruction::I64Const(self.return_addr as i64))?;
        Ok(())
    }
}

impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for ExpectedRaSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        // x86_64 always uses 64-bit addresses
        sink.instruction(ctx, &Instruction::I64Const(self.return_addr as i64))?;
        Ok(())
    }
}

impl<Context, E> wax_core::build::InstructionSource<Context, E> for ReturnAddressSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        // Load return address from local 23
        sink.instruction(ctx, &Instruction::LocalGet(23))?;

        // Subtract base_rip to get relative address: (return_addr - base_rip)
        sink.instruction(ctx, &Instruction::I64Const(self.base_rip as i64))?;
        sink.instruction(ctx, &Instruction::I64Sub)?;

        // Convert to function index (divide by 1 for x86_64, similar to rip_to_func_idx)
        // For x86_64, each function is 1 byte aligned (unlike RISC-V which is 2-byte aligned)
        sink.instruction(ctx, &Instruction::I32WrapI64)?;
        Ok(())
    }
}

impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for ReturnAddressSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        // Same logic as emit_instruction
        sink.instruction(ctx, &Instruction::LocalGet(23))?;
        sink.instruction(ctx, &Instruction::I64Const(self.base_rip as i64))?;
        sink.instruction(ctx, &Instruction::I64Sub)?;
        sink.instruction(ctx, &Instruction::I32WrapI64)?;
        Ok(())
    }
}

impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for ConditionSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        // For simple structs, we can delegate to emit_instruction
        self.emit_instruction(ctx, sink)
    }
}

impl<Context, E> wax_core::build::InstructionSource<Context, E> for ConditionSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        match self.condition_type {
            ConditionType::ZF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::ZF_LOCAL,
                    ),
                )?;
            }
            ConditionType::NZF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::SF_NE_OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::SF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
            }
            ConditionType::ZF_OR_SF_NE_OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::SF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
                sink.instruction(ctx, &Instruction::I32Or)?;
            }
            ConditionType::NZF_AND_SF_EQ_OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::SF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(ctx, &Instruction::I32And)?;
            }
            ConditionType::SF_EQ_OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::SF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::CF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::CF_LOCAL,
                    ),
                )?;
            }
            ConditionType::CF_OR_ZF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::CF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Or)?;
            }
            ConditionType::NCF_AND_NZF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::CF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(ctx, &Instruction::I32And)?;
            }
            ConditionType::NCF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::CF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::SF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::SF_LOCAL,
                    ),
                )?;
            }
            ConditionType::NSF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::SF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::OF_LOCAL,
                    ),
                )?;
            }
            ConditionType::NOF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::PF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::PF_LOCAL,
                    ),
                )?;
            }
            ConditionType::NPF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::PF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
        }
        Ok(())
    }
}

impl X86Recompiler {
    fn rip_to_func_idx<Context, E, F>(&self, rctx: &dyn ReactorContext<Context, E, FnType = F>, rip: u64) -> Option<FuncIdx> {
        if let Some(gate) = &self.slot_assigner {
            gate.slot_for_pc(rip).map(FuncIdx)
        } else {
            Some(FuncIdx((rip.wrapping_sub(self.base_rip)) as u32))
        }
    }

    fn init_function<Context, E, F>(
        &self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _rip: u64,
        inst_len: u32,
        _num_temps: u32,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, wasm_encoder::ValType)> + '_)) -> F
                  + '_),
    ) -> Result<usize, E> {
        let mark = rctx.locals_mark();
        rctx.layout_mut().rewind(&mark);
        rctx.declare_trap_locals();
        let _cell = rctx.alloc_cell();
        let fn_type = f(&mut rctx.layout().iter_since(&mark).collect::<alloc::vec::Vec<_>>().into_iter());
        rctx.next_with(ctx, fn_type, inst_len)
    }

    fn emit_memory_address<Context, E, F>(
        &self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
    ) -> Result<(), E> {
        use iced_x86::Register;
        let base  = inst.memory_base();
        let index = inst.memory_index();
        let scale = inst.memory_index_scale();
        let disp  = inst.memory_displacement64();
        let mut have_value = false;
        if base != Register::None {
            if let Some((local, _sz, _z, bit)) = Self::resolve_reg(base) {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(local))?;
                if bit > 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(bit as i64))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                }
                have_value = true;
            } else {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                return Ok(());
            }
        }
        if index != Register::None {
            if let Some((idx_local, _sz, _z, bit)) = Self::resolve_reg(index) {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(idx_local))?;
                if bit > 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(bit as i64))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                }
                if scale != 1 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(scale as i64))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                }
                if have_value {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                }
                have_value = true;
            } else {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                return Ok(());
            }
        }
        if disp != 0 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(disp as i64))?;
            if have_value {
                rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            }
            have_value = true;
        }
        if !have_value {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        }
        Ok(())
    }

    fn handle_memory_rmw<Context, E, F, Op>(
        &mut self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
        size_bits: u32,
        mut operation: Op,
    ) -> Result<Option<()>, E>
    where
        Op: FnMut(&mut Self, &mut Context, &dyn ReactorContext<Context, E, FnType = F>, usize) -> Result<(), E>,
    {
        use iced_x86::OpKind;
        self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
        self.emit_memory_load(ctx, rctx, tail_idx, size_bits, false)?;
        match inst.op1_kind() {
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to32 => {
                self.emit_i64_const(ctx, rctx, tail_idx, inst.immediate64() as i64)?;
            }
            OpKind::Register => {
                if let Some((r_local, r_size, _rz, bit)) = Self::resolve_reg(inst.op1_register()) {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r_local))?;
                    if bit > 0 {
                        self.emit_mask_shift_for_read(ctx, rctx, tail_idx, r_size, bit)?;
                    }
                    match r_size {
                        8  => { rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;   rctx.feed(ctx, tail_idx, &Instruction::I64And)?; }
                        16 => { rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF))?; rctx.feed(ctx, tail_idx, &Instruction::I64And)?; }
                        _  => {}
                    }
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        }
        operation(self, ctx, rctx, tail_idx)?;
        self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
        self.emit_memory_store(ctx, rctx, tail_idx, size_bits)?;
        Ok(Some(()))
    }

    fn handle_binary<Context, E, F, T>(
        &mut self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
        mut cb: T,
    ) -> Result<Option<()>, E>
    where
        T: FnMut(&mut Self, &mut Context, &dyn ReactorContext<Context, E, FnType = F>, usize, Operand, u32, u32, u32) -> Result<(), E>,
    {
        use iced_x86::OpKind;
        let op0 = inst.op0_kind();
        let op1 = inst.op1_kind();
        if op0 == OpKind::Memory {
            match op1 {
                OpKind::Immediate8
                | OpKind::Immediate16
                | OpKind::Immediate32
                | OpKind::Immediate64
                | OpKind::Immediate8to32 => {
                    self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
                    let imm = inst.immediate64() as i64;
                    self.emit_i64_const(ctx, rctx, tail_idx, imm)?;
                    let size_bits = match op1 {
                        OpKind::Immediate8 => 8, OpKind::Immediate16 => 16,
                        OpKind::Immediate8to32 | OpKind::Immediate32 => 32,
                        OpKind::Immediate64 => 64, _ => 64,
                    };
                    self.emit_memory_store(ctx, rctx, tail_idx, size_bits)?;
                    return Ok(Some(()));
                }
                OpKind::Register => {
                    if let Some((r_local, r_size, _rz, bit)) = Self::resolve_reg(inst.op1_register()) {
                        self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r_local))?;
                        if bit > 0 { self.emit_mask_shift_for_read(ctx, rctx, tail_idx, r_size, bit)?; }
                        match r_size {
                            8  => { rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;   rctx.feed(ctx, tail_idx, &Instruction::I64And)?; }
                            16 => { rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF))?; rctx.feed(ctx, tail_idx, &Instruction::I64And)?; }
                            _  => {}
                        }
                        self.emit_memory_store(ctx, rctx, tail_idx, r_size)?;
                        return Ok(Some(()));
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        }
        let dst_info = match op0 {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            _ => None,
        };
        let (dst_local, dst_size, _dst_zero_ext32, dst_bit_offset) = match dst_info {
            Some(v) => v,
            None => return Ok(None),
        };
        let src = match op1 {
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to32 => Operand::Imm(inst.immediate64() as i64),
            OpKind::Register => {
                if let Some((r_local, r_size, _z, bit)) = Self::resolve_reg(inst.op1_register()) {
                    Operand::RegWithSize(r_local, r_size, bit)
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };
        cb(self, ctx, rctx, tail_idx, src, dst_local, dst_size, dst_bit_offset)?;
        Ok(Some(()))
    }

    fn resolve_reg(reg: Register) -> Option<(u32, u32, bool, u32)> {
        match reg {
            Register::RAX => Some((0, 64, false, 0)),
            Register::RCX => Some((1, 64, false, 0)),
            Register::RDX => Some((2, 64, false, 0)),
            Register::RBX => Some((3, 64, false, 0)),
            Register::RSP => Some((4, 64, false, 0)),
            Register::RBP => Some((5, 64, false, 0)),
            Register::RSI => Some((6, 64, false, 0)),
            Register::RDI => Some((7, 64, false, 0)),
            Register::R8 => Some((8, 64, false, 0)),
            Register::R9 => Some((9, 64, false, 0)),
            Register::R10 => Some((10, 64, false, 0)),
            Register::R11 => Some((11, 64, false, 0)),
            Register::R12 => Some((12, 64, false, 0)),
            Register::R13 => Some((13, 64, false, 0)),
            Register::R14 => Some((14, 64, false, 0)),
            Register::R15 => Some((15, 64, false, 0)),
            Register::EAX => Some((0, 32, true, 0)),
            Register::ECX => Some((1, 32, true, 0)),
            Register::EDX => Some((2, 32, true, 0)),
            Register::EBX => Some((3, 32, true, 0)),
            Register::ESP => Some((4, 32, true, 0)),
            Register::EBP => Some((5, 32, true, 0)),
            Register::ESI => Some((6, 32, true, 0)),
            Register::EDI => Some((7, 32, true, 0)),
            Register::R8D => Some((8, 32, true, 0)),
            Register::R9D => Some((9, 32, true, 0)),
            Register::R10D => Some((10, 32, true, 0)),
            Register::R11D => Some((11, 32, true, 0)),
            Register::R12D => Some((12, 32, true, 0)),
            Register::R13D => Some((13, 32, true, 0)),
            Register::R14D => Some((14, 32, true, 0)),
            Register::R15D => Some((15, 32, true, 0)),
            Register::AX => Some((0, 16, false, 0)),
            Register::CX => Some((1, 16, false, 0)),
            Register::DX => Some((2, 16, false, 0)),
            Register::BX => Some((3, 16, false, 0)),
            Register::SP => Some((4, 16, false, 0)),
            Register::BP => Some((5, 16, false, 0)),
            Register::SI => Some((6, 16, false, 0)),
            Register::DI => Some((7, 16, false, 0)),
            Register::R8W => Some((8, 16, false, 0)),
            Register::R9W => Some((9, 16, false, 0)),
            Register::R10W => Some((10, 16, false, 0)),
            Register::R11W => Some((11, 16, false, 0)),
            Register::R12W => Some((12, 16, false, 0)),
            Register::R13W => Some((13, 16, false, 0)),
            Register::R14W => Some((14, 16, false, 0)),
            Register::R15W => Some((15, 16, false, 0)),
            Register::AL => Some((0, 8, false, 0)),
            Register::CL => Some((1, 8, false, 0)),
            Register::DL => Some((2, 8, false, 0)),
            Register::BL => Some((3, 8, false, 0)),
            Register::R8L => Some((8, 8, false, 0)),
            Register::R9L => Some((9, 8, false, 0)),
            Register::R10L => Some((10, 8, false, 0)),
            Register::R11L => Some((11, 8, false, 0)),
            Register::R12L => Some((12, 8, false, 0)),
            Register::R13L => Some((13, 8, false, 0)),
            Register::R14L => Some((14, 8, false, 0)),
            Register::R15L => Some((15, 8, false, 0)),
            Register::AH => Some((0, 8, false, 8)),
            Register::CH => Some((1, 8, false, 8)),
            Register::DH => Some((2, 8, false, 8)),
            Register::BH => Some((3, 8, false, 8)),
            _ => None,
        }
    }

    pub fn translate_bytes<Context, E, F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        bytes: &[u8],
        rip: u64,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, wasm_encoder::ValType)> + '_)) -> F
                  + '_),
    ) -> Result<(), E> {
        let mut dec = Decoder::with_ip(64, bytes, rip, DecoderOptions::NONE);
        while dec.can_decode() {
            let inst = dec.decode();
            let inst_len = inst.len() as u32;
            let inst_rip = dec.ip() - inst_len as u64;

            if let Some(gate) = &self.slot_assigner {
                if gate.slot_for_pc(inst_rip).is_none() {
                    continue;
                }
            }

            let tail_idx = self.init_function(ctx, rctx, inst_rip, inst_len, 4, f)?;
            rctx.feed(ctx, tail_idx, &Instruction::I32Const(inst_rip as i32))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(16))?;

            {
                use crate::{ArchTag, InstructionInfo, TrapAction};
                let insn_info = InstructionInfo {
                    pc: inst_rip,
                    len: inst_len,
                    arch: ArchTag::X86_64,
                    class: Self::classify_mnemonic(inst.mnemonic()),
                };
                if rctx.on_instruction(&insn_info, ctx)? == TrapAction::Skip {
                    continue;
                }
            }

            let undecidable_option = (match inst.mnemonic() {
                Mnemonic::Add => {
                    if inst.op0_kind() == OpKind::Memory {
                        let size_bits = match inst.op1_kind() {
                            OpKind::Immediate8 => 8,
                            OpKind::Immediate16 => 16,
                            OpKind::Immediate8to32 => 32,
                            OpKind::Immediate32 => 32,
                            OpKind::Immediate64 => 64,
                            OpKind::Register => {
                                if let Some((_, r_size, _, _)) = Self::resolve_reg(inst.op1_register()) {
                                    r_size
                                } else {
                                    64
                                }
                            }
                            _ => 64,
                        };
                        self.handle_memory_rmw(ctx, rctx, tail_idx, &inst, size_bits, |this, ctx, rctx, tail_idx| {
                            this.emit_i64_add(ctx, rctx, tail_idx)
                        })
                    } else {
                        self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                                match src {
                                    Operand::Imm(i) => this.emit_i64_const(ctx, rctx, tail_idx, i)?,
                                    Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
                                    Operand::RegWithSize(r, sz, bit) => {
                                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                                        this.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                                    }
                                }
                                this.emit_i64_add(ctx, rctx, tail_idx)?;
                                if dst_size == 64 && dst_bit_offset == 0 {
                                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                                } else if dst_size == 32 {
                                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                                } else {
                                    this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                                }
                            },
                        )
                    }
                }
                Mnemonic::Imul => self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => this.emit_i64_const(ctx, rctx, tail_idx, i)?,
                            Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
                            Operand::RegWithSize(r, sz, bit) => {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                            }
                        }
                        this.emit_i64_mul(ctx, rctx, tail_idx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                        }
                    },
                ),
                Mnemonic::And => self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => this.emit_i64_const(ctx, rctx, tail_idx, i)?,
                            Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
                            Operand::RegWithSize(r, sz, bit) => {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                            }
                        }
                        this.emit_i64_and(ctx, rctx, tail_idx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                        }
                    },
                ),
                Mnemonic::Or => self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => this.emit_i64_const(ctx, rctx, tail_idx, i)?,
                            Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
                            Operand::RegWithSize(r, sz, bit) => {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                            }
                        }
                        this.emit_i64_or(ctx, rctx, tail_idx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                        }
                    },
                ),
                Mnemonic::Xor => self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => this.emit_i64_const(ctx, rctx, tail_idx, i)?,
                            Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
                            Operand::RegWithSize(r, sz, bit) => {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                            }
                        }
                        this.emit_i64_xor(ctx, rctx, tail_idx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                        }
                    },
                ),
                Mnemonic::Shl => self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => this.emit_i64_const(ctx, rctx, tail_idx, i)?,
                            Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
                            Operand::RegWithSize(r, sz, bit) => {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                            }
                        }
                        this.emit_i64_shl(ctx, rctx, tail_idx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                        }
                    },
                ),
                Mnemonic::Shr | Mnemonic::Sar => {
                    self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                            match src {
                                Operand::Imm(i) => this.emit_i64_const(ctx, rctx, tail_idx, i)?,
                                Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
                                Operand::RegWithSize(r, sz, bit) => {
                                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                                    this.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                                }
                            }
                            if inst.mnemonic() == iced_x86::Mnemonic::Sar {
                                this.emit_i64_shr_s(ctx, rctx, tail_idx)?;
                            } else {
                                this.emit_i64_shr_u(ctx, rctx, tail_idx)?;
                            }
                            if dst_size == 64 && dst_bit_offset == 0 {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                            } else if dst_size == 32 {
                                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                            } else {
                                this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                            }
                        },
                    )
                }
                Mnemonic::Mov => {
                    self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                            match inst.op1_kind() {
                                OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                                    if let Operand::Imm(i) = src {
                                        this.emit_i64_const(ctx, rctx, tail_idx, i)?;
                                    }
                                }
                                OpKind::Register => {
                                    match src {
                                        Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
                                        Operand::RegWithSize(r, sz, bit) => {
                                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                                            this.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                                        }
                                        _ => return rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
                                    }
                                }
                                OpKind::Memory => {
                                    this.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                                    this.emit_memory_load(ctx, rctx, tail_idx, dst_size, false)?;
                                }
                                _ => return rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
                            }
                            if dst_size == 64 && dst_bit_offset == 0 {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                            } else if dst_size == 32 {
                                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                            } else {
                                this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                            }
                        },
                    )
                }
                Mnemonic::Lea => {
                    if inst.op0_kind() != OpKind::Register || inst.op1_kind() != OpKind::Memory {
                        Ok(None)
                    } else if let Some((dst_local, _dst_size, _z, _bit)) = Self::resolve_reg(inst.op0_register()) {
                        self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst_local))?;
                        Ok(Some(()))
                    } else {
                        Ok(None)
                    }
                }
                Mnemonic::Movsx => {
                    self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                            match inst.op1_kind() {
                                OpKind::Register => {
                                    match src {
                                        Operand::RegWithSize(r_local, r_size, r_bit) => {
                                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r_local))?;
                                            if r_bit > 0 {
                                                this.emit_mask_shift_for_read(ctx, rctx, tail_idx, r_size, r_bit)?;
                                            }
                                            match r_size {
                                                8 => {
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(56))?;
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(56))?;
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
                                                }
                                                16 => {
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(48))?;
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(48))?;
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
                                                }
                                                32 => rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?,
                                                _ => {}
                                            }
                                        }
                                        _ => return rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
                                    }
                                }
                                OpKind::Memory => {
                                    this.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                                    let mem_size_bits = match inst.memory_size() {
                                        iced_x86::MemorySize::UInt8 => 8,
                                        iced_x86::MemorySize::UInt16 => 16,
                                        iced_x86::MemorySize::UInt32 => 32,
                                        iced_x86::MemorySize::UInt64 => 64,
                                        _ => 64,
                                    };
                                    this.emit_memory_load(ctx, rctx, tail_idx, mem_size_bits, true)?;
                                }
                                _ => return rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
                            }
                            if dst_size == 64 && dst_bit_offset == 0 {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                            } else if dst_size == 32 {
                                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                            } else {
                                this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                            }
                        },
                    )
                }
                Mnemonic::Movzx => {
                    self.handle_binary(ctx, rctx, tail_idx, &inst, |this, ctx, rctx, tail_idx, src, dst, dst_size, dst_bit_offset| {
                            match inst.op1_kind() {
                                OpKind::Register => {
                                    match src {
                                        Operand::RegWithSize(r_local, r_size, r_bit) => {
                                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r_local))?;
                                            if r_bit > 0 {
                                                this.emit_mask_shift_for_read(ctx, rctx, tail_idx, r_size, r_bit)?;
                                            }
                                            match r_size {
                                                8 => {
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                                }
                                                16 => {
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF))?;
                                                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                                }
                                                _ => {}
                                            }
                                        }
                                        _ => return rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
                                    }
                                }
                                OpKind::Memory => {
                                    this.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                                    let mem_size_bits = match inst.memory_size() {
                                        iced_x86::MemorySize::UInt8 => 8,
                                        iced_x86::MemorySize::UInt16 => 16,
                                        iced_x86::MemorySize::UInt32 => 32,
                                        iced_x86::MemorySize::UInt64 => 64,
                                        _ => 64,
                                    };
                                    this.emit_memory_load(ctx, rctx, tail_idx, mem_size_bits, false)?;
                                }
                                _ => return rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
                            }
                            if dst_size == 64 && dst_bit_offset == 0 {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                            } else if dst_size == 32 {
                                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
                            } else {
                                this.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)
                            }
                        },
                    )
                }
                Mnemonic::Xchg => {
                    if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Register {
                        if let (Some((dst_local, _, _, _)), Some((src_local, _, _, _))) = (Self::resolve_reg(inst.op0_register()), Self::resolve_reg(inst.op1_register())) {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst_local))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(17))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst_local))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(17))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(src_local))?;
                            Ok(Some(()))
                        } else {
                            Ok(None)
                        }
                    } else if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Memory {
                        if let Some((dst_local, _dst_size, _, _)) = Self::resolve_reg(inst.op0_register()) {
                            self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                            self.emit_memory_load(ctx, rctx, tail_idx, _dst_size, false)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(17))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst_local))?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64Store(wasm_encoder::MemArg { offset: 0, align: 3, memory_index: 0 }))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(17))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst_local))?;
                            Ok(Some(()))
                        } else {
                            Ok(None)
                        }
                    } else {
                        Ok(None)
                    }
                }
                Mnemonic::Test => self.handle_test(ctx, rctx, tail_idx, &inst),
                Mnemonic::Cmp => self.handle_cmp(ctx, rctx, tail_idx, &inst),
                Mnemonic::Jmp => self.handle_jmp(ctx, rctx, tail_idx, &inst),
                Mnemonic::Je => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::ZF),
                Mnemonic::Jne => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::NZF),
                Mnemonic::Jl => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::SF_NE_OF),
                Mnemonic::Jle => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::ZF_OR_SF_NE_OF),
                Mnemonic::Jg => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::NZF_AND_SF_EQ_OF),
                Mnemonic::Jge => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::SF_EQ_OF),
                Mnemonic::Jb => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::CF),
                Mnemonic::Jbe => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::CF_OR_ZF),
                Mnemonic::Ja => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::NCF_AND_NZF),
                Mnemonic::Jae => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::NCF),
                Mnemonic::Js => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::SF),
                Mnemonic::Jns => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::NSF),
                Mnemonic::Jo => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::OF),
                Mnemonic::Jno => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::NOF),
                Mnemonic::Jp => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::PF),
                Mnemonic::Jnp => self.handle_conditional_jump(ctx, rctx, tail_idx, &inst, ConditionType::NPF),
                Mnemonic::Call => self.handle_call(ctx, rctx, tail_idx, &inst),
                Mnemonic::Ret => self.handle_ret(ctx, rctx, tail_idx, &inst),
                Mnemonic::Push => self.handle_push(ctx, rctx, tail_idx, &inst),
                Mnemonic::Pop => self.handle_pop(ctx, rctx, tail_idx, &inst),
                Mnemonic::Pushf | Mnemonic::Pushfd | Mnemonic::Pushfq => {
                    let (size_bits, rsp_sub) = if inst.mnemonic() == Mnemonic::Pushf { (16, 2) } else if inst.mnemonic() == Mnemonic::Pushfd { (32, 4) } else { (64, 8) };
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(rsp_sub))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(X86Recompiler::CF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(X86Recompiler::PF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(2))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(X86Recompiler::ZF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(6))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(X86Recompiler::SF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(7))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(X86Recompiler::OF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(11))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    if size_bits == 16 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    } else if size_bits == 32 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    }
                    self.emit_memory_store(ctx, rctx, tail_idx, size_bits)?;
                    Ok(Some(()))
                }
                Mnemonic::Popf | Mnemonic::Popfd | Mnemonic::Popfq => {
                    let (size_bits, rsp_add) = if inst.mnemonic() == Mnemonic::Popf { (16, 2) } else if inst.mnemonic() == Mnemonic::Popfd { (32, 4) } else { (64, 8) };
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
                    self.emit_memory_load(ctx, rctx, tail_idx, size_bits, false)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(rsp_add))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::CF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(2))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::PF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(6))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::ZF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(7))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::SF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(11))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::OF_LOCAL))?;
                    Ok(Some(()))
                }
                _ => Ok(None),
            })?;

            if undecidable_option.is_none() {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
            }

            dec.set_ip(inst_rip + 1);
        }
        Ok(())
    }

    fn handle_test<Context, E, F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let src = match inst.op1_kind() {
            OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => Operand::Imm(inst.immediate64() as i64),
            OpKind::Register => {
                if let Some((r_local, r_size, _z, bit)) = Self::resolve_reg(inst.op1_register()) {
                    Operand::RegWithSize(r_local, r_size, bit)
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        let dst_info = match inst.op0_kind() {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            _ => None,
        };

        if dst_info.is_none() { return Ok(None); }
        let (dst_local, dst_size, _, dst_bit_offset) = dst_info.unwrap();

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst_local))?;
        if dst_bit_offset > 0 {
            self.emit_mask_shift_for_read(ctx, rctx, tail_idx, dst_size, dst_bit_offset)?;
        }

        match src {
            Operand::Imm(i) => self.emit_i64_const(ctx, rctx, tail_idx, i)?,
            Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
            Operand::RegWithSize(r, sz, bit) => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                if bit > 0 {
                    self.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                }
            }
        }

        self.emit_i64_and(ctx, rctx, tail_idx)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eq)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::ZF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::SF_LOCAL))?;
        self.set_cf(ctx, rctx, tail_idx, false)?;
        self.set_of(ctx, rctx, tail_idx, false)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Popcnt)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::PF_LOCAL))?;
        Ok(Some(()))
    }

    fn handle_cmp<Context, E, F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let src = match inst.op1_kind() {
            OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => Operand::Imm(inst.immediate64() as i64),
            OpKind::Register => {
                if let Some((r_local, r_size, _, bit)) = Self::resolve_reg(inst.op1_register()) {
                    Operand::RegWithSize(r_local, r_size, bit)
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        let dst_info = match inst.op0_kind() {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            _ => None,
        };

        if dst_info.is_none() { return Ok(None); }
        let (dst_local, dst_size, _, dst_bit_offset) = dst_info.unwrap();

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst_local))?;
        if dst_bit_offset > 0 {
            self.emit_mask_shift_for_read(ctx, rctx, tail_idx, dst_size, dst_bit_offset)?;
        }

        match src {
            Operand::Imm(i) => self.emit_i64_const(ctx, rctx, tail_idx, i)?,
            Operand::Reg(r) => rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?,
            Operand::RegWithSize(r, sz, bit) => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r))?;
                if bit > 0 {
                    self.emit_mask_shift_for_read(ctx, rctx, tail_idx, sz, bit)?;
                }
            }
        }

        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(23))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(23))?;
        self.emit_i64_sub(ctx, rctx, tail_idx)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(24))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eq)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::ZF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(24))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::SF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(23))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtU)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::CF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(23))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(24))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::OF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(24))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Popcnt)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(X86Recompiler::PF_LOCAL))?;
        Ok(Some(()))
    }

    fn handle_jmp<Context, E, F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let target = match inst.op0_kind() {
            OpKind::NearBranch64 | OpKind::NearBranch32 | OpKind::NearBranch16 => (inst.ip() as i64 + inst.near_branch64() as i64) as u64,
            _ => return Ok(None),
        };

        {
            use crate::{JumpInfo, JumpKind, TrapAction};
            let jmp_info = JumpInfo::direct(inst.ip(), target, JumpKind::DirectJump);
            if rctx.on_jump(&jmp_info, ctx)? == TrapAction::Skip {
                return Ok(Some(()));
            }
        }
        let Some(target_func_idx) = self.rip_to_func_idx(rctx, target) else {
            rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
            return Ok(Some(()));
        };
        rctx.jmp(ctx, tail_idx, target_func_idx, rctx.locals_mark().total_locals)?;
        Ok(Some(()))
    }

    fn handle_conditional_jump<Context, E, F>(
        &self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
        condition_type: ConditionType,
    ) -> Result<Option<()>, E> {
        let target = match inst.op0_kind() {
            OpKind::NearBranch64 | OpKind::NearBranch32 | OpKind::NearBranch16 => (inst.ip() as i64 + inst.near_branch64() as i64) as u64,
            _ => return Ok(None),
        };

        {
            use crate::{JumpInfo, JumpKind, TrapAction};
            let jcc_info = JumpInfo::direct(inst.ip(), target, JumpKind::ConditionalBranch);
            if rctx.on_jump(&jcc_info, ctx)? == TrapAction::Skip {
                return Ok(Some(()));
            }
        }
        let Some(target_func_idx) = self.rip_to_func_idx(rctx, target) else {
            rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
            return Ok(Some(()));
        };
        let condition = ConditionSnippet { condition_type };
        let params = JumpCallParams::conditional_jump(target_func_idx, rctx.locals_mark().total_locals, &condition, rctx.pool());
        rctx.ji_with_params(ctx, tail_idx, params)?;
        Ok(Some(()))
    }

    fn handle_call<Context, E, F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let return_addr = inst.next_ip();
        let target = match inst.op0_kind() {
            OpKind::NearBranch64 | OpKind::NearBranch32 | OpKind::NearBranch16 => (inst.ip() as i64 + inst.near_branch64() as i64) as u64,
            _ => return Ok(None),
        };

        let use_speculative = self.enable_speculative_calls && rctx.escape_tag().is_some();

        if use_speculative {
            let escape_tag = rctx.escape_tag().unwrap();
            let Some(target_func) = self.rip_to_func_idx(rctx, target) else {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                return Ok(Some(()));
            };

            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(8))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
            self.emit_memory_store(ctx, rctx, tail_idx, 64)?;

            let expected_ra_snippet = ExpectedRaSnippet { return_addr };
            let params = yecta::JumpCallParams::call(target_func, rctx.locals_mark().total_locals, escape_tag, rctx.pool())
                .with_fixup(X86Recompiler::EXPECTED_RA_LOCAL, &expected_ra_snippet);
            rctx.ji_with_params(ctx, tail_idx, params)?;
            return Ok(Some(()));
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(8))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
            self.emit_memory_store(ctx, rctx, tail_idx, 64)?;

            {
                use crate::{JumpInfo, JumpKind, TrapAction};
                let call_info = JumpInfo::direct(inst.ip(), target, JumpKind::Call);
                if rctx.on_jump(&call_info, ctx)? == TrapAction::Skip {
                    return Ok(Some(()));
                }
            }
            let Some(target_func_idx) = self.rip_to_func_idx(rctx, target) else {
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                return Ok(Some(()));
            };
            rctx.jmp(ctx, tail_idx, target_func_idx, rctx.locals_mark().total_locals)?;
            Ok(Some(()))
        }
    }

    fn handle_ret<Context, E, F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let stack_cleanup = if inst.op_count() > 0 { match inst.op0_kind() { OpKind::Immediate16 | OpKind::Immediate32 => inst.immediate16() as u64, _ => 0 } } else { 0 };
        let use_speculative = self.enable_speculative_calls && rctx.escape_tag().is_some();

        if use_speculative {
            let escape_tag = rctx.escape_tag().unwrap();
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            self.emit_memory_load(ctx, rctx, tail_idx, 64, false)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(X86Recompiler::EXPECTED_RA_LOCAL))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Eq)?;
            rctx.feed(ctx, tail_idx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(8 + stack_cleanup as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::Return)?;
            rctx.feed(ctx, tail_idx, &Instruction::Else)?;
            rctx.ret(ctx, tail_idx, rctx.locals_mark().total_locals, escape_tag)?;
            rctx.feed(ctx, tail_idx, &Instruction::End)?;
            return Ok(Some(()));
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            self.emit_memory_load(ctx, rctx, tail_idx, 64, false)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(23))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(8 + stack_cleanup as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
            let return_addr_snippet = ReturnAddressSnippet { base_rip: self.base_rip };
            {
                use crate::{JumpInfo, JumpKind, TrapAction};
                let ret_info = JumpInfo::indirect(inst.ip(), 23, JumpKind::Return);
                if rctx.on_jump(&ret_info, ctx)? == TrapAction::Skip {
                    return Ok(Some(()));
                }
            }
            let params = yecta::JumpCallParams::indirect_jump(&return_addr_snippet, rctx.locals_mark().total_locals, rctx.pool());
            rctx.ji_with_params(ctx, tail_idx, params)?;
            Ok(Some(()))
        }
    }

    fn handle_push<Context, E, F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let (operand_size, stack_decrement) = match inst.op0_kind() {
            OpKind::Register => {
                if let Some((_, reg_size, _, _)) = Self::resolve_reg(inst.op0_register()) {
                    match reg_size { 64 => (64, 8), 32 => (32, 4), 16 => (16, 2), _ => (64, 8) }
                } else { return Ok(None); }
            }
            OpKind::Immediate8 | OpKind::Immediate8to32 | OpKind::Immediate32 | OpKind::Memory => (64, 8),
            OpKind::Immediate16 => (16, 2),
            _ => return Ok(None),
        };

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(stack_decrement))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;

        match inst.op0_kind() {
            OpKind::Register => {
                if let Some((local, size, _, bit)) = Self::resolve_reg(inst.op0_register()) {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(local))?;
                    if bit > 0 { self.emit_mask_shift_for_read(ctx, rctx, tail_idx, size, bit)?; }
                }
            }
            OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                self.emit_i64_const(ctx, rctx, tail_idx, inst.immediate64() as i64)?;
            }
            OpKind::Memory => {
                self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                self.emit_memory_load(ctx, rctx, tail_idx, 64, false)?;
            }
            _ => return Ok(None),
        }
        self.emit_memory_store(ctx, rctx, tail_idx, operand_size)?;
        Ok(Some(()))
    }

    fn handle_pop<Context, E, F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let (operand_size, stack_increment) = match inst.op0_kind() {
            OpKind::Register => {
                if let Some((_, reg_size, _, _)) = Self::resolve_reg(inst.op0_register()) {
                    match reg_size { 64 => (64, 8), 32 => (32, 4), 16 => (16, 2), _ => (64, 8) }
                } else { return Ok(None); }
            }
            OpKind::Memory => (64, 8),
            _ => return Ok(None),
        };

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
        self.emit_memory_load(ctx, rctx, tail_idx, operand_size, false)?;

        match inst.op0_kind() {
            OpKind::Register => {
                if let Some((local, size, _, bit)) = Self::resolve_reg(inst.op0_register()) {
                    if size == 64 && bit == 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(local))?;
                    } else {
                        self.emit_subreg_write_rmw(ctx, rctx, tail_idx, local, size, bit)?;
                    }
                }
            }
            OpKind::Memory => {
                self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Store(wasm_encoder::MemArg { offset: 0, align: 3, memory_index: 0 }))?;
            }
            _ => return Ok(None),
        }

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(stack_increment))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
        Ok(Some(()))
    }
}
