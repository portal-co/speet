use crate::*;

use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use wax_core::build::InstructionSource;
use yecta::{EscapeTag, FuncIdx, JumpCallParams, Pool, Reactor, TableIdx, TypeIdx};

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

impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for ConditionSnippet {
    fn emit(
        &self,
        ctx: &mut Context, sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        // For simple structs, we can delegate to emit_instruction
        self.emit_instruction(ctx, sink)
    }
}

impl<Context, E> wax_core::build::InstructionSource<Context, E> for ConditionSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context, sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        match self.condition_type {
            ConditionType::ZF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::ZF_LOCAL,
                ))?;
            }
            ConditionType::NZF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::ZF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::SF_NE_OF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::SF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::OF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
            }
            ConditionType::ZF_OR_SF_NE_OF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::ZF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::SF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::OF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
                sink.instruction(ctx, &Instruction::I32Or)?;
            }
            ConditionType::NZF_AND_SF_EQ_OF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::ZF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::SF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::OF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(ctx, &Instruction::I32And)?;
            }
            ConditionType::SF_EQ_OF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::SF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::OF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::CF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::CF_LOCAL,
                ))?;
            }
            ConditionType::CF_OR_ZF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::CF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::ZF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Or)?;
            }
            ConditionType::NCF_AND_NZF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::CF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::ZF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(ctx, &Instruction::I32And)?;
            }
            ConditionType::NCF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::CF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::SF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::SF_LOCAL,
                ))?;
            }
            ConditionType::NSF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::SF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::OF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::OF_LOCAL,
                ))?;
            }
            ConditionType::NOF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::OF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::PF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::PF_LOCAL,
                ))?;
            }
            ConditionType::NPF => {
                sink.instruction(ctx, &Instruction::LocalGet(
                    X86Recompiler::<Context, E, wasm_encoder::Function>::PF_LOCAL,
                ))?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
        }
        Ok(())
    }
}

impl<Context, E, F: InstructionSink<Context, E>> X86Recompiler<Context, E, F> {
    fn rip_to_func_idx(&self, rip: u64) -> FuncIdx {
        FuncIdx((rip.wrapping_sub(self.base_rip) / 1) as u32)
    }

    fn init_function(
        &mut self,
        _rip: u64,
        inst_len: u32,
        num_temps: u32,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, wasm_encoder::ValType)> + '_)) -> F
                  + '_),
    ) {
        // For simplicity, model 16 general purpose 64-bit regs as locals 0-15
        // PC in local 16 (i32)
        // Condition flags: ZF(17), SF(18), CF(19), OF(20), PF(21) as i32
        // Temps after that
        let locals = [
            (16, wasm_encoder::ValType::I64),        // registers
            (1, wasm_encoder::ValType::I32),         // PC
            (5, wasm_encoder::ValType::I32),         // condition flags: ZF, SF, CF, OF, PF
            (num_temps, wasm_encoder::ValType::I64), // temps
        ];
        // Pass instruction length to yecta so fallthrough is controlled by instruction size
        self.reactor.next_with(f(&mut locals.into_iter()), inst_len);
    }

    fn resolve_reg(reg: Register) -> Option<(u32, u32, bool, u32)> {
        // Return (local_index, size_bits, writes_zero_extend32, bit_offset)
        // bit_offset is the starting bit of the subregister within the 64-bit local (e.g., AH is offset 8)
        match reg {
            // 64-bit
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

            // 32-bit forms - writing to 32-bit zero-extends to 64
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

            // 16-bit forms
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

            // 8-bit low forms
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

            // 8-bit high forms (AH/CH/DH/BH) -> offset 8 within the low 16-bit register
            Register::AH => Some((0, 8, false, 8)),
            Register::CH => Some((1, 8, false, 8)),
            Register::DH => Some((2, 8, false, 8)),
            Register::BH => Some((3, 8, false, 8)),

            _ => None,
        }
    }

    /// Translate a sequence of bytes starting at `rip` to wasm using the provided function-local builder
    /// This creates one yecta function per instruction; the `inst_len` is passed to yecta so fallthroughs are the correct distance.
    pub fn translate_bytes(
        &mut self,
        ctx: &mut Context,
        bytes: &[u8],
        rip: u64,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, wasm_encoder::ValType)> + '_)) -> F
                  + '_),
    ) -> Result<(), E> {
        let mut dec = Decoder::with_ip(64, bytes, rip, DecoderOptions::NONE);
        while dec.can_decode() {
            let inst = dec.decode();
            let inst_len = inst.len() as u32;
            let inst_rip = dec.ip() - inst_len as u64; // decoder advanced

            self.init_function(inst_rip, inst_len, 4, f);
            // store rip into local 16
            self.reactor.feed(ctx, &Instruction::I32Const(inst_rip as i32))?;
            self.reactor.feed(ctx, &Instruction::LocalSet(16))?;

            // Try to handle the mnemonic; each arm returns Result<Option<()>, E> where None means undecidable operands
            let undecidable_option = (match inst.mnemonic() {
                Mnemonic::Add => {
                    // Handle memory destination case first
                    if inst.op0_kind() == OpKind::Memory {
                        let size_bits = match inst.op1_kind() {
                            OpKind::Immediate8 => 8,
                            OpKind::Immediate16 => 16,
                            OpKind::Immediate8to32 => 32,
                            OpKind::Immediate32 => 32,
                            OpKind::Immediate64 => 64,
                            OpKind::Register => {
                                if let Some((_, r_size, _, _)) =
                                    Self::resolve_reg(inst.op1_register())
                                {
                                    r_size
                                } else {
                                    64
                                }
                            }
                            _ => 64,
                        };
                        self.handle_memory_rmw(ctx, &inst, size_bits, |this, ctx| this.emit_i64_add(ctx))
                    } else {
                        self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                            this.reactor.feed(ctx, &Instruction::LocalGet(dst))?;
                            match src {
                                Operand::Imm(i) => {
                                    this.emit_i64_const(ctx, i)?;
                                }
                                Operand::Reg(r) => {
                                    this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                }
                                Operand::RegWithSize(r, sz, bit) => {
                                    this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                    this.emit_mask_shift_for_read(ctx, sz, bit)?;
                                }
                            }
                            this.emit_i64_add(ctx)?;
                            if dst_size == 64 && dst_bit_offset == 0 {
                                this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                            } else if dst_size == 32 {
                                this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                                this.reactor.feed(ctx, &Instruction::I64And)?;
                                this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                            } else {
                                this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                            }
                        })
                    }
                }
                Mnemonic::Imul => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        this.reactor.feed(ctx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => {
                                this.emit_i64_const(ctx, i)?;
                            }
                            Operand::Reg(r) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                            }
                            Operand::RegWithSize(r, sz, bit) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, sz, bit)?;
                            }
                        }
                        this.emit_i64_mul(ctx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                            this.reactor.feed(ctx, &Instruction::I64And)?;
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                        }
                    })
                }
                Mnemonic::And => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        this.reactor.feed(ctx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => {
                                this.emit_i64_const(ctx, i)?;
                            }
                            Operand::Reg(r) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                            }
                            Operand::RegWithSize(r, sz, bit) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, sz, bit)?;
                            }
                        }
                        this.emit_i64_and(ctx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                            this.reactor.feed(ctx, &Instruction::I64And)?;
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                        }
                    })
                }
                Mnemonic::Or => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        this.reactor.feed(ctx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => {
                                this.emit_i64_const(ctx, i)?;
                            }
                            Operand::Reg(r) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                            }
                            Operand::RegWithSize(r, sz, bit) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, sz, bit)?;
                            }
                        }
                        this.emit_i64_or(ctx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                            this.reactor.feed(ctx, &Instruction::I64And)?;
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                        }
                    })
                }
                Mnemonic::Xor => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        this.reactor.feed(ctx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => {
                                this.emit_i64_const(ctx, i)?;
                            }
                            Operand::Reg(r) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                            }
                            Operand::RegWithSize(r, sz, bit) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, sz, bit)?;
                            }
                        }
                        this.emit_i64_xor(ctx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                            this.reactor.feed(ctx, &Instruction::I64And)?;
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                        }
                    })
                }
                Mnemonic::Shl => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        this.reactor.feed(ctx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => {
                                this.emit_i64_const(ctx, i)?;
                            }
                            Operand::Reg(r) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                            }
                            Operand::RegWithSize(r, sz, bit) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, sz, bit)?;
                            }
                        }
                        this.emit_i64_shl(ctx)?;
                        if dst_size == 64 && dst_bit_offset == 0 {
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                            this.reactor.feed(ctx, &Instruction::I64And)?;
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                        }
                    })
                }
                Mnemonic::Shr | Mnemonic::Sar => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        this.reactor.feed(ctx, &Instruction::LocalGet(dst))?;
                        match src {
                            Operand::Imm(i) => {
                                this.emit_i64_const(ctx, i)?;
                            }
                            Operand::Reg(r) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                            }
                            Operand::RegWithSize(r, sz, bit) => {
                                this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                this.emit_mask_shift_for_read(ctx, sz, bit)?;
                            }
                        }
                        // choose arithmetic vs logical based on mnemonic
                        // if mnemonic was Sar use arithmetic (shr_s), otherwise Shr uses logical (shr_u)
                        if inst.mnemonic() == iced_x86::Mnemonic::Sar {
                            this.emit_i64_shr_s(ctx)?;
                        } else {
                            this.emit_i64_shr_u(ctx)?;
                        }
                        if dst_size == 64 && dst_bit_offset == 0 {
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                            this.reactor.feed(ctx, &Instruction::I64And)?;
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                        }
                    })
                }
                // MOV: moves between regs/mem/imm
                Mnemonic::Mov => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        match inst.op1_kind() {
                            OpKind::Immediate8
                            | OpKind::Immediate16
                            | OpKind::Immediate32
                            | OpKind::Immediate64
                            | OpKind::Immediate8to32 => {
                                if let Operand::Imm(i) = src {
                                    this.emit_i64_const(ctx, i)?;
                                }
                                if dst_size == 64 && dst_bit_offset == 0 {
                                    this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                                } else if dst_size == 32 {
                                    this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                                    this.reactor.feed(ctx, &Instruction::I64And)?;
                                    this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                                } else {
                                    this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                                }
                            }
                            OpKind::Register => {
                                match src {
                                    Operand::Reg(r) => {
                                        this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                    }
                                    Operand::RegWithSize(r, sz, bit) => {
                                        this.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                                        this.emit_mask_shift_for_read(ctx, sz, bit)?;
                                    }
                                    _ => return this.reactor.feed(ctx, &Instruction::Unreachable),
                                }
                                if dst_size == 64 && dst_bit_offset == 0 {
                                    this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                                } else if dst_size == 32 {
                                    this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                                    this.reactor.feed(ctx, &Instruction::I64And)?;
                                    this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                                } else {
                                    this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                                }
                            }
                            OpKind::Memory => {
                                // load from memory into dst
                                this.emit_memory_address(ctx, &inst)?;
                                // load according to destination size (zero-extend)
                                this.emit_memory_load(ctx, dst_size, false)?;
                                if dst_size == 64 && dst_bit_offset == 0 {
                                    this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                                } else if dst_size == 32 {
                                    this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                                    this.reactor.feed(ctx, &Instruction::I64And)?;
                                    this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                                } else {
                                    this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                                }
                            }
                            _ => return this.reactor.feed(ctx, &Instruction::Unreachable),
                        }
                    })
                }
                // LEA: compute effective address into dest register
                Mnemonic::Lea => {
                    // dest must be register, src must be memory
                    if inst.op0_kind() != OpKind::Register || inst.op1_kind() != OpKind::Memory {
                        Ok(None)
                    } else if let Some((dst_local, _dst_size, _z, _bit)) =
                        Self::resolve_reg(inst.op0_register())
                    {
                        self.emit_memory_address(ctx, &inst)?;
                        self.reactor.feed(ctx, &Instruction::LocalSet(dst_local))?;
                        Ok(Some(()))
                    } else {
                        Ok(None)
                    }
                }
                // MOVSX: sign-extend from smaller reg/mem into reg
                Mnemonic::Movsx => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        match inst.op1_kind() {
                            OpKind::Register => {
                                match src {
                                    Operand::RegWithSize(r_local, r_size, r_bit) => {
                                        this.reactor.feed(ctx, &Instruction::LocalGet(r_local))?;
                                        if r_bit > 0 {
                                            this.emit_mask_shift_for_read(ctx, r_size, r_bit)?;
                                        }
                                        match r_size {
                                            8 => {
                                                // sign-extend 8-bit: (v << 56) >> 56 (arith)
                                                this.reactor.feed(ctx, &Instruction::I64Const(56))?;
                                                this.reactor.feed(ctx, &Instruction::I64Shl)?;
                                                this.reactor.feed(ctx, &Instruction::I64Const(56))?;
                                                this.reactor.feed(ctx, &Instruction::I64ShrS)?;
                                            }
                                            16 => {
                                                this.reactor.feed(ctx, &Instruction::I64Const(48))?;
                                                this.reactor.feed(ctx, &Instruction::I64Shl)?;
                                                this.reactor.feed(ctx, &Instruction::I64Const(48))?;
                                                this.reactor.feed(ctx, &Instruction::I64ShrS)?;
                                            }
                                            32 => {
                                                // use i64.extend_i32_s if available via instruction
                                                this.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                                            }
                                            64 => { /* already 64-bit */ }
                                            _ => {}
                                        }
                                    }
                                    _ => return this.reactor.feed(ctx, &Instruction::Unreachable),
                                }
                            }
                            OpKind::Memory => {
                                this.emit_memory_address(ctx, &inst)?;
                                // memory signed load of operand size; determine size from inst.memory_size if available
                                // fallback to 64
                                // For now assume 8/16/32/64 based on op1_operand size: use inst.memory_size().size() if possible
                                let mem_size_bits = match inst.memory_size() {
                                    // may exist
                                    iced_x86::MemorySize::UInt8 => 8,
                                    iced_x86::MemorySize::UInt16 => 16,
                                    iced_x86::MemorySize::UInt32 => 32,
                                    iced_x86::MemorySize::UInt64 => 64,
                                    _ => 64,
                                };
                                this.emit_memory_load(ctx, mem_size_bits, true)?;
                            }
                            _ => return this.reactor.feed(ctx, &Instruction::Unreachable),
                        }
                        if dst_size == 64 && dst_bit_offset == 0 {
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                            this.reactor.feed(ctx, &Instruction::I64And)?;
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                        }
                    })
                }
                // MOVZX: zero-extend from smaller reg/mem into reg
                Mnemonic::Movzx => {
                    self.handle_binary(ctx, &inst, |this, ctx, src, dst, dst_size, dst_bit_offset| {
                        match inst.op1_kind() {
                            OpKind::Register => {
                                match src {
                                    Operand::RegWithSize(r_local, r_size, r_bit) => {
                                        this.reactor.feed(ctx, &Instruction::LocalGet(r_local))?;
                                        if r_bit > 0 {
                                            this.emit_mask_shift_for_read(ctx, r_size, r_bit)?;
                                        }
                                        match r_size {
                                            8 => {
                                                this.reactor.feed(ctx, &Instruction::I64Const(0xFF))?;
                                                this.reactor.feed(ctx, &Instruction::I64And)?;
                                            }
                                            16 => {
                                                this.reactor
                                                    .feed(ctx, &Instruction::I64Const(0xFFFF))?;
                                                this.reactor.feed(ctx, &Instruction::I64And)?;
                                            }
                                            32 => { /* already zero-extended */ }
                                            64 => { /* already 64 */ }
                                            _ => {}
                                        }
                                    }
                                    _ => return this.reactor.feed(ctx, &Instruction::Unreachable),
                                }
                            }
                            OpKind::Memory => {
                                this.emit_memory_address(ctx, &inst)?;
                                let mem_size_bits = match inst.memory_size() {
                                    iced_x86::MemorySize::UInt8 => 8,
                                    iced_x86::MemorySize::UInt16 => 16,
                                    iced_x86::MemorySize::UInt32 => 32,
                                    iced_x86::MemorySize::UInt64 => 64,
                                    _ => 64,
                                };
                                this.emit_memory_load(ctx, mem_size_bits, false)?;
                            }
                            _ => return this.reactor.feed(ctx, &Instruction::Unreachable),
                        }
                        if dst_size == 64 && dst_bit_offset == 0 {
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else if dst_size == 32 {
                            this.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                            this.reactor.feed(ctx, &Instruction::I64And)?;
                            this.reactor.feed(ctx, &Instruction::LocalSet(dst))
                        } else {
                            this.emit_subreg_write_rmw(ctx, dst, dst_size, dst_bit_offset)
                        }
                    })
                }
                // XCHG: exchange reg and reg/mem
                Mnemonic::Xchg => {
                    // support reg,reg and reg,mem
                    if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Register {
                        if let (
                            Some((dst_local, _dst_size, _z, _dbit)),
                            Some((src_local, _src_size, _sz, _sbit)),
                        ) = (
                            Self::resolve_reg(inst.op0_register()),
                            Self::resolve_reg(inst.op1_register()),
                        ) {
                            // temp in local 17
                            self.reactor.feed(ctx, &Instruction::LocalGet(dst_local))?;
                            self.reactor.feed(ctx, &Instruction::LocalSet(17))?;
                            self.reactor.feed(ctx, &Instruction::LocalGet(src_local))?;
                            self.reactor.feed(ctx, &Instruction::LocalSet(dst_local))?;
                            self.reactor.feed(ctx, &Instruction::LocalGet(17))?;
                            self.reactor.feed(ctx, &Instruction::LocalSet(src_local))?;
                            Ok(Some(()))
                        } else {
                            Ok(None)
                        }
                    } else if inst.op0_kind() == OpKind::Register
                        && inst.op1_kind() == OpKind::Memory
                    {
                        // mem <-> reg: load mem, store reg to mem, set reg to loaded value
                        if let Some((dst_local, _dst_size, _z, _dbit)) =
                            Self::resolve_reg(inst.op0_register())
                        {
                            // compute addr
                            self.emit_memory_address(ctx, &inst)?;
                            // load old value with register size
                            self.emit_memory_load(ctx, _dst_size, false)?;
                            // store old value to temp
                            self.reactor.feed(ctx, &Instruction::LocalSet(17))?;
                            // store register value to memory
                            self.reactor.feed(ctx, &Instruction::LocalGet(dst_local))?;
                            self.reactor
                                .feed(ctx, &Instruction::I64Store(wasm_encoder::MemArg {
                                    offset: 0,
                                    align: 3,
                                    memory_index: 0,
                                }))?;
                            // set reg to loaded
                            self.reactor.feed(ctx, &Instruction::LocalGet(17))?;
                            self.reactor.feed(ctx, &Instruction::LocalSet(dst_local))?;
                            Ok(Some(()))
                        } else {
                            Ok(None)
                        }
                    } else {
                        Ok(None)
                    }
                }
                Mnemonic::Test => self.handle_test(ctx, &inst),
                Mnemonic::Cmp => self.handle_cmp(ctx, &inst),
                Mnemonic::Jmp => self.handle_jmp(ctx, &inst),
                Mnemonic::Je => self.handle_conditional_jump(ctx, &inst, ConditionType::ZF),
                Mnemonic::Jne => self.handle_conditional_jump(ctx, &inst, ConditionType::NZF),
                Mnemonic::Jl => self.handle_conditional_jump(ctx, &inst, ConditionType::SF_NE_OF),
                Mnemonic::Jle => self.handle_conditional_jump(ctx, &inst, ConditionType::ZF_OR_SF_NE_OF),
                Mnemonic::Jg => {
                    self.handle_conditional_jump(ctx, &inst, ConditionType::NZF_AND_SF_EQ_OF)
                }
                Mnemonic::Jge => self.handle_conditional_jump(ctx, &inst, ConditionType::SF_EQ_OF),
                Mnemonic::Jb => self.handle_conditional_jump(ctx, &inst, ConditionType::CF),
                Mnemonic::Jbe => self.handle_conditional_jump(ctx, &inst, ConditionType::CF_OR_ZF),
                Mnemonic::Ja => self.handle_conditional_jump(ctx, &inst, ConditionType::NCF_AND_NZF),
                Mnemonic::Jae => self.handle_conditional_jump(ctx, &inst, ConditionType::NCF),
                Mnemonic::Js => self.handle_conditional_jump(ctx, &inst, ConditionType::SF),
                Mnemonic::Jns => self.handle_conditional_jump(ctx, &inst, ConditionType::NSF),
                Mnemonic::Jo => self.handle_conditional_jump(ctx, &inst, ConditionType::OF),
                Mnemonic::Jno => self.handle_conditional_jump(ctx, &inst, ConditionType::NOF),
                Mnemonic::Jp => self.handle_conditional_jump(ctx, &inst, ConditionType::PF),
                Mnemonic::Jnp => self.handle_conditional_jump(ctx, &inst, ConditionType::NPF),
                Mnemonic::Pushf | Mnemonic::Pushfd | Mnemonic::Pushfq => {
                    // PUSHF variants: push flags with operand-size 16/32/64
                    let (size_bits, rsp_sub) = if inst.mnemonic() == Mnemonic::Pushf {
                        (16u32, 2i64)
                    } else if inst.mnemonic() == Mnemonic::Pushfd {
                        (32u32, 4i64)
                    } else {
                        // Pushfq
                        (64u32, 8i64)
                    };

                    // decrement RSP by rsp_sub
                    self.reactor.feed(ctx, &Instruction::LocalGet(4))?;
                    self.reactor.feed(ctx, &Instruction::I64Const(rsp_sub))?;
                    self.reactor.feed(ctx, &Instruction::I64Sub)?;
                    self.reactor.feed(ctx, &Instruction::LocalSet(4))?;

                    // address (RSP) on stack
                    self.reactor.feed(ctx, &Instruction::LocalGet(4))?;

                    // build flags value on stack (i64 accumulator)
                    self.reactor.feed(ctx, &Instruction::I64Const(0))?; // acc = 0
                                                                   // CF -> bit 0
                    self.reactor.feed(ctx, &Instruction::LocalGet(Self::CF_LOCAL))?;
                    self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(0))?;
                    self.reactor.feed(ctx, &Instruction::I64Shl)?;
                    self.reactor.feed(ctx, &Instruction::I64Or)?;
                    // PF -> bit 2
                    self.reactor.feed(ctx, &Instruction::LocalGet(Self::PF_LOCAL))?;
                    self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(2))?;
                    self.reactor.feed(ctx, &Instruction::I64Shl)?;
                    self.reactor.feed(ctx, &Instruction::I64Or)?;
                    // ZF -> bit 6
                    self.reactor.feed(ctx, &Instruction::LocalGet(Self::ZF_LOCAL))?;
                    self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(6))?;
                    self.reactor.feed(ctx, &Instruction::I64Shl)?;
                    self.reactor.feed(ctx, &Instruction::I64Or)?;
                    // SF -> bit 7
                    self.reactor.feed(ctx, &Instruction::LocalGet(Self::SF_LOCAL))?;
                    self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(7))?;
                    self.reactor.feed(ctx, &Instruction::I64Shl)?;
                    self.reactor.feed(ctx, &Instruction::I64Or)?;
                    // OF -> bit 11
                    self.reactor.feed(ctx, &Instruction::LocalGet(Self::OF_LOCAL))?;
                    self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(11))?;
                    self.reactor.feed(ctx, &Instruction::I64Shl)?;
                    self.reactor.feed(ctx, &Instruction::I64Or)?;

                    // mask down if needed
                    if size_bits == 16 {
                        self.reactor.feed(ctx, &Instruction::I64Const(0xFFFF))?;
                        self.reactor.feed(ctx, &Instruction::I64And)?;
                    } else if size_bits == 32 {
                        self.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
                        self.reactor.feed(ctx, &Instruction::I64And)?;
                    }

                    // store flags at [RSP]
                    self.emit_memory_store(ctx, size_bits)?;
                    Ok(Some(()))
                }

                Mnemonic::Popf | Mnemonic::Popfd | Mnemonic::Popfq => {
                    // POPF variants: pop flags with operand-size 16/32/64
                    let (size_bits, rsp_add) = if inst.mnemonic() == Mnemonic::Popf {
                        (16u32, 2i64)
                    } else if inst.mnemonic() == Mnemonic::Popfd {
                        (32u32, 4i64)
                    } else {
                        (64u32, 8i64)
                    };

                    // load value from [RSP] with appropriate size
                    self.reactor.feed(ctx, &Instruction::LocalGet(4))?;
                    self.emit_memory_load(ctx, size_bits, false)?; // pushes i64 value (zero-extended)
                                                              // store popped value into temp local 22
                    self.reactor.feed(ctx, &Instruction::LocalSet(22))?;

                    // increment RSP by rsp_add
                    self.reactor.feed(ctx, &Instruction::LocalGet(4))?;
                    self.reactor.feed(ctx, &Instruction::I64Const(rsp_add))?;
                    self.reactor.feed(ctx, &Instruction::I64Add)?;
                    self.reactor.feed(ctx, &Instruction::LocalSet(4))?;

                    // Now extract bits from local 22 into flag locals. For sizes <64, the value is zero-extended.
                    // extract CF (bit 0)
                    self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
                    self.reactor.feed(ctx, &Instruction::I64Const(0))?;
                    self.reactor.feed(ctx, &Instruction::I64ShrU)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(1))?;
                    self.reactor.feed(ctx, &Instruction::I64And)?;
                    self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                    self.reactor.feed(ctx, &Instruction::LocalSet(Self::CF_LOCAL))?;
                    // extract PF (bit 2)
                    self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
                    self.reactor.feed(ctx, &Instruction::I64Const(2))?;
                    self.reactor.feed(ctx, &Instruction::I64ShrU)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(1))?;
                    self.reactor.feed(ctx, &Instruction::I64And)?;
                    self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                    self.reactor.feed(ctx, &Instruction::LocalSet(Self::PF_LOCAL))?;
                    // extract ZF (bit 6)
                    self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
                    self.reactor.feed(ctx, &Instruction::I64Const(6))?;
                    self.reactor.feed(ctx, &Instruction::I64ShrU)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(1))?;
                    self.reactor.feed(ctx, &Instruction::I64And)?;
                    self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                    self.reactor.feed(ctx, &Instruction::LocalSet(Self::ZF_LOCAL))?;
                    // extract SF (bit 7)
                    self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
                    self.reactor.feed(ctx, &Instruction::I64Const(7))?;
                    self.reactor.feed(ctx, &Instruction::I64ShrU)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(1))?;
                    self.reactor.feed(ctx, &Instruction::I64And)?;
                    self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                    self.reactor.feed(ctx, &Instruction::LocalSet(Self::SF_LOCAL))?;
                    // extract OF (bit 11)
                    self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
                    self.reactor.feed(ctx, &Instruction::I64Const(11))?;
                    self.reactor.feed(ctx, &Instruction::I64ShrU)?;
                    self.reactor.feed(ctx, &Instruction::I64Const(1))?;
                    self.reactor.feed(ctx, &Instruction::I64And)?;
                    self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                    self.reactor.feed(ctx, &Instruction::LocalSet(Self::OF_LOCAL))?;

                    Ok(Some(()))
                }

                _ => Ok(None),
            })?;

            // If handler returned None (undecidable operands), emit unreachable
            if undecidable_option.is_none() {
                self.reactor.feed(ctx, &Instruction::Unreachable)?;
            }

            // Rewind decoder to the byte after the current instruction RIP
            // Use inst_rip + 1 to allow variable-length instructions to be found at any byte
            dec.set_ip(inst_rip + 1);
        }
        Ok(())
    }

    // Helper: handle read-modify-write operations for memory destinations
    fn handle_memory_rmw<Op>(
        &mut self,
        ctx: &mut Context,
        inst: &IxInst,
        size_bits: u32,
        mut operation: Op,
    ) -> Result<Option<()>, E>
    where
        Op: FnMut(&mut Self, &mut Context) -> Result<(), E>,
    {
        use iced_x86::OpKind;

        // Load current value from memory
        self.emit_memory_address(ctx, &inst)?;
        self.emit_memory_load(ctx, size_bits, false)?;

        // Get source value based on operand kind
        match inst.op1_kind() {
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to32 => {
                let imm = inst.immediate64() as i64;
                self.emit_i64_const(ctx, imm)?;
            }
            OpKind::Register => {
                if let Some((r_local, r_size, _rz, bit)) = Self::resolve_reg(inst.op1_register()) {
                    self.reactor.feed(ctx, &Instruction::LocalGet(r_local))?;
                    if bit > 0 {
                        self.emit_mask_shift_for_read(ctx, r_size, bit)?;
                    }
                    // Apply mask for sub-registers
                    match r_size {
                        8 => {
                            self.reactor.feed(ctx, &Instruction::I64Const(0xFF))?;
                            self.reactor.feed(ctx, &Instruction::I64And)?;
                        }
                        16 => {
                            self.reactor.feed(ctx, &Instruction::I64Const(0xFFFF))?;
                            self.reactor.feed(ctx, &Instruction::I64And)?;
                        }
                        32 => { /* already zero-extended */ }
                        64 => { /* full */ }
                        _ => {}
                    }
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        }

        // Perform the operation (stack now has: old_value, src_value)
        operation(self, ctx)?;

        // Store result back to memory
        self.emit_memory_address(ctx, &inst)?;
        self.emit_memory_store(ctx, size_bits)?;

        Ok(Some(()))
    }

    // Helper: emit memory effective address for the instruction's memory operand
    fn emit_memory_address(&mut self, ctx: &mut Context, inst: &IxInst) -> Result<(), E> {
        use iced_x86::Register;
        // Build address: base + index*scale + displacement
        let base = inst.memory_base();
        let index = inst.memory_index();
        let scale = inst.memory_index_scale();
        let disp = inst.memory_displacement64();

        // Start with 0
        let mut have_value = false;
        if base != Register::None {
            if let Some((local, _sz, _z, bit)) = Self::resolve_reg(base) {
                self.reactor.feed(ctx, &Instruction::LocalGet(local))?;
                if bit > 0 {
                    // shift right by bit to align subregister
                    self.reactor.feed(ctx, &Instruction::I64Const(bit as i64))?;
                    self.reactor.feed(ctx, &Instruction::I64ShrU)?;
                }
                have_value = true;
            } else {
                self.reactor.feed(ctx, &Instruction::Unreachable)?;
                return Ok(());
            }
        }
        if index != Register::None {
            if let Some((idx_local, _sz, _z, bit)) = Self::resolve_reg(index) {
                if have_value {
                    self.reactor.feed(ctx, &Instruction::LocalGet(idx_local))?;
                } else {
                    self.reactor.feed(ctx, &Instruction::LocalGet(idx_local))?;
                    have_value = true;
                }
                if bit > 0 {
                    self.reactor.feed(ctx, &Instruction::I64Const(bit as i64))?;
                    self.reactor.feed(ctx, &Instruction::I64ShrU)?;
                }
                // multiply by scale
                if scale != 1 {
                    self.reactor.feed(ctx, &Instruction::I64Const(scale as i64))?;
                    self.reactor.feed(ctx, &Instruction::I64Mul)?;
                }
                if base != Register::None {
                    self.reactor.feed(ctx, &Instruction::I64Add)?;
                }
            } else {
                self.reactor.feed(ctx, &Instruction::Unreachable)?;
                return Ok(());
            }
        }
        if disp != 0 {
            if have_value {
                self.reactor.feed(ctx, &Instruction::I64Const(disp as i64))?;
                self.reactor.feed(ctx, &Instruction::I64Add)?;
            } else {
                self.reactor.feed(ctx, &Instruction::I64Const(disp as i64))?;
                have_value = true;
            }
        }
        // If nothing contributed, push zero
        if !have_value {
            self.reactor.feed(ctx, &Instruction::I64Const(0))?;
        }
        Ok(())
    }

    fn handle_binary<T>(&mut self, ctx: &mut Context, inst: &IxInst, mut cb: T) -> Result<Option<()>, E>
    where
        T: FnMut(&mut Self, &mut Context, Operand, u32, u32, u32) -> Result<(), E>,
    {
        // Try to extract dest register and src (imm or reg)
        use iced_x86::OpKind;

        let op0 = inst.op0_kind();
        let op1 = inst.op1_kind();

        // If destination is memory, handle store here.
        if op0 == OpKind::Memory {
            // source can be immediate or register
            match op1 {
                OpKind::Immediate8
                | OpKind::Immediate16
                | OpKind::Immediate32
                | OpKind::Immediate64
                | OpKind::Immediate8to32 => {
                    // emit address then immediate then store sized by immediate kind
                    self.emit_memory_address(ctx, &inst)?;
                    let imm = inst.immediate64() as i64;
                    self.emit_i64_const(ctx, imm)?;
                    let size_bits = match op1 {
                        OpKind::Immediate8 => 8,
                        OpKind::Immediate16 => 16,
                        OpKind::Immediate8to32 => 32,
                        OpKind::Immediate32 => 32,
                        OpKind::Immediate64 => 64,
                        _ => 64,
                    };
                    self.emit_memory_store(ctx, size_bits)?;
                    return Ok(Some(()));
                }
                OpKind::Register => {
                    if let Some((r_local, r_size, _rz, bit)) =
                        Self::resolve_reg(inst.op1_register())
                    {
                        // compute address
                        self.emit_memory_address(ctx, &inst)?;
                        // get source value and narrow if sub-register
                        self.reactor.feed(ctx, &Instruction::LocalGet(r_local))?;
                        if bit > 0 {
                            self.emit_mask_shift_for_read(ctx, r_size, bit)?;
                        }
                        // narrow by mask if needed
                        match r_size {
                            8 => {
                                self.reactor.feed(ctx, &Instruction::I64Const(0xFF))?;
                                self.reactor.feed(ctx, &Instruction::I64And)?;
                            }
                            16 => {
                                self.reactor.feed(ctx, &Instruction::I64Const(0xFFFF))?;
                                self.reactor.feed(ctx, &Instruction::I64And)?;
                            }
                            32 => { /* lower 32 bits are already zero-extended in locals for 32-bit regs */
                            }
                            64 => { /* full */ }
                            _ => {}
                        }
                        // store to memory using source size
                        self.emit_memory_store(ctx, r_size)?;
                        return Ok(Some(()));
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        }

        // Determine destination local and size
        let dst_info = match op0 {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            OpKind::Memory => None, // memory destinations not supported yet
            _ => None,
        };

        if dst_info.is_none() {
            return Ok(None);
        }
        let (dst_local, dst_size, _dst_zero_ext32, dst_bit_offset) = dst_info.unwrap();

        // Determine source operand and size semantics
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
            OpKind::Memory => {
                return Ok(None);
            }
            _ => return Ok(None),
        };

        // Generate code with knowledge of dst size; handle sub-register writes
        match src {
            Operand::Imm(_i) => {
                // immediate -> callback will emit and set dest
                cb(self, ctx, src, dst_local, dst_size, dst_bit_offset)?;
                // if dst is an 8/16-bit subregister we need to perform RMW; the callback should leave the new value on stack then we do RMW
                // but since our callbacks currently perform LocalSet(dst_local) directly for full-register writes, we need to adjust pattern:
            }
            Operand::RegWithSize(src_local, src_size, src_bit) => {
                cb(
                    self,
                    ctx,
                    Operand::RegWithSize(src_local, src_size, src_bit),
                    dst_local,
                    dst_size,
                    dst_bit_offset,
                )?;
            }
            _ => {
                cb(self, ctx, src, dst_local, dst_size, dst_bit_offset)?;
            }
        }

        Ok(Some(()))
    }

    fn handle_test(&mut self, ctx: &mut Context, inst: &IxInst) -> Result<Option<()>, E> {
        // TEST performs bitwise AND and sets flags, but doesn't store result
        let src = match inst.op1_kind() {
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
            OpKind::Memory => {
                return Ok(None);
            } // Memory operands not supported yet
            _ => return Ok(None),
        };

        // Get destination (must be register for TEST)
        let dst_info = match inst.op0_kind() {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            _ => None,
        };

        if dst_info.is_none() {
            return Ok(None);
        }
        let (dst_local, dst_size, _dst_zero_ext32, dst_bit_offset) = dst_info.unwrap();

        // Get dst value
        self.reactor.feed(ctx, &Instruction::LocalGet(dst_local))?;
        if dst_bit_offset > 0 {
            self.emit_mask_shift_for_read(ctx, dst_size, dst_bit_offset)?;
        }

        // Get src value
        match src {
            Operand::Imm(i) => {
                self.emit_i64_const(ctx, i)?;
            }
            Operand::Reg(r) => {
                self.reactor.feed(ctx, &Instruction::LocalGet(r))?;
            }
            Operand::RegWithSize(r, sz, bit) => {
                self.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                if bit > 0 {
                    self.emit_mask_shift_for_read(ctx, sz, bit)?;
                }
            }
        }

        // Perform AND
        self.emit_i64_and(ctx)?;

        // Store result temporarily and set flags
        self.reactor.feed(ctx, &Instruction::LocalTee(22))?; // temp = result, result still on stack

        // ZF: result == 0
        self.reactor.feed(ctx, &Instruction::I64Const(0))?;
        self.reactor.feed(ctx, &Instruction::I64Eq)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(Self::ZF_LOCAL))?;

        // SF: result < 0
        self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
        self.reactor.feed(ctx, &Instruction::I64Const(63))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::I32Const(1))?;
        self.reactor.feed(ctx, &Instruction::I32And)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(Self::SF_LOCAL))?;

        // CF and OF are always cleared for TEST
        self.set_cf(ctx, false)?;
        self.set_of(ctx, false)?;

        // PF: parity of lowest byte
        self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
        self.reactor.feed(ctx, &Instruction::I64Const(0xFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        // Simple parity check - count bits (simplified)
        self.reactor.feed(ctx, &Instruction::I32Popcnt)?;
        self.reactor.feed(ctx, &Instruction::I32Const(1))?;
        self.reactor.feed(ctx, &Instruction::I32And)?; // 1 if odd parity, 0 if even
        self.reactor.feed(ctx, &Instruction::I32Eqz)?; // 1 if even parity, 0 if odd
        self.reactor.feed(ctx, &Instruction::LocalSet(Self::PF_LOCAL))?;

        Ok(Some(()))
    }

    fn handle_cmp(&mut self, ctx: &mut Context, inst: &IxInst) -> Result<Option<()>, E> {
        // CMP performs subtraction and sets flags, but doesn't store result
        let src = match inst.op1_kind() {
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
            OpKind::Memory => {
                return Ok(None);
            } // Memory operands not supported yet
            _ => return Ok(None),
        };

        // Get destination
        let dst_info = match inst.op0_kind() {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            OpKind::Memory => None, // Memory destinations not supported yet
            _ => None,
        };

        if dst_info.is_none() {
            return Ok(None);
        }
        let (dst_local, dst_size, _dst_zero_ext32, dst_bit_offset) = dst_info.unwrap();

        // Get dst value
        self.reactor.feed(ctx, &Instruction::LocalGet(dst_local))?;
        if dst_bit_offset > 0 {
            self.emit_mask_shift_for_read(ctx, dst_size, dst_bit_offset)?;
        }

        // Get src value
        match src {
            Operand::Imm(i) => {
                self.emit_i64_const(ctx, i)?;
            }
            Operand::Reg(r) => {
                self.reactor.feed(ctx, &Instruction::LocalGet(r))?;
            }
            Operand::RegWithSize(r, sz, bit) => {
                self.reactor.feed(ctx, &Instruction::LocalGet(r))?;
                if bit > 0 {
                    self.emit_mask_shift_for_read(ctx, sz, bit)?;
                }
            }
        }

        // Store operands for flag computation
        self.reactor.feed(ctx, &Instruction::LocalSet(23))?; // src
        self.reactor.feed(ctx, &Instruction::LocalSet(22))?; // dst

        // Compute result = dst - src
        self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
        self.reactor.feed(ctx, &Instruction::LocalGet(23))?;
        self.emit_i64_sub(ctx)?;

        // Store result and set flags
        self.reactor.feed(ctx, &Instruction::LocalTee(24))?; // result

        // ZF: result == 0
        self.reactor.feed(ctx, &Instruction::I64Const(0))?;
        self.reactor.feed(ctx, &Instruction::I64Eq)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(Self::ZF_LOCAL))?;

        // SF: result < 0
        self.reactor.feed(ctx, &Instruction::LocalGet(24))?;
        self.reactor.feed(ctx, &Instruction::I64Const(63))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::I32Const(1))?;
        self.reactor.feed(ctx, &Instruction::I32And)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(Self::SF_LOCAL))?;

        // CF: for subtraction, CF if dst < src (unsigned)
        self.reactor.feed(ctx, &Instruction::LocalGet(22))?;
        self.reactor.feed(ctx, &Instruction::LocalGet(23))?;
        self.reactor.feed(ctx, &Instruction::I64LtU)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(Self::CF_LOCAL))?;

        // OF: overflow for subtraction
        // Overflow occurs when operands have different signs and result has different sign from dst
        self.reactor.feed(ctx, &Instruction::LocalGet(22))?; // dst
        self.reactor.feed(ctx, &Instruction::I64Const(63))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
        self.reactor.feed(ctx, &Instruction::LocalGet(23))?; // src
        self.reactor.feed(ctx, &Instruction::I64Const(63))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
        self.reactor.feed(ctx, &Instruction::I32Xor)?; // 1 if different signs
        self.reactor.feed(ctx, &Instruction::LocalGet(22))?; // dst
        self.reactor.feed(ctx, &Instruction::I64Const(63))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
        self.reactor.feed(ctx, &Instruction::LocalGet(24))?; // result
        self.reactor.feed(ctx, &Instruction::I64Const(63))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
        self.reactor.feed(ctx, &Instruction::I32Xor)?; // 1 if dst and result have different signs
        self.reactor.feed(ctx, &Instruction::I32And)?; // OF set
        self.reactor.feed(ctx, &Instruction::LocalSet(Self::OF_LOCAL))?;

        // PF: parity of lowest byte
        self.reactor.feed(ctx, &Instruction::LocalGet(24))?;
        self.reactor.feed(ctx, &Instruction::I64Const(0xFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::I32Popcnt)?;
        self.reactor.feed(ctx, &Instruction::I32Const(1))?;
        self.reactor.feed(ctx, &Instruction::I32And)?;
        self.reactor.feed(ctx, &Instruction::I32Eqz)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(Self::PF_LOCAL))?;

        Ok(Some(()))
    }

    fn handle_jmp(&mut self, ctx: &mut Context, inst: &IxInst) -> Result<Option<()>, E> {
        // JMP target
        let target = match inst.op0_kind() {
            OpKind::NearBranch64 | OpKind::NearBranch32 | OpKind::NearBranch16 => {
                let offset = inst.near_branch64() as i64;
                let current_rip = inst.ip() as i64;
                (current_rip + offset) as u64
            }
            _ => return Ok(None),
        };

        let target_func_idx = self.rip_to_func_idx(target);
        let pool = self.pool;

        // Unconditional jump
        self.reactor.jmp(ctx, target_func_idx, 0)?; // No params for now

        Ok(Some(()))
    }

    fn handle_conditional_jump(
        &mut self,
        ctx: &mut Context,
        inst: &IxInst,
        condition_type: ConditionType,
    ) -> Result<Option<()>, E> {
        let target = match inst.op0_kind() {
            OpKind::NearBranch64 | OpKind::NearBranch32 | OpKind::NearBranch16 => {
                let offset = inst.near_branch64() as i64;
                let current_rip = inst.ip() as i64;
                (current_rip + offset) as u64
            }
            _ => return Ok(None),
        };

        let target_func_idx = self.rip_to_func_idx(target);
        let pool = self.pool;

        // Use the proper yecta conditional jump API
        let condition = ConditionSnippet { condition_type };
        let params = JumpCallParams::conditional_jump(target_func_idx, 0, &condition, pool);
        self.reactor.ji_with_params(ctx, params)?;

        Ok(Some(()))
    }
}

