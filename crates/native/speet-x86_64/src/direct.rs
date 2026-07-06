use crate::*;

use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use wax_core::build::InstructionSource;
use yecta::layout::CellIdx;
use yecta::{FuncIdx, JumpCallParams, LocalDeclarator};

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

        // Keep this as i64: indirect jump plumbing expects architecture state words.
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
                        X86Recompiler::<Context, E>::ZF_LOCAL,
                    ),
                )?;
            }
            ConditionType::NZF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::SF_NE_OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::SF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
            }
            ConditionType::ZF_OR_SF_NE_OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::SF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
                sink.instruction(ctx, &Instruction::I32Or)?;
            }
            ConditionType::NZF_AND_SF_EQ_OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::SF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::OF_LOCAL,
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
                        X86Recompiler::<Context, E>::SF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Xor)?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::CF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::CF_LOCAL,
                    ),
                )?;
            }
            ConditionType::CF_OR_ZF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::CF_LOCAL,
                    ),
                )?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Or)?;
            }
            ConditionType::NCF_AND_NZF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::CF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::ZF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
                sink.instruction(ctx, &Instruction::I32And)?;
            }
            ConditionType::NCF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::CF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::SF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::SF_LOCAL,
                    ),
                )?;
            }
            ConditionType::NSF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::SF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::OF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::OF_LOCAL,
                    ),
                )?;
            }
            ConditionType::NOF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::OF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
            ConditionType::PF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::PF_LOCAL,
                    ),
                )?;
            }
            ConditionType::NPF => {
                sink.instruction(
                    ctx,
                    &Instruction::LocalGet(
                        X86Recompiler::<Context, E>::PF_LOCAL,
                    ),
                )?;
                sink.instruction(ctx, &Instruction::I32Eqz)?;
            }
        }
        Ok(())
    }
}

impl<Context, E> X86Recompiler<Context, E> {
    fn rip_to_func_idx<F>(&self, rctx: &dyn ReactorContext<Context, E, FnType = F>, rip: u64) -> Option<FuncIdx> {
        if let Some(gate) = &self.slot_assigner {
            gate.slot_for_pc(rip).map(FuncIdx)
        } else if rip >= self.base_rip && rip < self.base_rip + self.text_len as u64 {
            // Every byte offset is a potential slot here (no slot assigner
            // installed), but a target outside the actually-translated
            // range can never be a real instruction — most commonly a
            // garbage decode from a misaligned re-decode (the decoder
            // advances 1 byte at a time to catch every possible slot,
            // so it also re-decodes from the middle of real instructions,
            // producing call/jmp targets with meaningless operands).
            // Returning `None` routes these through `oob_jump`, which
            // traps at runtime instead of emitting a statically invalid
            // `Instruction::Call` that fails WASM validation at compile
            // time.
            Some(FuncIdx((rip.wrapping_sub(self.base_rip)) as u32))
        } else {
            None
        }
    }

    fn lookup_plt_import(&self, target: u64) -> Option<(u32, &str)> {
        let by_addr = self.plt_by_addr.as_ref()?;
        let imports = self.plt_imports.as_ref()?;
        let sym = by_addr.get(&target)?;
        let idx = imports.get(sym)?;
        Some((*idx, sym.as_str()))
    }

    fn emit_plt_import_call<F>(
        &self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        import_idx: u32,
        symbol: &str,
    ) -> Result<(), E> {
        let args = match symbol.strip_prefix('_').unwrap_or(symbol) {
            "execve" => [7u32, 6, 2], // RDI, RSI, RDX
            _ => return Ok(()),
        };
        for local in args {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(local))?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::Call(import_idx))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(0))?; // RAX
        Ok(())
    }

    fn init_function<F>(
        &mut self,
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
        let mut unit = ();
        let extra: &mut dyn LocalDeclarator = match self.memory_access.as_deref_mut() {
            Some(m) => m as &mut dyn LocalDeclarator,
            None => &mut unit,
        };
        rctx.declare_trap_locals(extra);
        let _cell = rctx.alloc_cell();
        let fn_type = f(&mut rctx.layout().iter_since(&mark).collect::<alloc::vec::Vec<_>>().into_iter());
        rctx.next_with(ctx, fn_type, inst_len)
    }

    fn emit_memory_address<F>(
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

    fn handle_memory_rmw<F, Op>(
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
        // Stack: [result]. Save result, push addr, restore — WASM store wants [addr, value].
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(22))?;
        self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        self.emit_memory_store(ctx, rctx, tail_idx, size_bits)?;
        Ok(Some(()))
    }

    fn handle_binary<F, T>(
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
                | OpKind::Immediate8to32
                | OpKind::Immediate8to64 => {
                    self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
                    let imm = inst.immediate64() as i64;
                    self.emit_i64_const(ctx, rctx, tail_idx, imm)?;
                    let size_bits = match op1 {
                        OpKind::Immediate8 => 8, OpKind::Immediate16 => 16,
                        OpKind::Immediate8to32 | OpKind::Immediate32 => 32,
                        OpKind::Immediate8to64 | OpKind::Immediate64 => 64,
                        _ => 64,
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
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => Operand::Imm(inst.immediate64() as i64),
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

    /// Map an XMM register to its guest-state WASM local (raw i64 bits). The
    /// XMM file occupies locals `XMM_BASE_LOCAL .. +16`; scalar SSE only.
    fn resolve_xmm(reg: Register) -> Option<u32> {
        use iced_x86::Register;
        let n: u32 = match reg {
            Register::XMM0 => 0, Register::XMM1 => 1, Register::XMM2 => 2, Register::XMM3 => 3,
            Register::XMM4 => 4, Register::XMM5 => 5, Register::XMM6 => 6, Register::XMM7 => 7,
            Register::XMM8 => 8, Register::XMM9 => 9, Register::XMM10 => 10, Register::XMM11 => 11,
            Register::XMM12 => 12, Register::XMM13 => 13, Register::XMM14 => 14, Register::XMM15 => 15,
            _ => return None,
        };
        Some(Self::XMM_BASE_LOCAL + n)
    }

    // ── SSE scalar floating point ────────────────────────────────────────────
    // XMM registers are stored as raw i64 bit patterns (low 64 bits). Handlers
    // reinterpret i64↔f64/f32 around native WASM FP ops; loads/stores move the
    // bits unchanged. Scratch GP temporaries 23/24 are used by `Ucomis*`.
    const FP_TMP_A: u32 = 23;
    const FP_TMP_B: u32 = 24;

    /// Stack: i64 bits → fp value (f32 if `f32` else f64).
    fn sse_bits_to_fp<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, f32: bool) -> Result<(), E> {
        if f32 {
            rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
            rctx.feed(ctx, tail_idx, &Instruction::F32ReinterpretI32)
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::F64ReinterpretI64)
        }
    }

    /// Stack: fp value → i64 bits (zero-extended for f32).
    fn sse_fp_to_bits<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, f32: bool) -> Result<(), E> {
        if f32 {
            rctx.feed(ctx, tail_idx, &Instruction::I32ReinterpretF32)?;
            rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::I64ReinterpretF64)
        }
    }

    /// Push operand `op1` (the source) as i64 bits. Handles an XMM register, a
    /// GP register (for `cvtsi`/`movq`), or memory. Returns `false` if the
    /// operand kind is unsupported.
    fn sse_push_src<F>(&mut self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst, size_bits: u32) -> Result<bool, E> {
        use iced_x86::OpKind;
        match inst.op1_kind() {
            OpKind::Register => {
                let r = inst.op1_register();
                if let Some(x) = Self::resolve_xmm(r) {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(x))?;
                } else if let Some((g, _gs, _z, bit)) = Self::resolve_reg(r) {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(g))?;
                    if bit > 0 { self.emit_mask_shift_for_read(ctx, rctx, tail_idx, _gs, bit)?; }
                    if size_bits == 32 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    }
                } else {
                    return Ok(false);
                }
            }
            OpKind::Memory => {
                self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
                self.emit_memory_load(ctx, rctx, tail_idx, size_bits, false)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Write the i64 on top of stack into GP register `dst` (size 32 or 64),
    /// zero-extending 32-bit results (mirrors the `Mov` reg-dst path).
    fn sse_write_gpr<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, dst: u32, size_bits: u32) -> Result<(), E> {
        if size_bits == 32 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))
    }

    /// Integer size (bits) of operand `op1` for `cvtsi2s*` (register or memory).
    fn sse_int_src_size(inst: &IxInst) -> u32 {
        use iced_x86::OpKind;
        match inst.op1_kind() {
            OpKind::Register => Self::resolve_reg(inst.op1_register()).map(|r| r.1).unwrap_or(64),
            OpKind::Memory => (inst.memory_size().size() as u32) * 8,
            _ => 64,
        }
    }

    /// Translate an SSE scalar FP instruction. Returns `Ok(None)` if `inst` is
    /// not an SSE instruction this backend recognizes (so the caller falls
    /// through to `unsupported_insns`).
    fn handle_sse<F>(&mut self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        use iced_x86::{Mnemonic, OpKind};
        let m = inst.mnemonic();

        // (wasm f64 op, wasm f32 op) for scalar binary arithmetic.
        let arith: Option<(Instruction, Instruction)> = match m {
            Mnemonic::Addsd => Some((Instruction::F64Add, Instruction::F32Add)),
            Mnemonic::Addss => Some((Instruction::F32Add, Instruction::F32Add)),
            Mnemonic::Subsd => Some((Instruction::F64Sub, Instruction::F32Sub)),
            Mnemonic::Subss => Some((Instruction::F32Sub, Instruction::F32Sub)),
            Mnemonic::Mulsd => Some((Instruction::F64Mul, Instruction::F32Mul)),
            Mnemonic::Mulss => Some((Instruction::F32Mul, Instruction::F32Mul)),
            Mnemonic::Divsd => Some((Instruction::F64Div, Instruction::F32Div)),
            Mnemonic::Divss => Some((Instruction::F32Div, Instruction::F32Div)),
            Mnemonic::Minsd => Some((Instruction::F64Min, Instruction::F32Min)),
            Mnemonic::Minss => Some((Instruction::F32Min, Instruction::F32Min)),
            Mnemonic::Maxsd => Some((Instruction::F64Max, Instruction::F32Max)),
            Mnemonic::Maxss => Some((Instruction::F32Max, Instruction::F32Max)),
            _ => None,
        };
        if let Some((op64, op32)) = arith {
            let f32 = matches!(m, Mnemonic::Addss | Mnemonic::Subss | Mnemonic::Mulss | Mnemonic::Divss | Mnemonic::Minss | Mnemonic::Maxss);
            let dst = match Self::resolve_xmm(inst.op0_register()) { Some(d) => d, None => return Ok(None) };
            let sz = if f32 { 32 } else { 64 };
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
            self.sse_bits_to_fp(ctx, rctx, tail_idx, f32)?;
            if !self.sse_push_src(ctx, rctx, tail_idx, inst, sz)? { return Ok(None); }
            self.sse_bits_to_fp(ctx, rctx, tail_idx, f32)?;
            rctx.feed(ctx, tail_idx, if f32 { &op32 } else { &op64 })?;
            self.sse_fp_to_bits(ctx, rctx, tail_idx, f32)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
            return Ok(Some(()));
        }

        match m {
            // ── unary sqrt: dst = sqrt(src) ──
            Mnemonic::Sqrtsd | Mnemonic::Sqrtss => {
                let f32 = m == Mnemonic::Sqrtss;
                let dst = match Self::resolve_xmm(inst.op0_register()) { Some(d) => d, None => return Ok(None) };
                if !self.sse_push_src(ctx, rctx, tail_idx, inst, if f32 { 32 } else { 64 })? { return Ok(None); }
                self.sse_bits_to_fp(ctx, rctx, tail_idx, f32)?;
                rctx.feed(ctx, tail_idx, if f32 { &Instruction::F32Sqrt } else { &Instruction::F64Sqrt })?;
                self.sse_fp_to_bits(ctx, rctx, tail_idx, f32)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                Ok(Some(()))
            }

            // ── bitwise (abs/neg/copysign idioms) on raw bits ──
            Mnemonic::Andpd | Mnemonic::Andps | Mnemonic::Pand
            | Mnemonic::Orpd | Mnemonic::Orps | Mnemonic::Por
            | Mnemonic::Xorpd | Mnemonic::Xorps | Mnemonic::Pxor
            | Mnemonic::Andnpd | Mnemonic::Andnps => {
                let dst = match Self::resolve_xmm(inst.op0_register()) { Some(d) => d, None => return Ok(None) };
                let andn = matches!(m, Mnemonic::Andnpd | Mnemonic::Andnps);
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                if andn {
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?; // ~dst
                }
                if !self.sse_push_src(ctx, rctx, tail_idx, inst, 64)? { return Ok(None); }
                let op = match m {
                    Mnemonic::Andpd | Mnemonic::Andps | Mnemonic::Pand
                    | Mnemonic::Andnpd | Mnemonic::Andnps => Instruction::I64And,
                    Mnemonic::Orpd | Mnemonic::Orps | Mnemonic::Por => Instruction::I64Or,
                    _ => Instruction::I64Xor,
                };
                rctx.feed(ctx, tail_idx, &op)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                Ok(Some(()))
            }

            // ── moves: movsd/movss + the aligned/unaligned reg-copy forms ──
            Mnemonic::Movsd | Mnemonic::Movss
            | Mnemonic::Movaps | Mnemonic::Movapd | Mnemonic::Movups | Mnemonic::Movupd
            | Mnemonic::Movdqa | Mnemonic::Movdqu => {
                // Disambiguate the string MOVSD from the SSE one: require an XMM operand.
                let op0_xmm = Self::resolve_xmm(inst.op0_register());
                let op1_xmm = Self::resolve_xmm(inst.op1_register());
                if op0_xmm.is_none() && op1_xmm.is_none() { return Ok(None); }
                let sz = if m == Mnemonic::Movss { 32 } else { 64 };
                if inst.op0_kind() == OpKind::Memory {
                    // store [mem], xmm
                    let src = match op1_xmm { Some(s) => s, None => return Ok(None) };
                    self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src))?;
                    self.emit_memory_store(ctx, rctx, tail_idx, sz)?;
                } else {
                    // load/copy into xmm dst
                    let dst = match op0_xmm { Some(d) => d, None => return Ok(None) };
                    if !self.sse_push_src(ctx, rctx, tail_idx, inst, sz)? { return Ok(None); }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                }
                Ok(Some(()))
            }

            // ── movq / movd: xmm↔gpr, xmm↔mem ──
            Mnemonic::Movq | Mnemonic::Movd => {
                let sz = if m == Mnemonic::Movd { 32 } else { 64 };
                if let Some(dst) = Self::resolve_xmm(inst.op0_register()) {
                    // dst is xmm ← gpr/mem/xmm
                    if !self.sse_push_src(ctx, rctx, tail_idx, inst, sz)? { return Ok(None); }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                    Ok(Some(()))
                } else if inst.op0_kind() == OpKind::Memory {
                    let src = match Self::resolve_xmm(inst.op1_register()) { Some(s) => s, None => return Ok(None) };
                    self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src))?;
                    if sz == 32 { rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?; rctx.feed(ctx, tail_idx, &Instruction::I64And)?; }
                    self.emit_memory_store(ctx, rctx, tail_idx, sz)?;
                    Ok(Some(()))
                } else if let Some((dst, dsz, _z, bit)) = Self::resolve_reg(inst.op0_register()) {
                    // dst is gpr ← xmm (op1)
                    if bit != 0 { return Ok(None); }
                    let src = match Self::resolve_xmm(inst.op1_register()) { Some(s) => s, None => return Ok(None) };
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src))?;
                    self.sse_write_gpr(ctx, rctx, tail_idx, dst, dsz)?;
                    Ok(Some(()))
                } else {
                    Ok(None)
                }
            }

            // ── compares: set ZF/PF/CF, clear SF/OF (NaN ⇒ ZF=PF=CF=1) ──
            Mnemonic::Ucomisd | Mnemonic::Ucomiss | Mnemonic::Comisd | Mnemonic::Comiss => {
                let f32 = matches!(m, Mnemonic::Ucomiss | Mnemonic::Comiss);
                let a = match Self::resolve_xmm(inst.op0_register()) { Some(d) => d, None => return Ok(None) };
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(a))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::FP_TMP_A))?;
                if !self.sse_push_src(ctx, rctx, tail_idx, inst, if f32 { 32 } else { 64 })? { return Ok(None); }
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::FP_TMP_B))?;
                let (ne, eq, lt) = if f32 {
                    (Instruction::F32Ne, Instruction::F32Eq, Instruction::F32Lt)
                } else {
                    (Instruction::F64Ne, Instruction::F64Eq, Instruction::F64Lt)
                };
                // Helper sequences (pushed inline): a_fp / b_fp / uno = a!=a | b!=b.
                let push_a = |this: &mut Self, ctx: &mut Context| -> Result<(), E> {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::FP_TMP_A))?;
                    this.sse_bits_to_fp(ctx, rctx, tail_idx, f32)
                };
                let push_b = |this: &mut Self, ctx: &mut Context| -> Result<(), E> {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::FP_TMP_B))?;
                    this.sse_bits_to_fp(ctx, rctx, tail_idx, f32)
                };
                let push_uno = |this: &mut Self, ctx: &mut Context| -> Result<(), E> {
                    push_a(this, ctx)?; push_a(this, ctx)?; rctx.feed(ctx, tail_idx, &ne)?;
                    push_b(this, ctx)?; push_b(this, ctx)?; rctx.feed(ctx, tail_idx, &ne)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32Or)
                };
                // PF = uno
                push_uno(self, ctx)?;
                self.emit_flag_set(ctx, rctx, tail_idx, 4)?;
                // CF = (a < b) | uno
                push_a(self, ctx)?; push_b(self, ctx)?; rctx.feed(ctx, tail_idx, &lt)?;
                push_uno(self, ctx)?; rctx.feed(ctx, tail_idx, &Instruction::I32Or)?;
                self.emit_flag_set(ctx, rctx, tail_idx, 2)?;
                // ZF = (a == b) | uno
                push_a(self, ctx)?; push_b(self, ctx)?; rctx.feed(ctx, tail_idx, &eq)?;
                push_uno(self, ctx)?; rctx.feed(ctx, tail_idx, &Instruction::I32Or)?;
                self.emit_flag_set(ctx, rctx, tail_idx, 0)?;
                self.set_sf(ctx, rctx, tail_idx, false)?;
                self.set_of(ctx, rctx, tail_idx, false)?;
                Ok(Some(()))
            }

            // ── int → fp (signed) ──
            Mnemonic::Cvtsi2sd | Mnemonic::Cvtsi2ss => {
                let f32 = m == Mnemonic::Cvtsi2ss;
                let dst = match Self::resolve_xmm(inst.op0_register()) { Some(d) => d, None => return Ok(None) };
                let isz = Self::sse_int_src_size(inst);
                if !self.sse_push_src(ctx, rctx, tail_idx, inst, isz)? { return Ok(None); }
                if isz <= 32 {
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, if f32 { &Instruction::F32ConvertI32S } else { &Instruction::F64ConvertI32S })?;
                } else {
                    rctx.feed(ctx, tail_idx, if f32 { &Instruction::F32ConvertI64S } else { &Instruction::F64ConvertI64S })?;
                }
                self.sse_fp_to_bits(ctx, rctx, tail_idx, f32)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                Ok(Some(()))
            }

            // ── fp → int (truncating / rounding); saturating to avoid traps ──
            Mnemonic::Cvttsd2si | Mnemonic::Cvttss2si | Mnemonic::Cvtsd2si | Mnemonic::Cvtss2si => {
                let f32 = matches!(m, Mnemonic::Cvttss2si | Mnemonic::Cvtss2si);
                let round = matches!(m, Mnemonic::Cvtsd2si | Mnemonic::Cvtss2si);
                let (dst, dsz, bit) = match Self::resolve_reg(inst.op0_register()) {
                    Some((d, s, _z, b)) => (d, s, b),
                    None => return Ok(None),
                };
                if bit != 0 { return Ok(None); }
                if !self.sse_push_src(ctx, rctx, tail_idx, inst, if f32 { 32 } else { 64 })? { return Ok(None); }
                self.sse_bits_to_fp(ctx, rctx, tail_idx, f32)?;
                if round { rctx.feed(ctx, tail_idx, if f32 { &Instruction::F32Nearest } else { &Instruction::F64Nearest })?; }
                let to64 = dsz == 64;
                let conv = match (f32, to64) {
                    (false, true)  => Instruction::I64TruncSatF64S,
                    (false, false) => Instruction::I32TruncSatF64S,
                    (true,  true)  => Instruction::I64TruncSatF32S,
                    (true,  false) => Instruction::I32TruncSatF32S,
                };
                rctx.feed(ctx, tail_idx, &conv)?;
                if !to64 { rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?; }
                self.sse_write_gpr(ctx, rctx, tail_idx, dst, dsz)?;
                Ok(Some(()))
            }

            // ── fp width conversions ──
            Mnemonic::Cvtsd2ss | Mnemonic::Cvtss2sd => {
                let demote = m == Mnemonic::Cvtsd2ss; // f64 → f32
                let dst = match Self::resolve_xmm(inst.op0_register()) { Some(d) => d, None => return Ok(None) };
                if !self.sse_push_src(ctx, rctx, tail_idx, inst, if demote { 64 } else { 32 })? { return Ok(None); }
                self.sse_bits_to_fp(ctx, rctx, tail_idx, !demote)?; // src precision
                rctx.feed(ctx, tail_idx, if demote { &Instruction::F32DemoteF64 } else { &Instruction::F64PromoteF32 })?;
                self.sse_fp_to_bits(ctx, rctx, tail_idx, demote)?;  // dst precision
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                Ok(Some(()))
            }

            _ => Ok(None),
        }
    }

    pub fn translate_bytes<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        bytes: &[u8],
        rip: u64,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, wasm_encoder::ValType)> + '_)) -> F
                  + '_),
    ) -> Result<(), E> {
        #[cfg(feature = "logging")]
        log::trace!(target: "speet::x86_64", "translate_bytes rip={:#x} len={}", rip, bytes.len());
        self.text_len = bytes.len() as u32;
        let mut dec = Decoder::with_ip(64, bytes, rip, DecoderOptions::NONE);
        // Every byte offset is a potential function slot (x86-64 instructions
        // aren't fixed-width, so a computed/indirect jump could land mid-
        // instruction relative to how we happened to disassemble this byte
        // stream) — `next_pos` tracks that byte stride explicitly.
        //
        // `Decoder::set_ip` only changes the *reported* IP, not the decoder's
        // read cursor (`position`) — using it alone (as this loop used to)
        // left the cursor advancing by each instruction's real length while
        // the reported `inst_rip` silently drifted by only 1 byte per
        // iteration, desyncing after the very first instruction. Explicitly
        // calling `set_position` keeps the cursor and the reported RIP in
        // lockstep with the intended 1-byte stride.
        let mut next_pos: usize = 0;
        while next_pos < bytes.len() {
            dec.set_position(next_pos).expect("next_pos < bytes.len()");
            let inst_rip = rip + next_pos as u64;
            dec.set_ip(inst_rip);
            if !dec.can_decode() {
                break;
            }
            let inst = dec.decode();
            let inst_len = inst.len() as u32;

            if let Some(gate) = &self.slot_assigner {
                if gate.slot_for_pc(inst_rip).is_none() {
                    next_pos += 1;
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
                    next_pos += 1;
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
                            OpKind::Immediate8to64 => 64,
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
                    } else if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Memory {
                        if let Some((dst, dst_size, _z, dst_bit_offset)) = Self::resolve_reg(inst.op0_register()) {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                            self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                            self.emit_memory_load(ctx, rctx, tail_idx, dst_size, false)?;
                            self.emit_i64_add(ctx, rctx, tail_idx)?;
                            if dst_size == 64 && dst_bit_offset == 0 {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                            } else if dst_size == 32 {
                                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                            } else {
                                self.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)?;
                            }
                            Ok(Some(()))
                        } else {
                            Ok(None)
                        }
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
                Mnemonic::Sub => {
                    if inst.op0_kind() == OpKind::Memory {
                        let size_bits = match inst.op1_kind() {
                            OpKind::Immediate8 => 8,
                            OpKind::Immediate16 => 16,
                            OpKind::Immediate8to32 => 32,
                            OpKind::Immediate8to64 => 64,
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
                            this.emit_i64_sub(ctx, rctx, tail_idx)
                        })
                    } else if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Memory {
                        if let Some((dst, dst_size, _z, dst_bit_offset)) = Self::resolve_reg(inst.op0_register()) {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                            self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                            self.emit_memory_load(ctx, rctx, tail_idx, dst_size, false)?;
                            self.emit_i64_sub(ctx, rctx, tail_idx)?;
                            if dst_size == 64 && dst_bit_offset == 0 {
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                            } else if dst_size == 32 {
                                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                            } else {
                                self.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)?;
                            }
                            Ok(Some(()))
                        } else {
                            Ok(None)
                        }
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
                                this.emit_i64_sub(ctx, rctx, tail_idx)?;
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
                    if inst.op0_kind() == OpKind::Memory {
                        match inst.op1_kind() {
                            OpKind::Immediate8
                            | OpKind::Immediate16
                            | OpKind::Immediate32
                            | OpKind::Immediate64
                            | OpKind::Immediate8to32 => {
                                let size_bits = match inst.op1_kind() {
                                    OpKind::Immediate8 => 8,
                                    OpKind::Immediate16 => 16,
                                    OpKind::Immediate8to32 | OpKind::Immediate32 => 32,
                                    OpKind::Immediate64 => 64,
                                    _ => 64,
                                };
                                self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                                self.emit_i64_const(ctx, rctx, tail_idx, inst.immediate64() as i64)?;
                                self.emit_memory_store(ctx, rctx, tail_idx, size_bits)?;
                                Ok(Some(()))
                            }
                            OpKind::Register => {
                                if let Some((src_local, src_size, _z, bit)) = Self::resolve_reg(inst.op1_register()) {
                                    self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?;
                                    if bit > 0 {
                                        self.emit_mask_shift_for_read(ctx, rctx, tail_idx, src_size, bit)?;
                                    }
                                    match src_size {
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
                                    self.emit_memory_store(ctx, rctx, tail_idx, src_size)?;
                                    Ok(Some(()))
                                } else {
                                    Ok(None)
                                }
                            }
                            _ => Ok(None),
                        }
                    } else {
                        if let (OpKind::Register, Some((dst, dst_size, _z, dst_bit_offset))) =
                            (inst.op0_kind(), Self::resolve_reg(inst.op0_register()))
                        {
                            let src_supported = match inst.op1_kind() {
                                OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                                    self.emit_i64_const(ctx, rctx, tail_idx, inst.immediate64() as i64)?;
                                    true
                                }
                                OpKind::Register => {
                                    if let Some((src_local, src_size, _z, bit)) = Self::resolve_reg(inst.op1_register()) {
                                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?;
                                        if bit > 0 {
                                            self.emit_mask_shift_for_read(ctx, rctx, tail_idx, src_size, bit)?;
                                        }
                                        match src_size {
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
                                        true
                                    } else {
                                        false
                                    }
                                }
                                OpKind::Memory => {
                                    self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                                    self.emit_memory_load(ctx, rctx, tail_idx, dst_size, false)?;
                                    true
                                }
                                _ => false,
                            };
                            if !src_supported {
                                Ok(None)
                            } else {
                                if dst_size == 64 && dst_bit_offset == 0 {
                                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                                } else if dst_size == 32 {
                                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
                                } else {
                                    self.emit_subreg_write_rmw(ctx, rctx, tail_idx, dst, dst_size, dst_bit_offset)?;
                                }
                                Ok(Some(()))
                            }
                        } else {
                            Ok(None)
                        }
                    }
                }
                Mnemonic::Nop => Ok(Some(())),
                // HLT halts the (guest) CPU; asm-arch's backend lowers WASM
                // `Unreachable` to a native `hlt` (naive.rs), so this is the
                // exact round-trip inverse — not a fallback/gap.
                Mnemonic::Hlt => { rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?; Ok(Some(())) }
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
                    let scratch = rctx.layout().local(self.tmp_slot, 0); // i64 scratch temp
                    if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Register {
                        if let (Some((dst_local, _, _, _)), Some((src_local, _, _, _))) = (Self::resolve_reg(inst.op0_register()), Self::resolve_reg(inst.op1_register())) {
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst_local))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(scratch))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst_local))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(src_local))?;
                            Ok(Some(()))
                        } else {
                            Ok(None)
                        }
                    } else if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Memory {
                        if let Some((dst_local, _dst_size, _, _)) = Self::resolve_reg(inst.op0_register()) {
                            self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                            self.emit_memory_load(ctx, rctx, tail_idx, _dst_size, false)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(scratch))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst_local))?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64Store(wasm_encoder::MemArg { offset: 0, align: 3, memory_index: 0 }))?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch))?;
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
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::CF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::PF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(2))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::ZF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(6))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::SF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(7))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::OF_LOCAL))?;
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
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::CF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(2))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::PF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(6))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::ZF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(7))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::SF_LOCAL))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(11))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(1))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::OF_LOCAL))?;
                    Ok(Some(()))
                }
                Mnemonic::Cmove | Mnemonic::Cmovne | Mnemonic::Cmovl | Mnemonic::Cmovle
                | Mnemonic::Cmovg | Mnemonic::Cmovge | Mnemonic::Cmovb | Mnemonic::Cmovbe
                | Mnemonic::Cmova | Mnemonic::Cmovae | Mnemonic::Cmovs | Mnemonic::Cmovns
                | Mnemonic::Cmovo | Mnemonic::Cmovno | Mnemonic::Cmovp | Mnemonic::Cmovnp => {
                    let condition_type = match inst.mnemonic() {
                        Mnemonic::Cmove => ConditionType::ZF,
                        Mnemonic::Cmovne => ConditionType::NZF,
                        Mnemonic::Cmovl => ConditionType::SF_NE_OF,
                        Mnemonic::Cmovle => ConditionType::ZF_OR_SF_NE_OF,
                        Mnemonic::Cmovg => ConditionType::NZF_AND_SF_EQ_OF,
                        Mnemonic::Cmovge => ConditionType::SF_EQ_OF,
                        Mnemonic::Cmovb => ConditionType::CF,
                        Mnemonic::Cmovbe => ConditionType::CF_OR_ZF,
                        Mnemonic::Cmova => ConditionType::NCF_AND_NZF,
                        Mnemonic::Cmovae => ConditionType::NCF,
                        Mnemonic::Cmovs => ConditionType::SF,
                        Mnemonic::Cmovns => ConditionType::NSF,
                        Mnemonic::Cmovo => ConditionType::OF,
                        Mnemonic::Cmovno => ConditionType::NOF,
                        Mnemonic::Cmovp => ConditionType::PF,
                        _ => ConditionType::NPF, // Cmovnp
                    };
                    self.handle_cmovcc(ctx, rctx, tail_idx, &inst, condition_type)
                }
                Mnemonic::Not => {
                    match inst.op0_kind() {
                        OpKind::Register => {
                            if let Some((local, size_bits, _z, bit_offset)) = Self::resolve_reg(inst.op0_register()) {
                                match size_bits {
                                    64 => {
                                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(local))?;
                                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1))?;
                                        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
                                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(local))?;
                                        Ok(Some(()))
                                    }
                                    32 => {
                                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(local))?;
                                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1))?;
                                        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
                                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                                        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(local))?;
                                        Ok(Some(()))
                                    }
                                    16 | 8 => {
                                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(local))?;
                                        if bit_offset > 0 {
                                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(bit_offset as i64))?;
                                            rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                                        }
                                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1))?;
                                        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
                                        self.emit_subreg_merge_write(ctx, rctx, tail_idx, local, size_bits, bit_offset)?;
                                        Ok(Some(()))
                                    }
                                    _ => Ok(None),
                                }
                            } else {
                                Ok(None)
                            }
                        }
                        OpKind::Memory => {
                            let size_bits = (inst.memory_size().size() as u32) * 8;
                            self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                            self.emit_memory_load(ctx, rctx, tail_idx, size_bits, false)?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1))?;
                            rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(22))?;
                            self.emit_memory_address(ctx, rctx, tail_idx, &inst)?;
                            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
                            self.emit_memory_store(ctx, rctx, tail_idx, size_bits)?;
                            Ok(Some(()))
                        }
                        _ => Ok(None),
                    }
                }
                Mnemonic::Div => self.handle_div(ctx, rctx, tail_idx, &inst, false),
                Mnemonic::Idiv => self.handle_div(ctx, rctx, tail_idx, &inst, true),
                // SSE scalar floating point (and a few packed bitwise idioms).
                _ => self.handle_sse(ctx, rctx, tail_idx, &inst),
            })?;

            if undecidable_option.is_none() {
                let name = alloc::format!("{:?}", inst.mnemonic());
                self.unsupported_insns.insert(name);
                // Seal (not just feed) the Unreachable: every byte offset is
                // attempted as a slot, so this is routinely a garbage decode
                // from a misaligned re-disassembly, not a real instruction.
                // A bare `feed` leaves the entry open for yecta's fall-
                // through merge to append the *next* slot's translation
                // after this Unreachable, which can splice unrelated
                // control-flow (If/Else/Return/Throw) into a function whose
                // declared result type no longer matches what actually
                // falls through — failing WASM validation at compile time
                // for what should just be a runtime trap.
                rctx.seal_fn(ctx, tail_idx, &Instruction::Unreachable)?;
                next_pos += 1;
                continue;
            }

            next_pos += 1;
        }
        // Seal any functions not explicitly terminated by a branch/call.
        let _ = rctx.seal_remaining(ctx);
        Ok(())
    }

    fn handle_test<F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
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
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::ZF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::SF_LOCAL))?;
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
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::PF_LOCAL))?;
        Ok(Some(()))
    }

    fn handle_cmp<F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
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
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::ZF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(24))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::SF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(23))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtU)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::CF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(23))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(24))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(63))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::OF_LOCAL))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(24))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Popcnt)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::PF_LOCAL))?;
        Ok(Some(()))
    }

    fn handle_jmp<F>(&self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let target = match inst.op0_kind() {
            // iced-x86's decoder already resolves this to the absolute
            // target (next_ip + sign-extended displacement) using the ip
            // we set via `set_ip` before decoding — adding `inst.ip()`
            // again here double-counts it.
            OpKind::NearBranch64 | OpKind::NearBranch32 | OpKind::NearBranch16 => inst.near_branch64(),
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
            rctx.oob_jump(ctx, tail_idx, target, rctx.locals_mark().total_locals)?;
            return Ok(Some(()));
        };
        rctx.jmp(ctx, tail_idx, target_func_idx, rctx.locals_mark().total_locals)?;
        Ok(Some(()))
    }

    /// CMOVcc: build the "would-write" and "unchanged" 64-bit candidates and
    /// pick between them with `Select` + one `LocalSet`. Unlike an
    /// unconditional `mov`, CMOVcc performs *no write at all* when the
    /// condition is false — so a false 32-bit CMOVcc must NOT zero-extend
    /// (the false candidate is the untouched old register), and a 16-bit
    /// CMOVcc's true candidate must merge into the dest's untouched upper 48
    /// bits (since a real 16-bit write, when it happens, never touches them).
    fn handle_cmovcc<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
        condition_type: ConditionType,
    ) -> Result<Option<()>, E> {
        let Some((dst, dst_size, _z, _dst_bit)) = Self::resolve_reg(inst.op0_register()) else {
            return Ok(None);
        };
        // Push the source value (the candidate new value) as i64.
        match inst.op1_kind() {
            OpKind::Register => {
                let Some((src_local, src_size, _z, bit)) = Self::resolve_reg(inst.op1_register()) else {
                    return Ok(None);
                };
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?;
                self.emit_mask_shift_for_read(ctx, rctx, tail_idx, src_size, bit)?;
            }
            OpKind::Memory => {
                self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
                self.emit_memory_load(ctx, rctx, tail_idx, dst_size, false)?;
            }
            _ => return Ok(None),
        }
        // Stack: [src]. Build [true_val, false_val] per dest width — CMOVcc
        // is only ever 16/32/64-bit (never 8-bit), so `resolve_reg` always
        // gives `_dst_bit == 0` here.
        match dst_size {
            64 => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
            }
            32 => {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
            }
            16 => {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(!0xFFFFi64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(dst))?;
            }
            _ => return Ok(None),
        }
        self.push_condition(ctx, rctx, tail_idx, condition_type)?;
        rctx.feed(ctx, tail_idx, &Instruction::Select)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(dst))?;
        Ok(Some(()))
    }

    /// DIV/IDIV. Scoped to the common compiler-generated pattern: the
    /// dividend's incoming high half (RDX/EDX/DX, or AH for the 8-bit form)
    /// is ignored, and the low register is treated as the *entire* dividend,
    /// zero/sign-extended to 64 bits per `signed`. This is exact for the
    /// `xor edx,edx; div` / `cdq; idiv` idiom compilers emit, but wrong for a
    /// genuine 128-bit dividend. Divide-by-zero and signed-overflow
    /// (`INT64_MIN / -1`) both trap via WASM's native `i64.div_u/s`
    /// semantics, matching real DIV/IDIV's `#DE` exception — no explicit
    /// guard needed.
    fn handle_div<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
        signed: bool,
    ) -> Result<Option<()>, E> {
        let size_bits = match inst.op0_kind() {
            OpKind::Register => match Self::resolve_reg(inst.op0_register()) {
                Some((_, s, _, _)) => s,
                None => return Ok(None),
            },
            OpKind::Memory => (inst.memory_size().size() as u32) * 8,
            _ => return Ok(None),
        };

        // Divisor → scratch local 22, zero/sign-extended to i64.
        match inst.op0_kind() {
            OpKind::Register => {
                let Some((r_local, r_size, _z, bit)) = Self::resolve_reg(inst.op0_register()) else {
                    return Ok(None);
                };
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(r_local))?;
                self.emit_mask_shift_for_read(ctx, rctx, tail_idx, r_size, bit)?;
                if signed {
                    match r_size {
                        8 => rctx.feed(ctx, tail_idx, &Instruction::I64Extend8S)?,
                        16 => rctx.feed(ctx, tail_idx, &Instruction::I64Extend16S)?,
                        32 => rctx.feed(ctx, tail_idx, &Instruction::I64Extend32S)?,
                        _ => {}
                    }
                }
            }
            OpKind::Memory => {
                self.emit_memory_address(ctx, rctx, tail_idx, inst)?;
                self.emit_memory_load(ctx, rctx, tail_idx, size_bits, signed)?;
            }
            _ => return Ok(None),
        }
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(22))?;

        // Dividend: the low register (RAX/EAX/AX; AX for the 8-bit form).
        let dividend_size = if size_bits == 8 { 16 } else { size_bits };
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(0))?;
        self.emit_mask_shift_for_read(ctx, rctx, tail_idx, dividend_size, 0)?;
        if signed {
            match dividend_size {
                16 => rctx.feed(ctx, tail_idx, &Instruction::I64Extend16S)?,
                32 => rctx.feed(ctx, tail_idx, &Instruction::I64Extend32S)?,
                _ => {}
            }
        }
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(23))?;

        // Quotient → scratch local 24 (written to its destination below).
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(23))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, if signed { &Instruction::I64DivS } else { &Instruction::I64DivU })?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(24))?;

        // Remainder → high register (RDX/EDX/DX, or AH for the 8-bit form).
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(23))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(22))?;
        rctx.feed(ctx, tail_idx, if signed { &Instruction::I64RemS } else { &Instruction::I64RemU })?;
        match size_bits {
            64 => { rctx.feed(ctx, tail_idx, &Instruction::LocalSet(2))?; }
            32 => {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(2))?;
            }
            16 => self.emit_subreg_merge_write(ctx, rctx, tail_idx, 2, 16, 0)?,
            8 => self.emit_subreg_merge_write(ctx, rctx, tail_idx, 0, 8, 8)?, // AH
            _ => return Ok(None),
        }

        // Quotient → low register (RAX/EAX/AX/AL).
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(24))?;
        match size_bits {
            64 => { rctx.feed(ctx, tail_idx, &Instruction::LocalSet(0))?; }
            32 => {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(0))?;
            }
            16 => self.emit_subreg_merge_write(ctx, rctx, tail_idx, 0, 16, 0)?,
            8 => self.emit_subreg_merge_write(ctx, rctx, tail_idx, 0, 8, 0)?, // AL
            _ => return Ok(None),
        }
        Ok(Some(()))
    }

    /// Push the i32 boolean value of `condition_type` onto the WASM stack,
    /// mirroring `ConditionSnippet::emit_instruction` but through `rctx.feed`
    /// directly — used by `Cmovcc`, which needs the condition as a plain
    /// stack value rather than as a `JumpCallParams` snippet.
    fn push_condition<F>(
        &self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        condition_type: ConditionType,
    ) -> Result<(), E> {
        match condition_type {
            ConditionType::ZF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::ZF_LOCAL))?;
            }
            ConditionType::NZF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::ZF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
            }
            ConditionType::SF_NE_OF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::SF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::OF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?;
            }
            ConditionType::ZF_OR_SF_NE_OF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::ZF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::SF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::OF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Or)?;
            }
            ConditionType::NZF_AND_SF_EQ_OF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::ZF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::SF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::OF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
            }
            ConditionType::SF_EQ_OF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::SF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::OF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
            }
            ConditionType::CF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::CF_LOCAL))?;
            }
            ConditionType::CF_OR_ZF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::CF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::ZF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Or)?;
            }
            ConditionType::NCF_AND_NZF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::CF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::ZF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
            }
            ConditionType::NCF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::CF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
            }
            ConditionType::SF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::SF_LOCAL))?;
            }
            ConditionType::NSF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::SF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
            }
            ConditionType::OF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::OF_LOCAL))?;
            }
            ConditionType::NOF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::OF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
            }
            ConditionType::PF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::PF_LOCAL))?;
            }
            ConditionType::NPF => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::PF_LOCAL))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?;
            }
        }
        Ok(())
    }

    fn handle_conditional_jump<F>(
        &self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        inst: &IxInst,
        condition_type: ConditionType,
    ) -> Result<Option<()>, E> {
        let target = match inst.op0_kind() {
            // iced-x86's decoder already resolves this to the absolute
            // target (next_ip + sign-extended displacement) using the ip
            // we set via `set_ip` before decoding — adding `inst.ip()`
            // again here double-counts it.
            OpKind::NearBranch64 | OpKind::NearBranch32 | OpKind::NearBranch16 => inst.near_branch64(),
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
            rctx.oob_jump(ctx, tail_idx, target, rctx.locals_mark().total_locals)?;
            return Ok(Some(()));
        };
        let condition = ConditionSnippet { condition_type };
        let params = JumpCallParams::conditional_jump(target_func_idx, rctx.locals_mark().total_locals, &condition, rctx.pool());
        rctx.ji_with_params(ctx, tail_idx, params)?;
        Ok(Some(()))
    }

    fn handle_call<F>(&mut self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let return_addr = inst.next_ip();
        let target = match inst.op0_kind() {
            // iced-x86's decoder already resolves this to the absolute
            // target (next_ip + sign-extended displacement) using the ip
            // we set via `set_ip` before decoding — adding `inst.ip()`
            // again here double-counts it.
            OpKind::NearBranch64 | OpKind::NearBranch32 | OpKind::NearBranch16 => inst.near_branch64(),
            _ => return Ok(None),
        };

        let use_speculative = self.enable_speculative_calls && rctx.escape_tag().is_some();

        if use_speculative {
            let escape_tag = rctx.escape_tag().unwrap();
            let Some(target_func) = self.rip_to_func_idx(rctx, target) else {
                rctx.oob_jump(ctx, tail_idx, target, rctx.locals_mark().total_locals)?;
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
                .with_fixup(Self::EXPECTED_RA_LOCAL, &expected_ra_snippet);
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
            if let Some((import_idx, sym)) = self.lookup_plt_import(target) {
                self.emit_plt_import_call(ctx, rctx, tail_idx, import_idx, sym)?;
                return Ok(Some(()));
            }
            let Some(target_func_idx) = self.rip_to_func_idx(rctx, target) else {
                rctx.oob_jump(ctx, tail_idx, target, rctx.locals_mark().total_locals)?;
                return Ok(Some(()));
            };
            rctx.jmp(ctx, tail_idx, target_func_idx, rctx.locals_mark().total_locals)?;
            Ok(Some(()))
        }
    }

    fn handle_ret<F>(&mut self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
        let stack_cleanup = if inst.op_count() > 0 { match inst.op0_kind() { OpKind::Immediate16 | OpKind::Immediate32 => inst.immediate16() as u64, _ => 0 } } else { 0 };
        let use_speculative = self.enable_speculative_calls && rctx.escape_tag().is_some();

        if use_speculative {
            let escape_tag = rctx.escape_tag().unwrap();
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            self.emit_memory_load(ctx, rctx, tail_idx, 64, false)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::EXPECTED_RA_LOCAL))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Eq)?;
            rctx.feed(ctx, tail_idx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(4))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(8 + stack_cleanup as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(4))?;
            // The generated function's ABI is (register_file) -> (register_file)
            // — a bare `Return` must push the full, current register file as
            // its results, matching the shape `ret`'s `Throw` below pushes.
            for p in 0..rctx.locals_mark().total_locals {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(p))?;
            }
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

    fn handle_push<F>(&mut self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
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

    fn handle_pop<F>(&mut self, ctx: &mut Context, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, inst: &IxInst) -> Result<Option<()>, E> {
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
