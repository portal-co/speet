//! Minimal x86_64 to WebAssembly recompiler
//!
//! Supports a small subset of integer instructions for demonstration purposes.
#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use wax_core::build::InstructionSink;
use wasm_encoder::Instruction;
use yecta::{EscapeTag, FuncIdx, JumpCallParams, Pool, Reactor, TableIdx, TypeIdx};
use wax_core::build::InstructionSource;


use iced_x86::{Decoder, DecoderOptions, Instruction as IxInst, Mnemonic, Register, OpKind};

#[derive(Clone, Copy)]
enum ConditionType {
    ZF, NZF, SF_NE_OF, ZF_OR_SF_NE_OF, NZF_AND_SF_EQ_OF, SF_EQ_OF,
    CF, CF_OR_ZF, NCF_AND_NZF, NCF, SF, NSF, OF, NOF, PF, NPF,
}

// Struct to represent a condition that can be used as a Snippet
#[derive(Clone, Copy)]
struct ConditionSnippet {
    condition_type: ConditionType,
}

impl<E> wax_core::build::InstructionOperatorSource<E> for ConditionSnippet {
    fn emit(&self, sink: &mut (dyn wax_core::build::InstructionOperatorSink<E> + '_)) -> Result<(), E> {
        // For simple structs, we can delegate to emit_instruction
        self.emit_instruction(sink)
    }
}

impl<E> wax_core::build::InstructionSource<E> for ConditionSnippet {
    fn emit_instruction(&self, sink: &mut (dyn wax_core::build::InstructionSink<E> + '_)) -> Result<(), E> {
        match self.condition_type {
            ConditionType::ZF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::ZF_LOCAL))?;
            }
            ConditionType::NZF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::ZF_LOCAL))?;
                sink.instruction(&Instruction::I32Eqz)?;
            }
            ConditionType::SF_NE_OF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::SF_LOCAL))?;
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::OF_LOCAL))?;
                sink.instruction(&Instruction::I32Xor)?;
            }
            ConditionType::ZF_OR_SF_NE_OF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::ZF_LOCAL))?;
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::SF_LOCAL))?;
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::OF_LOCAL))?;
                sink.instruction(&Instruction::I32Xor)?;
                sink.instruction(&Instruction::I32Or)?;
            }
            ConditionType::NZF_AND_SF_EQ_OF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::ZF_LOCAL))?;
                sink.instruction(&Instruction::I32Eqz)?;
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::SF_LOCAL))?;
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::OF_LOCAL))?;
                sink.instruction(&Instruction::I32Xor)?;
                sink.instruction(&Instruction::I32Eqz)?;
                sink.instruction(&Instruction::I32And)?;
            }
            ConditionType::SF_EQ_OF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::SF_LOCAL))?;
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::OF_LOCAL))?;
                sink.instruction(&Instruction::I32Xor)?;
                sink.instruction(&Instruction::I32Eqz)?;
            }
            ConditionType::CF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::CF_LOCAL))?;
            }
            ConditionType::CF_OR_ZF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::CF_LOCAL))?;
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::ZF_LOCAL))?;
                sink.instruction(&Instruction::I32Or)?;
            }
            ConditionType::NCF_AND_NZF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::CF_LOCAL))?;
                sink.instruction(&Instruction::I32Eqz)?;
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::ZF_LOCAL))?;
                sink.instruction(&Instruction::I32Eqz)?;
                sink.instruction(&Instruction::I32And)?;
            }
            ConditionType::NCF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::CF_LOCAL))?;
                sink.instruction(&Instruction::I32Eqz)?;
            }
            ConditionType::SF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::SF_LOCAL))?;
            }
            ConditionType::NSF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::SF_LOCAL))?;
                sink.instruction(&Instruction::I32Eqz)?;
            }
            ConditionType::OF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::OF_LOCAL))?;
            }
            ConditionType::NOF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::OF_LOCAL))?;
                sink.instruction(&Instruction::I32Eqz)?;
            }
            ConditionType::PF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::PF_LOCAL))?;
            }
            ConditionType::NPF => {
                sink.instruction(&Instruction::LocalGet(X86Recompiler::<E, wasm_encoder::Function>::PF_LOCAL))?;
                sink.instruction(&Instruction::I32Eqz)?;
            }
        }
        Ok(())
    }
}

/// Simple x86_64 recompiler for integer ops
pub struct X86Recompiler<E, F: InstructionSink<E>> {
    reactor: Reactor<E, F>,
    pool: Pool,
    escape_tag: Option<EscapeTag>,
    base_rip: u64,
    hints: Vec<u8>,
}

impl<E, F: InstructionSink<E>> X86Recompiler<E, F> {
    pub fn new() -> Self {
        Self::new_with_base_rip(0)
    }

    pub fn new_with_base_rip(base_rip: u64) -> Self {
        Self {
            reactor: Reactor::default(),
            pool: Pool { table: TableIdx(0), ty: TypeIdx(0) },
            escape_tag: None,
            base_rip,
            hints: Vec::new(),
        }
    }

    pub fn base_func_offset(&self) -> u32 {
        self.reactor.base_func_offset()
    }

    pub fn set_base_func_offset(&mut self, offset: u32) {
        self.reactor.set_base_func_offset(offset);
    }

    fn rip_to_func_idx(&self, rip: u64) -> FuncIdx {
        FuncIdx((rip.wrapping_sub(self.base_rip) / 1) as u32)
    }

    fn init_function(&mut self, _rip: u64, inst_len: u32, num_temps: u32, f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, wasm_encoder::ValType)> + '_)) -> F + '_)) {
        // For simplicity, model 16 general purpose 64-bit regs as locals 0-15
        // PC in local 16 (i32)
        // Condition flags: ZF(17), SF(18), CF(19), OF(20), PF(21) as i32
        // Temps after that
        let locals = [
            (16, wasm_encoder::ValType::I64),  // registers
            (1, wasm_encoder::ValType::I32),   // PC
            (5, wasm_encoder::ValType::I32),   // condition flags: ZF, SF, CF, OF, PF
            (num_temps, wasm_encoder::ValType::I64)  // temps
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

    fn emit_i64_const(&mut self, value: i64) -> Result<(), E> {
        self.reactor.feed(&Instruction::I64Const(value))
    }

    fn emit_i64_add(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64Add) }
    fn emit_i64_sub(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64Sub) }
    fn emit_i64_mul(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64Mul) }
    fn emit_i64_and(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64And) }
    fn emit_i64_or(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64Or) }
    fn emit_i64_xor(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64Xor) }
    fn emit_i64_shl(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64Shl) }
    fn emit_i64_shr_u(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64ShrU) }
    fn emit_i64_shr_s(&mut self) -> Result<(), E> { self.reactor.feed(&Instruction::I64ShrS) }

    // Condition flag helpers
    const ZF_LOCAL: u32 = 17;
    const SF_LOCAL: u32 = 18;
    const CF_LOCAL: u32 = 19;
    const OF_LOCAL: u32 = 20;
    const PF_LOCAL: u32 = 21;

    fn set_zf(&mut self, value: bool) -> Result<(), E> {
        self.reactor.feed(&Instruction::I32Const(if value { 1 } else { 0 }))?;
        self.reactor.feed(&Instruction::LocalSet(Self::ZF_LOCAL))
    }

    fn set_sf(&mut self, value: bool) -> Result<(), E> {
        self.reactor.feed(&Instruction::I32Const(if value { 1 } else { 0 }))?;
        self.reactor.feed(&Instruction::LocalSet(Self::SF_LOCAL))
    }

    fn set_cf(&mut self, value: bool) -> Result<(), E> {
        self.reactor.feed(&Instruction::I32Const(if value { 1 } else { 0 }))?;
        self.reactor.feed(&Instruction::LocalSet(Self::CF_LOCAL))
    }

    fn set_of(&mut self, value: bool) -> Result<(), E> {
        self.reactor.feed(&Instruction::I32Const(if value { 1 } else { 0 }))?;
        self.reactor.feed(&Instruction::LocalSet(Self::OF_LOCAL))
    }

    fn set_pf(&mut self, value: bool) -> Result<(), E> {
        self.reactor.feed(&Instruction::I32Const(if value { 1 } else { 0 }))?;
        self.reactor.feed(&Instruction::LocalSet(Self::PF_LOCAL))
    }



    // Helper to compute parity flag (even number of 1 bits in lowest byte)
    fn compute_parity(&mut self) -> Result<(), E> {
        // Assume value is on stack (i64)
        // Extract lowest byte: value & 0xFF
        self.reactor.feed(&Instruction::I64Const(0xFF))?;
        self.reactor.feed(&Instruction::I64And)?;
        // Count bits: use popcnt if available, otherwise simulate
        // For simplicity, we'll implement a basic parity check
        // This is a simplified version - real parity counts all bits in lowest byte
        self.reactor.feed(&Instruction::I32WrapI64)?;
        // Simple parity: check if number of 1s is even
        // For now, just set to 0 (even parity) - this is a simplification
        self.reactor.feed(&Instruction::Drop)?;
        self.reactor.feed(&Instruction::I32Const(0))?; // Assume even parity for simplicity
        self.reactor.feed(&Instruction::LocalSet(Self::PF_LOCAL))
    }

    // Helper to set flags after arithmetic operation
    fn set_flags_after_operation(&mut self, result: i64, operand1: i64, operand2: i64, is_subtraction: bool) -> Result<(), E> {
        // ZF: result == 0
        self.set_zf(result == 0)?;

        // SF: result < 0 (for signed)
        self.set_sf(result < 0)?;

        // For CF and OF, we need to detect carry/borrow and overflow
        // This is simplified - real implementation would need proper overflow detection

        // CF: carry flag (simplified)
        if is_subtraction {
            // For subtraction: CF if borrow occurred (operand1 < operand2 for unsigned)
            self.set_cf((operand1 as u64) < (operand2 as u64))?;
        } else {
            // For addition: CF if result < operand1 (unsigned overflow)
            self.set_cf((result as u64) < (operand1 as u64))?;
        }

        // OF: overflow flag (simplified - check if sign changed unexpectedly)
        let op1_sign = operand1 < 0;
        let op2_sign = operand2 < 0;
        let result_sign = result < 0;
        if is_subtraction {
            // For subtraction: overflow if (op1 positive, op2 negative, result negative) or (op1 negative, op2 positive, result positive)
            let overflow = (op1_sign && !op2_sign && !result_sign) || (!op1_sign && op2_sign && result_sign);
            self.set_of(overflow)?;
        } else {
            // For addition: overflow if both operands same sign but result different sign
            let overflow = (op1_sign == op2_sign) && (op1_sign != result_sign);
            self.set_of(overflow)?;
        }

        // PF: parity (simplified)
        self.compute_parity()?;

        Ok(())
    }

    fn emit_memory_load(&mut self, size_bits: u32, signed: bool) -> Result<(), E> {
        use wasm_encoder::MemArg;
        match (size_bits, signed) {
            (8, true) => self.reactor.feed(&Instruction::I64Load8S(MemArg { offset: 0, align: 0, memory_index: 0 })),
            (8, false) => self.reactor.feed(&Instruction::I64Load8U(MemArg { offset: 0, align: 0, memory_index: 0 })),
            (16, true) => self.reactor.feed(&Instruction::I64Load16S(MemArg { offset: 0, align: 1, memory_index: 0 })),
            (16, false) => self.reactor.feed(&Instruction::I64Load16U(MemArg { offset: 0, align: 1, memory_index: 0 })),
            (32, true) => self.reactor.feed(&Instruction::I64Load32S(MemArg { offset: 0, align: 2, memory_index: 0 })),
            (32, false) => self.reactor.feed(&Instruction::I64Load32U(MemArg { offset: 0, align: 2, memory_index: 0 })),
            (64, _) => self.reactor.feed(&Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index: 0 })),
            _ => { self.reactor.feed(&Instruction::Unreachable) }
        }
    }

    fn emit_memory_store(&mut self, size_bits: u32) -> Result<(), E> {
        use wasm_encoder::MemArg;
        match size_bits {
            8 => self.reactor.feed(&Instruction::I64Store8(MemArg { offset: 0, align: 0, memory_index: 0 })),
            16 => self.reactor.feed(&Instruction::I64Store16(MemArg { offset: 0, align: 1, memory_index: 0 })),
            32 => self.reactor.feed(&Instruction::I64Store32(MemArg { offset: 0, align: 2, memory_index: 0 })),
            64 => self.reactor.feed(&Instruction::I64Store(MemArg { offset: 0, align: 3, memory_index: 0 })),
            _ => { self.reactor.feed(&Instruction::Unreachable) }
        }
    }

    // Helpers for sub-register read/write
    fn emit_mask_shift_for_read(&mut self, size_bits: u32, bit_offset: u32) -> Result<(), E> {
        // shift right by bit_offset then mask size_bits
        if bit_offset > 0 {
            self.reactor.feed(&Instruction::I64Const(bit_offset as i64))?;
            self.reactor.feed(&Instruction::I64ShrU)?;
        }
        match size_bits {
            64 => { /* no mask */ }
            32 => { /* locals model 32-bit values as zero-extended into 64, so no mask needed for reads */ }
            16 => { self.reactor.feed(&Instruction::I64Const(0xFFFF))?; self.reactor.feed(&Instruction::I64And)?; }
            8 => { self.reactor.feed(&Instruction::I64Const(0xFF))?; self.reactor.feed(&Instruction::I64And)?; }
            _ => {}
        }
        Ok(())
    }

    fn emit_subreg_write_rmw(&mut self, local: u32, size_bits: u32, bit_offset: u32) -> Result<(), E> {
        // Assume new value is on stack (i64) and we need to write it into `local` at bit_offset preserving other bits.
        // Steps:
        // local_val = local.get(local)
        // mask = ((1 << size_bits) - 1) << bit_offset
        // cleared = local_val & ~mask
        // new_shifted = (new_value & ((1<<size_bits)-1)) << bit_offset
        // combined = cleared | new_shifted
        // local.set(local, combined)

        // local.get(local)
        self.reactor.feed(&Instruction::LocalGet(local))?;
        // store original in temp (we rely on temp locals being available after local 16). We'll use LocalSet(17) and LocalGet(17).
        self.reactor.feed(&Instruction::LocalSet(17))?;
        // compute mask = (1<<size_bits)-1
        let mask: i64 = if size_bits == 64 { -1i64 } else { ((1u128 << size_bits) - 1) as i64 };
        self.reactor.feed(&Instruction::I64Const(mask))?;
        // shift mask left by bit_offset
        if bit_offset > 0 {
            self.reactor.feed(&Instruction::I64Const(bit_offset as i64))?;
            self.reactor.feed(&Instruction::I64Shl)?;
        }
        // invert mask -> ~mask
        self.reactor.feed(&Instruction::I64Const(-1))?; // -1 is all ones
        self.reactor.feed(&Instruction::I64Xor)?; // ~mask = mask ^ -1
        // get original
        self.reactor.feed(&Instruction::LocalGet(17))?;
        // cleared = original & ~mask
        self.reactor.feed(&Instruction::I64And)?;
        // now compute new_shifted: we assume new_value is currently on top of stack
        // mask = (1<<size_bits)-1 (again)
        let small_mask: i64 = if size_bits == 64 { -1i64 } else { ((1u128 << size_bits) - 1) as i64 };
        self.reactor.feed(&Instruction::I64Const(small_mask))?;
        self.reactor.feed(&Instruction::I64And)?; // new_value & small_mask
        if bit_offset > 0 {
            self.reactor.feed(&Instruction::I64Const(bit_offset as i64))?;
            self.reactor.feed(&Instruction::I64Shl)?; // << bit_offset
        }
        // combined = cleared | new_shifted
        self.reactor.feed(&Instruction::I64Or)?;
        // store back into local
        self.reactor.feed(&Instruction::LocalSet(local))?;
        Ok(())
    }

    /// Translate a sequence of bytes starting at `rip` to wasm using the provided function-local builder
    /// This creates one yecta function per instruction; the `inst_len` is passed to yecta so fallthroughs are the correct distance.
    pub fn translate_bytes(
        &mut self,
        bytes: &[u8],
        rip: u64,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, wasm_encoder::ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        let mut dec = Decoder::with_ip(64, bytes, rip, DecoderOptions::NONE);
        while dec.can_decode() {
            let inst = dec.decode();
            let inst_len = inst.len() as u32;
            let inst_rip = dec.ip() - inst_len as u64; // decoder advanced

            self.init_function(inst_rip, inst_len, 4, f);
            // store rip into local 16
            self.reactor.feed(&Instruction::I32Const(inst_rip as i32))?;
            self.reactor.feed(&Instruction::LocalSet(16))?;

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
                                if let Some((_, r_size, _, _)) = Self::resolve_reg(inst.op1_register()) {
                                    r_size
                                } else { 64 }
                            }
                            _ => 64,
                        };
                        self.handle_memory_rmw(&inst, size_bits, |this| this.emit_i64_add())
                    } else {
                        self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    this.reactor.feed(&Instruction::LocalGet(dst))?;
                    match src {
                        Operand::Imm(i) => { this.emit_i64_const(i)?; }
                        Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                        Operand::RegWithSize(r, sz, bit) => {
                            this.reactor.feed(&Instruction::LocalGet(r))?;
                            this.emit_mask_shift_for_read(sz, bit)?;
                        }
                    }
                    this.emit_i64_add()?;
                    if dst_size == 64 && dst_bit_offset == 0 {
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else if dst_size == 32 {
                        this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                        this.reactor.feed(&Instruction::I64And)?;
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else {
                        this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                    }
                })
                    }
                },
                Mnemonic::Imul => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    this.reactor.feed(&Instruction::LocalGet(dst))?;
                    match src {
                        Operand::Imm(i) => { this.emit_i64_const(i)?; }
                        Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                        Operand::RegWithSize(r, sz, bit) => {
                            this.reactor.feed(&Instruction::LocalGet(r))?;
                            this.emit_mask_shift_for_read(sz, bit)?;
                        }
                    }
                    this.emit_i64_mul()?;
                    if dst_size == 64 && dst_bit_offset == 0 {
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else if dst_size == 32 {
                        this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                        this.reactor.feed(&Instruction::I64And)?;
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else {
                        this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                    }
                }),
                Mnemonic::And => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    this.reactor.feed(&Instruction::LocalGet(dst))?;
                    match src {
                        Operand::Imm(i) => { this.emit_i64_const(i)?; }
                        Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                        Operand::RegWithSize(r, sz, bit) => {
                            this.reactor.feed(&Instruction::LocalGet(r))?;
                            this.emit_mask_shift_for_read(sz, bit)?;
                        }
                    }
                    this.emit_i64_and()?;
                    if dst_size == 64 && dst_bit_offset == 0 {
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else if dst_size == 32 {
                        this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                        this.reactor.feed(&Instruction::I64And)?;
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else {
                        this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                    }
                }),
                Mnemonic::Or => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    this.reactor.feed(&Instruction::LocalGet(dst))?;
                    match src {
                        Operand::Imm(i) => { this.emit_i64_const(i)?; }
                        Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                        Operand::RegWithSize(r, sz, bit) => {
                            this.reactor.feed(&Instruction::LocalGet(r))?;
                            this.emit_mask_shift_for_read(sz, bit)?;
                        }
                    }
                    this.emit_i64_or()?;
                    if dst_size == 64 && dst_bit_offset == 0 {
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else if dst_size == 32 {
                        this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                        this.reactor.feed(&Instruction::I64And)?;
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else {
                        this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                    }
                }),
                Mnemonic::Xor => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    this.reactor.feed(&Instruction::LocalGet(dst))?;
                    match src {
                        Operand::Imm(i) => { this.emit_i64_const(i)?; }
                        Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                        Operand::RegWithSize(r, sz, bit) => {
                            this.reactor.feed(&Instruction::LocalGet(r))?;
                            this.emit_mask_shift_for_read(sz, bit)?;
                        }
                    }
                    this.emit_i64_xor()?;
                    if dst_size == 64 && dst_bit_offset == 0 {
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else if dst_size == 32 {
                        this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                        this.reactor.feed(&Instruction::I64And)?;
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else {
                        this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                    }
                }),
                Mnemonic::Shl => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    this.reactor.feed(&Instruction::LocalGet(dst))?;
                    match src {
                        Operand::Imm(i) => { this.emit_i64_const(i)?; }
                        Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                        Operand::RegWithSize(r, sz, bit) => {
                            this.reactor.feed(&Instruction::LocalGet(r))?;
                            this.emit_mask_shift_for_read(sz, bit)?;
                        }
                    }
                    this.emit_i64_shl()?;
                    if dst_size == 64 && dst_bit_offset == 0 {
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else if dst_size == 32 {
                        this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                        this.reactor.feed(&Instruction::I64And)?;
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else {
                        this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                    }
                }),
                Mnemonic::Shr | Mnemonic::Sar => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    this.reactor.feed(&Instruction::LocalGet(dst))?;
                    match src {
                        Operand::Imm(i) => { this.emit_i64_const(i)?; }
                        Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                        Operand::RegWithSize(r, sz, bit) => {
                            this.reactor.feed(&Instruction::LocalGet(r))?;
                            this.emit_mask_shift_for_read(sz, bit)?;
                        }
                    }
                    // choose arithmetic vs logical based on mnemonic
                    // if mnemonic was Sar use arithmetic (shr_s), otherwise Shr uses logical (shr_u)
                    if inst.mnemonic() == iced_x86::Mnemonic::Sar {
                        this.emit_i64_shr_s()?;
                    } else {
                        this.emit_i64_shr_u()?;
                    }
                    if dst_size == 64 && dst_bit_offset == 0 {
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else if dst_size == 32 {
                        this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                        this.reactor.feed(&Instruction::I64And)?;
                        this.reactor.feed(&Instruction::LocalSet(dst))
                    } else {
                        this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                    }
                }),
                // MOV: moves between regs/mem/imm
                Mnemonic::Mov => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    match inst.op1_kind() {
                        OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                            if let Operand::Imm(i) = src { this.emit_i64_const(i)?; }
                            if dst_size == 64 && dst_bit_offset == 0 {
                                this.reactor.feed(&Instruction::LocalSet(dst))
                            } else if dst_size == 32 {
                                this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                                this.reactor.feed(&Instruction::I64And)?;
                                this.reactor.feed(&Instruction::LocalSet(dst))
                            } else {
                                this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                            }
                        }
                        OpKind::Register => {
                            match src {
                                Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                                Operand::RegWithSize(r, sz, bit) => { this.reactor.feed(&Instruction::LocalGet(r))?; this.emit_mask_shift_for_read(sz, bit)?; }
                                _ => return this.reactor.feed(&Instruction::Unreachable),
                            }
                            if dst_size == 64 && dst_bit_offset == 0 {
                                this.reactor.feed(&Instruction::LocalSet(dst))
                            } else if dst_size == 32 {
                                this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                                this.reactor.feed(&Instruction::I64And)?;
                                this.reactor.feed(&Instruction::LocalSet(dst))
                            } else {
                                this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                            }
                        }
                        OpKind::Memory => {
                            // load from memory into dst
                            this.emit_memory_address(&inst)?;
                            // load according to destination size (zero-extend)
                            this.emit_memory_load(dst_size, false)?;
                            if dst_size == 64 && dst_bit_offset == 0 {
                                this.reactor.feed(&Instruction::LocalSet(dst))
                            } else if dst_size == 32 {
                                this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
                                this.reactor.feed(&Instruction::I64And)?;
                                this.reactor.feed(&Instruction::LocalSet(dst))
                            } else {
                                this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset)
                            }
                        }
                        _ => return this.reactor.feed(&Instruction::Unreachable),
                    }
                }),
                // LEA: compute effective address into dest register
                Mnemonic::Lea => {
                    // dest must be register, src must be memory
                    if inst.op0_kind() != OpKind::Register || inst.op1_kind() != OpKind::Memory { Ok(None) }
                    else if let Some((dst_local, _dst_size, _z, _bit)) = Self::resolve_reg(inst.op0_register()) {
                        self.emit_memory_address(&inst)?;
                        self.reactor.feed(&Instruction::LocalSet(dst_local))?;
                        Ok(Some(()))
                    } else { Ok(None) }
                }
                // MOVSX: sign-extend from smaller reg/mem into reg
                Mnemonic::Movsx => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    match inst.op1_kind() {
                        OpKind::Register => {
                            match src {
                                Operand::RegWithSize(r_local, r_size, r_bit) => {
                                    this.reactor.feed(&Instruction::LocalGet(r_local))?;
                                    if r_bit > 0 { this.emit_mask_shift_for_read(r_size, r_bit)?; }
                                    match r_size {
                                        8 => {
                                            // sign-extend 8-bit: (v << 56) >> 56 (arith)
                                            this.reactor.feed(&Instruction::I64Const(56))?; this.reactor.feed(&Instruction::I64Shl)?;
                                            this.reactor.feed(&Instruction::I64Const(56))?; this.reactor.feed(&Instruction::I64ShrS)?;
                                        }
                                        16 => {
                                            this.reactor.feed(&Instruction::I64Const(48))?; this.reactor.feed(&Instruction::I64Shl)?;
                                            this.reactor.feed(&Instruction::I64Const(48))?; this.reactor.feed(&Instruction::I64ShrS)?;
                                        }
                                        32 => {
                                            // use i64.extend_i32_s if available via instruction
                                            this.reactor.feed(&Instruction::I64ExtendI32S)?;
                                        }
                                        64 => { /* already 64-bit */ }
                                        _ => {}
                                    }
                                }
                                _ => return this.reactor.feed(&Instruction::Unreachable),
                            }
                        }
                        OpKind::Memory => {
                            this.emit_memory_address(&inst)?;
                            // memory signed load of operand size; determine size from inst.memory_size if available
                            // fallback to 64
                            // For now assume 8/16/32/64 based on op1_operand size: use inst.memory_size().size() if possible
                            let mem_size_bits = match inst.memory_size() { // may exist
                                iced_x86::MemorySize::UInt8 => 8,
                                iced_x86::MemorySize::UInt16 => 16,
                                iced_x86::MemorySize::UInt32 => 32,
                                iced_x86::MemorySize::UInt64 => 64,
                                _ => 64,
                            };
                            this.emit_memory_load(mem_size_bits, true)?;
                        }
                        _ => return this.reactor.feed(&Instruction::Unreachable),
                    }
                    if dst_size == 64 && dst_bit_offset == 0 { this.reactor.feed(&Instruction::LocalSet(dst)) }
                    else if dst_size == 32 { this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?; this.reactor.feed(&Instruction::I64And)?; this.reactor.feed(&Instruction::LocalSet(dst)) }
                    else { this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset) }
                }),
                // MOVZX: zero-extend from smaller reg/mem into reg
                Mnemonic::Movzx => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    match inst.op1_kind() {
                        OpKind::Register => {
                            match src {
                                Operand::RegWithSize(r_local, r_size, r_bit) => {
                                    this.reactor.feed(&Instruction::LocalGet(r_local))?;
                                    if r_bit > 0 { this.emit_mask_shift_for_read(r_size, r_bit)?; }
                                    match r_size {
                                        8 => { this.reactor.feed(&Instruction::I64Const(0xFF))?; this.reactor.feed(&Instruction::I64And)?; }
                                        16 => { this.reactor.feed(&Instruction::I64Const(0xFFFF))?; this.reactor.feed(&Instruction::I64And)?; }
                                        32 => { /* already zero-extended */ }
                                        64 => { /* already 64 */ }
                                        _ => {}
                                    }
                                }
                                _ => return this.reactor.feed(&Instruction::Unreachable),
                            }
                        }
                        OpKind::Memory => {
                            this.emit_memory_address(&inst)?;
                            let mem_size_bits = match inst.memory_size() {
                                iced_x86::MemorySize::UInt8 => 8,
                                iced_x86::MemorySize::UInt16 => 16,
                                iced_x86::MemorySize::UInt32 => 32,
                                iced_x86::MemorySize::UInt64 => 64,
                                _ => 64,
                            };
                            this.emit_memory_load(mem_size_bits, false)?;
                        }
                        _ => return this.reactor.feed(&Instruction::Unreachable),
                    }
                    if dst_size == 64 && dst_bit_offset == 0 { this.reactor.feed(&Instruction::LocalSet(dst)) }
                    else if dst_size == 32 { this.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?; this.reactor.feed(&Instruction::I64And)?; this.reactor.feed(&Instruction::LocalSet(dst)) }
                    else { this.emit_subreg_write_rmw(dst, dst_size, dst_bit_offset) }
                }),
                // XCHG: exchange reg and reg/mem
                Mnemonic::Xchg => {
                    // support reg,reg and reg,mem
                    if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Register {
                        if let (Some((dst_local, _dst_size, _z, _dbit)), Some((src_local, _src_size, _sz, _sbit))) = (Self::resolve_reg(inst.op0_register()), Self::resolve_reg(inst.op1_register())) {
                            // temp in local 17
                            self.reactor.feed(&Instruction::LocalGet(dst_local))?;
                            self.reactor.feed(&Instruction::LocalSet(17))?;
                            self.reactor.feed(&Instruction::LocalGet(src_local))?;
                            self.reactor.feed(&Instruction::LocalSet(dst_local))?;
                            self.reactor.feed(&Instruction::LocalGet(17))?;
                            self.reactor.feed(&Instruction::LocalSet(src_local))?;
                            Ok(Some(()))
                        } else { Ok(None) }
                    } else if inst.op0_kind() == OpKind::Register && inst.op1_kind() == OpKind::Memory {
                        // mem <-> reg: load mem, store reg to mem, set reg to loaded value
                        if let Some((dst_local, _dst_size, _z, _dbit)) = Self::resolve_reg(inst.op0_register()) {
                            // compute addr
                            self.emit_memory_address(&inst)?;
                            // load old value with register size
                            self.emit_memory_load(_dst_size, false)?;
                            // store old value to temp
                            self.reactor.feed(&Instruction::LocalSet(17))?;
                            // store register value to memory
                            self.reactor.feed(&Instruction::LocalGet(dst_local))?;
                            self.reactor.feed(&Instruction::I64Store(wasm_encoder::MemArg { offset: 0, align: 3, memory_index: 0 }))?;
                            // set reg to loaded
                            self.reactor.feed(&Instruction::LocalGet(17))?;
                            self.reactor.feed(&Instruction::LocalSet(dst_local))?;
                            Ok(Some(()))
                        } else { Ok(None) }
                    } else { Ok(None) }
                },
                 Mnemonic::Test => self.handle_test(&inst),
                 Mnemonic::Cmp => self.handle_cmp(&inst),
                 Mnemonic::Jmp => self.handle_jmp(&inst),
                 Mnemonic::Je => self.handle_conditional_jump(&inst, ConditionType::ZF),
                 Mnemonic::Jne => self.handle_conditional_jump(&inst, ConditionType::NZF),
                 Mnemonic::Jl => self.handle_conditional_jump(&inst, ConditionType::SF_NE_OF),
                 Mnemonic::Jle => self.handle_conditional_jump(&inst, ConditionType::ZF_OR_SF_NE_OF),
                 Mnemonic::Jg => self.handle_conditional_jump(&inst, ConditionType::NZF_AND_SF_EQ_OF),
                 Mnemonic::Jge => self.handle_conditional_jump(&inst, ConditionType::SF_EQ_OF),
                 Mnemonic::Jb => self.handle_conditional_jump(&inst, ConditionType::CF),
                 Mnemonic::Jbe => self.handle_conditional_jump(&inst, ConditionType::CF_OR_ZF),
                 Mnemonic::Ja => self.handle_conditional_jump(&inst, ConditionType::NCF_AND_NZF),
                 Mnemonic::Jae => self.handle_conditional_jump(&inst, ConditionType::NCF),
                 Mnemonic::Js => self.handle_conditional_jump(&inst, ConditionType::SF),
                 Mnemonic::Jns => self.handle_conditional_jump(&inst, ConditionType::NSF),
                 Mnemonic::Jo => self.handle_conditional_jump(&inst, ConditionType::OF),
                 Mnemonic::Jno => self.handle_conditional_jump(&inst, ConditionType::NOF),
                 Mnemonic::Jp => self.handle_conditional_jump(&inst, ConditionType::PF),
                 Mnemonic::Jnp => self.handle_conditional_jump(&inst, ConditionType::NPF),
                 _ => Ok(None),
             })?;

            // If handler returned None (undecidable operands), emit unreachable
            if undecidable_option.is_none() {
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // Rewind decoder to the byte after the current instruction RIP
            // Use inst_rip + 1 to allow variable-length instructions to be found at any byte
            dec.set_ip(inst_rip + 1);
        }
        Ok(())
    }

    // Helper: handle read-modify-write operations for memory destinations
    fn handle_memory_rmw<Op>(&mut self, inst: &IxInst, size_bits: u32, mut operation: Op) -> Result<Option<()>, E>
    where
        Op: FnMut(&mut Self) -> Result<(), E>,
    {
        use iced_x86::OpKind;
        
        // Load current value from memory
        self.emit_memory_address(&inst)?;
        self.emit_memory_load(size_bits, false)?;
        
        // Get source value based on operand kind
        match inst.op1_kind() {
            OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                let imm = inst.immediate64() as i64;
                self.emit_i64_const(imm)?;
            }
            OpKind::Register => {
                if let Some((r_local, r_size, _rz, bit)) = Self::resolve_reg(inst.op1_register()) {
                    self.reactor.feed(&Instruction::LocalGet(r_local))?;
                    if bit > 0 {
                        self.emit_mask_shift_for_read(r_size, bit)?;
                    }
                    // Apply mask for sub-registers
                    match r_size {
                        8 => { self.reactor.feed(&Instruction::I64Const(0xFF))?; self.reactor.feed(&Instruction::I64And)?; }
                        16 => { self.reactor.feed(&Instruction::I64Const(0xFFFF))?; self.reactor.feed(&Instruction::I64And)?; }
                        32 => { /* already zero-extended */ }
                        64 => { /* full */ }
                        _ => {}
                    }
                } else { return Ok(None); }
            }
            _ => return Ok(None),
        }
        
        // Perform the operation (stack now has: old_value, src_value)
        operation(self)?;
        
        // Store result back to memory
        self.emit_memory_address(&inst)?;
        self.emit_memory_store(size_bits)?;
        
        Ok(Some(()))
    }

    // Helper: emit memory effective address for the instruction's memory operand
    fn emit_memory_address(&mut self, inst: &IxInst) -> Result<(), E> {
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
                self.reactor.feed(&Instruction::LocalGet(local))?;
                if bit > 0 {
                    // shift right by bit to align subregister
                    self.reactor.feed(&Instruction::I64Const(bit as i64))?;
                    self.reactor.feed(&Instruction::I64ShrU)?;
                }
                have_value = true;
            } else { self.reactor.feed(&Instruction::Unreachable)?; return Ok(()); }
        }
        if index != Register::None {
            if let Some((idx_local, _sz, _z, bit)) = Self::resolve_reg(index) {
                if have_value {
                    self.reactor.feed(&Instruction::LocalGet(idx_local))?;
                } else {
                    self.reactor.feed(&Instruction::LocalGet(idx_local))?; have_value = true;
                }
                if bit > 0 {
                    self.reactor.feed(&Instruction::I64Const(bit as i64))?;
                    self.reactor.feed(&Instruction::I64ShrU)?;
                }
                // multiply by scale
                if scale != 1 {
                    self.reactor.feed(&Instruction::I64Const(scale as i64))?;
                    self.reactor.feed(&Instruction::I64Mul)?;
                }
                if base != Register::None {
                    self.reactor.feed(&Instruction::I64Add)?;
                }
            } else { self.reactor.feed(&Instruction::Unreachable)?; return Ok(()); }
        }
        if disp != 0 {
            if have_value {
                self.reactor.feed(&Instruction::I64Const(disp as i64))?;
                self.reactor.feed(&Instruction::I64Add)?;
            } else {
                self.reactor.feed(&Instruction::I64Const(disp as i64))?;
                have_value = true;
            }
        }
        // If nothing contributed, push zero
        if !have_value {
            self.reactor.feed(&Instruction::I64Const(0))?;
        }
        Ok(())
    }

fn handle_binary<T>(&mut self, inst: &IxInst, mut cb: T) -> Result<Option<()>, E>
    where
        T: FnMut(&mut Self, Operand, u32, u32, u32) -> Result<(), E>,
    {
        // Try to extract dest register and src (imm or reg)
        use iced_x86::OpKind;

        let op0 = inst.op0_kind();
        let op1 = inst.op1_kind();

        // If destination is memory, handle store here.
        if op0 == OpKind::Memory {
            // source can be immediate or register
            match op1 {
                OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                    // emit address then immediate then store sized by immediate kind
                    self.emit_memory_address(&inst)?;
                    let imm = inst.immediate64() as i64;
                    self.emit_i64_const(imm)?;
                    let size_bits = match op1 {
                        OpKind::Immediate8 => 8,
                        OpKind::Immediate16 => 16,
                        OpKind::Immediate8to32 => 32,
                        OpKind::Immediate32 => 32,
                        OpKind::Immediate64 => 64,
                        _ => 64,
                    };
                    self.emit_memory_store(size_bits)?;
                    return Ok(Some(()));
                }
                OpKind::Register => {
                    if let Some((r_local, r_size, _rz, bit)) = Self::resolve_reg(inst.op1_register()) {
                        // compute address
                        self.emit_memory_address(&inst)?;
                        // get source value and narrow if sub-register
                        self.reactor.feed(&Instruction::LocalGet(r_local))?;
                        if bit > 0 {
                            self.emit_mask_shift_for_read(r_size, bit)?;
                        }
                        // narrow by mask if needed
                        match r_size {
                            8 => { self.reactor.feed(&Instruction::I64Const(0xFF))?; self.reactor.feed(&Instruction::I64And)?; }
                            16 => { self.reactor.feed(&Instruction::I64Const(0xFFFF))?; self.reactor.feed(&Instruction::I64And)?; }
                            32 => { /* lower 32 bits are already zero-extended in locals for 32-bit regs */ }
                            64 => { /* full */ }
                            _ => {}
                        }
                        // store to memory using source size
                        self.emit_memory_store(r_size)?;
                        return Ok(Some(()));
                    } else { return Ok(None); }
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

        if dst_info.is_none() { return Ok(None); }
        let (dst_local, dst_size, _dst_zero_ext32, dst_bit_offset) = dst_info.unwrap();

        // Determine source operand and size semantics
        let src = match op1 {
            OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                Operand::Imm(inst.immediate64() as i64)
            }
            OpKind::Register => {
                if let Some((r_local, r_size, _z, bit)) = Self::resolve_reg(inst.op1_register()) { Operand::RegWithSize(r_local, r_size, bit) } else { return Ok(None); }
            }
            OpKind::Memory => { return Ok(None); }
            _ => return Ok(None),
        };

        // Generate code with knowledge of dst size; handle sub-register writes
         match src {
             Operand::Imm(_i) => {
                 // immediate -> callback will emit and set dest
 cb(self, src, dst_local, dst_size, dst_bit_offset)?;
                 // if dst is an 8/16-bit subregister we need to perform RMW; the callback should leave the new value on stack then we do RMW
                 // but since our callbacks currently perform LocalSet(dst_local) directly for full-register writes, we need to adjust pattern:
             }
             Operand::RegWithSize(src_local, src_size, src_bit) => {
                 cb(self, Operand::RegWithSize(src_local, src_size, src_bit), dst_local, dst_size, dst_bit_offset)?;
             }
             _ => { cb(self, src, dst_local, dst_size, dst_bit_offset)?; }
         }


        Ok(Some(()))
    }

    fn handle_test(&mut self, inst: &IxInst) -> Result<Option<()>, E> {
        // TEST performs bitwise AND and sets flags, but doesn't store result
        let src = match inst.op1_kind() {
            OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                Operand::Imm(inst.immediate64() as i64)
            }
            OpKind::Register => {
                if let Some((r_local, r_size, _z, bit)) = Self::resolve_reg(inst.op1_register()) {
                    Operand::RegWithSize(r_local, r_size, bit)
                } else { return Ok(None); }
            }
            OpKind::Memory => { return Ok(None); } // Memory operands not supported yet
            _ => return Ok(None),
        };

        // Get destination (must be register for TEST)
        let dst_info = match inst.op0_kind() {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            _ => None,
        };

        if dst_info.is_none() { return Ok(None); }
        let (dst_local, dst_size, _dst_zero_ext32, dst_bit_offset) = dst_info.unwrap();

        // Get dst value
        self.reactor.feed(&Instruction::LocalGet(dst_local))?;
        if dst_bit_offset > 0 {
            self.emit_mask_shift_for_read(dst_size, dst_bit_offset)?;
        }

        // Get src value
        match src {
            Operand::Imm(i) => { self.emit_i64_const(i)?; }
            Operand::Reg(r) => { self.reactor.feed(&Instruction::LocalGet(r))?; }
            Operand::RegWithSize(r, sz, bit) => {
                self.reactor.feed(&Instruction::LocalGet(r))?;
                if bit > 0 { self.emit_mask_shift_for_read(sz, bit)?; }
            }
        }

        // Perform AND
        self.emit_i64_and()?;

        // Store result temporarily and set flags
        self.reactor.feed(&Instruction::LocalTee(22))?; // temp = result, result still on stack

        // ZF: result == 0
        self.reactor.feed(&Instruction::I64Const(0))?;
        self.reactor.feed(&Instruction::I64Eq)?;
        self.reactor.feed(&Instruction::I32WrapI64)?;
        self.reactor.feed(&Instruction::LocalSet(Self::ZF_LOCAL))?;

        // SF: result < 0
        self.reactor.feed(&Instruction::LocalGet(22))?;
        self.reactor.feed(&Instruction::I64Const(63))?;
        self.reactor.feed(&Instruction::I64ShrS)?;
        self.reactor.feed(&Instruction::I32WrapI64)?;
        self.reactor.feed(&Instruction::I32Const(1))?;
        self.reactor.feed(&Instruction::I32And)?;
        self.reactor.feed(&Instruction::LocalSet(Self::SF_LOCAL))?;

        // CF and OF are always cleared for TEST
        self.set_cf(false)?;
        self.set_of(false)?;

        // PF: parity of lowest byte
        self.reactor.feed(&Instruction::LocalGet(22))?;
        self.reactor.feed(&Instruction::I64Const(0xFF))?;
        self.reactor.feed(&Instruction::I64And)?;
        self.reactor.feed(&Instruction::I32WrapI64)?;
        // Simple parity check - count bits (simplified)
        self.reactor.feed(&Instruction::I32Popcnt)?;
        self.reactor.feed(&Instruction::I32Const(1))?;
        self.reactor.feed(&Instruction::I32And)?; // 1 if odd parity, 0 if even
        self.reactor.feed(&Instruction::I32Eqz)?; // 1 if even parity, 0 if odd
        self.reactor.feed(&Instruction::LocalSet(Self::PF_LOCAL))?;

        Ok(Some(()))
    }

    fn handle_cmp(&mut self, inst: &IxInst) -> Result<Option<()>, E> {
        // CMP performs subtraction and sets flags, but doesn't store result
        let src = match inst.op1_kind() {
            OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate8to32 => {
                Operand::Imm(inst.immediate64() as i64)
            }
            OpKind::Register => {
                if let Some((r_local, r_size, _z, bit)) = Self::resolve_reg(inst.op1_register()) {
                    Operand::RegWithSize(r_local, r_size, bit)
                } else { return Ok(None); }
            }
            OpKind::Memory => { return Ok(None); } // Memory operands not supported yet
            _ => return Ok(None),
        };

        // Get destination
        let dst_info = match inst.op0_kind() {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            OpKind::Memory => None, // Memory destinations not supported yet
            _ => None,
        };

        if dst_info.is_none() { return Ok(None); }
        let (dst_local, dst_size, _dst_zero_ext32, dst_bit_offset) = dst_info.unwrap();

        // Get dst value
        self.reactor.feed(&Instruction::LocalGet(dst_local))?;
        if dst_bit_offset > 0 {
            self.emit_mask_shift_for_read(dst_size, dst_bit_offset)?;
        }

        // Get src value
        match src {
            Operand::Imm(i) => { self.emit_i64_const(i)?; }
            Operand::Reg(r) => { self.reactor.feed(&Instruction::LocalGet(r))?; }
            Operand::RegWithSize(r, sz, bit) => {
                self.reactor.feed(&Instruction::LocalGet(r))?;
                if bit > 0 { self.emit_mask_shift_for_read(sz, bit)?; }
            }
        }

        // Store operands for flag computation
        self.reactor.feed(&Instruction::LocalSet(23))?; // src
        self.reactor.feed(&Instruction::LocalSet(22))?; // dst

        // Compute result = dst - src
        self.reactor.feed(&Instruction::LocalGet(22))?;
        self.reactor.feed(&Instruction::LocalGet(23))?;
        self.emit_i64_sub()?;

        // Store result and set flags
        self.reactor.feed(&Instruction::LocalTee(24))?; // result

        // ZF: result == 0
        self.reactor.feed(&Instruction::I64Const(0))?;
        self.reactor.feed(&Instruction::I64Eq)?;
        self.reactor.feed(&Instruction::I32WrapI64)?;
        self.reactor.feed(&Instruction::LocalSet(Self::ZF_LOCAL))?;

        // SF: result < 0
        self.reactor.feed(&Instruction::LocalGet(24))?;
        self.reactor.feed(&Instruction::I64Const(63))?;
        self.reactor.feed(&Instruction::I64ShrS)?;
        self.reactor.feed(&Instruction::I32WrapI64)?;
        self.reactor.feed(&Instruction::I32Const(1))?;
        self.reactor.feed(&Instruction::I32And)?;
        self.reactor.feed(&Instruction::LocalSet(Self::SF_LOCAL))?;

        // CF: for subtraction, CF if dst < src (unsigned)
        self.reactor.feed(&Instruction::LocalGet(22))?;
        self.reactor.feed(&Instruction::LocalGet(23))?;
        self.reactor.feed(&Instruction::I64LtU)?;
        self.reactor.feed(&Instruction::I32WrapI64)?;
        self.reactor.feed(&Instruction::LocalSet(Self::CF_LOCAL))?;

        // OF: overflow for subtraction
        // Overflow occurs when operands have different signs and result has different sign from dst
        self.reactor.feed(&Instruction::LocalGet(22))?; // dst
        self.reactor.feed(&Instruction::I64Const(63))?;
        self.reactor.feed(&Instruction::I64ShrS)?;
        self.reactor.feed(&Instruction::LocalGet(23))?; // src
        self.reactor.feed(&Instruction::I64Const(63))?;
        self.reactor.feed(&Instruction::I64ShrS)?;
        self.reactor.feed(&Instruction::I32Xor)?; // 1 if different signs
        self.reactor.feed(&Instruction::LocalGet(22))?; // dst
        self.reactor.feed(&Instruction::I64Const(63))?;
        self.reactor.feed(&Instruction::I64ShrS)?;
        self.reactor.feed(&Instruction::LocalGet(24))?; // result
        self.reactor.feed(&Instruction::I64Const(63))?;
        self.reactor.feed(&Instruction::I64ShrS)?;
        self.reactor.feed(&Instruction::I32Xor)?; // 1 if dst and result have different signs
        self.reactor.feed(&Instruction::I32And)?; // OF set
        self.reactor.feed(&Instruction::LocalSet(Self::OF_LOCAL))?;

        // PF: parity of lowest byte
        self.reactor.feed(&Instruction::LocalGet(24))?;
        self.reactor.feed(&Instruction::I64Const(0xFF))?;
        self.reactor.feed(&Instruction::I64And)?;
        self.reactor.feed(&Instruction::I32WrapI64)?;
        self.reactor.feed(&Instruction::I32Popcnt)?;
        self.reactor.feed(&Instruction::I32Const(1))?;
        self.reactor.feed(&Instruction::I32And)?;
        self.reactor.feed(&Instruction::I32Eqz)?;
        self.reactor.feed(&Instruction::LocalSet(Self::PF_LOCAL))?;

        Ok(Some(()))
    }

    fn handle_jmp(&mut self, inst: &IxInst) -> Result<Option<()>, E> {
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
        self.reactor.jmp(target_func_idx, 0)?; // No params for now

        Ok(Some(()))
    }

    fn handle_conditional_jump(&mut self, inst: &IxInst, condition_type: ConditionType) -> Result<Option<()>, E> {
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
        self.reactor.ji_with_params(params)?;

        Ok(Some(()))
    }


}

enum Operand {
    Imm(i64),
    Reg(u32),
    RegWithSize(u32, u32, u32),
}

impl From<i64> for Operand {
    fn from(v: i64) -> Self { Operand::Imm(v) }
}

impl From<u32> for Operand {
    fn from(v: u32) -> Self { Operand::Reg(v) }
}

impl Operand {
    fn with_bit(self, bit_offset: u32) -> Self {
        match self {
            Operand::RegWithSize(r, s, _) => Operand::RegWithSize(r, s, bit_offset),
            other => other,
        }
    }
}

