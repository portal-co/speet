//! Minimal x86_64 to WebAssembly recompiler
//!
//! Supports a small subset of integer instructions for demonstration purposes.
#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use wax_core::build::InstructionSink;
use wasm_encoder::Instruction;
use yecta::{EscapeTag, FuncIdx, Pool, Reactor, TableIdx, TypeIdx};

use iced_x86::{Decoder, DecoderOptions, Instruction as IxInst, Mnemonic, Register, OpKind};

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
        // PC in local 16 (i32), temps after that
        let locals = [ (16, wasm_encoder::ValType::I64), (1, wasm_encoder::ValType::I32), (num_temps, wasm_encoder::ValType::I64) ];
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
                Mnemonic::Add => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
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
                }),
                Mnemonic::Sub => self.handle_binary(&inst, |this, src, dst, dst_size, dst_bit_offset| {
                    this.reactor.feed(&Instruction::LocalGet(dst))?;
                    match src {
                        Operand::Imm(i) => { this.emit_i64_const(i)?; }
                        Operand::Reg(r) => { this.reactor.feed(&Instruction::LocalGet(r))?; }
                        Operand::RegWithSize(r, sz, bit) => {
                            this.reactor.feed(&Instruction::LocalGet(r))?;
                            this.emit_mask_shift_for_read(sz, bit)?;
                        }
                    }
                    this.emit_i64_sub()?;
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

fn handle_binary<T>(&mut self, inst: &IxInst, mut cb: T) -> Result<Option<()>, E>
    where
        T: FnMut(&mut Self, Operand, u32, u32, u32) -> Result<(), E>,
    {
        // Try to extract dest register and src (imm or reg)
        use iced_x86::OpKind;

        let op0 = inst.op0_kind();
        let op1 = inst.op1_kind();

        // Determine destination local and size
        let dst_info = match op0 {
            OpKind::Register => Self::resolve_reg(inst.op0_register()),
            OpKind::Memory => None, // memory destinations not supported yet
            _ => None,
        };

        if dst_info.is_none() { return Ok(None); }
        let (dst_local, dst_size, dst_zero_ext32, dst_bit_offset) = dst_info.unwrap();

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
