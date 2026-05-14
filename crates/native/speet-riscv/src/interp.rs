//! [`RiscVThompsonInterp`] — Thompson-threaded RISC-V interpreter.
//!
//! Each WASM function handles one opcode class and tail-calls the lookup stub
//! on exit, sharing the same function type as compiled RISC-V functions.
//! Integer ALU and control-flow instructions are fully implemented; LOAD/STORE
//! delegate to `InterpBuildCtx::memory` when present, else emit `unreachable`.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use speet_interp::builder::InterpBodyBuilder;
use speet_interp::context::{FlatMemorySink, InterpBuildCtx};
use speet_memory::{LoadKind, StoreKind};
use alloc::borrow::Cow;
use wasm_encoder::{BlockType, Instruction, MemArg, ValType};
use wax_core::build::InstructionSink;

// ── Major-opcode to handler-slot mapping ─────────────────────────────────────
//
// Slot assignment (matches InterpBodyBuilder::num_handler_fns = 13):
const SLOT_LOAD:      u32 = 0;
const SLOT_STORE:     u32 = 1;
const SLOT_OP_IMM:    u32 = 2;
const SLOT_OP:        u32 = 3;
const SLOT_BRANCH:    u32 = 4;
const SLOT_JAL:       u32 = 5;
const SLOT_JALR:      u32 = 6;
const SLOT_LUI:       u32 = 7;
const SLOT_AUIPC:     u32 = 8;
const SLOT_OP_IMM32:  u32 = 9;
const SLOT_OP32:      u32 = 10;
const SLOT_SYSTEM:    u32 = 11;
const SLOT_FALLBACK:  u32 = 12;
const N_HANDLERS:     u32 = 13;

/// Maps RISC-V 7-bit major opcodes to handler slot indices.
fn build_opcode_table() -> [u32; 128] {
    let mut t = [SLOT_FALLBACK; 128];
    t[0x03] = SLOT_LOAD;
    t[0x23] = SLOT_STORE;
    t[0x13] = SLOT_OP_IMM;
    t[0x33] = SLOT_OP;
    t[0x63] = SLOT_BRANCH;
    t[0x6F] = SLOT_JAL;
    t[0x67] = SLOT_JALR;
    t[0x37] = SLOT_LUI;
    t[0x17] = SLOT_AUIPC;
    t[0x1B] = SLOT_OP_IMM32;
    t[0x3B] = SLOT_OP32;
    t[0x73] = SLOT_SYSTEM;
    t
}

// ── Scratch local layout (indices relative to total_params) ──────────────────
const SI_INSTR:   u32 = 0; // i32: raw instruction word
const SI_RD:      u32 = 1; // i32: destination register index
const SI_RS1:     u32 = 2; // i32: source register 1 index
const SI_RS2:     u32 = 3; // i32: source register 2 index
const SI_FUNCT3:  u32 = 4; // i32: 3-bit function code
const SI_FUNCT7:  u32 = 5; // i32: 7-bit function code
const SJ_IMM:     u32 = 0; // i64: sign-extended immediate
const SJ_RS1V:    u32 = 1; // i64: value of x[rs1]
const SJ_RS2V:    u32 = 2; // i64: value of x[rs2]
const SJ_RESULT:  u32 = 3; // i64: computed result
const SJ_NEXT_PC: u32 = 4; // i64: next program counter
const N_I32_SCRATCH: u32 = 6;
const N_I64_SCRATCH: u32 = 5;

// ── RiscVThompsonInterp ───────────────────────────────────────────────────────

/// Thompson-threaded RISC-V interpreter builder.
///
/// Plug this into [`OobInterp::register_with_builder`] and
/// [`OobInterp::emit_with_builder`] to replace the default `unreachable` stub
/// with a real per-opcode interpreter.
pub struct RiscVThompsonInterp<Context, E> {
    /// Total WASM param count of the shared function type (arch regs + target_pc).
    pub total_params: u32,
    /// Whether RV64 instructions are enabled (uses i64 for integer regs).
    pub enable_rv64: bool,
    _marker: PhantomData<fn(&mut Context) -> E>,
}

impl<Context, E> RiscVThompsonInterp<Context, E> {
    /// Create a new `RiscVThompsonInterp`.
    ///
    /// `total_params` must equal the param count used when generating the
    /// shared function type (the value returned by `setup_traps`).
    pub fn new(total_params: u32, enable_rv64: bool) -> Self {
        Self { total_params, enable_rv64, _marker: PhantomData }
    }

    /// Local index of `target_pc` (last param).
    fn tpc(&self) -> u32 { self.total_params - 1 }

    /// Local index of arch PC register (param 64).
    fn pc(&self) -> u32 { 64 }

    /// Scratch i32 local at offset `n`.
    fn si(&self, n: u32) -> u32 { self.total_params + n }

    /// Scratch i64 local at offset `n`.
    fn sj(&self, n: u32) -> u32 { self.total_params + N_I32_SCRATCH + n }

    /// Local declarations for handler functions.
    fn handler_locals() -> Vec<(u32, ValType)> {
        vec![(N_I32_SCRATCH, ValType::I32), (N_I64_SCRATCH, ValType::I64)]
    }
}

impl<Context, E> InterpBodyBuilder<Context, E> for RiscVThompsonInterp<Context, E> {
    fn num_handler_fns(&self) -> u32 { N_HANDLERS }

    fn dispatch_fn_locals(&self) -> Vec<(u32, ValType)> {
        vec![(1, ValType::I32)] // one i32 scratch for opcode extraction
    }

    fn handler_fn_locals(&self, _i: u32) -> Vec<(u32, ValType)> {
        Self::handler_locals()
    }

    fn build_interp(
        &mut self,
        dispatch_sink: &mut dyn InstructionSink<Context, E>,
        handler_sinks: &mut [Box<dyn InstructionSink<Context, E>>],
        ctx: &mut Context,
        ictx: &mut InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        self.build_dispatch(dispatch_sink, ctx, ictx)?;
        self.build_handler_load(handler_sinks[SLOT_LOAD as usize].as_mut(), ctx, ictx)?;
        self.build_handler_store(handler_sinks[SLOT_STORE as usize].as_mut(), ctx, ictx)?;
        self.build_handler_op_imm(handler_sinks[SLOT_OP_IMM as usize].as_mut(), ctx, ictx)?;
        self.build_handler_op(handler_sinks[SLOT_OP as usize].as_mut(), ctx, ictx)?;
        self.build_handler_branch(handler_sinks[SLOT_BRANCH as usize].as_mut(), ctx, ictx)?;
        self.build_handler_jal(handler_sinks[SLOT_JAL as usize].as_mut(), ctx, ictx)?;
        self.build_handler_jalr(handler_sinks[SLOT_JALR as usize].as_mut(), ctx, ictx)?;
        self.build_handler_lui(handler_sinks[SLOT_LUI as usize].as_mut(), ctx, ictx)?;
        self.build_handler_auipc(handler_sinks[SLOT_AUIPC as usize].as_mut(), ctx, ictx)?;
        self.build_handler_stub(handler_sinks[SLOT_OP_IMM32 as usize].as_mut(), ctx)?;
        self.build_handler_stub(handler_sinks[SLOT_OP32 as usize].as_mut(), ctx)?;
        self.build_handler_stub(handler_sinks[SLOT_SYSTEM as usize].as_mut(), ctx)?;
        self.build_handler_stub(handler_sinks[SLOT_FALLBACK as usize].as_mut(), ctx)
    }
}

impl<Context, E> RiscVThompsonInterp<Context, E> {
    // ── Dispatch function ─────────────────────────────────────────────────────

    fn build_dispatch(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        let opcode_scratch = self.total_params; // the one i32 local declared above

        // Read 32-bit instruction word from guest memory at target_pc.
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I32WrapI64)?;
        sink.instruction(ctx, &Instruction::I32Load(MemArg {
            offset: 0, align: 2, memory_index: ictx.insn_mem_idx,
        }))?;
        // Extract bits [6:0] (major opcode).
        sink.instruction(ctx, &Instruction::I32Const(0x7F))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::LocalSet(opcode_scratch))?;

        // Emit 128-entry br_table.  Wrap in 13 nested blocks so each arm
        // tail-calls the right handler function.
        //
        // Pattern: N nested blocks + br_table[0..127] with slots mapped to
        // handler indices.  After each `end`, fall-through executes
        // `return_call handler[k]`.
        let opcode_table = build_opcode_table();

        // Open 13 case blocks (outermost = case 12, innermost = case 0).
        for _ in 0..N_HANDLERS {
            sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?;
        }

        // The br_table maps opcode → branch depth.
        // Depth 0 exits the innermost block (case 0).
        // Depth 12 exits the outermost block (case 12/fallback).
        let targets: Vec<u32> = opcode_table.iter().map(|&s| s).collect();
        sink.instruction(ctx, &Instruction::LocalGet(opcode_scratch))?;
        sink.instruction(ctx, &Instruction::BrTable(Cow::Borrowed(&targets), SLOT_FALLBACK))?;
        sink.instruction(ctx, &Instruction::Unreachable)?;

        // Arms: innermost first (case 0 = LOAD), outermost last (case 12 = FALLBACK).
        for slot in 0..N_HANDLERS {
            sink.instruction(ctx, &Instruction::End)?; // end block for this slot
            // Forward all params to the handler and tail-call it.
            for p in 0..self.total_params {
                sink.instruction(ctx, &Instruction::LocalGet(p))?;
            }
            sink.instruction(ctx, &Instruction::ReturnCall(ictx.handler_func_indices[slot as usize]))?;
        }

        sink.instruction(ctx, &Instruction::End) // end function
    }

    // ── Handler: unreachable stub ─────────────────────────────────────────────

    fn build_handler_stub(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        let n = self.total_params;
        for p in 0..n {
            sink.instruction(ctx, &Instruction::LocalGet(p))?;
            sink.instruction(ctx, &Instruction::Drop)?;
        }
        sink.instruction(ctx, &Instruction::Unreachable)?;
        sink.instruction(ctx, &Instruction::End)
    }

    // ── Common prologue helpers ───────────────────────────────────────────────

    /// Emit: read instr word from insn_mem at target_pc → local SI_INSTR.
    fn emit_read_instr(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I32WrapI64)?;
        sink.instruction(ctx, &Instruction::I32Load(MemArg {
            offset: 0, align: 2, memory_index: ictx.insn_mem_idx,
        }))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.si(SI_INSTR)))
    }

    /// Emit: decode rd, rs1, funct3 from SI_INSTR.
    fn emit_decode_rrf(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        let instr = self.si(SI_INSTR);
        // rd = (instr >> 7) & 0x1F
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(7))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x1F))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.si(SI_RD)))?;
        // rs1 = (instr >> 15) & 0x1F
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(15))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x1F))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.si(SI_RS1)))?;
        // funct3 = (instr >> 12) & 0x7
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(12))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(7))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.si(SI_FUNCT3)))
    }

    /// Emit: decode rs2, funct7 from SI_INSTR.
    fn emit_decode_rs2_funct7(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        let instr = self.si(SI_INSTR);
        // rs2 = (instr >> 20) & 0x1F
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(20))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x1F))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.si(SI_RS2)))?;
        // funct7 = (instr >> 25) & 0x7F
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(25))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x7F))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.si(SI_FUNCT7)))
    }

    /// Emit I-type immediate: sign_extend(instr >> 20, 12) → SJ_IMM.
    fn emit_imm_i_type(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalGet(self.si(SI_INSTR)))?;
        sink.instruction(ctx, &Instruction::I32Const(20))?;
        sink.instruction(ctx, &Instruction::I32ShrS)?; // arithmetic shift → sign extended
        sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_IMM)))
    }

    /// Emit B-type immediate → SJ_IMM.
    fn emit_imm_b_type(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        let instr = self.si(SI_INSTR);
        // imm[12] = instr[31]; imm[10:5] = instr[30:25]; imm[4:1] = instr[11:8]; imm[11] = instr[7]
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(19))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x1000))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        // imm[10:5]
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(20))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x7E0))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::I32Or)?;
        // imm[4:1]
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(7))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x1E))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::I32Or)?;
        // imm[11]
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(4))?;
        sink.instruction(ctx, &Instruction::I32Shl)?;
        sink.instruction(ctx, &Instruction::I32Const(0x800))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::I32Or)?;
        // sign extend from bit 12: shl 19, shr_s 19
        sink.instruction(ctx, &Instruction::I32Const(19))?;
        sink.instruction(ctx, &Instruction::I32Shl)?;
        sink.instruction(ctx, &Instruction::I32Const(19))?;
        sink.instruction(ctx, &Instruction::I32ShrS)?;
        sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_IMM)))
    }

    /// Emit J-type immediate (JAL) → SJ_IMM.
    fn emit_imm_j_type(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        let instr = self.si(SI_INSTR);
        // imm[20] = instr[31]; imm[10:1] = instr[30:21]; imm[11] = instr[20]; imm[19:12] = instr[19:12]
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(11))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x100000))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        // imm[10:1]
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(20))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x7FE))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::I32Or)?;
        // imm[11]
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(9))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(0x800))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::I32Or)?;
        // imm[19:12]
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(0xFF000u32 as i32))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::I32Or)?;
        // sign extend from bit 20: shl 11, shr_s 11
        sink.instruction(ctx, &Instruction::I32Const(11))?;
        sink.instruction(ctx, &Instruction::I32Shl)?;
        sink.instruction(ctx, &Instruction::I32Const(11))?;
        sink.instruction(ctx, &Instruction::I32ShrS)?;
        sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_IMM)))
    }

    /// Emit U-type immediate (LUI/AUIPC) → SJ_IMM (sign-extended i64).
    fn emit_imm_u_type(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        // Upper 20 bits already in place, lower 12 cleared.
        sink.instruction(ctx, &Instruction::LocalGet(self.si(SI_INSTR)))?;
        sink.instruction(ctx, &Instruction::I32Const(0xFFFFF000u32 as i32))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_IMM)))
    }

    // ── Dynamic register read/write ───────────────────────────────────────────

    /// Emit: push x[reg_idx_local] onto WASM stack as i64.
    ///
    /// Uses a 32-entry br_table dispatch.  x0 always returns 0.
    fn emit_dyn_xreg_read(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        reg_idx_local: u32,
    ) -> Result<(), E> {
        // block (result i64)  + 32 inner blocks.
        // br_table targets[k] = k, so: index k → br k → exit inner block k → land at case k.
        sink.instruction(ctx, &Instruction::Block(BlockType::Result(ValType::I64)))?;
        for _ in 0..32u32 {
            sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?;
        }
        let targets: Vec<u32> = (0..32).collect();
        sink.instruction(ctx, &Instruction::LocalGet(reg_idx_local))?;
        sink.instruction(ctx, &Instruction::BrTable(Cow::Borrowed(&targets), 31))?;
        sink.instruction(ctx, &Instruction::Unreachable)?;

        // Case 0 (x0 = hardwired 0): br 0 exits innermost block ($B0).
        // We're now in $B1..$B31 + $R (result block).  br 31 exits $R.
        sink.instruction(ctx, &Instruction::End)?; // end $B0
        sink.instruction(ctx, &Instruction::I64Const(0))?;
        sink.instruction(ctx, &Instruction::Br(31))?;

        // Cases 1..30.
        for r in 1..31u32 {
            sink.instruction(ctx, &Instruction::End)?; // end $Br
            sink.instruction(ctx, &Instruction::LocalGet(r))?;
            sink.instruction(ctx, &Instruction::Br(31 - r))?;
        }

        // Case 31: br 31 from br_table exits all inner blocks, land inside $R.
        sink.instruction(ctx, &Instruction::End)?; // end $B31
        sink.instruction(ctx, &Instruction::LocalGet(31))?;
        // fall through to end $R
        sink.instruction(ctx, &Instruction::End) // end result block — i64 on stack
    }

    /// Emit: pop i64 from WASM stack and store into x[rd_idx_local].
    ///
    /// rd=0 discards the value (x0 is immutable in RISC-V).
    fn emit_dyn_xreg_write(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        rd_idx_local: u32,
        scratch: u32, // a spare i64 local to save the value
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalSet(scratch))?;

        sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?; // $outer
        for _ in 0..32u32 {
            sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?;
        }
        let targets: Vec<u32> = (0..32).collect();
        sink.instruction(ctx, &Instruction::LocalGet(rd_idx_local))?;
        sink.instruction(ctx, &Instruction::BrTable(Cow::Borrowed(&targets), 31))?;
        sink.instruction(ctx, &Instruction::Unreachable)?;

        // Case 0: discard (x0 immutable).  br 31 exits $outer.
        sink.instruction(ctx, &Instruction::End)?; // end innermost $B0
        sink.instruction(ctx, &Instruction::Br(31))?;

        // Cases 1..30: write then exit $outer.
        for r in 1..31u32 {
            sink.instruction(ctx, &Instruction::End)?;
            sink.instruction(ctx, &Instruction::LocalGet(scratch))?;
            sink.instruction(ctx, &Instruction::LocalSet(r))?;
            sink.instruction(ctx, &Instruction::Br(31 - r))?;
        }

        // Case 31: write and fall through to end $outer.
        sink.instruction(ctx, &Instruction::End)?; // end $B31
        sink.instruction(ctx, &Instruction::LocalGet(scratch))?;
        sink.instruction(ctx, &Instruction::LocalSet(31))?;

        sink.instruction(ctx, &Instruction::End) // end $outer
    }

    // ── Forward all params and tail-call lookup stub ──────────────────────────

    /// Push all register params (with updated next_pc for the PC slot and
    /// target_pc slot) and tail-call the lookup stub.
    fn emit_forward_to_lookup(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        let pc_local = self.pc();
        let tpc_local = self.tpc();
        let next_pc = self.sj(SJ_NEXT_PC);

        // Push all params.  Slot 64 (PC) and the last slot (target_pc) use next_pc.
        for p in 0..self.total_params {
            if p == pc_local || p == tpc_local {
                sink.instruction(ctx, &Instruction::LocalGet(next_pc))?;
            } else {
                sink.instruction(ctx, &Instruction::LocalGet(p))?;
            }
        }
        sink.instruction(ctx, &Instruction::ReturnCall(ictx.oob.lookup_stub_func_idx))?;
        sink.instruction(ctx, &Instruction::End) // end function
    }

    // ── Handlers ─────────────────────────────────────────────────────────────

    fn build_handler_op_imm(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        self.emit_read_instr(sink, ctx, ictx)?;
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_imm_i_type(sink, ctx)?;
        // rs1_val = x[rs1]
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS1))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RS1V)))?;

        // Dispatch on funct3 for the operation.
        // result = rs1_val OP imm
        self.emit_op_imm_dispatch(sink, ctx)?;

        // Write result to rd.
        self.emit_dyn_xreg_write(sink, ctx, self.si(SI_RD), self.sj(SJ_RESULT))?;

        // next_pc = target_pc + 4
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }

    /// Emit funct3 dispatch for OP-IMM, leaving result in SJ_RESULT.
    fn emit_op_imm_dispatch(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        let rs1v = self.sj(SJ_RS1V);
        let imm  = self.sj(SJ_IMM);
        let res  = self.sj(SJ_RESULT);
        let f3   = self.si(SI_FUNCT3);
        let instr = self.si(SI_INSTR);

        // 8 inner blocks + result block.  funct3 is 0..7.
        sink.instruction(ctx, &Instruction::Block(BlockType::Result(ValType::I64)))?;
        for _ in 0..8u32 {
            sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?;
        }
        let targets: Vec<u32> = (0..8).collect();
        sink.instruction(ctx, &Instruction::LocalGet(f3))?;
        sink.instruction(ctx, &Instruction::BrTable(Cow::Borrowed(&targets), 7))?;
        sink.instruction(ctx, &Instruction::Unreachable)?;

        // f3=0: ADDI
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::Br(7))?;

        // f3=1: SLLI  (shamt = imm & 0x3F)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64Const(0x3F))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        sink.instruction(ctx, &Instruction::I64Shl)?;
        sink.instruction(ctx, &Instruction::Br(6))?;

        // f3=2: SLTI
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64LtS)?;
        sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
        sink.instruction(ctx, &Instruction::Br(5))?;

        // f3=3: SLTIU
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64LtU)?;
        sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
        sink.instruction(ctx, &Instruction::Br(4))?;

        // f3=4: XORI
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64Xor)?;
        sink.instruction(ctx, &Instruction::Br(3))?;

        // f3=5: SRLI (funct7=0) or SRAI (funct7=0x20)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64Const(0x3F))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        // Arithmetic shift if bit 30 of instr is set (imm[10] = instr[30]).
        sink.instruction(ctx, &Instruction::LocalGet(instr))?;
        sink.instruction(ctx, &Instruction::I32Const(30))?;
        sink.instruction(ctx, &Instruction::I32ShrU)?;
        sink.instruction(ctx, &Instruction::I32Const(1))?;
        sink.instruction(ctx, &Instruction::I32And)?;
        sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
        // SRAI: arithmetic shift
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64Const(0x3F))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        sink.instruction(ctx, &Instruction::I64ShrS)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RESULT)))?;
        sink.instruction(ctx, &Instruction::Else)?;
        // SRLI: logical shift — result still on stack from before the if
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64Const(0x3F))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        sink.instruction(ctx, &Instruction::I64ShrU)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RESULT)))?;
        sink.instruction(ctx, &Instruction::End)?; // end if
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_RESULT)))?;
        sink.instruction(ctx, &Instruction::Br(2))?;

        // f3=6: ORI
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64Or)?;
        sink.instruction(ctx, &Instruction::Br(1))?;

        // f3=7: ANDI (default)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(imm))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        // fall through to end result block

        sink.instruction(ctx, &Instruction::End)?; // end result block
        sink.instruction(ctx, &Instruction::LocalSet(res))
    }

    fn build_handler_op(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        self.emit_read_instr(sink, ctx, ictx)?;
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_decode_rs2_funct7(sink, ctx)?;
        // rs1_val, rs2_val
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS1))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RS1V)))?;
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS2))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RS2V)))?;

        self.emit_op_dispatch(sink, ctx)?;

        self.emit_dyn_xreg_write(sink, ctx, self.si(SI_RD), self.sj(SJ_RESULT))?;

        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }

    fn emit_op_dispatch(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        let rs1v  = self.sj(SJ_RS1V);
        let rs2v  = self.sj(SJ_RS2V);
        let res   = self.sj(SJ_RESULT);
        let f3    = self.si(SI_FUNCT3);
        let f7    = self.si(SI_FUNCT7);

        // funct3 dispatch (0..7), then within each arm check funct7 for SUB/SRA.
        sink.instruction(ctx, &Instruction::Block(BlockType::Result(ValType::I64)))?;
        for _ in 0..8u32 {
            sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?;
        }
        let targets: Vec<u32> = (0..8).collect();
        sink.instruction(ctx, &Instruction::LocalGet(f3))?;
        sink.instruction(ctx, &Instruction::BrTable(Cow::Borrowed(&targets), 7))?;
        sink.instruction(ctx, &Instruction::Unreachable)?;

        // f3=0: ADD (funct7=0) or SUB (funct7=0x20)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(f7))?;
        sink.instruction(ctx, &Instruction::I32Const(0x20))?;
        sink.instruction(ctx, &Instruction::I32Eq)?;
        sink.instruction(ctx, &Instruction::If(BlockType::Result(ValType::I64)))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Sub)?;
        sink.instruction(ctx, &Instruction::Else)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::Br(7))?;

        // f3=1: SLL
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Const(0x3F))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        sink.instruction(ctx, &Instruction::I64Shl)?;
        sink.instruction(ctx, &Instruction::Br(6))?;

        // f3=2: SLT
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64LtS)?;
        sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
        sink.instruction(ctx, &Instruction::Br(5))?;

        // f3=3: SLTU
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64LtU)?;
        sink.instruction(ctx, &Instruction::I64ExtendI32U)?;
        sink.instruction(ctx, &Instruction::Br(4))?;

        // f3=4: XOR
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Xor)?;
        sink.instruction(ctx, &Instruction::Br(3))?;

        // f3=5: SRL (funct7=0) or SRA (funct7=0x20)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(f7))?;
        sink.instruction(ctx, &Instruction::I32Const(0x20))?;
        sink.instruction(ctx, &Instruction::I32Eq)?;
        sink.instruction(ctx, &Instruction::If(BlockType::Result(ValType::I64)))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Const(0x3F))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        sink.instruction(ctx, &Instruction::I64ShrS)?;
        sink.instruction(ctx, &Instruction::Else)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Const(0x3F))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        sink.instruction(ctx, &Instruction::I64ShrU)?;
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::Br(2))?;

        // f3=6: OR
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Or)?;
        sink.instruction(ctx, &Instruction::Br(1))?;

        // f3=7: AND (default)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64And)?;

        sink.instruction(ctx, &Instruction::End)?; // end result block
        sink.instruction(ctx, &Instruction::LocalSet(res))
    }

    fn build_handler_lui(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        self.emit_read_instr(sink, ctx, ictx)?;
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_imm_u_type(sink, ctx)?;
        // rd = imm (upper 20 bits, sign-extended)
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_IMM)))?;
        self.emit_dyn_xreg_write(sink, ctx, self.si(SI_RD), self.sj(SJ_RESULT))?;

        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }

    fn build_handler_auipc(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        self.emit_read_instr(sink, ctx, ictx)?;
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_imm_u_type(sink, ctx)?;
        // rd = PC + imm
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_IMM)))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        self.emit_dyn_xreg_write(sink, ctx, self.si(SI_RD), self.sj(SJ_RESULT))?;

        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }

    fn build_handler_jal(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        self.emit_read_instr(sink, ctx, ictx)?;
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_imm_j_type(sink, ctx)?;
        // rd = PC + 4 (return address)
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        self.emit_dyn_xreg_write(sink, ctx, self.si(SI_RD), self.sj(SJ_RESULT))?;
        // next_pc = PC + imm
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_IMM)))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }

    fn build_handler_jalr(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        self.emit_read_instr(sink, ctx, ictx)?;
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_imm_i_type(sink, ctx)?;
        // rs1_val = x[rs1]
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS1))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RS1V)))?;
        // rd = PC + 4
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        self.emit_dyn_xreg_write(sink, ctx, self.si(SI_RD), self.sj(SJ_RESULT))?;
        // next_pc = (rs1_val + imm) & !1
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_RS1V)))?;
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_IMM)))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::I64Const(!1i64))?;
        sink.instruction(ctx, &Instruction::I64And)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }

    fn build_handler_branch(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        self.emit_read_instr(sink, ctx, ictx)?;
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_decode_rs2_funct7(sink, ctx)?;
        self.emit_imm_b_type(sink, ctx)?;
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS1))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RS1V)))?;
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS2))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RS2V)))?;

        // Evaluate condition based on funct3.
        self.emit_branch_condition(sink, ctx)?; // leaves i32 on stack (1=taken)

        // if taken: next_pc = PC + imm, else next_pc = PC + 4
        sink.instruction(ctx, &Instruction::If(BlockType::Empty))?;
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_IMM)))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;
        sink.instruction(ctx, &Instruction::Else)?;
        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;
        sink.instruction(ctx, &Instruction::End)?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }

    /// Emit funct3 dispatch for BRANCH conditions, leaving i32 on stack.
    fn emit_branch_condition(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
    ) -> Result<(), E> {
        let rs1v = self.sj(SJ_RS1V);
        let rs2v = self.sj(SJ_RS2V);
        let f3   = self.si(SI_FUNCT3);

        sink.instruction(ctx, &Instruction::Block(BlockType::Result(ValType::I32)))?;
        for _ in 0..8u32 {
            sink.instruction(ctx, &Instruction::Block(BlockType::Empty))?;
        }
        let targets: Vec<u32> = (0..8).collect();
        sink.instruction(ctx, &Instruction::LocalGet(f3))?;
        sink.instruction(ctx, &Instruction::BrTable(Cow::Borrowed(&targets), 7))?;
        sink.instruction(ctx, &Instruction::Unreachable)?;

        // f3=0: BEQ
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Eq)?;
        sink.instruction(ctx, &Instruction::Br(7))?;

        // f3=1: BNE
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64Ne)?;
        sink.instruction(ctx, &Instruction::Br(6))?;

        // f3=2: unused (0)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::I32Const(0))?;
        sink.instruction(ctx, &Instruction::Br(5))?;

        // f3=3: unused (0)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::I32Const(0))?;
        sink.instruction(ctx, &Instruction::Br(4))?;

        // f3=4: BLT
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64LtS)?;
        sink.instruction(ctx, &Instruction::Br(3))?;

        // f3=5: BGE
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64GeS)?;
        sink.instruction(ctx, &Instruction::Br(2))?;

        // f3=6: BLTU
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64LtU)?;
        sink.instruction(ctx, &Instruction::Br(1))?;

        // f3=7: BGEU (default)
        sink.instruction(ctx, &Instruction::End)?;
        sink.instruction(ctx, &Instruction::LocalGet(rs1v))?;
        sink.instruction(ctx, &Instruction::LocalGet(rs2v))?;
        sink.instruction(ctx, &Instruction::I64GeU)?;

        sink.instruction(ctx, &Instruction::End) // end result block — i32 on stack
    }

    fn build_handler_load(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &mut InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        if ictx.memory.is_none() {
            return self.build_handler_stub(sink, ctx);
        }
        self.emit_read_instr(sink, ctx, ictx)?;
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_imm_i_type(sink, ctx)?;
        // effective address = x[rs1] + imm
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS1))?;
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_IMM)))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        // emit_load via MemoryAccess.  Dispatch load kind from funct3.
        let f3_val = self.si(SI_FUNCT3); // we need the runtime funct3
        // For simplicity, emit LW (32-bit signed) — full funct3 dispatch is a follow-up.
        let mut mem_sink = FlatMemorySink::new(sink);
        if let Some(mem) = ictx.memory.as_mut() {
            mem.emit_load(ctx, &mut mem_sink, LoadKind::I32S)?;
        }
        let _ = f3_val;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_RESULT)))?;
        self.emit_dyn_xreg_write(sink, ctx, self.si(SI_RD), self.sj(SJ_RESULT))?;

        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }

    fn build_handler_store(
        &self,
        sink: &mut dyn InstructionSink<Context, E>,
        ctx: &mut Context,
        ictx: &mut InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E> {
        if ictx.memory.is_none() {
            return self.build_handler_stub(sink, ctx);
        }
        self.emit_read_instr(sink, ctx, ictx)?;
        // Decode rs1, rs2, funct3 (and implicitly the S-type immediate).
        self.emit_decode_rrf(sink, ctx)?;
        self.emit_decode_rs2_funct7(sink, ctx)?;
        // S-type immediate: imm[11:5] from instr[31:25], imm[4:0] from instr[11:7]
        {
            let instr = self.si(SI_INSTR);
            sink.instruction(ctx, &Instruction::LocalGet(instr))?;
            sink.instruction(ctx, &Instruction::I32Const(20))?;
            sink.instruction(ctx, &Instruction::I32ShrU)?;
            sink.instruction(ctx, &Instruction::I32Const(!0x1Fu32 as i32))?;
            sink.instruction(ctx, &Instruction::I32And)?;
            sink.instruction(ctx, &Instruction::LocalGet(instr))?;
            sink.instruction(ctx, &Instruction::I32Const(7))?;
            sink.instruction(ctx, &Instruction::I32ShrU)?;
            sink.instruction(ctx, &Instruction::I32Const(0x1F))?;
            sink.instruction(ctx, &Instruction::I32And)?;
            sink.instruction(ctx, &Instruction::I32Or)?;
            // sign extend from bit 11
            sink.instruction(ctx, &Instruction::I32Const(20))?;
            sink.instruction(ctx, &Instruction::I32Shl)?;
            sink.instruction(ctx, &Instruction::I32Const(20))?;
            sink.instruction(ctx, &Instruction::I32ShrS)?;
            sink.instruction(ctx, &Instruction::I64ExtendI32S)?;
            sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_IMM)))?;
        }
        // effective address = x[rs1] + imm
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS1))?;
        sink.instruction(ctx, &Instruction::LocalGet(self.sj(SJ_IMM)))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        let mut mem_sink = FlatMemorySink::new(sink);
        if let Some(mem) = ictx.memory.as_mut() {
            mem.emit_store_addr(ctx, &mut mem_sink)?;
        }
        // value = x[rs2]
        self.emit_dyn_xreg_read(sink, ctx, self.si(SI_RS2))?;
        // For simplicity emit SW (32-bit store) — full funct3 dispatch is a follow-up.
        let mut mem_sink2 = FlatMemorySink::new(sink);
        if let Some(mem) = ictx.memory.as_mut() {
            mem.emit_store_insn(ctx, &mut mem_sink2, StoreKind::I32)?;
        }

        sink.instruction(ctx, &Instruction::LocalGet(self.tpc()))?;
        sink.instruction(ctx, &Instruction::I64Const(4))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        sink.instruction(ctx, &Instruction::LocalSet(self.sj(SJ_NEXT_PC)))?;

        self.emit_forward_to_lookup(sink, ctx, ictx)
    }
}
