//! # speet-x86_64
//!
//! x86-64 to WebAssembly static recompiler.
//!
//! Translates a subset of x86-64 integer instructions to WebAssembly using
//! the [`yecta`] reactor for control-flow management.
//!
//! ## Supported instructions
//! Integer ALU (`ADD`, `SUB`, `IMUL`, `AND`, `OR`, `XOR`, `SHL`, `SHR`,
//! `SAR`, `NOT`, `NEG`, `INC`, `DEC`, `CMP`, `TEST`), data movement
//! (`MOV`, `MOVZX`, `MOVSX`, `MOVSXD`, `LEA`, `XCHG`, `PUSH`, `POP`),
//! control flow (`Jcc`, `JMP`, `CALL`, `RET`), and memory string operations.
//! Floating-point and SIMD instructions are recognised and classified but
//! emitted as `unreachable` placeholders.
//!
//! ## Local-variable layout
//! The generated WASM function uses a fixed local layout:
//! - Locals 0–15: the 16 x86-64 general-purpose registers (each `i64`).
//! - Local 16: the program counter / RIP (`i64`).
//! - Locals 17–21: condition flags ZF, SF, CF, OF, PF (each `i32`).
//! - Locals 22–24: scratch temporaries (`i64`).
//! - Local 25: expected return address for speculative-call optimisation (`i64`).
//!
//! See [`X86Recompiler::BASE_PARAMS`] for the total count.
//!
//! ## Speculative calls
//! When enabled via [`X86Recompiler::set_speculative_calls`], ABI-conformant
//! `CALL` instructions are lowered to direct WASM `call` instructions wrapped
//! in a try/catch block keyed by an [`EscapeTag`].  `RET` instructions check
//! the top of the shadow stack against the expected return address and take
//! the fast path when they match.
//!
//! ## Trap hooks
//! Instruction-level and jump-level traps can be installed with
//! [`X86Recompiler::set_instruction_trap`] and
//! [`X86Recompiler::set_jump_trap`] (see [`speet_traps`] for the trait
//! definitions).  Call [`X86Recompiler::setup_traps`] once before
//! translation to register the trap parameters in the function layout.
#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use yecta::{EscapeTag, Fed, LocalLayout, LocalPoolBackend, Mark, Pool, Reactor, SlotAssigner, TableIdx, TypeIdx, layout::CellIdx};
pub mod cfg;
pub mod direct;
use speet_traps::{
    insn::{ArchTag, InsnClass},
    InstructionInfo, InstructionTrap, JumpInfo, JumpKind, JumpTrap, TrapAction, TrapConfig,
};
/// x86-64 to WebAssembly recompiler.
///
/// Each instance manages a single [`yecta::Reactor`] and translates one
/// contiguous region of x86-64 bytes into a sequence of WASM functions
/// (one function per instruction) via [`direct::translate_bytes`].
///
/// # Type parameters
/// - `Context` / `E` / `F` – forwarded to the underlying [`yecta::Reactor`].
/// - `P` – the local-pool backend (defaults to [`yecta::LocalPool`]).
pub struct X86Recompiler<
    'cb,
    'ctx,
    Context,
    E,
    F: InstructionSink<Context, E>,
    P: yecta::LocalPoolBackend = yecta::LocalPool,
> {
    reactor: Reactor<Context, E, F, P>,
    pool: Pool<'cb, Context, E>,
    escape_tag: Option<EscapeTag>,
    base_rip: u64,
    hints: Vec<u8>,
    enable_speculative_calls: bool,
    /// Pluggable instruction-level and jump-level trap hooks.
    traps: TrapConfig<'cb, 'ctx, Context, E>,
    /// Total wasm function parameter count (recompiler params + trap params).
    total_params: u32,
    /// Unified layout: arch params + trap params, then per-function locals.
    layout: LocalLayout,
    /// Mark placed after all param slots.
    locals_mark: Mark,
    /// Optional slot assigner: controls which RIPs get function slots.
    slot_assigner: Option<alloc::boxed::Box<dyn SlotAssigner + Send + Sync>>,
}

impl<'cb, 'ctx, Context, E, F, P> X86Recompiler<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: yecta::LocalPoolBackend + Default,
{
    /// Returns the function-index base used by the underlying reactor.
    ///
    /// All WASM function indices emitted during translation are offset by
    /// this value, allowing multiple recompilers to share a single WASM
    /// module without index collisions.
    pub fn base_func_offset(&self) -> u32 {
        self.reactor.base_func_offset()
    }

    /// Sets the function-index base.  Must be called before translation
    /// to ensure correct inter-function references.
    pub fn set_base_func_offset(&mut self, offset: u32) {
        self.reactor.set_base_func_offset(offset);
    }
    /// Create a new recompiler with `base_rip = 0`.
    pub fn new() -> Self {
        Self::new_with_base_rip(0)
    }

    /// Create a new recompiler whose RIP-relative addresses are resolved
    /// relative to `base_rip`.
    pub fn new_with_base_rip(base_rip: u64) -> Self {
        let mut recomp = Self {
            reactor: Reactor::default(),
            pool: { static T: TableIdx = TableIdx(0); Pool { handler: &T, ty: TypeIdx(0) } },
            escape_tag: None,
            base_rip,
            hints: Vec::new(),
            enable_speculative_calls: false,
            traps: TrapConfig::new(),
            total_params: Self::BASE_PARAMS,
            layout: LocalLayout::empty(),
            locals_mark: Mark {
                slot_count: 0,
                total_locals: 0,
            },
            slot_assigner: None,
        };
        recomp.setup_traps();
        recomp
    }

    /// Install a slot assigner to control which guest RIPs receive WASM function slots.
    ///
    /// When set, `rip_to_func_idx` uses `SlotAssigner::slot_for_pc` instead of the
    /// legacy `(rip - base_rip) / 1` formula, fixing the latent PC→slot-index bug.
    pub fn set_slot_assigner(&mut self, gate: impl SlotAssigner + Send + Sync + 'static) {
        self.slot_assigner = Some(alloc::boxed::Box::new(gate));
    }

    /// Return the total WASM function slots declared by the installed slot assigner.
    ///
    /// Panics if no slot assigner has been installed via `set_slot_assigner`.
    pub fn count_fns(&self) -> u32 {
        self.slot_assigner
            .as_ref()
            .expect("set_slot_assigner must be called before count_fns")
            .total_slots()
    }

    /// Enable or disable speculative call optimization
    ///
    /// When enabled, ABI-compliant x86_64 `call` instructions are lowered to native WASM `call`
    /// instructions wrapped in try-catch blocks. Returns (`ret`) compare stack top with expected
    /// return address and use direct WASM return for ABI-compliant cases.
    ///
    /// Requires an escape tag to be configured via `set_escape_tag()`.
    pub fn set_speculative_calls(&mut self, enable: bool) {
        self.enable_speculative_calls = enable;
    }

    /// Check if speculative call optimization is enabled
    pub fn is_speculative_calls_enabled(&self) -> bool {
        self.enable_speculative_calls
    }

    /// Set the escape tag used for exception-based control flow in speculative calls
    pub fn set_escape_tag(&mut self, tag: Option<EscapeTag>) {
        self.escape_tag = tag;
    }

    /// Get the current escape tag
    pub fn get_escape_tag(&self) -> Option<EscapeTag> {
        self.escape_tag
    }

    /// Get the local index for the expected return address used in speculative calls
    pub fn expected_ra_local() -> u32 {
        Self::EXPECTED_RA_LOCAL
    }

    fn emit_i64_const(&mut self, ctx: &mut Context, value: i64, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(value))
    }

    fn emit_i64_add(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Add)
    }
    fn emit_i64_sub(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Sub)
    }
    fn emit_i64_mul(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Mul)
    }
    fn emit_i64_and(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64And)
    }
    fn emit_i64_or(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Or)
    }
    fn emit_i64_xor(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Xor)
    }
    fn emit_i64_shl(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Shl)
    }
    fn emit_i64_shr_u(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64ShrU)
    }
    fn emit_i64_shr_s(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64ShrS)
    }

    // Condition flag helpers
    const ZF_LOCAL: u32 = 17;
    const SF_LOCAL: u32 = 18;
    const CF_LOCAL: u32 = 19;
    const OF_LOCAL: u32 = 20;
    const PF_LOCAL: u32 = 21;
    // Temporary locals: 22, 23, 24 are used in direct.rs
    // Expected return address for speculative calls
    const EXPECTED_RA_LOCAL: u32 = 25;

    /// x86-64 base parameter count:
    /// 16 GPRs (i64) + PC (i32) + ZF + SF + CF + OF + PF (5×i32) + 4 temp i64s
    /// + expected_RA (i64) = 26.
    pub const BASE_PARAMS: u32 = 26;

    /// Install an instruction trap.
    pub fn set_instruction_trap(
        &mut self,
        trap: &'cb mut (dyn InstructionTrap<Context, E> + 'ctx),
    ) {
        self.traps.set_instruction_trap(trap);
    }

    /// Remove the instruction trap.
    pub fn clear_instruction_trap(&mut self) {
        self.traps.clear_instruction_trap();
    }

    /// Install a jump trap.
    pub fn set_jump_trap(
        &mut self,
        trap: &'cb mut (dyn JumpTrap<Context, E> + 'ctx),
    ) {
        self.traps.set_jump_trap(trap);
    }

    /// Remove the jump trap.
    pub fn clear_jump_trap(&mut self) {
        self.traps.clear_jump_trap();
    }

    /// **Phase 1** — register trap parameters and compute `total_params`.
    pub fn setup_traps(&mut self) -> u32 {
        self.layout = LocalLayout::empty();
        self.layout.append(16, wasm_encoder::ValType::I64); // GPRs (params 0-15)
        self.layout.append(1, wasm_encoder::ValType::I32); // PC (param 16)
        self.layout.append(5, wasm_encoder::ValType::I32); // ZF/SF/CF/OF/PF (params 17-21)
        self.layout.append(4, wasm_encoder::ValType::I64); // temps + expected_RA (params 22-25)
        self.traps.declare_params(CellIdx(0),&mut self.layout);
        self.locals_mark = self.layout.mark();
        self.total_params = self.locals_mark.total_locals;
        self.total_params
    }

    /// The current total wasm function parameter count.
    pub fn total_params(&self) -> u32 {
        self.total_params
    }

    /// Classify an x86-64 mnemonic into [`InsnClass`] flags.
    pub(crate) fn classify_mnemonic(mnemonic: iced_x86::Mnemonic) -> InsnClass {
        use iced_x86::Mnemonic;
        match mnemonic {
            // Memory
            Mnemonic::Mov
            | Mnemonic::Movzx
            | Mnemonic::Movsx
            | Mnemonic::Movsxd
            | Mnemonic::Lea
            | Mnemonic::Xchg
            | Mnemonic::Push
            | Mnemonic::Pop
            | Mnemonic::Movsb
            | Mnemonic::Movsw
            | Mnemonic::Movsd
            | Mnemonic::Movsq
            | Mnemonic::Stosb
            | Mnemonic::Stosw
            | Mnemonic::Stosd
            | Mnemonic::Stosq
            | Mnemonic::Lodsb
            | Mnemonic::Lodsw
            | Mnemonic::Lodsd
            | Mnemonic::Lodsq
            | Mnemonic::Scasb
            | Mnemonic::Scasw
            | Mnemonic::Scasd
            | Mnemonic::Scasq => InsnClass::MEMORY,
            // Conditional branches
            Mnemonic::Jo
            | Mnemonic::Jno
            | Mnemonic::Jb
            | Mnemonic::Jae
            | Mnemonic::Je
            | Mnemonic::Jne
            | Mnemonic::Jbe
            | Mnemonic::Ja
            | Mnemonic::Js
            | Mnemonic::Jns
            | Mnemonic::Jp
            | Mnemonic::Jnp
            | Mnemonic::Jl
            | Mnemonic::Jge
            | Mnemonic::Jle
            | Mnemonic::Jg
            | Mnemonic::Jcxz
            | Mnemonic::Jecxz
            | Mnemonic::Jrcxz
            | Mnemonic::Loop
            | Mnemonic::Loope
            | Mnemonic::Loopne => InsnClass::BRANCH,
            // Direct/indirect unconditional jump
            Mnemonic::Jmp => InsnClass::BRANCH,
            // Call
            Mnemonic::Call => InsnClass::CALL,
            // Return
            Mnemonic::Ret | Mnemonic::Retf | Mnemonic::Iret | Mnemonic::Iretd | Mnemonic::Iretq => {
                InsnClass::RETURN
            }
            // Syscall / privileged
            Mnemonic::Syscall
            | Mnemonic::Sysenter
            | Mnemonic::Int
            | Mnemonic::Int1
            | Mnemonic::Int3
            | Mnemonic::Into => InsnClass::PRIVILEGED,
            // Float
            Mnemonic::Fld
            | Mnemonic::Fst
            | Mnemonic::Fstp
            | Mnemonic::Fadd
            | Mnemonic::Fsub
            | Mnemonic::Fmul
            | Mnemonic::Fdiv
            | Mnemonic::Fsqrt
            | Mnemonic::Movss
            | Mnemonic::Movsd
            | Mnemonic::Addss
            | Mnemonic::Addsd
            | Mnemonic::Subss
            | Mnemonic::Subsd
            | Mnemonic::Mulss
            | Mnemonic::Mulsd
            | Mnemonic::Divss
            | Mnemonic::Divsd
            | Mnemonic::Sqrtss
            | Mnemonic::Sqrtsd => InsnClass::FLOAT,
            _ => InsnClass::OTHER,
        }
    }

    fn set_zf(&mut self, ctx: &mut Context, value: bool, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalSet(Self::ZF_LOCAL))
    }

    fn set_sf(&mut self, ctx: &mut Context, value: bool, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalSet(Self::SF_LOCAL))
    }

    fn set_cf(&mut self, ctx: &mut Context, value: bool, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalSet(Self::CF_LOCAL))
    }

    fn set_of(&mut self, ctx: &mut Context, value: bool, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalSet(Self::OF_LOCAL))
    }

    fn set_pf(&mut self, ctx: &mut Context, value: bool, tail_idx: usize) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalSet(Self::PF_LOCAL))
    }

    // Helper to compute parity flag (even number of 1 bits in lowest byte)
    fn compute_parity(&mut self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        // Assume value is on stack (i64)
        // Extract lowest byte: value & 0xFF
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(0xFF))?;
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64And)?;
        // Count bits: use popcnt if available, otherwise simulate
        // For simplicity, we'll implement a basic parity check
        // This is a simplified version - real parity counts all bits in lowest byte
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I32WrapI64)?;
        // Simple parity: check if number of 1s is even
        // For now, just set to 0 (even parity) - this is a simplification
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::Drop)?;
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I32Const(0))?; // Assume even parity for simplicity
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalSet(Self::PF_LOCAL))
    }

    // Helper to set flags after arithmetic operation
    fn set_flags_after_operation(
        &mut self,
        ctx: &mut Context,
        result: i64,
        operand1: i64,
        operand2: i64,
        is_subtraction: bool,
        tail_idx: usize,
    ) -> Result<(), E> {
        // ZF: result == 0
        self.set_zf(ctx, result == 0, tail_idx)?;

        // SF: result < 0 (for signed)
        self.set_sf(ctx, result < 0, tail_idx)?;

        // For CF and OF, we need to detect carry/borrow and overflow
        // This is simplified - real implementation would need proper overflow detection

        // CF: carry flag (simplified)
        if is_subtraction {
            // For subtraction: CF if borrow occurred (operand1 < operand2 for unsigned)
            self.set_cf(ctx, (operand1 as u64) < (operand2 as u64), tail_idx)?;
        } else {
            // For addition: CF if result < operand1 (unsigned overflow)
            self.set_cf(ctx, (result as u64) < (operand1 as u64), tail_idx)?;
        }

        // OF: overflow flag (simplified - check if sign changed unexpectedly)
        let op1_sign = operand1 < 0;
        let op2_sign = operand2 < 0;
        let result_sign = result < 0;
        if is_subtraction {
            // For subtraction: overflow if (op1 positive, op2 negative, result negative) or (op1 negative, op2 positive, result positive)
            let overflow =
                (op1_sign && !op2_sign && !result_sign) || (!op1_sign && op2_sign && result_sign);
            self.set_of(ctx, overflow, tail_idx)?;
        } else {
            // For addition: overflow if both operands same sign but result different sign
            let overflow = (op1_sign == op2_sign) && (op1_sign != result_sign);
            self.set_of(ctx, overflow, tail_idx)?;
        }

        // PF: parity (simplified)
        self.compute_parity(ctx, tail_idx)?;

        Ok(())
    }

    fn emit_memory_load(
        &mut self,
        ctx: &mut Context,
        size_bits: u32,
        signed: bool,
        tail_idx: usize,
    ) -> Result<(), E> {
        use wasm_encoder::MemArg;
        match (size_bits, signed) {
            (8, true) => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Load8S(MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                }),
            ),
            (8, false) => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Load8U(MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                }),
            ),
            (16, true) => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Load16S(MemArg {
                    offset: 0,
                    align: 1,
                    memory_index: 0,
                }),
            ),
            (16, false) => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Load16U(MemArg {
                    offset: 0,
                    align: 1,
                    memory_index: 0,
                }),
            ),
            (32, true) => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Load32S(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }),
            ),
            (32, false) => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Load32U(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }),
            ),
            (64, _) => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }),
            ),
            _ => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::Unreachable),
        }
    }

    fn emit_memory_store(&mut self, ctx: &mut Context, size_bits: u32, tail_idx: usize) -> Result<(), E> {
        use wasm_encoder::MemArg;
        match size_bits {
            8 => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Store8(MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                }),
            ),
            16 => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Store16(MemArg {
                    offset: 0,
                    align: 1,
                    memory_index: 0,
                }),
            ),
            32 => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Store32(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }),
            ),
            64 => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Store(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }),
            ),
            _ => Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::Unreachable),
        }
    }

    // Helpers for sub-register read/write
    fn emit_mask_shift_for_read(
        &mut self,
        ctx: &mut Context,
        size_bits: u32,
        bit_offset: u32,
        tail_idx: usize,
    ) -> Result<(), E> {
        // shift right by bit_offset then mask size_bits
        if bit_offset > 0 {
            Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(bit_offset as i64))?;
            Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64ShrU)?;
        }
        match size_bits {
            64 => { /* no mask */ }
            32 => { /* locals model 32-bit values as zero-extended into 64, so no mask needed for reads */
            }
            16 => {
                Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(0xFFFF))?;
                Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64And)?;
            }
            8 => {
                Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(0xFF))?;
                Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64And)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn emit_subreg_write_rmw(
        &mut self,
        ctx: &mut Context,
        local: u32,
        size_bits: u32,
        bit_offset: u32,
        tail_idx: usize,
    ) -> Result<(), E> {
        // Assume new value is on stack (i64) and we need to write it into `local` at bit_offset preserving other bits.
        // Steps:
        // local_val = local.get(local)
        // mask = ((1 << size_bits) - 1) << bit_offset
        // cleared = local_val & ~mask
        // new_shifted = (new_value & ((1<<size_bits)-1)) << bit_offset
        // combined = cleared | new_shifted
        // local.set(local, combined)

        // local.get(local)
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalGet(local))?;
        // store original in temp (we rely on temp locals being available after local 16). We'll use LocalSet(17) and LocalGet(17).
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalSet(17))?;
        // compute mask = (1<<size_bits)-1
        let mask: i64 = if size_bits == 64 {
            -1i64
        } else {
            ((1u128 << size_bits) - 1) as i64
        };
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(mask))?;
        // shift mask left by bit_offset
        if bit_offset > 0 {
            Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(bit_offset as i64))?;
            Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Shl)?;
        }
        // invert mask -> ~mask
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(-1))?; // -1 is all ones
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Xor)?; // ~mask = mask ^ -1
                                                       // get original
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalGet(17))?;
        // cleared = original & ~mask
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64And)?;
        // now compute new_shifted: we assume new_value is currently on top of stack
        // mask = (1<<size_bits)-1 (again)
        let small_mask: i64 = if size_bits == 64 {
            -1i64
        } else {
            ((1u128 << size_bits) - 1) as i64
        };
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(small_mask))?;
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64And)?; // new_value & small_mask
        if bit_offset > 0 {
            Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Const(bit_offset as i64))?;
            Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Shl)?; // << bit_offset
        }
        // combined = cleared | new_shifted
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::I64Or)?;
        // store back into local
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, &Instruction::LocalSet(local))?;
        Ok(())
    }
}
enum Operand {
    Imm(i64),
    Reg(u32),
    RegWithSize(u32, u32, u32),
}

impl From<i64> for Operand {
    fn from(v: i64) -> Self {
        Operand::Imm(v)
    }
}

impl From<u32> for Operand {
    fn from(v: u32) -> Self {
        Operand::Reg(v)
    }
}

impl Operand {
    fn with_bit(self, bit_offset: u32) -> Self {
        match self {
            Operand::RegWithSize(r, s, _) => Operand::RegWithSize(r, s, bit_offset),
            other => other,
        }
    }
}

// ── Recompile impl ────────────────────────────────────────────────────────────

use alloc::string::String;
use speet_link_core::{
    context::ReactorContext,
    recompiler::Recompile,
    unit::{BinaryUnit, FuncType},
};
use wasm_encoder::ValType;

impl<'cb, 'ctx, Context, E, F, P> Recompile<Context, E, F>
    for X86Recompiler<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: yecta::LocalPoolBackend + Default,
{
    /// New `base_rip` for the next binary.
    type BinaryArgs = u64;

    fn reset_for_next_binary(
        &mut self,
        _ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        new_base_rip: u64,
    ) {
        self.base_rip = new_base_rip;
        self.hints.clear();
    }

    fn drain_unit(
        &mut self,
        ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        entry_points: Vec<(String, u32)>,
    ) -> BinaryUnit<F> {
        // Build the uniform function type: all params are i64 (x86-64 GPRs,
        // PC, flags, temps are all i64 or i32, but each function shares the
        // same total_params-wide signature).
        let param_types: alloc::vec::Vec<ValType> =
            (0..self.total_params).map(|_| ValType::I64).collect();
        let func_type = FuncType::from_val_types(&param_types, &[]);

        let base = ctx.base_func_offset();
        let fns = ctx.drain_fns();
        let count = fns.len();
        BinaryUnit {
            fns,
            base_func_offset: base,
            entry_points,
            func_types: alloc::vec![func_type; count],
            data_segments: alloc::vec![],
            data_init_fn: None,
        }
    }
}
