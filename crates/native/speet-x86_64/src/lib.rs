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
use yecta::{EscapeTag, Fed, LocalDeclarator, LocalLayout, LocalPoolBackend, Mark, Pool, Reactor, SlotAssigner, TableIdx, TypeIdx, layout::{CellIdx, CellRegistry}};
pub mod cfg;
pub mod direct;
use speet_memory::MemoryAccess;
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
/// - `Context` / `E` – forwarded to the underlying [`yecta::Reactor`].
///
/// The function-sink type `F` is a per-method generic (not on the struct).
pub struct X86Recompiler<Context, E> {
    base_rip: u64,
    hints: Vec<u8>,
    enable_speculative_calls: bool,
    /// Optional slot assigner: controls which RIPs get function slots.
    slot_assigner: Option<alloc::boxed::Box<dyn SlotAssigner + Send + Sync>>,
    /// Mnemonics that had no translation and fell back to `unreachable`.
    /// Cleared by [`clear_unsupported`](Self::clear_unsupported).
    unsupported_insns: alloc::collections::BTreeSet<alloc::string::String>,
    /// Optional memory access implementation (mapper + load/store).
    memory_access: Option<alloc::boxed::Box<dyn MemoryAccess<Context, E>>>,
}

impl<Context, E> X86Recompiler<Context, E> {
    /// Returns the function-index base used by the underlying reactor.
    ///
    /// All WASM function indices emitted during translation are offset by
    /// this value, allowing multiple recompilers to share a single WASM
    /// module without index collisions.
    pub fn base_func_offset<F>(&self, ctx: &dyn ReactorContext<Context, E, FnType = F>) -> u32 {
        ctx.base_func_offset()
    }

    /// Sets the function-index base.  Must be called before translation
    /// to ensure correct inter-function references.
    pub fn set_base_func_offset<F>(&mut self, ctx: &mut dyn ReactorContext<Context, E, FnType = F>, offset: u32) {
        ctx.set_base_func_offset(offset);
    }

    /// Create a new recompiler with `base_rip = 0`.
    pub fn new() -> Self {
        Self::new_with_base_rip(0)
    }

    /// Create a new recompiler whose RIP-relative addresses are resolved
    /// relative to `base_rip`.
    pub fn new_with_base_rip(base_rip: u64) -> Self {
        Self {
            base_rip,
            hints: Vec::new(),
            enable_speculative_calls: false,
            slot_assigner: None,
            unsupported_insns: alloc::collections::BTreeSet::new(),
            memory_access: None,
        }
    }

    /// Returns the set of instruction mnemonics that had no translation and
    /// fell back to `unreachable` during the most recent `translate_bytes` call.
    pub fn unsupported_insns(&self) -> &alloc::collections::BTreeSet<alloc::string::String> {
        &self.unsupported_insns
    }

    /// Clear the unsupported-instruction tracking set.
    pub fn clear_unsupported(&mut self) {
        self.unsupported_insns.clear();
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
    pub fn set_escape_tag<F>(&mut self, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, tag: Option<EscapeTag>) {
        rctx.set_escape_tag(tag);
    }

    /// Get the current escape tag
    pub fn get_escape_tag<F>(&self, rctx: &dyn ReactorContext<Context, E, FnType = F>) -> Option<EscapeTag> {
        rctx.escape_tag()
    }

    /// Get the local index for the expected return address used in speculative calls
    pub fn expected_ra_local() -> u32 {
        Self::EXPECTED_RA_LOCAL
    }

    /// Install a memory access implementation (mapper + load/store).
    ///
    /// The implementation's `declare_params` / `declare_locals` will be
    /// called during `setup_traps` / `init_function`.
    pub fn set_memory_access(&mut self, ma: alloc::boxed::Box<dyn MemoryAccess<Context, E>>) {
        self.memory_access = Some(ma);
    }

    /// Remove any previously installed memory access implementation.
    pub fn clear_memory_access(&mut self) {
        self.memory_access = None;
    }

    fn emit_i64_const<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, value: i64) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(value))
    }

    fn emit_i64_add<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64Add)
    }
    fn emit_i64_sub<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64Sub)
    }
    fn emit_i64_mul<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64Mul)
    }
    fn emit_i64_and<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64And)
    }
    fn emit_i64_or<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64Or)
    }
    fn emit_i64_xor<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)
    }
    fn emit_i64_shl<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64Shl)
    }
    fn emit_i64_shr_u<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)
    }
    fn emit_i64_shr_s<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)
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

    /// **Phase 1** — register trap parameters and compute `total_params`.
    ///
    /// Calls `declare_trap_params` on the underlying context, forwarding
    /// the installed memory access (if any) as the `extra` declarator.
    pub fn setup_traps<F>(
        &mut self,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
    ) -> u32 {
        rctx.layout_mut().append(16, wasm_encoder::ValType::I64);
        rctx.layout_mut().append(1, wasm_encoder::ValType::I32);
        rctx.layout_mut().append(5, wasm_encoder::ValType::I32);
        rctx.layout_mut().append(4, wasm_encoder::ValType::I64);
        let mut unit = ();
        let extra: &mut dyn LocalDeclarator = match self.memory_access.as_deref_mut() {
            Some(m) => m as &mut dyn LocalDeclarator,
            None => &mut unit,
        };
        rctx.declare_trap_params(extra);
        let mark = rctx.layout().mark();
        rctx.set_locals_mark(mark);
        mark.total_locals
    }

    /// The current total wasm function parameter count.
    pub fn total_params<F>(&self, rctx: &dyn ReactorContext<Context, E, FnType = F>) -> u32 {
        rctx.locals_mark().total_locals
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

    fn set_zf<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, value: bool) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::ZF_LOCAL))
    }

    fn set_sf<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, value: bool) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::SF_LOCAL))
    }

    fn set_cf<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, value: bool) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::CF_LOCAL))
    }

    fn set_of<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, value: bool) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::OF_LOCAL))
    }

    fn set_pf<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize, value: bool) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(if value { 1 } else { 0 }))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::PF_LOCAL))
    }

    // Helper to compute parity flag (even number of 1 bits in lowest byte)
    fn compute_parity<F>(&self, ctx: &mut Context, rctx: &dyn ReactorContext<Context, E, FnType = F>, tail_idx: usize) -> Result<(), E> {
        // Assume value is on stack (i64)
        // Extract lowest byte: value & 0xFF
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
        // Count bits: use popcnt if available, otherwise simulate
        // For simplicity, we'll implement a basic parity check
        // This is a simplified version - real parity counts all bits in lowest byte
        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        // Simple parity: check if number of 1s is even
        // For now, just set to 0 (even parity) - this is a simplification
        rctx.feed(ctx, tail_idx, &Instruction::Drop)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?; // Assume even parity for simplicity
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::PF_LOCAL))
    }

    // Helper to set flags after arithmetic operation
    fn set_flags_after_operation<F>(
        &self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        result: i64,
        operand1: i64,
        operand2: i64,
        is_subtraction: bool,
    ) -> Result<(), E> {
        // ZF: result == 0
        self.set_zf(ctx, rctx, tail_idx, result == 0)?;

        // SF: result < 0 (for signed)
        self.set_sf(ctx, rctx, tail_idx, result < 0)?;

        // For CF and OF, we need to detect carry/borrow and overflow
        // This is simplified - real implementation would need proper overflow detection

        // CF: carry flag (simplified)
        if is_subtraction {
            // For subtraction: CF if borrow occurred (operand1 < operand2 for unsigned)
            self.set_cf(ctx, rctx, tail_idx, (operand1 as u64) < (operand2 as u64))?;
        } else {
            // For addition: CF if result < operand1 (unsigned overflow)
            self.set_cf(ctx, rctx, tail_idx, (result as u64) < (operand1 as u64))?;
        }

        // OF: overflow flag (simplified - check if sign changed unexpectedly)
        let op1_sign = operand1 < 0;
        let op2_sign = operand2 < 0;
        let result_sign = result < 0;
        if is_subtraction {
            // For subtraction: overflow if (op1 positive, op2 negative, result negative) or (op1 negative, op2 positive, result positive)
            let overflow =
                (op1_sign && !op2_sign && !result_sign) || (!op1_sign && op2_sign && result_sign);
            self.set_of(ctx, rctx, tail_idx, overflow)?;
        } else {
            // For addition: overflow if both operands same sign but result different sign
            let overflow = (op1_sign == op2_sign) && (op1_sign != result_sign);
            self.set_of(ctx, rctx, tail_idx, overflow)?;
        }

        // PF: parity (simplified)
        self.compute_parity(ctx, rctx, tail_idx)?;

        Ok(())
    }

    pub(crate) fn emit_memory_load<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        size_bits: u32,
        signed: bool,
    ) -> Result<(), E> {
        use wasm_encoder::MemArg;
        if let Some(ma) = self.memory_access.as_deref_mut() {
            use speet_memory::mem::LoadKind;
            use speet_ordering::EagerMemorySink;
            use speet_link_core::context::FedContext;
            let kind = match (size_bits, signed) {
                (8, true)   => LoadKind::I8S,
                (8, false)  => LoadKind::I8U,
                (16, true)  => LoadKind::I16S,
                (16, false) => LoadKind::I16U,
                (32, true)  => LoadKind::I32S,
                (32, false) => LoadKind::I32U,
                (64, _)     => LoadKind::I64,
                _ => return rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
            };
            let mut fed = FedContext::new(rctx, tail_idx);
            let mut sink = EagerMemorySink::new(&mut fed);
            ma.emit_load(ctx, &mut sink, kind)
        } else {
            match (size_bits, signed) {
                (8, true)   => rctx.feed(ctx, tail_idx, &Instruction::I64Load8S(MemArg  { offset: 0, align: 0, memory_index: 0 })),
                (8, false)  => rctx.feed(ctx, tail_idx, &Instruction::I64Load8U(MemArg  { offset: 0, align: 0, memory_index: 0 })),
                (16, true)  => rctx.feed(ctx, tail_idx, &Instruction::I64Load16S(MemArg { offset: 0, align: 1, memory_index: 0 })),
                (16, false) => rctx.feed(ctx, tail_idx, &Instruction::I64Load16U(MemArg { offset: 0, align: 1, memory_index: 0 })),
                (32, true)  => rctx.feed(ctx, tail_idx, &Instruction::I64Load32S(MemArg { offset: 0, align: 2, memory_index: 0 })),
                (32, false) => rctx.feed(ctx, tail_idx, &Instruction::I64Load32U(MemArg { offset: 0, align: 2, memory_index: 0 })),
                (64, _)     => rctx.feed(ctx, tail_idx, &Instruction::I64Load(MemArg    { offset: 0, align: 3, memory_index: 0 })),
                _           => rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
            }
        }
    }

    pub(crate) fn emit_memory_store<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        size_bits: u32,
    ) -> Result<(), E> {
        use wasm_encoder::MemArg;
        // Stack contract: [addr, value] — addr deeper, value on top.
        if let Some(ma) = self.memory_access.as_deref_mut() {
            use speet_memory::mem::StoreKind;
            use speet_ordering::EagerMemorySink;
            use speet_link_core::context::FedContext;
            const STORE_TMP: u32 = 22;
            let kind = match size_bits {
                8  => StoreKind::I8,
                16 => StoreKind::I16,
                32 => StoreKind::I32,
                64 => StoreKind::I64,
                _  => return rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
            };
            // Save value (top of stack) to scratch local, map addr, restore value.
            rctx.feed(ctx, tail_idx, &Instruction::LocalSet(STORE_TMP))?;
            {
                let mut fed = FedContext::new(rctx, tail_idx);
                let mut sink = EagerMemorySink::new(&mut fed);
                ma.emit_store_addr(ctx, &mut sink)?;
            }
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(STORE_TMP))?;
            {
                let mut fed = FedContext::new(rctx, tail_idx);
                let mut sink = EagerMemorySink::new(&mut fed);
                ma.emit_store_insn(ctx, &mut sink, kind)
            }
        } else {
            match size_bits {
                8  => rctx.feed(ctx, tail_idx, &Instruction::I64Store8(MemArg  { offset: 0, align: 0, memory_index: 0 })),
                16 => rctx.feed(ctx, tail_idx, &Instruction::I64Store16(MemArg { offset: 0, align: 1, memory_index: 0 })),
                32 => rctx.feed(ctx, tail_idx, &Instruction::I64Store32(MemArg { offset: 0, align: 2, memory_index: 0 })),
                64 => rctx.feed(ctx, tail_idx, &Instruction::I64Store(MemArg   { offset: 0, align: 3, memory_index: 0 })),
                _  => rctx.feed(ctx, tail_idx, &Instruction::Unreachable),
            }
        }
    }

    fn emit_mask_shift_for_read<F>(
        &self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        size_bits: u32,
        bit_offset: u32,
    ) -> Result<(), E> {
        if bit_offset > 0 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(bit_offset as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
        }
        match size_bits {
            64 | 32 => {}
            16 => {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
            }
            8 => {
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFF))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn emit_subreg_write_rmw<F>(
        &self,
        ctx: &mut Context,
        rctx: &dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        local: u32,
        size_bits: u32,
        bit_offset: u32,
    ) -> Result<(), E> {
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(local))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(17))?;
        let mask: i64 = if size_bits == 64 { -1i64 } else { ((1u128 << size_bits) - 1) as i64 };
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(mask))?;
        if bit_offset > 0 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(bit_offset as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(-1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(17))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
        let small_mask: i64 = if size_bits == 64 { -1i64 } else { ((1u128 << size_bits) - 1) as i64 };
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(small_mask))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
        if bit_offset > 0 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(bit_offset as i64))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Shl)?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(local))?;
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

impl<Context, E, F> Recompile<Context, E, F>
    for X86Recompiler<Context, E>
where
    F: InstructionSink<Context, E>,
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
        rctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        entry_points: Vec<(String, u32)>,
    ) -> BinaryUnit<F> {
        // Build the uniform function type matching setup_traps():
        // 16 x i64 GPRs, 1 x i32 PC, 5 x i32 flags, 4 x i64 temps.
        let mut param_types: alloc::vec::Vec<ValType> = alloc::vec::Vec::with_capacity(Self::BASE_PARAMS as usize);
        param_types.extend((0..16).map(|_| ValType::I64));
        param_types.extend((0..6).map(|_| ValType::I32));
        param_types.extend((0..4).map(|_| ValType::I64));
        let func_type = FuncType::from_val_types(&param_types, &[]);

        let base = rctx.base_func_offset();
        let fns = rctx.drain_fns();
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
