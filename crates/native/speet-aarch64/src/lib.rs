//! # AArch64 to WebAssembly Recompiler
//!
//! Translates a subset of AArch64 machine code to WebAssembly using the
//! [`yecta`] reactor for control-flow management.
//!
//! ## Supported instructions
//! Integer ALU with immediate (`ADD`, `SUB`, `ADDS`, `SUBS`, `CMP`),
//! integer ALU register (`ADD`, `SUB` shifted/extended), logical shifted
//! (`AND`, `ORR`, `EOR`, `BIC`, `ORN`, `EON`), logical immediate
//! (`AND`, `ORR`, `EOR`), move-wide (`MOVZ`, `MOVK`, `MOVN`),
//! bitfield (`UBFM`/`SBFM`/`BFM` — shifts, sign/zero-extends),
//! multiply (`MADD`, `MSUB`, `SMADDL`, `UMADDL`, `SMULH`, `UMULH`),
//! division (`UDIV`, `SDIV`), register shifts (`LSLV`, `LSRV`, `ASRV`, `RORV`),
//! conditional select (`CSEL`, `CSINC`, `CSINV`, `CSNEG`),
//! PC-relative address (`ADR`, `ADRP`),
//! control flow (`B`, `BL`, `BR`, `BLR`, `RET`, `CBZ`, `CBNZ`, `B.cond`),
//! system (`MRS`/`MSR` NZCV), exception (`BRK`),
//! scalar loads/stores unsigned-offset (`LDR`, `LDRB`, `LDRH`, `LDRSW`,
//! `STR`, `STRB`, `STRH`), pre/post-index loads/stores, register-offset
//! loads/stores, load/store pairs (`LDP`, `STP`, `LDPSW`),
//! floating-point arithmetic (`FADD`, `FSUB`, `FMUL`, `FDIV`, `FNMUL`, `FMIN`,
//! `FMAX`, `FMINNM`, `FMAXNM`, `FMADD`, `FMSUB`, `FNMADD`, `FNMSUB`),
//! FP unary (`FABS`, `FNEG`, `FSQRT`, `FMOV`), FP conversions
//! (`SCVTF`, `UCVTF`, `FCVTZS`, `FCVTZU`), FP compare (`FCMP`), FP select (`FCSEL`).
//!
//! ## Local-variable layout (one WASM function per instruction)
//! - Params  0–30: x0–x30 general-purpose registers (each `i64`).
//! - Param  31:    program counter / PC (`i64`).
//! - Params 32–35: condition flags N, Z, C, V (each `i32`).
//! - Params 36–38: scratch temporaries (`i64`).
//! - Params 39–70: V0–V31 floating-point registers (`f64`).
//!
//! x31 is XZR in most contexts: reads emit `i64.const 0`, writes are dropped.

#![no_std]
extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::string::String;

use disarm64::decoder_full::Mnemonic;
use speet_link_core::{FedContext, ReactorContext};
use speet_memory::MemoryAccess;
use speet_traps::{
    InstructionInfo, JumpInfo, JumpKind, TrapAction,
    insn::{ArchTag, InsnClass},
    LocalDeclarator,
};
use wasm_encoder::{Ieee64, Instruction, ValType};
use yecta::{FuncIdx, JumpCallParams, LocalLayout, LocalSlot, Mark};

pub mod direct;
use direct::A64IndirectTarget;

/// AArch64 to WebAssembly recompiler.
pub struct AArch64Recompiler<Context, E> {
    pub(crate) base_pc: u64,
    unsupported_insns: BTreeSet<String>,
    pub(crate) memory_access: Option<alloc::boxed::Box<dyn MemoryAccess<Context, E>>>,
    /// x0–x30 (31 × i64).
    pub(crate) gpr_slot: LocalSlot,
    /// PC (1 × i64).
    pub(crate) pc_slot: LocalSlot,
    /// N, Z, C, V (4 × i32).
    pub(crate) nzcv_slot: LocalSlot,
    /// Three scratch temporaries (3 × i64): tmp0=result, tmp1=operand-a, tmp2=addr-scratch.
    pub(crate) tmp_slot: LocalSlot,
    /// V0–V31 floating-point registers (32 × f64).
    pub(crate) fp_slot: LocalSlot,
    /// Stack pointer (1 × i64).  AArch64 encodes SP as register 31 in load/store and
    /// `ADD`/`SUB` immediate forms; x31 remains XZR everywhere else.
    pub(crate) sp_slot: LocalSlot,
    /// Guest-address → symbolic-label hook table (compile-time PLT/
    /// external-call redirects). Shared type with `speet-x86_64` via
    /// `speet_plugin_api::external_target` instead of each arch keeping
    /// its own `plt_by_addr`/`plt_imports` `BTreeMap` pair — see
    /// `docs/guides/thin-runtime-genericity.md` principle 1.
    hooks: Option<speet_plugin_api::external_target::PltHookTable>,
    hook_library: speet_plugin_api::external_target::LibraryId,
    stub_for_pc_import_idx: Option<u32>,
}

impl<Context, E> AArch64Recompiler<Context, E> {
    /// AArch64 base parameter count: 31 GPRs (i64) + PC (i64) + 4 NZCV flags
    /// (i32) + 3 scratch temporaries (i64) + 32 FP registers (f64) + SP
    /// (i64) = 72. See the module doc's local-variable layout.
    pub const BASE_PARAMS: u32 = 72;

    /// WASM param/local index of the guest SP register. Stable regardless of
    /// trap params, which `setup_traps` always appends *after*
    /// `BASE_PARAMS` (SP is the last of the fixed slots). A C-callable entry
    /// (e.g. `speet-rt`'s shim) must seed this argument position with a
    /// valid, freshly allocated guest stack — guest registers are WASM
    /// params (see `docs/guides/thin-runtime-genericity.md`), so an
    /// externally-invoked entry function is only correctly callable if the
    /// caller supplies real initial values for every one of them, not just
    /// whatever a bare C prototype leaves as garbage.
    pub const SP_PARAM_INDEX: u32 = Self::BASE_PARAMS - 1;

    /// WASM param/local index of the guest link register (X30/LR) — always
    /// `30`, since `setup_traps` appends the 31 GPRs (x0–x30) first (see the
    /// module doc's local-variable layout), stable regardless of trap
    /// params appended afterward. A C-callable entry must seed this
    /// position with the **halt sentinel** address (see
    /// `speet_recompile::frontend::halt_addr` and
    /// `docs/guides/thin-runtime-genericity.md` principle 4) so a guest
    /// `ret` that's never overwritten LR — i.e. really is the program's
    /// final return, most commonly `main` returning with no crt0 chain —
    /// lands on the reserved halt stub instead of on whatever garbage LR
    /// held at entry.
    pub const LR_PARAM_INDEX: u32 = 30;

    pub fn new_with_base_pc(base_pc: u64) -> Self {
        Self {
            base_pc,
            unsupported_insns: BTreeSet::new(),
            memory_access: None,
            gpr_slot: LocalSlot::default(),
            pc_slot: LocalSlot::default(),
            nzcv_slot: LocalSlot::default(),
            tmp_slot: LocalSlot::default(),
            fp_slot: LocalSlot::default(),
            sp_slot: LocalSlot::default(),
            hooks: None,
            hook_library: speet_plugin_api::external_target::LibraryId::MAIN_IMAGE,
            stub_for_pc_import_idx: None,
        }
    }

    /// WASM import index for `env.__speet_stub_for_pc` (fn-ptr arg rewrite).
    pub fn set_stub_for_pc_import_idx(&mut self, idx: u32) {
        self.stub_for_pc_import_idx = Some(idx);
    }

    /// Install compile-time PLT/external-call hooks (integrated runtime).
    /// Checked at every decode slot's PC, not just resolved `bl`/`b`
    /// targets — see `docs/guides/thin-runtime-genericity.md` principle 2.
    pub fn set_plt_hooks(&mut self, hooks: speet_plugin_api::external_target::PltHookTable) {
        self.hooks = Some(hooks);
    }

    /// Which [`LibraryId`] [`lookup_hook`] uses (default: main image).
    pub fn set_hook_library(&mut self, library: speet_plugin_api::external_target::LibraryId) {
        self.hook_library = library;
    }

    pub(crate) fn arg_needs_fn_ptr_rewrite(&self, label: &str, arg_idx: usize) -> bool {
        let bare = label.strip_prefix('_').unwrap_or(label);
        speet_abi_stubs::fn_ptr_arg_indices(label)
            .or_else(|| speet_abi_stubs::fn_ptr_arg_indices(bare))
            .is_some_and(|indices| indices.contains(&arg_idx))
    }

    pub fn new() -> Self {
        Self::new_with_base_pc(0)
    }

    pub fn unsupported_insns(&self) -> &BTreeSet<String> {
        &self.unsupported_insns
    }

    pub fn clear_unsupported(&mut self) {
        self.unsupported_insns.clear();
    }

    pub fn set_memory_access(&mut self, ma: alloc::boxed::Box<dyn MemoryAccess<Context, E>>) {
        self.memory_access = Some(ma);
    }

    /// Register trap parameters and compute the total WASM local count.
    /// Must be called once before translation.
    pub fn setup_traps<F>(
        &mut self,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        _ctx: &mut Context,
    ) -> u32 {
        self.gpr_slot  = rctx.layout_mut().append(31, ValType::I64); // x0–x30
        self.pc_slot   = rctx.layout_mut().append(1,  ValType::I64); // PC
        self.nzcv_slot = rctx.layout_mut().append(4,  ValType::I32); // N, Z, C, V
        self.tmp_slot  = rctx.layout_mut().append(3,  ValType::I64); // 3 scratch i64
        self.fp_slot   = rctx.layout_mut().append(32, ValType::F64); // V0–V31
        self.sp_slot   = rctx.layout_mut().append(1,  ValType::I64); // SP

        let mut unit = ();
        let extra: &mut dyn LocalDeclarator = match self.memory_access.as_deref_mut() {
            Some(m) => m as &mut dyn LocalDeclarator,
            None => &mut unit,
        };
        rctx.declare_trap_params(extra);
        let mark = rctx.layout().mark();
        rctx.set_locals_mark(mark);
        // `DirectMemory::declare_locals` allocates the address-mapper's own
        // scratch local (e.g. for alias-check flushes) — without this call,
        // `emit_load`/`emit_store_addr` panic the first time they run since
        // that scratch slot was never assigned. Must run *after* the params
        // mark is captured above: `declare_locals` is a distinct concern from
        // `declare_params` (see `ReactorContext::declare_trap_params` vs
        // `declare_trap_locals`) and appends a genuine per-function *local*,
        // not a wasm function *param* — calling it before the mark folded
        // this scratch into the param list, corrupting every function's
        // exported/expected type signature by one extra i32.
        rctx.declare_trap_locals(extra);
        mark.total_locals
    }

    /// Classify an AArch64 mnemonic into [`InsnClass`] flags.
    pub(crate) fn classify_mnemonic(mnemonic: Mnemonic) -> InsnClass {
        match mnemonic {
            Mnemonic::r#ldr
            | Mnemonic::r#ldrb
            | Mnemonic::r#ldrh
            | Mnemonic::r#ldrsb
            | Mnemonic::r#ldrsh
            | Mnemonic::r#ldrsw
            | Mnemonic::r#str
            | Mnemonic::r#strb
            | Mnemonic::r#strh
            | Mnemonic::r#ldp
            | Mnemonic::r#ldpsw
            | Mnemonic::r#stp => InsnClass::MEMORY,

            Mnemonic::r#b | Mnemonic::r#cbz | Mnemonic::r#cbnz => InsnClass::BRANCH,
            Mnemonic::r#br => InsnClass::BRANCH | InsnClass::INDIRECT,

            Mnemonic::r#bl => InsnClass::CALL,
            Mnemonic::r#blr => InsnClass::CALL | InsnClass::INDIRECT,

            Mnemonic::r#ret | Mnemonic::r#retaa | Mnemonic::r#retab => {
                InsnClass::RETURN | InsnClass::INDIRECT
            }

            Mnemonic::r#svc | Mnemonic::r#hvc | Mnemonic::r#smc => InsnClass::PRIVILEGED,

            Mnemonic::r#fadd | Mnemonic::r#fsub | Mnemonic::r#fmul | Mnemonic::r#fdiv
            | Mnemonic::r#fcmp | Mnemonic::r#fcmpe | Mnemonic::r#fcvt | Mnemonic::r#fmov
            | Mnemonic::r#fsqrt | Mnemonic::r#fmadd | Mnemonic::r#fmsub
            | Mnemonic::r#fabs | Mnemonic::r#fneg | Mnemonic::r#fcsel
            | Mnemonic::r#scvtf | Mnemonic::r#ucvtf
            | Mnemonic::r#fcvtzs | Mnemonic::r#fcvtzu => InsnClass::FLOAT,

            Mnemonic::r#ldadd | Mnemonic::r#stlr | Mnemonic::r#ldaxr
            | Mnemonic::r#stlxr | Mnemonic::r#cas => InsnClass::ATOMIC | InsnClass::MEMORY,

            _ => InsnClass::OTHER,
        }
    }

    /// Convert a guest PC to a WASM function index relative to base_pc.
    pub(crate) fn pc_to_func_idx(&self, pc: u64) -> Option<FuncIdx> {
        let offset = pc.checked_sub(self.base_pc)?;
        Some(FuncIdx((offset / 4) as u32))
    }

    pub(crate) fn lookup_hook(&self, target: u64) -> Option<&speet_plugin_api::external_target::PltHook> {
        self.hooks.as_ref()?.lookup(self.hook_library, target)
    }

    /// Emit "make the redirected host call, then behave like a `ret`" for a
    /// hooked guest address — correct regardless of whether it was reached
    /// via `bl` (which just set `x30 = pc+4`) or a tail `b` (whose caller's
    /// `x30` is preserved unchanged through the chain): either way, `x30`
    /// holds a valid return address at this point, so jumping to it is
    /// safe. See `docs/guides/thin-runtime-genericity.md` principle 2 and
    /// `speet_plugin_api::external_target::CallingConvention`.
    pub(crate) fn emit_hook_call_and_return<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        pc: u64,
        hook: &speet_plugin_api::external_target::PltHook,
    ) -> Result<(), E> {
        use speet_plugin_api::external_target::PltHookTarget;
        let PltHookTarget::WasmImport { import_idx } = hook.target else {
            return Ok(());
        };
        for (i, &reg) in hook.convention.arg_locals.iter().enumerate() {
            self.emit_gpr_get(ctx, rctx, tail_idx, reg)?;
            if self.arg_needs_fn_ptr_rewrite(&hook.label, i) {
                if let Some(stub_idx) = self.stub_for_pc_import_idx {
                    rctx.feed(ctx, tail_idx, &Instruction::Call(stub_idx))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
                    rctx.feed(
                        ctx,
                        tail_idx,
                        &Instruction::If(wasm_encoder::BlockType::Empty),
                    )?;
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                    rctx.feed(ctx, tail_idx, &Instruction::End)?;
                }
            } else if hook.convention.wraps_i32(i) {
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
            }
        }
        rctx.feed(ctx, tail_idx, &Instruction::Call(import_idx))?;
        if let Some(result_reg) = hook.convention.result_local {
            if hook.convention.result_extend_i32 {
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
            }
            self.emit_gpr_set(ctx, rctx, tail_idx, result_reg)?;
        }

        let total = rctx.locals_mark().total_locals;
        let lr_local = rctx.layout().local(self.gpr_slot, 30);
        {
            let info = JumpInfo::indirect(pc, lr_local, JumpKind::Return);
            if rctx.on_jump(&info, ctx)? == TrapAction::Skip {
                return Ok(());
            }
        }
        let target_snippet = A64IndirectTarget {
            gpr_local: lr_local,
            base_pc: self.base_pc,
            base_func_offset: rctx.base_func_offset(),
        };
        let params = JumpCallParams::indirect_jump(&target_snippet, total, rctx.pool());
        rctx.ji_with_params(ctx, tail_idx, params)?;
        Ok(())
    }

    // ── GPR emit helpers ──────────────────────────────────────────────────────

    /// Push GPR `reg` (0–30) onto the WASM stack.  Register 31 → `i64.const 0`.
    pub(crate) fn emit_gpr_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg >= 31 {
            return rctx.feed(ctx, tail_idx, &Instruction::I64Const(0));
        }
        for instr in rctx.layout().emit_get(self.gpr_slot, reg) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// Pop the stack top into GPR `reg` (0–30).  Register 31 → `drop`.
    pub(crate) fn emit_gpr_set<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg >= 31 {
            return rctx.feed(ctx, tail_idx, &Instruction::Drop);
        }
        for instr in rctx.layout().emit_set(self.gpr_slot, reg) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// Push SP onto the WASM stack.
    pub(crate) fn emit_sp_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
    ) -> Result<(), E> {
        for instr in rctx.layout().emit_get(self.sp_slot, 0) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// Pop the stack top into SP.
    pub(crate) fn emit_sp_set<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
    ) -> Result<(), E> {
        for instr in rctx.layout().emit_set(self.sp_slot, 0) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// Load/store base register: encoding 31 is SP, not XZR.
    pub(crate) fn emit_addr_reg_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg == 31 {
            self.emit_sp_get(ctx, rctx, tail_idx)
        } else {
            self.emit_gpr_get(ctx, rctx, tail_idx, reg)
        }
    }

    /// Load/store base writeback: encoding 31 is SP, not XZR.
    pub(crate) fn emit_addr_reg_set<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg == 31 {
            self.emit_sp_set(ctx, rctx, tail_idx)
        } else {
            self.emit_gpr_set(ctx, rctx, tail_idx, reg)
        }
    }

    // ── FP register emit helpers ──────────────────────────────────────────────

    /// Push FP register `reg` (0–31) onto the WASM stack as f64.
    pub(crate) fn emit_fp_get<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg >= 32 {
            return rctx.feed(ctx, tail_idx, &Instruction::F64Const(Ieee64::from(0.0)));
        }
        for instr in rctx.layout().emit_get(self.fp_slot, reg) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// Pop the stack top (f64) into FP register `reg` (0–31).  reg >= 32 → `drop`.
    pub(crate) fn emit_fp_set<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        ctx: &mut Context,
        rctx: &RC,
        tail_idx: usize,
        reg: u32,
    ) -> Result<(), E> {
        if reg >= 32 {
            return rctx.feed(ctx, tail_idx, &Instruction::Drop);
        }
        for instr in rctx.layout().emit_set(self.fp_slot, reg) {
            rctx.feed(ctx, tail_idx, &instr)?;
        }
        Ok(())
    }

    /// WASM local index for NZCV flag `n` (0=N, 1=Z, 2=C, 3=V).
    pub(crate) fn nzcv_local<RC: ReactorContext<Context, E> + ?Sized>(
        &self,
        rctx: &RC,
        n: u32,
    ) -> u32 {
        rctx.layout().local(self.nzcv_slot, n)
    }

    // ── NZCV flag computation ─────────────────────────────────────────────────

    /// Set N/Z from result on stack (consumed); C=0, V=0.  Used after logical ops.
    pub(crate) fn set_nzcv_logical<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        dest: u32,
    ) -> Result<(), E> {
        let tmp0  = rctx.layout().local(self.tmp_slot, 0);
        let n_loc = rctx.layout().local(self.nzcv_slot, 0);
        let z_loc = rctx.layout().local(self.nzcv_slot, 1);
        let c_loc = rctx.layout().local(self.nzcv_slot, 2);
        let v_loc = rctx.layout().local(self.nzcv_slot, 3);

        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = 0, V = 0
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    /// Set N/Z/C/V after an ADD; result on stack.
    /// Caller must have stored operand `a` (first addend) into tmp1 before the add.
    /// `b_imm`: if Some, the immediate second addend (avoids needing tmp2 for V).
    pub(crate) fn set_nzcv_add<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        dest: u32,
        b_imm: Option<i64>,
    ) -> Result<(), E> {
        let tmp0  = rctx.layout().local(self.tmp_slot, 0);
        let tmp1  = rctx.layout().local(self.tmp_slot, 1);
        let tmp2  = rctx.layout().local(self.tmp_slot, 2);
        let n_loc = rctx.layout().local(self.nzcv_slot, 0);
        let z_loc = rctx.layout().local(self.nzcv_slot, 1);
        let c_loc = rctx.layout().local(self.nzcv_slot, 2);
        let v_loc = rctx.layout().local(self.nzcv_slot, 3);

        // result → tmp0 (keep on stack)
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = result_u64 < a_u64  (unsigned overflow / carry out)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtU)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;

        // V = signed overflow: same signs in, different sign out
        //   = ~((a ^ b) & SIGN) & ((result ^ a) & SIGN)  → i32 1/0
        // Get b: either constant or tmp2
        if let Some(b) = b_imm {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(b))?;
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
        }
        // (a ^ b) < 0  →  sign(a) != sign(b)  →  (a^b) has MSB set
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?; // i32: 1 if signs differ
        rctx.feed(ctx, tail_idx, &Instruction::I32Eqz)?; // i32: 1 if signs same
        // (result ^ a) < 0  →  sign(result) != sign(a)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    /// Set N/Z/C/V after a SUB (result = a - b); result on stack.
    /// Caller must store `a` in tmp1 and `b` in tmp2 before the sub
    /// (or pass `b_imm` to skip storing b in tmp2).
    pub(crate) fn set_nzcv_sub<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        dest: u32,
        b_imm: Option<i64>,
    ) -> Result<(), E> {
        let tmp0  = rctx.layout().local(self.tmp_slot, 0);
        let tmp1  = rctx.layout().local(self.tmp_slot, 1);
        let tmp2  = rctx.layout().local(self.tmp_slot, 2);
        let n_loc = rctx.layout().local(self.nzcv_slot, 0);
        let z_loc = rctx.layout().local(self.nzcv_slot, 1);
        let c_loc = rctx.layout().local(self.nzcv_slot, 2);
        let v_loc = rctx.layout().local(self.nzcv_slot, 3);

        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = a_u64 >= b_u64  (no borrow = carry out in ARM SUB convention)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
        if let Some(b) = b_imm {
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(b))?;
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::I64GeU)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;

        // V = signed overflow: different signs in, different sign out
        //   = ((a ^ b) & SIGN) & ((result ^ a) & SIGN)
        if let Some(b) = b_imm {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::I64Const(b))?;
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp2))?;
        }
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?; // 1 if signs differ
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp1))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }

    /// Legacy alias kept for callers that don't need proper C/V (conservative C=V=0).
    /// Prefer `set_nzcv_add` / `set_nzcv_sub` for flag-setting instructions.
    pub(crate) fn set_nzcv_and_store<F>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        tail_idx: usize,
        dest: u32,
    ) -> Result<(), E> {
        let tmp0   = rctx.layout().local(self.tmp_slot, 0);
        let n_loc  = rctx.layout().local(self.nzcv_slot, 0);
        let z_loc  = rctx.layout().local(self.nzcv_slot, 1);
        let c_loc  = rctx.layout().local(self.nzcv_slot, 2);
        let v_loc  = rctx.layout().local(self.nzcv_slot, 3);

        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(tmp0))?;

        // N = (result < 0)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(n_loc))?;

        // Z = (result == 0)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64Eqz)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(z_loc))?;

        // C = 0 (conservative)
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(c_loc))?;

        // V = 0 (conservative)
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0))?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(v_loc))?;

        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(tmp0))?;
        self.emit_gpr_set(ctx, rctx, tail_idx, dest)?;
        Ok(())
    }
}

/// Calling convention for a compile-time PLT redirect of `symbol` on AArch64.
pub fn plt_calling_convention(symbol: &str) -> speet_plugin_api::external_target::CallingConvention {
    use binary_io::BinArch;
    use speet_host_api::ImportManifest;
    speet_abi_stubs::plt_calling_convention(&ImportManifest::integrated_native(), BinArch::AArch64, symbol)
}
