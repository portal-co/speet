//! Security traps — control-flow integrity enforcement.
//!
//! Security traps validate control-flow transfers against a ground-truth
//! policy and redirect to a violation handler when the policy is violated.
//! Unlike [`tracing`](crate::tracing) traps, these traps may return
//! [`TrapAction::Skip`] to suppress the original transfer entirely and
//! replace it with a handler call.
//!
//! | Type | Trait | Policy |
//! |------|-------|--------|
//! | [`CfiReturnTrap`] | [`JumpTrap`] | Return-target bitmap derived from native CFI data |
//!
//! ## Relationship to hardening
//!
//! [`security`](self) traps enforce *static* policies compiled from the guest
//! binary's own CFI metadata (DWARF `.eh_frame`, `ENDBR` markers, …).
//! [`hardening`](crate::hardening) traps enforce *dynamic* invariants that
//! hold regardless of the binary's metadata.  Both can be stacked with
//! [`ChainedTrap`](crate::ChainedTrap).

use wasm_encoder::{Instruction, ValType};
use yecta::{FuncIdx, LocalDeclarator, LocalLayout, LocalSlot};

use crate::context::TrapContext;
use crate::insn::TrapAction;
use crate::jump::{JumpInfo, JumpKind, JumpTrap};

// ── CfiReturnTrap ─────────────────────────────────────────────────────────────

/// CFI return-target validation using a bitmap derived from native CFI data.
///
/// This trap fires on every `Return` (and optionally `IndirectJump`) and
/// checks the runtime return address against a **caller-supplied bitmap of
/// valid return-target addresses** built ahead of time from the guest binary's
/// own CFI annotations — DWARF `.eh_frame` / `.debug_frame` call-site
/// records, RISC-V compressed-call pairs, x86-64 `ENDBR` markers, or any
/// other architecture-specific CFI metadata.
///
/// ## Bitmap layout
///
/// The caller allocates a byte array in wasm linear memory and populates it
/// before recompilation begins.  Entry `n` covers guest addresses
/// `[bitmap_guest_base + n * granularity, bitmap_guest_base + (n+1) *
/// granularity)`.  The check emitted at each Return is:
///
/// ```text
/// ;; target_local holds the runtime return address (i32 or i64).
/// ;; Check: bitmap[(target - guest_base) / granularity] != 0
/// local.get  target_local
/// i32.const  guest_base_lo        ;; low 32 bits of bitmap_guest_base
/// i32.sub
/// i32.const  granularity_shift    ;; log2(granularity)
/// i32.shr_u                       ;; byte index into bitmap
/// ;; range-guard: if index >= bitmap_byte_len → violation
/// local.tee  <scratch>
/// i32.const  bitmap_byte_len
/// i32.lt_u
/// i32.eqz                         ;; 1 if out of range
/// if
///   <violation_handler jump>
/// end
/// local.get  <scratch>
/// i32.const  bitmap_wasm_addr
/// i32.add
/// i32.load8_u (offset=0)
/// i32.eqz                         ;; 1 if byte is zero (not a valid call site)
/// if
///   <violation_handler jump>
/// end
/// ```
///
/// A byte value of `0` means "not a valid return target"; any non-zero value
/// means "valid".  A full byte per entry (rather than a single bit) avoids
/// shift-and-mask arithmetic in the emitted wasm and is cache-friendly for
/// typical call-site densities.
///
/// ## Populating the bitmap
///
/// Build the bitmap from the guest binary's CFI tables before handing it to
/// the recompiler.  For each call instruction at guest address `call_pc` with
/// instruction length `call_len`:
///
/// ```text
/// valid_return_target = call_pc + call_len
/// bitmap[(valid_return_target - guest_base) / granularity] = 1
/// ```
///
/// For RISC-V with 4-byte calls: `granularity_shift = 2`.  Use
/// `granularity_shift = 1` when compressed/16-bit calls are present.
/// The bitmap slice must be written into wasm linear memory at
/// `bitmap_wasm_addr` before any translated functions execute.
///
/// ## Integration note
///
/// This trap fires on `Return` only when [`JumpInfo::target_local`] is
/// `Some`.  The recompiler must expose the runtime return address as a wasm
/// local (via `local.tee`) and record its index in `JumpInfo::target_local`
/// when constructing the `JumpInfo` for a `Return` transfer.
pub struct CfiReturnTrap {
    /// Guest address corresponding to byte 0 of the bitmap.
    pub bitmap_guest_base: u32,
    /// Length of the bitmap in bytes (= address range / granularity).
    pub bitmap_byte_len: u32,
    /// Address of the bitmap byte array in wasm linear memory.
    pub bitmap_wasm_addr: u32,
    /// `log2(granularity)`: bytes per bitmap entry as a shift.
    /// `0` = byte-granular, `1` = 16-bit aligned, `2` = 32-bit aligned.
    pub granularity_shift: u32,
    /// Wasm function to jump to on a CFI violation.
    pub violation_handler: FuncIdx,
    /// Number of wasm function parameters to pass to the violation handler.
    pub handler_params: u32,
    /// Scratch `i32` local for stashing the computed bitmap index.
    /// Set during [`declare_locals`](JumpTrap::declare_locals).
    index_local_slot: LocalSlot,
}

impl CfiReturnTrap {
    /// Construct a new `CfiReturnTrap`.
    ///
    /// `granularity_shift = 2` is appropriate for ISAs where all valid call
    /// sites are 4-byte-aligned (pure 32-bit RISC-V, MIPS, most A64).
    /// Use `granularity_shift = 1` when compressed/16-bit calls are present.
    pub fn new(
        bitmap_guest_base: u32,
        bitmap_byte_len: u32,
        bitmap_wasm_addr: u32,
        granularity_shift: u32,
        violation_handler: FuncIdx,
        handler_params: u32,
    ) -> Self {
        Self {
            bitmap_guest_base,
            bitmap_byte_len,
            bitmap_wasm_addr,
            granularity_shift,
            violation_handler,
            handler_params,
            index_local_slot: LocalSlot::default(),
        }
    }
}

impl LocalDeclarator for CfiReturnTrap {
    fn declare_locals(&mut self, locals: &mut LocalLayout) {
        self.index_local_slot = locals.append(1, ValType::I32);
    }
}

impl<Context, E> JumpTrap<Context, E> for CfiReturnTrap {
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E>,
    ) -> Result<TrapAction, E> {
        if !matches!(info.kind, JumpKind::Return | JumpKind::IndirectJump) {
            return Ok(TrapAction::Continue);
        }
        let target_local = match info.target_local {
            Some(l) => l,
            None => return Ok(TrapAction::Continue),
        };
        let index_local = trap_ctx.layout().local(self.index_local_slot, 0);
        let handler = self.violation_handler;
        let params = self.handler_params;

        // Compute bitmap index: (target - guest_base) >> granularity_shift
        trap_ctx.emit(ctx, &Instruction::LocalGet(target_local))?;
        trap_ctx.emit(ctx, &Instruction::I32Const(self.bitmap_guest_base as i32))?;
        trap_ctx.emit(ctx, &Instruction::I32Sub)?;
        if self.granularity_shift > 0 {
            trap_ctx.emit(ctx, &Instruction::I32Const(self.granularity_shift as i32))?;
            trap_ctx.emit(ctx, &Instruction::I32ShrU)?;
        }
        trap_ctx.emit(ctx, &Instruction::LocalTee(index_local))?;

        // Range guard: if index >= bitmap_byte_len → violation
        trap_ctx.emit(ctx, &Instruction::I32Const(self.bitmap_byte_len as i32))?;
        trap_ctx.emit(ctx, &Instruction::I32LtU)?;
        trap_ctx.emit(ctx, &Instruction::I32Eqz)?;
        trap_ctx.jump_if(ctx, handler, params)?;

        // Bitmap byte load: bitmap_wasm_addr + index
        trap_ctx.emit(ctx, &Instruction::LocalGet(index_local))?;
        trap_ctx.emit(ctx, &Instruction::I32Const(self.bitmap_wasm_addr as i32))?;
        trap_ctx.emit(ctx, &Instruction::I32Add)?;
        trap_ctx.emit(
            ctx,
            &Instruction::I32Load8U(wasm_encoder::MemArg {
                align: 0,
                offset: 0,
                memory_index: 0,
            }),
        )?;
        trap_ctx.emit(ctx, &Instruction::I32Eqz)?;
        trap_ctx.jump_if(ctx, handler, params)?;

        Ok(TrapAction::Continue)
    }
}
