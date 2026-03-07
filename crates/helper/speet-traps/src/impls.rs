//! Standard (batteries-included) trap implementations.
//!
//! These are ready-to-use implementations of [`InstructionTrap`] and
//! [`JumpTrap`] covering the most common use cases.  They can be composed
//! using `Vec<Box<dyn …>>` for combining multiple behaviours.
//!
//! | Type | Trait | Behaviour |
//! |------|-------|-----------|
//! | [`NullTrap`] | both | No-op; zero-cost when monomorphised |
//! | [`ChainedTrap`] | both | Run A then B; `Skip` short-circuits |
//! | [`CounterTrap`] | `InstructionTrap` | Increment a wasm global per `InsnClass` flag |
//! | [`CfiReturnTrap`] | `JumpTrap` | Validate return targets against a bitmap built from native CFI data |
//! | [`RopDetectTrap`] | `JumpTrap` | Track Call/Return depth; `depth < 0` → more returns than calls → violation |
//! | [`TraceLogTrap`] | `JumpTrap` | Before each jump emit a call to a logging wasm import |

use wasm_encoder::{Instruction, ValType};
use wax_core::build::InstructionSink;
use yecta::{FuncIdx, LocalLayout, LocalSlot};

use crate::context::TrapContext;
use crate::insn::{InsnClass, InstructionInfo, InstructionTrap, TrapAction};
use crate::jump::{JumpInfo, JumpKind, JumpTrap};

// ── NullTrap ──────────────────────────────────────────────────────────────────

/// A no-op trap that always returns [`TrapAction::Continue`].
///
/// Use this as a placeholder when the trait bound requires a concrete type but
/// no actual trap behaviour is needed.  When the generic parameter is
/// monomorphised to `NullTrap` the compiler will eliminate all trap overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullTrap;

impl<Context, E, F: InstructionSink<Context, E>> InstructionTrap<Context, E, F> for NullTrap {
    fn on_instruction(
        &mut self,
        _info: &InstructionInfo,
        _ctx: &mut Context,
        _trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        Ok(TrapAction::Continue)
    }
}

impl<Context, E, F: InstructionSink<Context, E>> JumpTrap<Context, E, F> for NullTrap {
    fn on_jump(
        &mut self,
        _info: &JumpInfo,
        _ctx: &mut Context,
        _trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        Ok(TrapAction::Continue)
    }
}

// ── ChainedTrap ───────────────────────────────────────────────────────────────

/// Compose two traps of the same kind: run `A` first, then `B`.
///
/// If `A` returns [`TrapAction::Skip`], `B`'s `on_instruction` / `on_jump` is
/// **not** called, and `A`'s `skip_snippet` is used.
///
/// Both traps append their parameter and local slots to the same shared
/// [`LocalLayout`] during [`declare_params`] / [`declare_locals`].  Because
/// they each receive a different [`LocalSlot`] handle, their indices will
/// never conflict regardless of insertion order.
///
/// [`declare_params`]: ChainedTrap::declare_params
/// [`declare_locals`]: ChainedTrap::declare_locals
pub struct ChainedTrap<A, B> {
    /// The first trap to run.
    pub a: A,
    /// The second trap to run (skipped if `a` returns [`TrapAction::Skip`]).
    pub b: B,
}

impl<A, B> ChainedTrap<A, B> {
    /// Construct a `ChainedTrap` that runs `a` first, then `b`.
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<Context, E, F, A, B> InstructionTrap<Context, E, F> for ChainedTrap<A, B>
where
    F: InstructionSink<Context, E>,
    A: InstructionTrap<Context, E, F>,
    B: InstructionTrap<Context, E, F>,
{
    fn declare_params(&mut self, params: &mut LocalLayout) {
        self.a.declare_params(params);
        self.b.declare_params(params);
    }

    fn declare_locals(&mut self, locals: &mut LocalLayout) {
        self.a.declare_locals(locals);
        self.b.declare_locals(locals);
    }

    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        if self.a.on_instruction(info, ctx, trap_ctx)? == TrapAction::Skip {
            return Ok(TrapAction::Skip);
        }
        self.b.on_instruction(info, ctx, trap_ctx)
    }

    fn skip_snippet(
        &self,
        info: &InstructionInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<(), E> {
        // Only A's snippet is used since B never ran (see on_instruction).
        self.a.skip_snippet(info, ctx, skip_ctx)
    }
}

impl<Context, E, F, A, B> JumpTrap<Context, E, F> for ChainedTrap<A, B>
where
    F: InstructionSink<Context, E>,
    A: JumpTrap<Context, E, F>,
    B: JumpTrap<Context, E, F>,
{
    fn declare_params(&mut self, params: &mut LocalLayout) {
        self.a.declare_params(params);
        self.b.declare_params(params);
    }

    fn declare_locals(&mut self, locals: &mut LocalLayout) {
        self.a.declare_locals(locals);
        self.b.declare_locals(locals);
    }

    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        if self.a.on_jump(info, ctx, trap_ctx)? == TrapAction::Skip {
            return Ok(TrapAction::Skip);
        }
        self.b.on_jump(info, ctx, trap_ctx)
    }

    fn skip_snippet(
        &self,
        info: &JumpInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<(), E> {
        self.a.skip_snippet(info, ctx, skip_ctx)
    }
}

// ── CounterTrap ───────────────────────────────────────────────────────────────

/// Increment a wasm global counter for each instruction matching a class mask.
///
/// Useful for profiling (count memory accesses, count branches, etc.) without
/// any per-function locals.  The counter is a wasm global of type `i32` at
/// index `global_idx`.
///
/// The trap emits:
/// ```text
/// global.get global_idx
/// i32.const 1
/// i32.add
/// global.set global_idx
/// ```
/// for every instruction whose `class` field has any bit in common with
/// `mask`.
pub struct CounterTrap {
    /// Wasm global index to increment.
    pub global_idx: u32,
    /// Instruction class mask — increment if `info.class.0 & mask.0 != 0`.
    /// Use `InsnClass(u32::MAX)` to count every instruction.
    pub mask: InsnClass,
}

impl<Context, E, F: InstructionSink<Context, E>> InstructionTrap<Context, E, F> for CounterTrap {
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        if self.mask == InsnClass::OTHER || info.class.contains(self.mask) {
            trap_ctx.emit(ctx, &Instruction::GlobalGet(self.global_idx))?;
            trap_ctx.emit(ctx, &Instruction::I32Const(1))?;
            trap_ctx.emit(ctx, &Instruction::I32Add)?;
            trap_ctx.emit(ctx, &Instruction::GlobalSet(self.global_idx))?;
        }
        Ok(TrapAction::Continue)
    }
}

// ── CfiReturnTrap ─────────────────────────────────────────────────────────────

/// CFI return-target validation using a bitmap derived from native CFI data.
///
/// This trap checks every `Return` (and optionally `IndirectJump`) against a
/// **caller-supplied bitmap of valid return-target addresses** built ahead of
/// time from the guest binary's own CFI annotations — DWARF `.eh_frame` /
/// `.debug_frame` call-site records, RISC-V compressed-call pairs, x86-64
/// `ENDBR` markers, or any other architecture-specific CFI metadata.
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
/// local.tee  <scratch>            ;; stash index
/// i32.const  bitmap_byte_len
/// i32.lt_u                        ;; 1 if in range
/// i32.eqz                         ;; 1 if out of range
/// if
///   ;; out of range → definitely not a valid call site → violation
///   <violation_handler jump>
/// end
/// local.get  <scratch>
/// i32.const  bitmap_wasm_addr     ;; base address of bitmap in wasm memory
/// i32.add                         ;; effective byte address
/// i32.load8_u (offset=0)          ;; load the byte
/// i32.eqz                         ;; 1 if the byte is zero (not a call site)
/// if
///   <violation_handler jump>
/// end
/// ```
///
/// A byte value of `0` means "not a valid return target"; any non-zero value
/// means "valid".  Using a full byte per entry (rather than a single bit)
/// avoids shift-and-mask arithmetic in the emitted wasm and is cache-friendly
/// for typical call-site densities.
///
/// ## Populating the bitmap
///
/// The caller builds the bitmap from the guest binary's CFI tables before
/// handing it to the recompiler.  For each call instruction at guest address
/// `call_pc` with instruction length `call_len`:
///
/// ```text
/// valid_return_target = call_pc + call_len
/// bitmap[(valid_return_target - guest_base) / granularity] = 1
/// ```
///
/// For RISC-V with 4-byte calls: `granularity = 4`; for compressed calls:
/// `granularity = 2`.  Using `granularity = 2` covers both.
///
/// The resulting bitmap slice must be written into wasm linear memory at
/// `bitmap_wasm_addr` before any translated functions execute.
///
/// ## Integration note
///
/// This trap fires on `Return` only when `JumpInfo::target_local` is `Some`.
/// The recompiler must ensure it exposes the runtime return address as a wasm
/// local (via `local.tee`) and records its index in `JumpInfo::target_local`
/// when constructing the `JumpInfo` for a `Return` transfer.  See
/// `docs/trap-hooks.md §7.5` for the per-architecture integration details.
pub struct CfiReturnTrap {
    /// Guest address corresponding to byte 0 of the bitmap.
    pub bitmap_guest_base: u32,
    /// Length of the bitmap in bytes (= address range / granularity).
    pub bitmap_byte_len: u32,
    /// Address of the bitmap byte array in wasm linear memory.
    pub bitmap_wasm_addr: u32,
    /// Address granularity: `1 << granularity_shift` bytes per bitmap entry.
    /// Typical values: 1 (byte-granular), 1 (→ shift=0 for byte), 2 (16-bit
    /// aligned, shift=1), 4 (32-bit aligned, shift=2).
    pub granularity_shift: u32,
    /// Wasm function to jump to on a CFI violation.
    pub violation_handler: FuncIdx,
    /// Number of wasm function parameters to pass to the violation handler.
    pub handler_params: u32,
    /// Slot for the scratch `i32` local used to stash the computed bitmap index.
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
            index_local_slot: LocalSlot(0), // overwritten in declare_locals
        }
    }
}

impl<Context, E, F: InstructionSink<Context, E>> JumpTrap<Context, E, F> for CfiReturnTrap {
    fn declare_locals(&mut self, locals: &mut LocalLayout) {
        self.index_local_slot = locals.append(1, ValType::I32);
    }

    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        if !matches!(info.kind, JumpKind::Return | JumpKind::IndirectJump) {
            return Ok(TrapAction::Continue);
        }
        let target_local = match info.target_local {
            Some(l) => l,
            None    => return Ok(TrapAction::Continue),
        };
        let index_local = trap_ctx.locals().local(self.index_local_slot, 0);
        let handler     = self.violation_handler;
        let params      = self.handler_params;

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
        trap_ctx.emit(ctx, &Instruction::I32Load8U(wasm_encoder::MemArg {
            align:        0,
            offset:       0,
            memory_index: 0,
        }))?;
        trap_ctx.emit(ctx, &Instruction::I32Eqz)?;
        trap_ctx.jump_if(ctx, handler, params)?;

        Ok(TrapAction::Continue)
    }
}

// ── RopDetectTrap ─────────────────────────────────────────────────────────────

/// ROP detection via per-function call/return depth tracking.
///
/// Within any translated function group, a legitimate execution either:
/// - makes more calls than returns (normal function body calling into
///   subroutines), or
/// - makes exactly as many returns as calls (balanced recursion / tail
///   return from a called block), or
/// - exits via a direct jump (tail-call, unreachable, syscall).
///
/// A Return that exceeds the number of Calls seen *within this group* means
/// the guest is consuming a return address that was placed on the stack by a
/// *different* code path — the signature of a ROP gadget.  This trap detects
/// that imbalance.
///
/// ## Cross-function state via parameters
///
/// The depth counter is declared as a wasm **parameter** (not a local) via
/// [`declare_params`](JumpTrap::declare_params).  Parameters survive
/// `return_call` chains, so the depth correctly accumulates across the
/// sequence of yecta wasm functions that together translate a single guest
/// basic block.
///
/// The recompiler must call [`TrapConfig::setup`] before translation and
/// store the returned `total_params`, then use it as the `params` argument
/// to every `jmp` / `ji` call so that the depth counter is forwarded along
/// each control-flow edge.
///
/// ## Mechanism
///
/// - **On `Call` or `IndirectCall`**: `depth++`
/// - **On `Return`**: `depth--`; if `depth < 0` → fire violation handler
///
/// Note that `depth` only tracks balance within the *current reachable yecta
/// function group*.  A gadget that begins execution with a balanced stack
/// will not be caught here.  For stronger guarantees, combine with
/// [`CfiReturnTrap`].
pub struct RopDetectTrap {
    /// Wasm function to jump to when the return depth goes negative.
    pub violation_handler: FuncIdx,
    /// Number of wasm function parameters to pass to the violation handler.
    pub handler_params: u32,
    /// Slot for the `i32` depth-counter parameter.
    /// Set during [`declare_params`](JumpTrap::declare_params).
    depth_param_slot: LocalSlot,
}

impl RopDetectTrap {
    /// Construct a `RopDetectTrap`.
    ///
    /// `violation_handler` is called (via `return_call`) when the return depth
    /// goes negative.  `handler_params` is the number of wasm parameters
    /// forwarded to the handler (typically the same `total_params` used for
    /// all `jmp` calls in this function group).
    pub fn new(violation_handler: FuncIdx, handler_params: u32) -> Self {
        Self {
            violation_handler,
            handler_params,
            depth_param_slot: LocalSlot(0), // overwritten in declare_params
        }
    }
}

impl<Context, E, F: InstructionSink<Context, E>> JumpTrap<Context, E, F> for RopDetectTrap {
    fn declare_params(&mut self, params: &mut LocalLayout) {
        self.depth_param_slot = params.append(1, ValType::I32);
    }

    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        let depth_local = trap_ctx.params().local(self.depth_param_slot, 0);
        match info.kind {
            JumpKind::Call | JumpKind::IndirectCall => {
                trap_ctx.emit(ctx, &Instruction::LocalGet(depth_local))?;
                trap_ctx.emit(ctx, &Instruction::I32Const(1))?;
                trap_ctx.emit(ctx, &Instruction::I32Add)?;
                trap_ctx.emit(ctx, &Instruction::LocalSet(depth_local))?;
            }
            JumpKind::Return => {
                trap_ctx.emit(ctx, &Instruction::LocalGet(depth_local))?;
                trap_ctx.emit(ctx, &Instruction::I32Const(1))?;
                trap_ctx.emit(ctx, &Instruction::I32Sub)?;
                trap_ctx.emit(ctx, &Instruction::LocalTee(depth_local))?;
                trap_ctx.emit(ctx, &Instruction::I32Const(0))?;
                trap_ctx.emit(ctx, &Instruction::I32LtS)?;
                let handler = self.violation_handler;
                let params  = self.handler_params;
                trap_ctx.jump_if(ctx, handler, params)?;
            }
            _ => {}
        }
        Ok(TrapAction::Continue)
    }
}

// ── TraceLogTrap ──────────────────────────────────────────────────────────────

/// Emit a call to a wasm import before each control-flow transfer.
///
/// The import is called with the following arguments (in wasm stack order):
///
/// ```text
/// i32: source_pc (truncated to 32 bits)
/// i32: target_pc (truncated, or 0 for indirect)
/// i32: JumpKind discriminant (see TraceLogTrap::kind_to_i32)
/// ```
///
/// The import must have wasm type `(i32, i32, i32) -> ()`.
///
/// This is intended for debugging and tracing; it adds a call per jump site.
pub struct TraceLogTrap {
    /// Index of the wasm function import to call.
    pub log_func_idx: u32,
}

impl TraceLogTrap {
    fn kind_to_i32(kind: JumpKind) -> i32 {
        match kind {
            JumpKind::DirectJump        => 0,
            JumpKind::ConditionalBranch => 1,
            JumpKind::Call              => 2,
            JumpKind::Return            => 3,
            JumpKind::IndirectJump      => 4,
            JumpKind::IndirectCall      => 5,
            JumpKind::Syscall           => 6,
        }
    }
}

impl<Context, E, F: InstructionSink<Context, E>> JumpTrap<Context, E, F> for TraceLogTrap {
    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        let target_i32 = info.target_pc.unwrap_or(0) as i32;
        trap_ctx.emit(ctx, &Instruction::I32Const(info.source_pc as i32))?;
        trap_ctx.emit(ctx, &Instruction::I32Const(target_i32))?;
        trap_ctx.emit(ctx, &Instruction::I32Const(Self::kind_to_i32(info.kind)))?;
        trap_ctx.emit(ctx, &Instruction::Call(self.log_func_idx))?;
        Ok(TrapAction::Continue)
    }
}
