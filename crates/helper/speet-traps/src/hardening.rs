//! Virtual machine and weird machine hardening — anti-\*OP defences.
//!
//! These traps defend against *-oriented programming attacks: sequences of
//! small code gadgets (returns, jumps, calls) chained together by an attacker
//! to perform arbitrary computation inside the translated guest.  Unlike
//! [`security`](crate::security) traps, which enforce a static policy
//! compiled from the binary's own metadata, hardening traps enforce
//! *dynamic invariants* that hold for any legitimate execution regardless of
//! what the binary's CFI tables say.
//!
//! | Type | Trait | Invariant enforced |
//! |------|-------|-------------------|
//! | [`RopDetectTrap`] | [`JumpTrap`] | Return depth never goes negative within a translated function group |
//!
//! ## Background: weird machines
//!
//! A *weird machine* is any computational substrate that an attacker can
//! drive through carefully crafted inputs — the translator itself, its data
//! structures, or the control-flow edges it exposes.  *-Oriented programming
//! attacks (ROP, JOP, COP, …) exploit unintended gadgets produced by the
//! translator to build a weird machine capable of executing attacker-chosen
//! payloads.
//!
//! Traps in this module harden the translated execution environment against
//! such attacks by detecting invariant violations at runtime inside the
//! generated wasm, before they can be exploited.
//!
//! ## Combining with CFI
//!
//! Hardening traps and CFI traps are complementary:
//!
//! - [`CfiReturnTrap`](crate::CfiReturnTrap) (static): validates that every
//!   return address was explicitly placed there by a matching `Call` site in
//!   the guest binary.
//! - [`RopDetectTrap`] (dynamic): validates that the *count* of Returns has
//!   never exceeded the count of Calls within the current function group,
//!   regardless of what the call sites say.
//!
//! Stack them with [`ChainedTrap`](crate::ChainedTrap) for defence in depth:
//!
//! ```ignore
//! let mut trap = ChainedTrap::new(
//!     CfiReturnTrap::new(base, len, wasm_addr, shift, handler, params),
//!     RopDetectTrap::new(handler, params),
//! );
//! linker.traps.set_jump_trap(&mut trap);
//! ```

// See docs/trap-hooks.md §9 and AGENTS.md §2 for the rationale behind using a
// wasm *parameter* (not a local) for RopDetectTrap's depth counter.

use wasm_encoder::{Instruction, ValType};
use wax_core::build::InstructionSink;
use yecta::{FuncIdx, LocalLayout, LocalSlot};

use crate::context::TrapContext;
use crate::insn::TrapAction;
use crate::jump::{JumpInfo, JumpKind, JumpTrap};

// ── RopDetectTrap ─────────────────────────────────────────────────────────────

/// ROP detection via per-function-group call/return depth tracking.
///
/// Within any translated yecta function group, a legitimate execution either:
/// - makes more calls than returns (normal function body calling subroutines),
/// - balances calls and returns exactly (tail return from a called block), or
/// - exits via a direct jump, `unreachable`, or syscall.
///
/// A `Return` that pushes the return count above the call count *within this
/// group* means the guest is consuming a return address that was not placed
/// there by any call visible in this group — the signature of a ROP gadget.
/// This trap detects that imbalance.
///
/// ## Cross-function state via parameters
///
/// The depth counter is declared as a wasm **parameter** (not a local) via
/// [`declare_params`](JumpTrap::declare_params).  Parameters survive
/// `return_call` chains, so the depth correctly accumulates across the
/// sequence of yecta wasm functions that together translate a single guest
/// basic block.
///
/// The recompiler must store the total parameter count returned by
/// [`TrapConfig::declare_params`](crate::TrapConfig::declare_params) +
/// [`LocalLayout::mark`](yecta::LocalLayout::mark) and use it as the
/// `params` argument to every `jmp` / `ji` call, so the depth counter is
/// forwarded on every control-flow edge.
///
/// ## Mechanism
///
/// | Event | Action |
/// |-------|--------|
/// | `Call` or `IndirectCall` | `depth += 1` |
/// | `Return` | `depth -= 1`; if `depth < 0` → jump to violation handler |
/// | All other jumps | No-op |
///
/// ## Limitations
///
/// `depth` only tracks balance *within the current reachable yecta function
/// group*.  A gadget that begins execution with a balanced stack will not be
/// caught here.  For stronger guarantees, combine with
/// [`CfiReturnTrap`](crate::CfiReturnTrap), which validates the actual target
/// address against the guest binary's call-site set.
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
    /// forwarded to the handler — typically the same `total_params` used for
    /// all `jmp` calls in this function group so the handler has access to the
    /// full guest register file.
    pub fn new(violation_handler: FuncIdx, handler_params: u32) -> Self {
        Self {
            violation_handler,
            handler_params,
            depth_param_slot: LocalSlot::default(),
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
        let depth_local = trap_ctx.layout().local(self.depth_param_slot, 0);
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
                let params = self.handler_params;
                trap_ctx.jump_if(ctx, handler, params)?;
            }
            _ => {}
        }
        Ok(TrapAction::Continue)
    }
}
