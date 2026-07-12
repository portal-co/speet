//! `speet-syscall` — generic syscall-to-WASM dispatch for static recompilers.
//!
//! Provides the [`SyscallTable`] / [`SyscallEntry`] data model and
//! [`WasmSyscallDispatcher`], an [`EcallCallback`](speet_riscv::EcallCallback)-
//! compatible type that emits an inline `br_table` dispatch over the runtime
//! syscall number at each guest `ecall` / `syscall` site.
//!
//! # Design overview
//!
//! At translation time the recompiler calls
//! `WasmSyscallDispatcher::call(ecall_info, ctx, callback_ctx)` for every
//! `ecall` instruction.  The dispatcher emits:
//!
//! ```text
//! ;; 1. Load syscall number register → i32 for br_table
//! local.get $syscall_num_local
//! i32.wrap_i64                    ;; (only when num local is i64)
//!
//! ;; 2. Dispatch block tree
//! block $exit
//!   block $unknown
//!     block $entry_N … block $entry_0
//!       br_table $entry_0 … $entry_N $unknown
//!     end ;; $entry_0
//!     … handler arm for syscall 0 …
//!     return_call $next_pc_func   ;; non-terminating syscalls only
//!     …
//!     end ;; $entry_N
//!     … handler arm for syscall N …
//!     return_call $next_pc_func   ;; or unreachable if terminating
//!   end ;; $unknown
//!   unreachable   ;; (or fallback call if configured)
//! end ;; $exit
//!
//! Non-terminating handler arms tail-call `$next_pc_func` directly. The
//! continuation is **not** emitted after the closing `end` of `$exit`: wasm-blitz
//! places post-block code at function entry (before the block body), which would
//! clobber register locals before the `br_table` runs.
//! ```
//!
//! Each handler arm emits optional saves, marshals WASI parameters from guest
//! register locals, calls the WASI import via a non-tail `call`, optionally
//! stores the result into a guest register local, and optionally negates a
//! non-zero WASI errno to the Linux convention.
//!
//! # Ambient-state forwarding
//!
//! Each non-terminating arm's `return_call` must forward all params of the
//! function, including injected trap params (e.g. `RopDetectTrap` depth
//! counter).  The caller provides `num_params` (the total param count for the
//! current chain function) so the dispatcher can emit `local.get 0 …
//! local.get num_params-1` before the `return_call`.  For typed forwarding
//! of injected params across cells, use
//! [`LocalDeclarator::translate_slot`](yecta::LocalDeclarator::translate_slot).
//!
//! # Relation to `ShimSpec`
//!
//! `ShimSpec` generates a *separate* WASM shim function for ABI bridging.
//! `WasmSyscallDispatcher` emits code *inline* inside the current chain
//! function.  The two mechanisms are complementary: use `ShimSpec` for
//! cross-cell tail-call bridges, and `WasmSyscallDispatcher` for the inline
//! ecall dispatch.

#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use wasm_encoder::Instruction;

// ── Param source ──────────────────────────────────────────────────────────────

/// Describes how to produce one parameter for a WASM handler function.
///
/// Each [`SyscallEntry`] carries a `Vec<ParamSource>` whose length matches the
/// handler's WASM parameter count.  The dispatcher emits code for each source
/// in order to push the handler's arguments onto the WASM stack before the
/// `call`.
#[derive(Debug, Clone)]
pub enum ParamSource {
    /// Read an `i64` local and wrap it to `i32` for the handler.
    ///
    /// Use for guest integer registers (which are `i64` in the RV64 layout)
    /// when the WASI function expects `i32` parameters.
    LocalI64AsI32(u32),

    /// Read an `i32` local directly (no truncation needed).
    LocalI32(u32),

    /// Push a constant `i32` value (e.g. a fixed fd or flags value).
    ConstI32(i32),

    /// Push a constant `i64` value.
    ConstI64(i64),
}

// ── Save pair ─────────────────────────────────────────────────────────────────

/// A single save operation: spill a local to a WASM global before the call.
///
/// Currently only global saves are supported.  Extend with `Place::Deref` for
/// memory saves when needed.
#[derive(Debug, Clone)]
pub struct SavePair {
    /// Source local index (read with `local.get`).
    pub local_idx: u32,
    /// Destination global index (written with `global.set`).
    pub global_idx: u32,
}

// ── Memory store ──────────────────────────────────────────────────────────────

/// Writes a local value to a specific linear-memory address before the call.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    /// Memory address to write to.
    pub addr: u32,
    /// Source local index (read with `local.get`).
    pub value_local: u32,
    /// If `true`, the local has type `i64` and must be wrapped to `i32` before storing.
    pub value_is_i64: bool,
}

// ── SyscallEntry ──────────────────────────────────────────────────────────────

/// One handler for a specific Linux syscall number.
///
/// The dispatcher emits one `block`/arm per entry.
#[derive(Debug, Clone)]
pub struct SyscallEntry {
    /// Absolute WASM function index of the handler (e.g. a WASI import).
    pub func_idx: u32,

    /// How to produce each parameter the handler expects.
    ///
    /// Length must equal the handler's WASM param count.
    pub param_map: Vec<ParamSource>,

    /// Globals to spill before marshalling handler params.
    ///
    /// Applied in order before `param_map` is pushed.
    pub saves: Vec<SavePair>,

    /// If `Some(local)`: store the handler's `i32` return value into this
    /// guest local as `i64` (sign-extended).  Use for syscalls that return a
    /// value in e.g. `a0` (local 10 on RV64 default layout).
    ///
    /// `None` means the return value is dropped (e.g. `proc_exit` or
    /// `fd_close` where the caller ignores the result).
    pub result_local: Option<u32>,

    /// If `true`, negate a non-zero WASI return value before storing it into
    /// `result_local`.  Converts `wasi_errno → -wasi_errno` for the Linux
    /// convention of returning `-errno` on error.
    pub negate_nonzero_result: bool,

    /// If `true`, the handler returns a value that needs to be stashed or dropped.
    /// Set `false` for functions that return no value (e.g. `proc_exit` which has type `(i32) -> ()`).
    pub has_return: bool,

    /// If `true`, this syscall never returns (e.g. `proc_exit`).
    ///
    /// The dispatcher emits `unreachable` after the call instead of branching to
    /// `$exit` and forwarding params via `return_call`.  This prevents infinite
    /// loops when the host mock returns normally.
    pub terminates: bool,

    /// Memory writes to perform before marshalling handler params.
    pub memory_stores: Vec<MemoryStore>,

    /// If `Some(offset)`: load an `i32` from this linear-memory offset on success
    /// (i.e. when the handler returns 0), extend it to `i64`, and store it in
    /// `result_local`.  Use for WASI calls that write their actual result to memory.
    pub load_mem_on_success: Option<u32>,
}

// ── SyscallTable ──────────────────────────────────────────────────────────────

/// Complete dispatch table: sorted list of `(syscall_number, SyscallEntry)`.
///
/// Must be sorted by syscall number in ascending order for correct `br_table`
/// index computation.  Build via [`SyscallTable::new`] which sorts on
/// construction.
#[derive(Debug, Clone)]
pub struct SyscallTable {
    entries: Vec<(u64, SyscallEntry)>,
}

impl SyscallTable {
    /// Construct from an unsorted slice of `(syscall_number, entry)` pairs.
    ///
    /// Sorts by syscall number in ascending order.  Duplicate numbers are
    /// allowed (only the first match is used by the dispatcher).
    pub fn new(mut entries: Vec<(u64, SyscallEntry)>) -> Self {
        entries.sort_unstable_by_key(|(n, _)| *n);
        Self { entries }
    }

    /// Return the sorted entries slice.
    pub fn entries(&self) -> &[(u64, SyscallEntry)] {
        &self.entries
    }
}

// ── WasmSyscallDispatcher ─────────────────────────────────────────────────────

/// An [`EcallCallback`](speet_riscv::EcallCallback)-compatible type that
/// emits inline `br_table` syscall dispatch at guest `ecall` sites.
///
/// See the [module documentation](self) for the emitted instruction sequence.
///
/// ## Construction
///
/// ```ignore
/// use speet_syscall::{WasmSyscallDispatcher, SyscallTable};
///
/// let dispatcher = WasmSyscallDispatcher {
///     table: &table,
///     syscall_num_local: 17,   // a7 = x17, local 17 in default RV64 layout
///     syscall_num_is_i64: true, // RV64 regs are i64 — wrap to i32 for br_table
///     num_params: 35,           // total chain-function param count
///     next_pc_func: 42,         // absolute WASM index of the PC+4 function
/// };
/// recompiler.set_ecall_callback(&mut dispatcher);
/// ```
pub struct WasmSyscallDispatcher<'t> {
    /// The dispatch table.  Entries must be sorted by syscall number.
    pub table: &'t SyscallTable,

    /// WASM local index of the register holding the syscall number.
    ///
    /// For RISC-V: `a7 = x17`, which occupies local 17 in the default layout
    /// where locals 0–31 map to `x0`–`x31`.
    pub syscall_num_local: u32,

    /// If `true`, the syscall-number local has type `i64` and must be wrapped
    /// to `i32` before the `br_table` index computation.
    ///
    /// Set `true` for RV64 (all integer registers are `i64`).
    /// Set `false` for RV32 or any ISA where the register is already `i32`.
    pub syscall_num_is_i64: bool,

    /// Total number of WASM parameters in the current chain function.
    ///
    /// Used to emit `local.get 0 … local.get num_params-1` before the
    /// trailing `return_call` to the next-PC function.
    pub num_params: u32,

    /// Absolute WASM function index of the function representing the
    /// instruction immediately following this `ecall` (i.e. PC + inst_len).
    ///
    /// The dispatcher ends with `return_call $next_pc_func`, forwarding all
    /// `num_params` parameters unchanged.
    pub next_pc_func: u32,
}

impl<'t, Context, E> speet_riscv::EcallCallback<Context, E> for WasmSyscallDispatcher<'t> {
    fn call(
        &mut self,
        _ecall: &speet_riscv::EcallInfo,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) {
        // Ignore any error: EcallCallback has no Result return; a failure here
        // means the module will be malformed, which the validator will catch.
        let _ = self.emit(ctx, cb);
    }
}

impl<'t> WasmSyscallDispatcher<'t> {
    /// Emit the full inline dispatch sequence into `cb`.
    fn emit<Context, E>(
        &self,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) -> Result<(), E> {
        let entries = self.table.entries();



        // ── 2. Build a dense br_table over the [min, max] syscall range ───
        //
        // We use a compact block-per-entry scheme.  The br_table index is the
        // raw syscall number; gaps between sparse entries branch to $unknown.
        //
        // Layout (innermost → outermost):
        //   block $exit
        //     block $unknown
        //       block $arm_0   ← innermost
        //         block $arm_1
        //           …
        //           block $arm_{N-1}   ← outermost arm block
        //             br_table [arm_0, arm_1, …, arm_{N-1}, $unknown] $unknown
        //           end → arm_{N-1} handler
        //           br $exit
        //         end → arm_{N-2} handler …
        //       end → arm_0 handler
        //       br $exit
        //     end → $unknown
        //     unreachable
        //   end → $exit
        //
        // The br_table target list maps each integer in [min_syscall, max_syscall]
        // to the appropriate arm block depth (or $unknown for gaps).

        if entries.is_empty() {
            // No entries: always unreachable.
            cb.emit(ctx, &Instruction::Unreachable)?;
            return Ok(());
        }

        let min_num = entries.first().unwrap().0;
        let max_num = entries.last().unwrap().0;
        let range = (max_num - min_num + 1) as usize;
        let n = entries.len();

        // Build br_table targets: index 0 = $exit block (depth n+1 from innermost arm).
        // Depth from innermost arm block:
        //   0 = break out of innermost arm block (arm_0's handler)
        //   1 = arm_1's handler
        //   …
        //   n-1 = arm_{n-1}'s handler
        //   n   = $unknown block
        //   n+1 = $exit block (never used as branch target directly)
        //
        // We want each syscall number to fall through to its arm's handler,
        // so we branch to the corresponding arm block depth.
        // Unknown syscalls go to $unknown (depth = n from innermost arm).

        // Build arm-depth lookup: for each integer in [min, max], what depth?
        // Arm i (0-indexed, sorted ascending) gets depth = (n-1) - i so that
        // the innermost arm block (depth 0) handles entry 0, etc.
        let unknown_depth = n as u32;

        let mut br_targets: Vec<u32> = alloc::vec![unknown_depth; range];
        for (arm_idx, (syscall_num, _)) in entries.iter().enumerate() {
            let offset = (syscall_num - min_num) as usize;
            // Depth of this arm's block from the innermost arm block:
            // innermost is arm 0 (depth 0), next is arm 1 (depth 1), etc.
            br_targets[offset] = arm_idx as u32;
        }

        // Emit outer blocks: $exit (outermost), then $unknown, then arm blocks.
        // $exit
        cb.emit(ctx, &Instruction::Block(wasm_encoder::BlockType::Empty))?;
        // $unknown
        cb.emit(ctx, &Instruction::Block(wasm_encoder::BlockType::Empty))?;
        // arm blocks (n total, arm_{n-1} outermost → arm_0 innermost)
        for _ in 0..n {
            cb.emit(ctx, &Instruction::Block(wasm_encoder::BlockType::Empty))?;
        }

        // ── 1. Load the syscall number onto the WASM stack as i32 ─────────
        cb.emit(ctx, &Instruction::LocalGet(self.syscall_num_local))?;
        if self.syscall_num_is_i64 {
            cb.emit(ctx, &Instruction::I32WrapI64)?;
        }

        // Subtract min_num from the syscall number to index into br_targets.
        if min_num > 0 {
            cb.emit(ctx, &Instruction::I32Const(min_num as i32))?;
            cb.emit(ctx, &Instruction::I32Sub)?;
        }

        // br_table: indexed by (syscall_num - min_num), default = $unknown.
        cb.emit(ctx, &Instruction::BrTable(
            alloc::borrow::Cow::Owned(br_targets),
            unknown_depth,
        ))?;

        // ── 3. Handler arms (innermost arm first = entry index 0) ─────────
        for (arm_idx, (_, entry)) in entries.iter().enumerate() {
            // Close the arm block opened for this arm index.
            cb.emit(ctx, &Instruction::End)?;

            // 3a. Saves
            for save in &entry.saves {
                cb.emit(ctx, &Instruction::LocalGet(save.local_idx))?;
                cb.emit(ctx, &Instruction::GlobalSet(save.global_idx))?;
            }

            // 3a_mem. Memory stores
            for store in &entry.memory_stores {
                cb.emit(ctx, &Instruction::I32Const(store.addr as i32))?;
                cb.emit(ctx, &Instruction::LocalGet(store.value_local))?;
                if store.value_is_i64 {
                    cb.emit(ctx, &Instruction::I32WrapI64)?;
                }
                cb.emit(ctx, &Instruction::I32Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 2, // 4-byte align
                    memory_index: 0,
                }))?;
            }

            // 3b. Push handler parameters
            for source in &entry.param_map {
                match source {
                    ParamSource::LocalI64AsI32(local) => {
                        cb.emit(ctx, &Instruction::LocalGet(*local))?;
                        cb.emit(ctx, &Instruction::I32WrapI64)?;
                    }
                    ParamSource::LocalI32(local) => {
                        cb.emit(ctx, &Instruction::LocalGet(*local))?;
                    }
                    ParamSource::ConstI32(v) => {
                        cb.emit(ctx, &Instruction::I32Const(*v))?;
                    }
                    ParamSource::ConstI64(v) => {
                        cb.emit(ctx, &Instruction::I64Const(*v))?;
                    }
                }
            }

            // 3c. Call the handler (non-tail)
            cb.emit_call(ctx, entry.func_idx)?;

            // 3d. Store result if requested
            if let Some(result_local) = entry.result_local {
                if entry.negate_nonzero_result || entry.load_mem_on_success.is_some() {
                    cb.emit(ctx, &Instruction::I64ExtendI32S)?;
                    cb.emit(ctx, &Instruction::LocalTee(result_local))?;
                    cb.emit(ctx, &Instruction::I64Const(0))?;
                    cb.emit(ctx, &Instruction::I64Ne)?;
                    cb.emit(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
                    if entry.negate_nonzero_result {
                        cb.emit(ctx, &Instruction::I64Const(0))?;
                        cb.emit(ctx, &Instruction::LocalGet(result_local))?;
                        cb.emit(ctx, &Instruction::I64Sub)?;
                        cb.emit(ctx, &Instruction::LocalSet(result_local))?;
                    }
                    if let Some(mem_offset) = entry.load_mem_on_success {
                        cb.emit(ctx, &Instruction::Else)?;
                        cb.emit(ctx, &Instruction::I32Const(mem_offset as i32))?;
                        cb.emit(ctx, &Instruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                        cb.emit(ctx, &Instruction::I64ExtendI32S)?;
                        cb.emit(ctx, &Instruction::LocalSet(result_local))?;
                    }
                    cb.emit(ctx, &Instruction::End)?;
                } else {
                    // Simple: extend to i64, store.
                    cb.emit(ctx, &Instruction::I64ExtendI32S)?;
                    cb.emit(ctx, &Instruction::LocalSet(result_local))?;
                }
            } else {
                // Drop the return value if the handler returns one.
                if entry.has_return {
                    cb.emit(ctx, &Instruction::Drop)?;
                }
            }

            // 3e. Terminate or continue to the next guest PC.
            if entry.terminates {
                // Non-returning syscall: trap rather than fall through.
                cb.emit(ctx, &Instruction::Unreachable)?;
            } else {
                self.emit_continue_to_next_pc(ctx, cb)?;
            }
        }

        // ── $unknown arm ──────────────────────────────────────────────────
        // Close $unknown block, emit unreachable.
        cb.emit(ctx, &Instruction::End)?; // close $unknown
        cb.emit(ctx, &Instruction::Unreachable)?;

        // ── Close $exit block ─────────────────────────────────────────────
        cb.emit(ctx, &Instruction::End)?;

        Ok(())
    }

    /// Forward the live register file and tail-call the next guest PC slot.
    fn emit_continue_to_next_pc<Context, E>(
        &self,
        ctx: &mut Context,
        cb: &mut speet_riscv::CallbackContext<'_, Context, E>,
    ) -> Result<(), E> {
        for p in 0..self.num_params {
            cb.emit(ctx, &Instruction::LocalGet(p))?;
        }
        cb.emit(ctx, &Instruction::ReturnCall(self.next_pc_func))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;
    use alloc::string::String;
    use alloc::format;
    use speet_riscv::CallbackContext;

    struct MockSink {
        instructions: Vec<String>,
    }

    impl wax_core::build::InstructionSink<(), Infallible> for MockSink {
        fn instruction(&mut self, _ctx: &mut (), instruction: &Instruction<'_>) -> Result<(), Infallible> {
            self.instructions.push(format!("{:?}", instruction));
            Ok(())
        }
    }

    #[test]
    fn test_syscall_dispatcher_emit() {
        // Build a SyscallTable with two entries:
        // Syscall 10 -> func_idx 100
        // Syscall 20 -> func_idx 200
        let entries = alloc::vec![
            (10, SyscallEntry {
                func_idx: 100,
                param_map: alloc::vec![ParamSource::LocalI64AsI32(1)],
                saves: alloc::vec![],
                result_local: Some(2),
                negate_nonzero_result: false,
                has_return: true,
                terminates: false,
                memory_stores: alloc::vec![],
                load_mem_on_success: None,
            }),
            (20, SyscallEntry {
                func_idx: 200,
                param_map: alloc::vec![ParamSource::ConstI32(42)],
                saves: alloc::vec![],
                result_local: None,
                negate_nonzero_result: false,
                has_return: false,
                terminates: false,
                memory_stores: alloc::vec![],
                load_mem_on_success: None,
            }),
        ];
        let table = SyscallTable::new(entries);

        let dispatcher = WasmSyscallDispatcher {
            table: &table,
            syscall_num_local: 10,
            syscall_num_is_i64: true,
            num_params: 5,
            next_pc_func: 300,
        };

        let mut sink = MockSink { instructions: Vec::new() };
        {
            let mut cb = CallbackContext::new(&mut sink);
            dispatcher.emit(&mut (), &mut cb).unwrap();
        }

        // Verify emitted instructions
        let insts = &sink.instructions;

        // Check loading the syscall number: LocalGet(10) followed by I32WrapI64
        assert!(insts.iter().any(|i| i.contains("LocalGet(10)")));
        assert!(insts.iter().any(|i| i.contains("I32WrapI64")));

        // Check that a BrTable is emitted
        assert!(insts.iter().any(|i| i.contains("BrTable")));

        // Check call targets
        assert!(insts.iter().any(|i| i.contains("Call(100)")));
        assert!(insts.iter().any(|i| i.contains("Call(200)")));

        // Check trailing parameter gets and ReturnCall (one per non-terminating arm)
        assert!(insts.iter().any(|i| i.contains("ReturnCall(300)")));
        assert_eq!(
            insts.iter().filter(|i| i.contains("ReturnCall(300)")).count(),
            2
        );
    }
}

