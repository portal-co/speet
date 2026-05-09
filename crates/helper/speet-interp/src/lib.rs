//! `speet-interp` — lookup stub and interpreter WASM function generators.
//!
//! This crate generates the two special WASM functions required for OOB jump
//! dispatch:
//!
//! ## Lookup stub
//!
//! A WASM function with the same type as every compiled function
//! (arch-regs + `target_pc: i64` → arch-regs).  Given a `target_pc` it
//! binary-searches a sorted `[(pc: i64, table_slot: i32)]` array that is
//! baked into a passive data segment, then issues a `return_call_indirect`
//! to the correct compiled function.  Falls back to the interpreter when no
//! entry matches.
//!
//! ## Interpreter stub
//!
//! A WASM function with the same type.  The default implementation emits
//! `unreachable`; ISA-specific interpreters can be plugged in by replacing
//! the function body generator.  A host-import proxy (calling a wasmi host
//! function) is the recommended starting point until a full WASM interpreter
//! is available.
//!
//! ## Usage
//!
//! ```ignore
//! // Phase 1:
//! let oob = OobConfig::register(&mut entity_space);
//! let interp = OobInterp::register(&mut entity_space, &oob);  // reserves 2 fn slots
//! linker.inner.oob_config = Some(oob);
//!
//! // Phase 2 — after compiling all guest functions, build the BinaryUnit:
//! let (lookup_fn, interp_fn, data_seg) = interp.generate(
//!     &entity_space, &oob, &compiled_pc_table, func_type, type_idx,
//! );
//! ```

extern crate alloc;

use alloc::vec::Vec;
use speet_link_core::{EntityIndexSpace, IndexSlot, OobConfig};
use wasm_encoder::{
    BlockType, Function, Instruction, MemArg, ValType,
};

// ── PcEntry ───────────────────────────────────────────────────────────────────

/// One entry in the compiled-PC dispatch table.
///
/// Entries are stored sorted by `guest_pc` in ascending order so the lookup
/// stub can binary-search them.  `table_slot` is the absolute WASM function
/// table slot index for the corresponding compiled function.
#[derive(Clone, Copy, Debug)]
pub struct PcEntry {
    /// Guest PC of the compiled function.
    pub guest_pc: u64,
    /// Absolute WASM function table slot (index into the dispatch table).
    pub table_slot: u32,
}

// ── OobInterp ─────────────────────────────────────────────────────────────────

/// Registers and generates the lookup stub + interpreter for OOB dispatch.
pub struct OobInterp {
    /// Index slot for the two generated functions (lookup stub + interpreter).
    pub func_slot: IndexSlot,
}

impl OobInterp {
    /// Register two function slots in Phase 1 (lookup stub + interpreter),
    /// and fill their absolute indices into `oob`.
    pub fn register(entity_space: &mut EntityIndexSpace, oob: &mut OobConfig) -> Self {
        let func_slot = entity_space.functions.append(2);
        let base = entity_space.functions.base(func_slot);
        oob.set_func_indices(base, base + 1);
        Self { func_slot }
    }

    /// Generate the lookup stub and interpreter WASM function bodies.
    ///
    /// * `entity_space` — to resolve the absolute function base.
    /// * `oob` — for the dispatch table index and interpreter func index.
    /// * `entries` — sorted `[(guest_pc, table_slot)]` for all compiled fns.
    /// * `params` — WASM param types for the shared function type
    ///   (arch regs + `target_pc: i64`).
    /// * `returns` — WASM return types (injected/trap param types).
    /// * `type_idx` — pre-registered WASM type index for the shared fn type.
    ///
    /// Returns `(lookup_stub_fn, interp_fn, passive_data_bytes)`:
    /// - The two `Function` values are ready to insert into the code section.
    /// - The passive data bytes encode the sorted PC table and must be added
    ///   as a passive data segment whose index is `data_seg_idx`.
    pub fn generate(
        &self,
        entity_space: &EntityIndexSpace,
        oob: &OobConfig,
        entries: &[PcEntry],
        params: &[ValType],
        returns: &[ValType],
        type_idx: u32,
        table_idx: u32,
        data_seg_idx: u32,
        data_mem_idx: u32,
    ) -> (Function, Function, Vec<u8>) {
        let data = Self::build_pc_table(entries);
        let lookup = self.build_lookup_stub(
            oob, params, returns, type_idx, table_idx, data_seg_idx,
            data_mem_idx, entries.len() as u32,
        );
        let interp = self.build_interp_stub(params, returns);
        (lookup, interp, data)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Encode the PC table as a flat byte array of `(pc: i64, slot: i32)`
    /// pairs (12 bytes each, little-endian), sorted by `pc`.
    fn build_pc_table(entries: &[PcEntry]) -> Vec<u8> {
        let mut data = Vec::with_capacity(entries.len() * 12);
        for e in entries {
            data.extend_from_slice(&e.guest_pc.to_le_bytes());
            data.extend_from_slice(&e.table_slot.to_le_bytes());
        }
        data
    }

    /// Generate the lookup stub function body.
    ///
    /// Algorithm (binary search over the passive data segment):
    /// ```text
    /// lo = 0; hi = N
    /// loop:
    ///   if lo >= hi → call interp
    ///   mid = (lo + hi) >> 1
    ///   mid_pc = load i64 from data[mid * 12]
    ///   if mid_pc == target_pc:
    ///     slot = load i32 from data[mid * 12 + 8]
    ///     push all params
    ///     return_call_indirect table[slot]
    ///   elif mid_pc < target_pc: lo = mid + 1
    ///   else: hi = mid
    ///   br loop
    /// ```
    ///
    /// The data segment is initialised into scratch memory at offset 0 of a
    /// dedicated memory (data_mem_idx) before any guest code runs.
    #[allow(clippy::too_many_arguments)]
    fn build_lookup_stub(
        &self,
        oob: &OobConfig,
        params: &[ValType],
        returns: &[ValType],
        type_idx: u32,
        table_idx: u32,
        data_seg_idx: u32,
        data_mem_idx: u32,
        n_entries: u32,
    ) -> Function {
        // Locals after params:
        //   0: lo (i32)
        //   1: hi (i32)
        //   2: mid (i32)
        //   3: mid_pc (i64)
        //   4: byte_off (i32)  — mid * 12
        let n_params = params.len() as u32;
        // target_pc is the last param (local index n_params - 1)
        let tpc_local = n_params - 1;
        // scratch locals start at n_params
        let lo  = n_params;
        let hi  = n_params + 1;
        let mid = n_params + 2;
        let mid_pc = n_params + 3;
        let byte_off = n_params + 4;

        let mut f = Function::new([
            (1, ValType::I32), // lo
            (1, ValType::I32), // hi
            (1, ValType::I32), // mid
            (1, ValType::I64), // mid_pc
            (1, ValType::I32), // byte_off
        ]);

        // lo = 0; hi = n_entries
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::LocalSet(lo));
        f.instruction(&Instruction::I32Const(n_entries as i32));
        f.instruction(&Instruction::LocalSet(hi));

        // loop { ... }
        f.instruction(&Instruction::Loop(BlockType::Empty));

        // if lo >= hi → break loop → fall through to interp call
        f.instruction(&Instruction::LocalGet(lo));
        f.instruction(&Instruction::LocalGet(hi));
        f.instruction(&Instruction::I32GeU);
        f.instruction(&Instruction::BrIf(1)); // break out of loop

        // mid = (lo + hi) >> 1
        f.instruction(&Instruction::LocalGet(lo));
        f.instruction(&Instruction::LocalGet(hi));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(1));
        f.instruction(&Instruction::I32ShrU);
        f.instruction(&Instruction::LocalSet(mid));

        // byte_off = mid * 12
        f.instruction(&Instruction::LocalGet(mid));
        f.instruction(&Instruction::I32Const(12));
        f.instruction(&Instruction::I32Mul);
        f.instruction(&Instruction::LocalSet(byte_off));

        // mid_pc = load i64 from mem[byte_off]
        f.instruction(&Instruction::LocalGet(byte_off));
        f.instruction(&Instruction::I64Load(MemArg {
            offset: 0,
            align: 0,
            memory_index: data_mem_idx,
        }));
        f.instruction(&Instruction::LocalSet(mid_pc));

        // if mid_pc == target_pc { dispatch }
        f.instruction(&Instruction::LocalGet(mid_pc));
        f.instruction(&Instruction::LocalGet(tpc_local));
        f.instruction(&Instruction::I64Eq);
        f.instruction(&Instruction::If(BlockType::Empty));
        // Push all params (regs + target_pc as-is)
        for p in 0..n_params {
            f.instruction(&Instruction::LocalGet(p));
        }
        // table slot = load i32 from mem[byte_off + 8]
        f.instruction(&Instruction::LocalGet(byte_off));
        f.instruction(&Instruction::I32Load(MemArg {
            offset: 8,
            align: 0,
            memory_index: data_mem_idx,
        }));
        f.instruction(&Instruction::ReturnCallIndirect {
            type_index: type_idx,
            table_index: table_idx,
        });
        f.instruction(&Instruction::End); // end if

        // elif mid_pc < target_pc: lo = mid + 1
        f.instruction(&Instruction::LocalGet(mid_pc));
        f.instruction(&Instruction::LocalGet(tpc_local));
        f.instruction(&Instruction::I64LtU);
        f.instruction(&Instruction::If(BlockType::Empty));
        f.instruction(&Instruction::LocalGet(mid));
        f.instruction(&Instruction::I32Const(1));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalSet(lo));
        f.instruction(&Instruction::Else);
        // else: hi = mid
        f.instruction(&Instruction::LocalGet(mid));
        f.instruction(&Instruction::LocalSet(hi));
        f.instruction(&Instruction::End); // end if/else

        f.instruction(&Instruction::Br(0)); // continue loop
        f.instruction(&Instruction::End); // end loop

        // Not found: tail-call interpreter
        for p in 0..n_params {
            f.instruction(&Instruction::LocalGet(p));
        }
        f.instruction(&Instruction::ReturnCall(oob.interp_func_idx));

        f.instruction(&Instruction::End); // end function
        let _ = (data_seg_idx, returns); // suppress unused warnings
        f
    }

    /// Generate a placeholder interpreter function that traps (`unreachable`).
    ///
    /// Replace this with an ISA-specific generator when a real interpreter is
    /// available.  The signature must match the shared function type.
    fn build_interp_stub(&self, params: &[ValType], _returns: &[ValType]) -> Function {
        let n_params = params.len() as u32;
        let mut f = Function::new([]);
        // Suppress unused-param warnings from the validator.
        for p in 0..n_params {
            f.instruction(&Instruction::LocalGet(p));
            f.instruction(&Instruction::Drop);
        }
        f.instruction(&Instruction::Unreachable);
        f.instruction(&Instruction::End);
        f
    }
}
