//! ABI shim emitter for cross-architecture function bridges.
//!
//! An ABI shim is a thin WASM function that adapts the calling convention of
//! one architecture's generated code to another.  For example, a JNI bridge
//! from x86-64 generated code to a RISC-V native function would:
//!
//! 1. Map caller register locals to callee register positions.
//! 2. Zero-fill callee positions that have no caller counterpart.
//! 3. Tail-call the callee.
//!
//! ## Usage
//!
//! ```ignore
//! use speet_link::shim::{ShimSpec, emit_shim};
//! use speet_link::unit::FuncType;
//! use wasm_encoder::ValType;
//!
//! let caller_sig = FuncType::from_val_types(&[ValType::I64; 26], &[]);
//! let callee_sig = FuncType::from_val_types(&[ValType::I64; 32], &[]);
//!
//! // Map first 26 callee params from the 26 caller params; rest → 0.
//! let param_map: Vec<Option<u32>> = (0..32u32).map(|i| {
//!     if i < 26 { Some(i) } else { None }
//! }).collect();
//!
//! let spec = ShimSpec {
//!     caller_sig,
//!     callee_func_idx: 42,
//!     callee_sig,
//!     param_map,
//! };
//! let shim_fn: wasm_encoder::Function = emit_shim(&spec);
//! ```

use alloc::vec::Vec;
use wasm_encoder::{Function, Instruction, ValType};

use crate::unit::FuncType;

// ── ShimSpec ──────────────────────────────────────────────────────────────────

/// Specification for a cross-ABI shim function.
pub struct ShimSpec {
    /// The signature of the shim itself (visible to callers).
    pub caller_sig: FuncType,
    /// Absolute WASM function index of the target callee.
    pub callee_func_idx: u32,
    /// The callee's expected signature.
    pub callee_sig: FuncType,
    /// Maps each callee parameter position to either:
    /// - `Some(local_idx)` — copy from the corresponding caller local, or
    /// - `None` — push `i32.const 0` (zero-fill).
    ///
    /// Length must equal the number of parameters in `callee_sig`.
    pub param_map: Vec<Option<u32>>,
}

// ── emit_shim ─────────────────────────────────────────────────────────────────

/// Emit a [`wasm_encoder::Function`] that adapts `caller_sig` to `callee_sig`
/// according to `spec.param_map`, then tail-calls the callee.
///
/// The emitted function body is:
/// ```wasm
/// (func (param <caller_sig.params>)
///   ;; For each callee param:
///   ;; if param_map[i] = Some(j): local.get j
///   ;; if param_map[i] = None:   i32.const 0  (or i64.const 0)
///   return_call <callee_func_idx>)
/// ```
pub fn emit_shim(spec: &ShimSpec) -> Function {
    // No extra locals needed — the shim only reads caller params and
    // immediately tail-calls the callee.
    let mut f = Function::new([]);

    // Push each callee argument.
    let callee_param_types: Vec<ValType> = spec.callee_sig.params_val_types().collect();
    for (i, opt_local) in spec.param_map.iter().enumerate() {
        let ty = callee_param_types
            .get(i)
            .copied()
            .unwrap_or(ValType::I32);
        match opt_local {
            Some(local_idx) => {
                f.instruction(&Instruction::LocalGet(*local_idx));
            }
            None => {
                // Zero-fill: use the callee parameter's type to pick the
                // appropriate constant instruction.
                match ty {
                    ValType::I64 | ValType::F64 => {
                        f.instruction(&Instruction::I64Const(0));
                    }
                    ValType::F32 => {
                        f.instruction(&Instruction::F32Const(0.0_f32.into()));
                    }
                    _ => {
                        f.instruction(&Instruction::I32Const(0));
                    }
                }
            }
        }
    }

    // Tail-call the callee.
    f.instruction(&Instruction::ReturnCall(spec.callee_func_idx));
    f.instruction(&Instruction::End);

    f
}
