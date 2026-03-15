//! ABI shim emitter — cross-architecture function bridges with rich place expressions.
//!
//! A shim is a thin WASM function that adapts one calling convention to another.
//! Its body has three phases:
//!
//! 1. **Saves** — copy values between [`Place`]s before any parameters are touched
//!    (e.g. snapshot a caller register into a global or memory slot).
//! 2. **Param map** — for each callee parameter, evaluate a [`ParamSource`] and push
//!    the result onto the stack.
//! 3. **Tail-call** — `return_call <callee_func_idx>`.
//!
//! ## Place expressions
//!
//! [`Place`] forms a small expression tree for storable / loadable locations:
//!
//! ```text
//! Place ::= Local(idx)
//!         | Global(idx)
//!         | Deref { base: Place, offset: u64, width: MemWidth, memory: u32 }
//! ```
//!
//! `Deref` dereferences memory: it evaluates `base` to get an address, adds
//! `offset`, and reads or writes `width` bytes in memory `memory`.
//!
//! A nested pointer stored in a global is just `Deref { base: Global(g), offset,
//! width, memory }` — for example, a JNI `JNIEnv**` looks like:
//!
//! ```ignore
//! // *env = *(JNIEnv*)global[0]  at offset 16 (function table entry)
//! ParamSource::Load(Place::Deref {
//!     base:   Box::new(Place::Deref {
//!                 base: Box::new(Place::Global(0)),
//!                 offset: 0,
//!                 width: MemWidth::I32,
//!                 memory: 0,
//!             }),
//!     offset: 16,
//!     width:  MemWidth::I32,
//!     memory: 0,
//! })
//! ```
//!
//! ## Saves
//!
//! A [`SavePair`] copies `src: Place` → `dst: Place` unconditionally, before
//! any parameter is evaluated.  Use this to preserve caller state that would
//! otherwise be overwritten:
//!
//! ```ignore
//! // Save the current stack-pointer local (local 6) to global 1 before the call.
//! SavePair { src: Place::Local(6), dst: Place::Global(1) }
//! ```
//!
//! ## Extra locals
//!
//! Shims that use `Place::Local(n)` where `n >= caller_sig.params.len()` need the
//! extra slots declared via [`ShimSpec::extra_locals`].  Each entry is `(count,
//! ValType)` — the same format as `wasm_encoder::Function::new`.

use alloc::boxed::Box;
use alloc::vec::Vec;
use wasm_encoder::{Function, Instruction, MemArg, ValType};

use crate::unit::FuncType;

// ── MemWidth ──────────────────────────────────────────────────────────────────

/// Width and signedness of a memory access inside a [`Place::Deref`].
///
/// Names follow the pattern `<result_type>[Sign][bits]`:
/// - No suffix → natural-width load/store (e.g. `I32` = 4-byte i32).
/// - `U` / `S` suffix → unsigned / signed extension of a narrower field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemWidth {
    // ── 32-bit results ────────────────────────────────────────────────────
    /// `i32.load` — 4-byte, no extension.
    I32,
    /// `i32.load8_u` — 1-byte, zero-extended to i32.
    I32U8,
    /// `i32.load8_s` — 1-byte, sign-extended to i32.
    I32S8,
    /// `i32.load16_u` — 2-byte, zero-extended to i32.
    I32U16,
    /// `i32.load16_s` — 2-byte, sign-extended to i32.
    I32S16,

    // ── 64-bit results ────────────────────────────────────────────────────
    /// `i64.load` — 8-byte, no extension.
    I64,
    /// `i64.load8_u` — 1-byte, zero-extended to i64.
    I64U8,
    /// `i64.load8_s` — 1-byte, sign-extended to i64.
    I64S8,
    /// `i64.load16_u` — 2-byte, zero-extended to i64.
    I64U16,
    /// `i64.load16_s` — 2-byte, sign-extended to i64.
    I64S16,
    /// `i64.load32_u` — 4-byte, zero-extended to i64.
    I64U32,
    /// `i64.load32_s` — 4-byte, sign-extended to i64.
    I64S32,

    // ── Floating-point ────────────────────────────────────────────────────
    /// `f32.load` — 4-byte IEEE 754 single.
    F32,
    /// `f64.load` — 8-byte IEEE 754 double.
    F64,
}

impl MemWidth {
    /// Natural alignment exponent (log₂ of the byte width being transferred).
    pub fn natural_align(&self) -> u32 {
        match self {
            MemWidth::I32U8  | MemWidth::I32S8
            | MemWidth::I64U8  | MemWidth::I64S8   => 0,
            MemWidth::I32U16 | MemWidth::I32S16
            | MemWidth::I64U16 | MemWidth::I64S16  => 1,
            MemWidth::I32    | MemWidth::F32
            | MemWidth::I64U32 | MemWidth::I64S32  => 2,
            MemWidth::I64    | MemWidth::F64        => 3,
        }
    }

    /// The WASM `ValType` produced by loading through this width.
    pub fn result_type(&self) -> ValType {
        match self {
            MemWidth::I32 | MemWidth::I32U8 | MemWidth::I32S8
            | MemWidth::I32U16 | MemWidth::I32S16 => ValType::I32,
            MemWidth::I64 | MemWidth::I64U8 | MemWidth::I64S8
            | MemWidth::I64U16 | MemWidth::I64S16
            | MemWidth::I64U32 | MemWidth::I64S32 => ValType::I64,
            MemWidth::F32 => ValType::F32,
            MemWidth::F64 => ValType::F64,
        }
    }

    fn memarg(&self, offset: u64, memory: u32) -> MemArg {
        MemArg { offset, align: self.natural_align(), memory_index: memory }
    }

    fn load_insn(&self, offset: u64, memory: u32) -> Instruction<'static> {
        let a = self.memarg(offset, memory);
        match self {
            MemWidth::I32    => Instruction::I32Load(a),
            MemWidth::I32U8  => Instruction::I32Load8U(a),
            MemWidth::I32S8  => Instruction::I32Load8S(a),
            MemWidth::I32U16 => Instruction::I32Load16U(a),
            MemWidth::I32S16 => Instruction::I32Load16S(a),
            MemWidth::I64    => Instruction::I64Load(a),
            MemWidth::I64U8  => Instruction::I64Load8U(a),
            MemWidth::I64S8  => Instruction::I64Load8S(a),
            MemWidth::I64U16 => Instruction::I64Load16U(a),
            MemWidth::I64S16 => Instruction::I64Load16S(a),
            MemWidth::I64U32 => Instruction::I64Load32U(a),
            MemWidth::I64S32 => Instruction::I64Load32S(a),
            MemWidth::F32    => Instruction::F32Load(a),
            MemWidth::F64    => Instruction::F64Load(a),
        }
    }

    fn store_insn(&self, offset: u64, memory: u32) -> Instruction<'static> {
        let a = self.memarg(offset, memory);
        match self {
            MemWidth::I32 | MemWidth::I32U8 | MemWidth::I32S8
            | MemWidth::I32U16 | MemWidth::I32S16  => match self {
                MemWidth::I32    => Instruction::I32Store(a),
                MemWidth::I32U8  => Instruction::I32Store8(a),
                MemWidth::I32S8  => Instruction::I32Store8(a),
                MemWidth::I32U16 => Instruction::I32Store16(a),
                MemWidth::I32S16 => Instruction::I32Store16(a),
                _ => unreachable!(),
            },
            MemWidth::I64 | MemWidth::I64U8 | MemWidth::I64S8
            | MemWidth::I64U16 | MemWidth::I64S16
            | MemWidth::I64U32 | MemWidth::I64S32 => match self {
                MemWidth::I64    => Instruction::I64Store(a),
                MemWidth::I64U8  => Instruction::I64Store8(a),
                MemWidth::I64S8  => Instruction::I64Store8(a),
                MemWidth::I64U16 => Instruction::I64Store16(a),
                MemWidth::I64S16 => Instruction::I64Store16(a),
                MemWidth::I64U32 => Instruction::I64Store32(a),
                MemWidth::I64S32 => Instruction::I64Store32(a),
                _ => unreachable!(),
            },
            MemWidth::F32 => Instruction::F32Store(a),
            MemWidth::F64 => Instruction::F64Store(a),
        }
    }
}

// ── Place ─────────────────────────────────────────────────────────────────────

/// A storable / loadable location in the generated WASM.
///
/// Can appear as either the source or the destination of a value transfer.
/// `Deref` nodes form a tree, enabling nested-pointer dereferences such as
/// `*(global[g] + offset)`.
#[derive(Clone, Debug)]
pub enum Place {
    /// A WASM local variable.
    Local(u32),

    /// A WASM mutable global variable.
    Global(u32),

    /// Memory dereference: evaluate `base` to get an address, add `offset`,
    /// then load or store `width` bytes in WASM linear memory `memory`.
    ///
    /// When loading, the result type is `width.result_type()`.
    /// When storing, the value being stored must match `width.result_type()`.
    Deref {
        /// Expression that evaluates to the base address (must be i32 or i64
        /// depending on the memory's index type).
        base: Box<Place>,
        /// Byte offset added to the base address.
        offset: u64,
        /// Byte width and sign-extension mode.
        width: MemWidth,
        /// Linear memory index (0 for the default memory).
        memory: u32,
    },
}

impl Place {
    /// Emit instructions that **load** from this place onto the WASM stack.
    fn emit_load(&self, f: &mut Function) {
        match self {
            Place::Local(idx) => {
                f.instruction(&Instruction::LocalGet(*idx));
            }
            Place::Global(idx) => {
                f.instruction(&Instruction::GlobalGet(*idx));
            }
            Place::Deref { base, offset, width, memory } => {
                // Push address, then load.
                base.emit_load(f);
                f.instruction(&width.load_insn(*offset, *memory));
            }
        }
    }

    /// Emit instructions that **store** the value currently on top of the
    /// WASM stack into this place.
    ///
    /// For `Deref` sinks the stack must contain only the value to store; the
    /// address is pushed internally before the store instruction.  For `Local`
    /// and `Global` the value is popped directly.
    ///
    /// `emit_value` is called to push the value; it is called *after* any
    /// address setup so that the final stack state is `[addr?, value]` as
    /// required by WASM store instructions.
    fn emit_store(&self, f: &mut Function, emit_value: &dyn Fn(&mut Function)) {
        match self {
            Place::Local(idx) => {
                emit_value(f);
                f.instruction(&Instruction::LocalSet(*idx));
            }
            Place::Global(idx) => {
                emit_value(f);
                f.instruction(&Instruction::GlobalSet(*idx));
            }
            Place::Deref { base, offset, width, memory } => {
                // WASM memory stores: [addr, value] → store.
                base.emit_load(f);   // push address
                emit_value(f);       // push value
                f.instruction(&width.store_insn(*offset, *memory));
            }
        }
    }
}

// ── ParamSource ───────────────────────────────────────────────────────────────

/// Source expression for a single callee parameter.
///
/// Replaces the previous `Option<u32>` in [`ShimSpec::param_map`].
#[derive(Clone, Debug)]
pub enum ParamSource {
    /// Load the value from this place.  The result type must match the callee
    /// parameter type; no truncation or extension is emitted.
    Load(Place),

    /// Synthesize a zero value.  The exact instruction (`i32.const 0`,
    /// `i64.const 0`, etc.) is chosen from the callee parameter type in the
    /// signature.
    Zero,

    /// Push a specific `i32` constant.
    I32Const(i32),

    /// Push a specific `i64` constant.
    I64Const(i64),

    /// Push a specific `f32` constant.
    F32Const(f32),

    /// Push a specific `f64` constant.
    F64Const(f64),
}

impl ParamSource {
    fn emit(&self, f: &mut Function, callee_ty: ValType) {
        match self {
            ParamSource::Load(place) => place.emit_load(f),
            ParamSource::Zero => emit_zero(f, callee_ty),
            ParamSource::I32Const(v) => { f.instruction(&Instruction::I32Const(*v)); }
            ParamSource::I64Const(v) => { f.instruction(&Instruction::I64Const(*v)); }
            ParamSource::F32Const(v) => { f.instruction(&Instruction::F32Const((*v).into())); }
            ParamSource::F64Const(v) => { f.instruction(&Instruction::F64Const((*v).into())); }
        }
    }
}

// ── SavePair ──────────────────────────────────────────────────────────────────

/// An unconditional copy from `src` to `dst`, performed before parameter
/// evaluation.
///
/// Use saves to snapshot caller state that would otherwise be clobbered by
/// the shim's tail-call.  For example, preserving a stack pointer:
///
/// ```ignore
/// SavePair { src: Place::Local(6), dst: Place::Global(1) }
/// ```
///
/// Or spilling a register into memory through a pointer in a global:
///
/// ```ignore
/// SavePair {
///     src: Place::Local(2),
///     dst: Place::Deref { base: Box::new(Place::Global(0)), offset: 8,
///                         width: MemWidth::I64, memory: 0 },
/// }
/// ```
#[derive(Clone, Debug)]
pub struct SavePair {
    /// Value to copy.
    pub src: Place,
    /// Destination to write the value into.
    pub dst: Place,
}

// ── ShimSpec ──────────────────────────────────────────────────────────────────

/// Specification for a cross-ABI shim function.
pub struct ShimSpec {
    /// The signature of the shim itself (visible to callers).
    pub caller_sig: FuncType,
    /// Absolute WASM function index of the target callee.
    pub callee_func_idx: u32,
    /// The callee's expected signature.
    pub callee_sig: FuncType,
    /// Source expression for each callee parameter.
    ///
    /// Length must equal the number of parameters in `callee_sig`.
    pub param_map: Vec<ParamSource>,
    /// Copies to perform *before* the param map is evaluated.
    ///
    /// Executed in order; later saves see the results of earlier saves.
    pub saves: Vec<SavePair>,
    /// Extra local variable groups to declare in the generated function body
    /// (in addition to the implicit caller parameters).
    ///
    /// Each entry is `(count, ValType)` — the same format accepted by
    /// `wasm_encoder::Function::new`.  Required when `saves` or `param_map`
    /// reference `Place::Local(n)` where `n >= caller_sig.params.len()`.
    pub extra_locals: Vec<(u32, ValType)>,
}

// ── emit_shim ─────────────────────────────────────────────────────────────────

/// Emit a [`wasm_encoder::Function`] from a [`ShimSpec`].
///
/// ## Generated body
///
/// ```wasm
/// (func (param …caller_sig…) (local …extra_locals…)
///   ;; Phase 1 — saves (in order)
///   <dst_0>.store(src_0.load())
///   <dst_1>.store(src_1.load())
///   …
///   ;; Phase 2 — callee argument construction
///   param_map[0].emit(callee_sig.params[0])
///   param_map[1].emit(callee_sig.params[1])
///   …
///   ;; Phase 3 — tail-call
///   return_call <callee_func_idx>
///   end)
/// ```
pub fn emit_shim(spec: &ShimSpec) -> Function {
    let mut f = Function::new(spec.extra_locals.iter().copied());

    // Phase 1 — saves.
    for pair in &spec.saves {
        let src = &pair.src;
        pair.dst.emit_store(&mut f, &|f| src.emit_load(f));
    }

    // Phase 2 — callee arguments.
    let callee_param_types: Vec<ValType> = spec.callee_sig.params_val_types().collect();
    for (i, src) in spec.param_map.iter().enumerate() {
        let ty = callee_param_types.get(i).copied().unwrap_or(ValType::I32);
        src.emit(&mut f, ty);
    }

    // Phase 3 — tail-call.
    f.instruction(&Instruction::ReturnCall(spec.callee_func_idx));
    f.instruction(&Instruction::End);

    f
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn emit_zero(f: &mut Function, ty: ValType) {
    match ty {
        ValType::I64 => { f.instruction(&Instruction::I64Const(0)); }
        ValType::F32 => { f.instruction(&Instruction::F32Const(0.0_f32.into())); }
        ValType::F64 => { f.instruction(&Instruction::F64Const(0.0_f64.into())); }
        _            => { f.instruction(&Instruction::I32Const(0)); }
    }
}
