//! [`FuncType`] and [`BinaryUnit`] — the atom of cross-binary linking.
//!
//! A [`BinaryUnit`] collects the compiled functions for one translated binary
//! together with their WASM types and any named entry-point exports.  The
//! [`Linker`](crate::linker::Linker) feeds these to a
//! [`LinkerPlugin`](crate::linker::LinkerPlugin) (e.g.
//! [`MegabinaryBuilder`](crate::builder::MegabinaryBuilder)) after each
//! translation pass.

use alloc::{string::String, vec::Vec};
use wasm_encoder::ValType;

// ── DataSegment ───────────────────────────────────────────────────────────────

/// A raw data blob stored as a passive WASM data segment.
///
/// Placement into linear memory is handled entirely at WASM runtime by the
/// data-init function associated with the [`BinaryUnit`] that owns this
/// segment.  The segment itself carries no address — the mapper emits the
/// destination address when the init function calls `memory.init`.
#[derive(Clone, Debug)]
pub struct DataSegment {
    /// Raw data bytes.
    pub data: Vec<u8>,
}

// ── ValType byte encoding ─────────────────────────────────────────────────────
//
// We store ValType discriminants as raw bytes so that `FuncType` can derive
// `Ord` / `Hash` without depending on wasm_encoder's trait impls.  The
// encoding follows the WASM binary format's valtype opcodes.

fn val_type_to_byte(v: ValType) -> u8 {
    match v {
        ValType::I32 => 0x7F,
        ValType::I64 => 0x7E,
        ValType::F32 => 0x7D,
        ValType::F64 => 0x7C,
        ValType::V128 => 0x7B,
        // Ref types: use a sentinel byte; the ref-type variants are not used
        // in the integer-only recompiler pipelines.
        ValType::Ref(_) => 0x00,
    }
}

fn byte_to_val_type(b: u8) -> Option<ValType> {
    match b {
        0x7F => Some(ValType::I32),
        0x7E => Some(ValType::I64),
        0x7D => Some(ValType::F32),
        0x7C => Some(ValType::F64),
        0x7B => Some(ValType::V128),
        _ => None, // ref types are not round-tripped here
    }
}

// ── FuncType ──────────────────────────────────────────────────────────────────

/// A WASM function signature stored as byte-encoded [`ValType`] discriminants.
///
/// Using raw bytes rather than `wasm_encoder::FuncType` lets `FuncType`
/// implement `Ord` and `Hash` without additional dependencies.  The byte
/// encoding follows the WASM binary format.
///
/// # Example
/// ```ignore
/// use speet_link::unit::FuncType;
/// use wasm_encoder::ValType;
///
/// let sig = FuncType::from_val_types(&[ValType::I32; 26], &[]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FuncType {
    /// Encoded parameter types.
    pub params: Vec<u8>,
    /// Encoded result types.
    pub results: Vec<u8>,
}

impl FuncType {
    /// Construct from slices of [`ValType`].
    pub fn from_val_types(params: &[ValType], results: &[ValType]) -> Self {
        Self {
            params: params.iter().copied().map(val_type_to_byte).collect(),
            results: results.iter().copied().map(val_type_to_byte).collect(),
        }
    }

    /// Construct a type with `n` identical parameter types and no results.
    pub fn uniform_params(n: u32, ty: ValType) -> Self {
        Self::from_val_types(&alloc::vec![ty; n as usize], &[])
    }

    /// Iterate over parameter [`ValType`]s.
    ///
    /// Byte values that do not correspond to a recognised primitive
    /// `ValType` are silently skipped (ref types are not round-tripped).
    pub fn params_val_types(&self) -> impl Iterator<Item = ValType> + '_ {
        self.params.iter().filter_map(|b| byte_to_val_type(*b))
    }

    /// Iterate over result [`ValType`]s.
    pub fn results_val_types(&self) -> impl Iterator<Item = ValType> + '_ {
        self.results.iter().filter_map(|b| byte_to_val_type(*b))
    }
}

// ── BinaryUnit ────────────────────────────────────────────────────────────────

/// The output of translating one contiguous binary (or binary fragment).
///
/// `F` is the instruction-sink type produced by the underlying reactor
/// (typically `wasm_encoder::Function`).
///
/// ## Fields
///
/// * `fns` — compiled WASM functions in order.
/// * `base_func_offset` — the absolute WASM function index of `fns[0]`.
/// * `entry_points` — `(symbol, absolute_wasm_func_index)` pairs for exports.
/// * `func_types` — per-function type, parallel to `fns`.
/// * `data_segments` — passive data blobs (no addresses).
/// * `data_init_fn` — optional `() → ()` function that loads the segments into
///   linear memory at their guest physical addresses.
pub struct BinaryUnit<F> {
    /// Compiled WASM functions, in order.
    pub fns: Vec<F>,
    /// Absolute WASM function index of `fns[0]`.
    pub base_func_offset: u32,
    /// Named exports: `(symbol_name, absolute_wasm_func_index)`.
    pub entry_points: Vec<(String, u32)>,
    /// Per-function type, parallel to `fns`.  `func_types[i]` is the type
    /// of `fns[i]`.
    pub func_types: Vec<FuncType>,
    /// Passive data segments associated with this binary unit.
    ///
    /// Each entry is a raw byte blob with no embedded address.  The physical
    /// destination is computed at WASM runtime by `data_init_fn`.
    pub data_segments: Vec<DataSegment>,
    /// Optional data-initialiser function (type `() → ()`).
    ///
    /// When present, this function emits `memory.init` + `data.drop` for
    /// every element of `data_segments`, using mapper calls to compute
    /// physical destinations at WASM runtime.
    ///
    /// [`MegabinaryBuilder`](crate::builder::MegabinaryBuilder) chains all
    /// per-unit init functions into a single start function (or
    /// `__speet_data_init` export) in the merged output module.
    pub data_init_fn: Option<(F, FuncType)>,
}
