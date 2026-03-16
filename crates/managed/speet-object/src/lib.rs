//! # speet-object
//!
//! Pluggable object model for managed-runtime recompilers (DEX, JVM, CLR, …).
//!
//! ## Object format
//!
//! Every heap object begins with a 36-byte header:
//!
//! ```text
//! Offset  Size  Field
//! 0       32    type_hash: [u8; 32]
//!                   SHA3-256 of the class name (no array brackets) for reference types.
//!                   All-zero except the last byte for primitive array element types
//!                   (e.g. [0,0,...,5] for `int[]`, using PrimitiveType discriminants).
//! 32       4    array_dim: u32 (little-endian)
//!                   0 for plain objects; n for n-dimensional arrays (e.g. 2 for `int[][]`).
//! 36      ...   data
//!                   array_dim == 0 → instance fields at fixed byte offsets
//!                   array_dim  > 0 → [length: u32 @ 36][elements starting @ 40]
//! ```
//!
//! ## Pluggability
//!
//! The [`ObjectModel`] trait abstracts over how objects are represented in
//! wasm, how they are allocated, and how fields and elements are accessed.
//! All methods emit wasm instructions directly into an [`InstructionSink`],
//! allowing implementations to write to the reactor without an intermediate buffer.
//!
//! Provided implementations:
//!
//! - [`LinearMemoryObjects`] — linear-memory layout as described above.
//! - [`NoObjectModel`] — stub that emits `unreachable` for every operation
//!   (used when object operations are not needed).
//!
//! ## Type identification
//!
//! - [`TypeHash`] — the 32-byte runtime type tag.
//! - [`PrimitiveType`] — discriminants for primitive array element types.

#![no_std]
extern crate alloc;

mod hash;
mod linear;
mod model;

pub use hash::{PrimitiveType, TypeHash};
pub use linear::{ARRAY_DATA_OFFSET, ARRAY_LENGTH_OFFSET, LinearMemoryObjects, OBJECT_HEADER_SIZE};
pub use model::{FieldValType, NoObjectModel, ObjectModel};
