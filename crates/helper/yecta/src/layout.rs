//! Re-exports [`LocalLayout`], [`LocalSlot`], and [`Mark`] from the
//! [`wasm_layout`] crate.
//!
//! The implementation lives in `wasm-layout` so that crates that only need
//! local-variable layout management (e.g. `speet-wasm`) don't have to pull in
//! the full yecta reactor.

pub use wasm_layout::{
    CellIdx, CellRegistry, CellSignature, FuncSignature, LocalAllocator, LocalDeclarator,
    LocalLayout, LocalSlot, Mark,
};
