//! Ingest ABI description files into a structured model for thin-runtime
//! redirect stub generation.
//!
//! Phase 1 (this crate): parse BridgeSupport XML → [`AbiSpec`]. Phase 2
//! (`speet-abi-codegen`) emits checked-in stub code; see
//! `docs/future/abi-spec-redirects.md`.

mod bridgesupport;
mod model;

pub use bridgesupport::{parse_bridgesupport, BridgeSupportError};
pub use model::{AbiArg, AbiFunction, AbiSpec, AbiType, AbiValueKind};
