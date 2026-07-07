//! Generate checked-in ABI redirect stub Rust from BridgeSupport specs.

mod generate;

pub use generate::{generate, CodegenConfig, GeneratedFile, StubArch};
