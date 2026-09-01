//! Shared test support for `speet-dex` integration tests: a toolchain-free
//! minimal DEX container encoder, a hand-rolled DEX instruction assembler,
//! and a `Reactor`/`wasmi`-based translate-and-run harness.

pub mod dex_builder;
pub mod harness;
pub mod insn;
