//! # speet-powerpc
//!
//! PowerPC to WebAssembly recompiler (stub).
//!
//! This crate is a placeholder for a future PowerPC architecture back-end
//! in the speet static-recompiler framework. It currently contains no
//! translation logic; the package exists so that downstream workspace
//! tooling can reserve the crate name and dependency slot.
//!
//! ## Planned scope
//! Once implemented this crate will expose a `PowerPCRecompiler` type
//! analogous to [`speet_x86_64::X86Recompiler`] and
//! [`speet_riscv::RiscVRecompiler`], translating PowerPC machine code to
//! WebAssembly via the [`yecta`] reactor.
#![no_std]
extern crate alloc;
