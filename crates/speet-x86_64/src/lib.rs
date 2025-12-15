//! Minimal x86_64 to WebAssembly recompiler
//!
//! Supports a small subset of integer instructions for demonstration purposes.
#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use wax_core::build::InstructionSink;
use yecta::{EscapeTag, Pool, Reactor, TableIdx, TypeIdx};
pub mod direct;
/// Simple x86_64 recompiler for integer ops
pub struct X86Recompiler<E, F: InstructionSink<E>> {
    reactor: Reactor<E, F>,
    pool: Pool,
    escape_tag: Option<EscapeTag>,
    base_rip: u64,
    hints: Vec<u8>,
}

impl<E, F: InstructionSink<E>> X86Recompiler<E, F> {
    pub fn base_func_offset(&self) -> u32 {
        self.reactor.base_func_offset()
    }

    pub fn set_base_func_offset(&mut self, offset: u32) {
        self.reactor.set_base_func_offset(offset);
    }
    pub fn new() -> Self {
        Self::new_with_base_rip(0)
    }

    pub fn new_with_base_rip(base_rip: u64) -> Self {
        Self {
            reactor: Reactor::default(),
            pool: Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            escape_tag: None,
            base_rip,
            hints: Vec::new(),
        }
    }
}
