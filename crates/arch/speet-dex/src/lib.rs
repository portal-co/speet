//! # DEX to WebAssembly Recompiler
//!
//! This crate provides a DEX (Dalvik Executable) to WebAssembly static recompiler
//! that translates DEX bytecode to WebAssembly using the yecta control flow library.
//!
//! ## Architecture
//!
//! The recompiler uses a register mapping approach where DEX registers are mapped
//! to WebAssembly local variables. All instructions from all methods are flattened
//! into a single sequence for efficient compilation.
//!
//! ## Usage
//!
//! ```ignore
//! use speet_dex::DexRecompiler;
//! use std::fs;
//!
//! // Load DEX file
//! let dex_data = fs::read("classes.dex")?;
//! let mut recompiler = DexRecompiler::new();
//!
//! // Parse and compile
//! recompiler.parse_dex(&dex_data)?;
//! // Lowering via yecta would go here
//! ```

#![no_std]
extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use wax_core::build::InstructionSink;

use dex::{Dex, DexReader, code::ExceptionType, jtype::Type, string::DexString};
use wasm_encoder::{Instruction, ValType};
use yecta::{EscapeTag, FuncIdx, LocalPool, LocalPoolBackend, Pool, Reactor, TableIdx, TypeIdx};

// Re-export shared abstractions
pub use speet_memory::{CallbackContext, MapperCallback};
pub use speet_ordering::{AtomicOpts, MemOrder, RmwOp, RmwWidth};
use speet_traps::{
    FunctionLayout, InstructionInfo, InstructionTrap, JumpInfo, JumpKind, JumpTrap, TrapAction,
    TrapConfig,
    insn::{ArchTag, InsnClass},
};

/// Flat representation of a DEX method
#[derive(Debug, Clone)]
pub struct FlatMethod {
    /// Method index in the DEX file
    pub idx: u64,
    /// Class index
    pub class_idx: u32,
    /// Method name index
    pub name: DexString,
    pub params: Vec<Type>,
    pub returns: Type,
    /// Register count
    pub registers_size: u16,
    /// Input register count
    pub ins_size: u16,
    /// Output register count
    pub outs_size: u16,
    /// Try/catch blocks
    pub tries: Vec<TryBlock>,
    /// Handlers
    pub handlers: Vec<Handler>,
    /// Instructions
    pub instructions: Vec<u16>,
    pub base: u64, // Base offset for function indices
}

/// Try/catch block information
#[derive(Debug, Clone)]
pub struct TryBlock {
    pub start_addr: u64,
    pub end_addr: u64,
    pub handlers: Vec<Handler>,
}

/// Exception handler
#[derive(Debug, Clone)]
pub struct Handler {
    pub addr: u64,
    pub type_idx: ExceptionType,
}

/// DEX Recompiler
pub struct DexRecompiler<
    'cb,
    'ctx,
    Context,
    E,
    F: InstructionSink<Context, E>,
    P: yecta::LocalPoolBackend = yecta::LocalPool,
> {
    reactor: Reactor<Context, E, F, P>,
    pool: Pool,
    escape_tag: Option<EscapeTag>,
    /// Base offset for function indices
    base_func_offset: u32,
    /// Flattened methods
    methods: Vec<FlatMethod>,
    /// Total instruction count
    total_instructions: usize,
    /// Memory ordering mode
    mem_order: MemOrder,
    /// Atomic options
    atomic_opts: AtomicOpts,
    /// Trap configuration
    traps: TrapConfig<'cb, 'ctx, Context, E, Reactor<Context, E, F, P>>,
    /// Total parameters
    total_params: u32,
}

impl<'cb, 'ctx, Context, E, F, P> DexRecompiler<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: yecta::LocalPoolBackend,
{
    /// Create a new DEX recompiler
    pub fn new() -> Self
    where
        P: Default,
    {
        Self {
            reactor: Reactor::default(),
            pool: Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            escape_tag: None,
            base_func_offset: 0,
            methods: Vec::new(),
            total_instructions: 0,
            mem_order: MemOrder::Strong,
            atomic_opts: AtomicOpts::NONE,
            traps: TrapConfig::new(),
            total_params: Self::BASE_PARAMS,
        }
    }

    /// Parse DEX file data into flat representation
    pub fn parse_dex(&mut self, data: &[u8]) -> Result<(), dex::Error> {
        let dex_file = DexReader::from_vec(data)?;

        // Clear existing data
        self.methods.clear();
        self.total_instructions = 0;

        // Process each class
        for class_def_result in dex_file.classes() {
            let class_def: dex::class::Class = class_def_result?;

            // Process each method in the class
            for method in class_def.methods() {
                let code: &dex::code::CodeItem = match method.code() {
                    Some(code) => code,
                    None => continue, // Abstract or native method
                };

                let mut flat_method = FlatMethod {
                    idx: method.id(),
                    class_idx: class_def.id(),
                    name: method.name().clone(),
                    params: method.params().clone(),
                    returns: method.return_type().clone(),
                    registers_size: code.registers_size(),
                    ins_size: code.ins_size(),
                    outs_size: code.outs_size(),
                    tries: Vec::new(),
                    handlers: Vec::new(),
                    instructions: code.insns().clone(),
                    base: self.total_instructions as u64,
                };

                // TODO: Parse instructions from insns (Vec<ushort>)
                // For now, skip instruction parsing

                // TODO: Process try/catch blocks
                for try_item in code.tries().iter() {
                    flat_method.tries.push(TryBlock {
                        start_addr: try_item.start_addr() as u64 + flat_method.base as u64,
                        end_addr: try_item.insn_count() as u64
                            + flat_method.base as u64
                            + try_item.start_addr() as u64,
                        handlers: try_item
                            .catch_handlers()
                            .iter()
                            .map(|handler| Handler {
                                addr: handler.addr() as u64 + flat_method.base as u64,
                                type_idx: handler.exception().clone(),
                            })
                            .collect(),
                    });
                }

                self.total_instructions += flat_method.instructions.len();
                self.methods.push(flat_method);
            }
        }

        Ok(())
    }

    /// Get the flattened methods
    pub fn methods(&self) -> &[FlatMethod] {
        &self.methods
    }

    /// Get total instruction count
    pub fn total_instructions(&self) -> usize {
        self.total_instructions
    }

    const BASE_PARAMS: u32 = 0; // TODO: Define based on architecture
}

impl<'cb, 'ctx, Context, E, F, P> DexRecompiler<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: yecta::LocalPoolBackend + Default,
{
    /// Create with custom configuration
    pub fn new_with_config(base_func_offset: u32) -> Self {
        Self {
            reactor: Reactor::with_base_func_offset(base_func_offset),
            base_func_offset,
            ..Self::new()
        }
    }
}
