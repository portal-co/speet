use speet_link_core::FedContext;
use yecta::Fed;
use crate::*;
use rv_asm::AmoOp;

#[cfg(feature = "logging")]
macro_rules! rlog {
    ($($arg:tt)*) => { log::debug!($($arg)*) };
}
#[cfg(not(feature = "logging"))]
macro_rules! rlog {
    ($($arg:tt)*) => {};
}
use speet_ordering::{
    RmwOp, RmwWidth, emit_fence, emit_load, emit_lr, emit_rmw, emit_sc, emit_store,
};

/// Snippet for setting expected_ra to a constant return address in speculative calls
struct ExpectedRaSnippet {
    return_addr: u64,
    enable_rv64: bool,
}

impl<Context, E> wax_core::build::InstructionSource<Context, E> for ExpectedRaSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        if self.enable_rv64 {
            sink.instruction(
                ctx,
                &wasm_encoder::Instruction::I64Const(self.return_addr as i64),
            )?;
        } else {
            sink.instruction(
                ctx,
                &wasm_encoder::Instruction::I32Const(self.return_addr as i32),
            )?;
        }
        Ok(())
    }
}

impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for ExpectedRaSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        if self.enable_rv64 {
            sink.instruction(
                ctx,
                &wasm_encoder::Instruction::I64Const(self.return_addr as i64),
            )?;
        } else {
            sink.instruction(
                ctx,
                &wasm_encoder::Instruction::I32Const(self.return_addr as i32),
            )?;
        }
        Ok(())
    }
}

impl<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>>
    RiscVRecompiler<'cb, 'ctx, Context, E, F>
{
    /// Helper to translate load instructions
    pub(crate) fn translate_load<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        base: Reg,
        offset: Imm,
        dest: Reg,
        op: LoadOp,
    ) -> Result<(), E> {
        if dest.0 == 0 {
            return Ok(()); // x0 is hardwired to zero
        }

        // Compute address: base + offset
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(ctx, rctx, tail_idx, offset)?;

        // Add instruction depends on whether we're using memory64 and RV64
        if self.enable_rv64 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            // If not using memory64, wrap to 32-bit address
            if !self.use_memory64 {
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
            }
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::I32Add)?;
            // RV32 + memory64: extend i32 address to i64.
            if self.use_memory64 {
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
            }
        }

        // Apply address mapping if provided (for paging support)
        if let Some(mapper) = self.mapper_callback.as_mut() {
            let mut fed = FedContext::new(rctx, tail_idx);
            let mut callback_ctx = CallbackContext::new(&mut fed);
            mapper.call(ctx, &mut callback_ctx)?;
        }

        // Save the effective address into the load-addr scratch local so that
        // emit_load can compare it against any pending lazy store addresses.
        let load_addr = self.load_addr_scratch_local(rctx.layout());
        let addr_type = self.addr_val_type();
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(load_addr))?;

        // Load from memory
        match op {
            LoadOp::I8 => {
                if self.enable_rv64 && self.use_memory64 {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I64Load8S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I32Load8S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    // If RV64 but not memory64, extend to i64
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U8 => {
                if self.enable_rv64 && self.use_memory64 {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I64Load8U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I32Load8U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    }
                }
            }
            LoadOp::I16 => {
                if self.enable_rv64 && self.use_memory64 {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I64Load16S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I32Load16S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U16 => {
                if self.enable_rv64 && self.use_memory64 {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I64Load16U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I32Load16U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    }
                }
            }
            LoadOp::I32 => {
                if self.enable_rv64 && self.use_memory64 {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I64Load32S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U32 => {
                // RV64 LWU instruction - load word unsigned (zero-extended)
                if self.use_memory64 {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I64Load32U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        addr_type,
                        self.atomic_opts,
                        Instruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                }
            }
            LoadOp::I64 => {
                // RV64 LD instruction - load double-word
                emit_load(
                    ctx,
                    rctx,
                    load_addr,
                    addr_type,
                    self.atomic_opts,
                    Instruction::I64Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }),
                    tail_idx,
                )?;
            }
        }

        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(dest)))?;
        Ok(())
    }

    /// Helper to translate store instructions
    pub(crate) fn translate_store<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        base: Reg,
        offset: Imm,
        src: Reg,
        op: StoreOp,
    ) -> Result<(), E> {
        let addr_type = self.addr_val_type();
        // Compute address: base + offset
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(ctx, rctx, tail_idx, offset)?;

        // Add instruction depends on whether we're using memory64 and RV64
        if self.enable_rv64 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            // If not using memory64, wrap to 32-bit address
            if !self.use_memory64 {
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
            }
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::I32Add)?;
            // RV32 + memory64: extend i32 address to i64.
            if self.use_memory64 {
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
            }
        }

        // Apply address mapping if provided (for paging support)
        if let Some(mapper) = self.mapper_callback.as_mut() {
            let mut fed = FedContext::new(rctx, tail_idx);
            let mut callback_ctx = CallbackContext::new(&mut fed);
            mapper.call(ctx, &mut callback_ctx)?;
        }

        // Load value to store
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(src)))?;

        // If RV64 but not memory64, need to wrap i64 value to i32 for 32-bit stores
        let need_wrap = self.enable_rv64 && !self.use_memory64 && !matches!(op, StoreOp::I64);
        if need_wrap {
            rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
        }

        // Store to memory
        match op {
            StoreOp::I8 => {
                if self.enable_rv64 && self.use_memory64 {
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        addr_type,
                        Instruction::I64Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        addr_type,
                        Instruction::I32Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                }
            }
            StoreOp::I16 => {
                if self.enable_rv64 && self.use_memory64 {
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        addr_type,
                        Instruction::I64Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        addr_type,
                        Instruction::I32Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                }
            }
            StoreOp::I32 => {
                if self.enable_rv64 && self.use_memory64 {
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        addr_type,
                        Instruction::I64Store32(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        addr_type,
                        Instruction::I32Store(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                }
            }
            StoreOp::I64 => {
                // RV64 SD instruction - store double-word
                emit_store(
                    ctx,
                    rctx,
                    self.mem_order,
                    self.atomic_opts,
                    addr_type,
                    Instruction::I64Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }),
                    tail_idx,
                )?;
            }
        }

        Ok(())
    }

    /// Helper to translate floating-point load instructions
    pub(crate) fn translate_fload<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        base: Reg,
        offset: Imm,
        dest: FReg,
        op: FLoadOp,
    ) -> Result<(), E> {
        // Compute address: base + offset
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(ctx, rctx, tail_idx, offset)?;
        if self.enable_rv64 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            if !self.use_memory64 {
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
            }
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::I32Add)?;
            if self.use_memory64 {
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
            }
        }

        // Apply address mapping if provided (for paging support)
        if let Some(mapper) = self.mapper_callback.as_mut() {
            let mut fed = FedContext::new(rctx, tail_idx);
            let mut callback_ctx = CallbackContext::new(&mut fed);
            mapper.call(ctx, &mut callback_ctx)?;
        }

        // Save the effective address for alias checks.
        let load_addr = self.load_addr_scratch_local(rctx.layout());
        let addr_type = self.addr_val_type();
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(load_addr))?;

        // Load from memory
        match op {
            FLoadOp::F32 => {
                emit_load(
                    ctx,
                    rctx,
                    load_addr,
                    addr_type,
                    self.atomic_opts,
                    Instruction::F32Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }),
                    tail_idx,
                )?;
                rctx.feed(ctx, tail_idx, &Instruction::F64PromoteF32)?;
            }
            FLoadOp::F64 => {
                emit_load(
                    ctx,
                    rctx,
                    load_addr,
                    addr_type,
                    self.atomic_opts,
                    Instruction::F64Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }),
                    tail_idx,
                )?;
            }
        }

        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(dest)))?;
        Ok(())
    }

    /// Helper to translate floating-point store instructions
    pub(crate) fn translate_fstore<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        base: Reg,
        offset: Imm,
        src: FReg,
        op: FStoreOp,
    ) -> Result<(), E> {
        let addr_type = self.addr_val_type();
        // Compute address: base + offset
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(ctx, rctx, tail_idx, offset)?;
        if self.enable_rv64 {
            rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
            if !self.use_memory64 {
                rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
            }
        } else {
            rctx.feed(ctx, tail_idx, &Instruction::I32Add)?;
            if self.use_memory64 {
                rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
            }
        }

        // Apply address mapping if provided (for paging support)
        if let Some(mapper) = self.mapper_callback.as_mut() {
            let mut fed = FedContext::new(rctx, tail_idx);
            let mut callback_ctx = CallbackContext::new(&mut fed);
            mapper.call(ctx, &mut callback_ctx)?;
        }

        // Load value to store
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(src)))?;

        // Store to memory
        match op {
            FStoreOp::F32 => {
                rctx.feed(ctx, tail_idx, &Instruction::F32DemoteF64)?;
                emit_store(
                    ctx,
                    rctx,
                    self.mem_order,
                    self.atomic_opts,
                    addr_type,
                    Instruction::F32Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }),
                    tail_idx,
                )?;
            }
            FStoreOp::F64 => {
                emit_store(
                    ctx,
                    rctx,
                    self.mem_order,
                    self.atomic_opts,
                    addr_type,
                    Instruction::F64Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }),
                    tail_idx,
                )?;
            }
        }

        Ok(())
    }

    /// Helper to emit sign-injection for single-precision floats
    pub(crate) fn emit_fsgnj_s<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        dest: FReg,
        src1: FReg,
        src2: FReg,
        op: FsgnjOp,
    ) -> Result<(), E> {
        // Sign injection uses bit manipulation on the float representation
        // Get magnitude from src1, sign from src2 (possibly modified)

        // Convert src1 to i32 to manipulate bits
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(src1)))?;
        self.unbox_f32(ctx, rctx, tail_idx)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32ReinterpretF32)?;

        // Mask to keep only magnitude (clear sign bit): 0x7FFFFFFF
        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0x7FFFFFFF))?;
        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;

        // Get sign bit from src2
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(src2)))?;
        self.unbox_f32(ctx, rctx, tail_idx)?;
        rctx.feed(ctx, tail_idx, &Instruction::I32ReinterpretF32)?;

        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly: mask with 0x80000000
                rctx.feed(ctx, tail_idx, &Instruction::I32Const(0x80000000_u32 as i32))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                rctx.feed(ctx, tail_idx, &Instruction::I32Const(0x80000000_u32 as i32))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Const(0x80000000_u32 as i32))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?; // Flip the sign bit
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits of src1 and src2
                // Need original src1 sign
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32ReinterpretF32)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?;
                rctx.feed(ctx, tail_idx, &Instruction::I32Const(0x80000000_u32 as i32))?;
                rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
            }
        }

        // Combine magnitude and sign
        rctx.feed(ctx, tail_idx, &Instruction::I32Or)?;
        rctx.feed(ctx, tail_idx, &Instruction::F32ReinterpretI32)?;
        self.nan_box_f32(ctx, rctx, tail_idx)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(dest)))?;

        Ok(())
    }

    /// Helper to emit sign-injection for double-precision floats
    pub(crate) fn emit_fsgnj_d<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        dest: FReg,
        src1: FReg,
        src2: FReg,
        op: FsgnjOp,
    ) -> Result<(), E> {
        // Similar to single-precision but using i64
        // Convert src1 to i64 to manipulate bits
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(src1)))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ReinterpretF64)?;

        // Mask to keep only magnitude (clear sign bit)
        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0x7FFFFFFFFFFFFFFF))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;

        // Get sign bit from src2
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(src2)))?;
        rctx.feed(ctx, tail_idx, &Instruction::I64ReinterpretF64)?;

        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64ReinterpretF64)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?;
                rctx.feed(ctx, tail_idx, &Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
            }
        }

        // Combine magnitude and sign
        rctx.feed(ctx, tail_idx, &Instruction::I64Or)?;
        rctx.feed(ctx, tail_idx, &Instruction::F64ReinterpretI64)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(dest)))?;

        Ok(())
    }

    /// Translate a block of RISC-V bytecode starting at the given address
    ///
    /// This method decodes and translates multiple instructions, creating separate
    /// functions for each instruction and linking them with jumps.
    ///
    /// # Arguments
    /// * `bytes` - The bytecode to translate
    /// * `start_pc` - The starting program counter address
    /// * `xlen` - The XLEN mode (RV32 or RV64)
    ///
    /// # Returns
    /// The number of bytes successfully translated
    pub fn translate_bytes<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        bytes: &[u8],
        start_pc: u32,
        xlen: Xlen,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> RC::FnType + '_),
    ) -> Result<usize, ()> {
        // Pre-scan: set total_func_count so pc_to_func_idx can clamp jump targets that
        // land outside the binary entirely.  The legacy formula maps each 2-byte-aligned
        // address to a slot, so there are bytes.len()/2 valid slots (0 .. len/2 - 1).
        if self.total_func_count.is_none() && self.slot_assigner.is_none() {
            self.total_func_count = Some((bytes.len() / 2) as u32);
        }

        let mut offset = 0;

        while offset < bytes.len() {
            // Need at least 2 bytes for a compressed instruction
            if offset + 1 >= bytes.len() {
                break;
            }

            // Read instruction bytes (little-endian)
            let inst_word = if offset + 3 < bytes.len() {
                u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ])
            } else {
                // Might be a compressed instruction at the end
                u32::from_le_bytes([bytes[offset], bytes[offset + 1], 0, 0])
            };

            let pc = start_pc + offset as u32;

            // Decode the instruction. On failure, create a trap slot to preserve
            // the 1:1 mapping between 2-byte-aligned slots and reactor functions,
            // then advance by 2 bytes to the next aligned slot.
            let (inst, is_compressed) = match Inst::decode(inst_word, xlen) {
                Ok(result) => result,
                Err(_) => {
                    rlog!("translate_bytes: junk slot at pc={:#x}, emitting unreachable", pc);
                    if let Ok(tail_idx) = self.init_function(ctx, rctx, pc, 1, 0, 0, f) {
                        let _ = rctx.feed(ctx, tail_idx, &Instruction::Unreachable);
                    }
                    offset += 2;
                    continue;
                }
            };

            // Slot gate: skip omitted instructions entirely (true slot omission).
            let inst_byte_len = match is_compressed {
                IsCompressed::Yes => 2,
                IsCompressed::No => 4,
            };
            if let Some(gate) = &self.slot_assigner {
                if gate.slot_for_pc(pc as u64).is_none() {
                    offset += inst_byte_len;
                    continue;
                }
            }

            // Translate the instruction
            if let Err(_) = self.translate_instruction(ctx, rctx, &inst, pc, is_compressed, f) {
                break;
            }
            // Advance by instruction size. For 4-byte instructions without a slot assigner,
            // the per-2-byte slot model requires a junk/unreachable function for the
            // intermediate 2-byte slot (offset+2) so that slot indices stay aligned with
            // reactor function indices.
            match is_compressed {
                IsCompressed::Yes => { offset += 2; }
                IsCompressed::No => {
                    if self.slot_assigner.is_none() && offset + 2 < bytes.len() {
                        let junk_pc = start_pc + offset as u32 + 2;
                        rlog!("translate_bytes: intermediate junk slot at pc={:#x}", junk_pc);
                        if let Ok(tail_idx) = self.init_function(ctx, rctx, junk_pc, 1, 0, 0, f) {
                            let _ = rctx.feed(ctx, tail_idx, &Instruction::Unreachable);
                        }
                    }
                    offset += 4;
                }
            }
        }

        // Seal any functions that were not explicitly terminated by a branch
        // or jump — the trailing instructions of a translated region have no
        // explicit successor and must be closed with unreachable; end.
        let _ = rctx.seal_remaining(ctx);

        Ok(offset)
    }

    /// Translate a single RISC-V instruction to WebAssembly
    ///
    /// This creates a separate function for the instruction at the given PC and
    /// handles jumps to other instructions using the yecta reactor's jump APIs.
    ///
    /// # Arguments
    /// * `inst` - The decoded RISC-V instruction
    /// * `pc` - Current program counter value
    /// * `is_compressed` - Whether the instruction is compressed (2 bytes vs 4 bytes)
    pub fn translate_instruction<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        inst: &Inst,
        pc: u32,
        is_compressed: IsCompressed,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> RC::FnType + '_),
    ) -> Result<(), E> {
        let inst_len = match is_compressed {
            IsCompressed::Yes => 1,
            IsCompressed::No => 2,
        };
        let tail_idx = self.init_function(ctx, rctx, pc, inst_len, 8, 2, f)?;
        rlog!("translate_instruction pc={:#x} tail_idx={} inst={:?}", pc, tail_idx, inst);
        self.emit_int_const(ctx, rctx, tail_idx, pc as i32)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::pc_local()))?;
        // x0 is hardwired to zero; assert the invariant so the constant-fold
        // layer can propagate it.
        self.emit_int_const(ctx, rctx, tail_idx, 0)?;
        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(0))?;

        // Fire instruction trap (if installed).
        let insn_info = InstructionInfo {
            pc: pc as u64,
            len: inst_len * 2, // inst_len is in 2-byte units; convert to bytes
            arch: ArchTag::RiscV,
            class: Self::classify_insn(inst),
        };
        if rctx.on_instruction(&insn_info, ctx)? == TrapAction::Skip
        {
            return Ok(());
        }

        match inst {
            // RV32I Base Integer Instruction Set

            // Lui: Load Upper Immediate
            // RISC-V Specification Quote:
            // "LUI (load upper immediate) is used to build 32-bit constants and uses the U-type format.
            // LUI places the 32-bit U-immediate value into the destination register rd, filling in the
            // lowest 12 bits with zeros."
            Inst::Lui { uimm, dest } => {
                self.emit_imm(ctx, rctx, tail_idx, *uimm)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
            }

            // Auipc: Add Upper Immediate to PC
            // RISC-V Specification Quote:
            // "AUIPC (add upper immediate to pc) is used to build pc-relative addresses and uses the
            // U-type format. AUIPC forms a 32-bit offset from the U-immediate, filling in the lowest
            // 12 bits with zeros, adds this offset to the address of the AUIPC instruction, then places
            // the result in register rd."
            Inst::Auipc { uimm, dest } => {
                if self.enable_rv64 {
                    // RV64: Use full 64-bit PC
                    rctx.feed(ctx, tail_idx, &Instruction::I64Const(pc as i64))?;
                    self.emit_imm(ctx, rctx, tail_idx, *uimm)?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                } else {
                    // RV32: Use 32-bit PC
                    rctx.feed(ctx, tail_idx, &Instruction::I32Const(pc as i32))?;
                    self.emit_imm(ctx, rctx, tail_idx, *uimm)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32Add)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Jal: Jump And Link
            // RISC-V Specification Quote:
            // "The jump and link (JAL) instruction uses the J-type format, where the J-immediate encodes
            // a signed offset in multiples of 2 bytes. The offset is sign-extended and added to the
            // address of the jump instruction to form the jump target address."
            Inst::Jal { offset, dest } => {
                let return_addr = pc as u64 + (inst_len * 2) as u64;
                let target_pc = if self.enable_rv64 {
                    (pc as i64).wrapping_add(offset.as_i32() as i64) as u64
                } else {
                    (pc as i32).wrapping_add(offset.as_i32()) as u32 as u64
                };

                // Check if this is an ABI-compliant call (dest == x1/ra) and speculative calls are enabled
                let use_speculative = self.enable_speculative_calls
                    && dest.0 == 1  // x1 is the return address register (ra)
                    && rctx.escape_tag().is_some();

                if use_speculative {
                    // Speculative call lowering: emit a native WASM call
                    // See SPECULATIVE_CALLS.md and set_speculative_calls() docs for the full pattern
                    let escape_tag = rctx.escape_tag().unwrap();
                    let Some(target_func) = self.pc_to_func_idx(target_pc) else {
                        // Target is omitted — unreachable in a correctly analyzed binary.
                        rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                        return Ok(());
                    };

                    // Save return address to ra (x1)
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32Const(return_addr as i32))?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;

                    // Create a snippet that sets expected_ra to the return address
                    let expected_ra_snippet = ExpectedRaSnippet {
                        return_addr,
                        enable_rv64: self.enable_rv64,
                    };

                    // Use fixups to set expected_ra (local 65) only for this call
                    let params = yecta::JumpCallParams::call(
                        target_func,
                        rctx.locals_mark().total_locals,
                        escape_tag,
                        rctx.pool(),
                    )
                    .with_fixup(Self::expected_ra_local(), &expected_ra_snippet);

                    // Emit the speculative call using yecta's ji_with_params API
                    // This wraps the call in a try-catch block and uses fixups mechanism
                    rctx.ji_with_params(ctx, tail_idx, params)?;

                    // After the call returns (via exception catch), execution continues here
                    // No manual validation needed - the ret() in the callee threw the state
                    return Ok(());
                } else {
                    // Non-speculative path: original jump-based implementation
                    if dest.0 != 0 {
                        if self.enable_rv64 {
                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
                        } else {
                            rctx.feed(ctx, tail_idx, &Instruction::I32Const(return_addr as i32))?;
                        }
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                    // Jump trap: Jal rd!=x0,ra → Call; others → DirectJump
                    let jal_kind = if dest.0 == 1 {
                        JumpKind::Call
                    } else {
                        JumpKind::DirectJump
                    };
                    let jal_info = JumpInfo::direct(pc as u64, target_pc, jal_kind);
                    if                         rctx.on_jump(&jal_info, ctx)?
                        == TrapAction::Skip
                    {
                        return Ok(());
                    }
                    self.jump_to_pc(ctx, rctx, tail_idx, target_pc, rctx.locals_mark().total_locals)?;
                    return Ok(());
                }
            }

            // Jalr: Jump And Link Register
            // RISC-V Specification Quote:
            // "The indirect jump instruction JALR (jump and link register) uses the I-type encoding.
            // The target address is obtained by adding the sign-extended 12-bit I-immediate to the
            // register rs1, then setting the least-significant bit of the result to zero."
            Inst::Jalr { offset, base, dest } => {
                let return_addr = pc as u64 + (inst_len * 2) as u64;

                // Check if this is an ABI-compliant return: jalr x0, ra, 0
                // (jump to ra with no link register saved)
                let is_abi_return = self.enable_speculative_calls
                    && dest.0 == 0       // x0 = no link (return, not call)
                    && base.0 == 1       // ra = return address register
                    && offset.as_i32() == 0
                    && rctx.escape_tag().is_some();

                if is_abi_return {
                    // Jump trap: jalr x0, ra, 0 → Return (indirect)
                    // Tee ra into load_addr_scratch_local so the trap can inspect it.
                    let scratch = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(Reg(1))))?;
                    // rv32+memory64: scratch is i64, extend before tee.
                    if !self.enable_rv64 && self.use_memory64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalTee(scratch))?;
                    rctx.feed(ctx, tail_idx, &Instruction::Drop)?; // balance the stack
                    let jalr_ret_info = JumpInfo::indirect(pc as u64, scratch, JumpKind::Return);
                    if                         rctx.on_jump(&jalr_ret_info, ctx)?
                        == TrapAction::Skip
                    {
                        return Ok(());
                    }

                    // ABI-compliant return: check if ra matches expected_ra
                    // If they match, use regular WASM Return; if not, throw escape tag
                    let escape_tag = rctx.escape_tag().unwrap();

                    // Load ra (current return address)
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(Reg(1))))?;

                    // Load expected_ra
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::expected_ra_local()))?;

                    // Compare them
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64Eq)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32Eq)?;
                    }

                    // If equal (ABI-compliant), use regular return; else throw exception
                    rctx.feed(ctx, tail_idx, &Instruction::If(wasm_encoder::BlockType::Empty))?;

                    // ABI-compliant return - use WASM Return
                    rctx.feed(ctx, tail_idx, &Instruction::Return)?;

                    rctx.feed(ctx, tail_idx, &Instruction::Else)?;

                    // Non-ABI-compliant return - throw escape tag with all register state
                    rctx.ret(ctx, tail_idx, rctx.locals_mark().total_locals, escape_tag)?;;

                    rctx.feed(ctx, tail_idx, &Instruction::End)?;
                    return Ok(());
                }

                // Check if this is an ABI-compliant call (dest == x1/ra) and speculative calls are enabled
                let use_speculative = self.enable_speculative_calls
                    && dest.0 == 1  // x1 is the return address register (ra)
                    && rctx.escape_tag().is_some();

                if use_speculative {
                    // Speculative call lowering for indirect calls
                    // Since JALR is indirect, we use dynamic dispatch through the pool table
                    let escape_tag = rctx.escape_tag().unwrap();

                    // Save return address to ra (x1)
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32Const(return_addr as i32))?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;

                    // Create a snippet that sets expected_ra to the return address
                    let expected_ra_snippet = ExpectedRaSnippet {
                        return_addr,
                        enable_rv64: self.enable_rv64,
                    };

                    // Compute target address: (base + offset) & ~1
                    // We need to create a snippet that computes the function index
                    struct JalrTargetSnippet {
                        base_local: u32,
                        offset: i32,
                        enable_rv64: bool,
                        base_pc: u64,
                    }

                    impl<Context, E> wax_core::build::InstructionSource<Context, E> for JalrTargetSnippet {
                        fn emit_instruction(
                            &self,
                            ctx: &mut Context,
                            sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
                        ) -> Result<(), E> {
                            // Compute: ((base + offset) & ~1 - base_pc) / 2
                            // This gives us the function index from the PC
                            sink.instruction(ctx, &Instruction::LocalGet(self.base_local))?;
                            if self.enable_rv64 {
                                sink.instruction(ctx, &Instruction::I64Const(self.offset as i64))?;
                                sink.instruction(ctx, &Instruction::I64Add)?;
                                sink.instruction(
                                    ctx,
                                    &Instruction::I64Const(0xFFFFFFFFFFFFFFFE_u64 as i64),
                                )?;
                                sink.instruction(ctx, &Instruction::I64And)?;
                                sink.instruction(ctx, &Instruction::I64Const(self.base_pc as i64))?;
                                sink.instruction(ctx, &Instruction::I64Sub)?;
                                sink.instruction(ctx, &Instruction::I64Const(1))?;
                                sink.instruction(ctx, &Instruction::I64ShrU)?;
                                sink.instruction(ctx, &Instruction::I32WrapI64)?;
                            } else {
                                sink.instruction(ctx, &Instruction::I32Const(self.offset))?;
                                sink.instruction(ctx, &Instruction::I32Add)?;
                                sink.instruction(
                                    ctx,
                                    &Instruction::I32Const(0xFFFFFFFE_u32 as i32),
                                )?;
                                sink.instruction(ctx, &Instruction::I32And)?;
                                sink.instruction(ctx, &Instruction::I32Const(self.base_pc as i32))?;
                                sink.instruction(ctx, &Instruction::I32Sub)?;
                                sink.instruction(ctx, &Instruction::I32Const(1))?;
                                sink.instruction(ctx, &Instruction::I32ShrU)?;
                            }
                            Ok(())
                        }
                    }

                    impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for JalrTargetSnippet {
                        fn emit(
                            &self,
                            ctx: &mut Context,
                            sink: &mut (
                                     dyn wax_core::build::InstructionOperatorSink<Context, E> + '_
                                 ),
                        ) -> Result<(), E> {
                            // Compute: ((base + offset) & ~1 - base_pc) / 2
                            // This gives us the function index from the PC
                            sink.instruction(ctx, &Instruction::LocalGet(self.base_local))?;
                            if self.enable_rv64 {
                                sink.instruction(ctx, &Instruction::I64Const(self.offset as i64))?;
                                sink.instruction(ctx, &Instruction::I64Add)?;
                                sink.instruction(
                                    ctx,
                                    &Instruction::I64Const(0xFFFFFFFFFFFFFFFE_u64 as i64),
                                )?;
                                sink.instruction(ctx, &Instruction::I64And)?;
                                sink.instruction(ctx, &Instruction::I64Const(self.base_pc as i64))?;
                                sink.instruction(ctx, &Instruction::I64Sub)?;
                                sink.instruction(ctx, &Instruction::I64Const(1))?;
                                sink.instruction(ctx, &Instruction::I64ShrU)?;
                                sink.instruction(ctx, &Instruction::I32WrapI64)?;
                            } else {
                                sink.instruction(ctx, &Instruction::I32Const(self.offset))?;
                                sink.instruction(ctx, &Instruction::I32Add)?;
                                sink.instruction(
                                    ctx,
                                    &Instruction::I32Const(0xFFFFFFFE_u32 as i32),
                                )?;
                                sink.instruction(ctx, &Instruction::I32And)?;
                                sink.instruction(ctx, &Instruction::I32Const(self.base_pc as i32))?;
                                sink.instruction(ctx, &Instruction::I32Sub)?;
                                sink.instruction(ctx, &Instruction::I32Const(1))?;
                                sink.instruction(ctx, &Instruction::I32ShrU)?;
                            }
                            Ok(())
                        }
                    }

                    let target_snippet = JalrTargetSnippet {
                        base_local: Self::reg_to_local(*base),
                        offset: offset.as_i32(),
                        enable_rv64: self.enable_rv64,
                        base_pc: self.base_pc,
                    };

                    // Use fixups to set expected_ra (local 65) only for this call
                    let mut fixups = alloc::collections::BTreeMap::new();
                    fixups.insert(
                        Self::expected_ra_local(),
                        &expected_ra_snippet as &(dyn yecta::Snippet<Context, E> + '_),
                    );

                    let params = yecta::JumpCallParams {
                        params: rctx.locals_mark().total_locals,
                        fixups,
                        target: yecta::Target::Dynamic {
                            idx: &target_snippet,
                        },
                        call: Some(escape_tag),
                        pool: rctx.pool(),
                        condition: None,
                        condition_hook: None,
                    };

                    // Emit the speculative call using yecta's ji_with_params API
                    rctx.ji_with_params(ctx, tail_idx, params)?;

                    // After the call returns (via exception catch), execution continues here
                    return Ok(());
                } else {
                    // Non-speculative path: original implementation
                    if dest.0 != 0 {
                        if self.enable_rv64 {
                            rctx.feed(ctx, tail_idx, &Instruction::I64Const(return_addr as i64))?;
                        } else {
                            rctx.feed(ctx, tail_idx, &Instruction::I32Const(return_addr as i32))?;
                        }
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                    // JALR is indirect, compute target and update PC.
                    // Tee the target into load_addr_scratch_local for the jump trap.
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*base)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *offset)?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(0xFFFFFFFFFFFFFFFE_u64 as i64))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64And)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32Const(0xFFFFFFFE_u32 as i32))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32And)?;
                        // rv32+memory64: scratch is i64, so extend before tee.
                        if self.use_memory64 {
                            rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                        }
                    }
                    let scratch = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &Instruction::LocalTee(scratch))?;
                    // PC is i32 for rv32, i64 for rv64.  After the rv32+memory64
                    // extend the tee leaves i64 on the stack; wrap back to i32 for PC.
                    if !self.enable_rv64 && self.use_memory64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::pc_local()))?;
                    // Jump trap: dest==1 → IndirectCall; dest==0 without base==ra → IndirectJump
                    let jalr_kind = if dest.0 == 1 {
                        JumpKind::IndirectCall
                    } else {
                        JumpKind::IndirectJump
                    };
                    let jalr_info = JumpInfo::indirect(pc as u64, scratch, jalr_kind);
                    if                         rctx.on_jump(&jalr_info, ctx)?
                        == TrapAction::Skip
                    {
                        return Ok(());
                    }
                    // For indirect jumps, seal with unreachable as we can't statically determine target
                    rctx.seal_fn(ctx, tail_idx, &Instruction::Unreachable)?;
                    return Ok(());
                }
            }

            // Branch Instructions
            // RISC-V Specification Quote:
            // "All branch instructions use the B-type instruction format. The 12-bit B-immediate encodes
            // signed offsets in multiples of 2 bytes. The offset is sign-extended and added to the address
            // of the branch instruction to give the target address."
            Inst::Beq { offset, src1, src2 } => {
                self.translate_branch(ctx, rctx, tail_idx, *src1, *src2, *offset, pc, inst_len, BranchOp::Eq)?;
                return Ok(()); // Branch handles control flow
            }

            Inst::Bne { offset, src1, src2 } => {
                self.translate_branch(ctx, rctx, tail_idx, *src1, *src2, *offset, pc, inst_len, BranchOp::Ne)?;
                return Ok(());
            }

            Inst::Blt { offset, src1, src2 } => {
                self.translate_branch(ctx, rctx, tail_idx, *src1, *src2, *offset, pc, inst_len, BranchOp::LtS)?;
                return Ok(());
            }

            Inst::Bge { offset, src1, src2 } => {
                self.translate_branch(ctx, rctx, tail_idx, *src1, *src2, *offset, pc, inst_len, BranchOp::GeS)?;
                return Ok(());
            }

            Inst::Bltu { offset, src1, src2 } => {
                self.translate_branch(ctx, rctx, tail_idx, *src1, *src2, *offset, pc, inst_len, BranchOp::LtU)?;
                return Ok(());
            }

            Inst::Bgeu { offset, src1, src2 } => {
                self.translate_branch(ctx, rctx, tail_idx, *src1, *src2, *offset, pc, inst_len, BranchOp::GeU)?;
                return Ok(());
            }

            // Load Instructions
            // RISC-V Specification Quote:
            // "Load and store instructions transfer a value between the registers and memory.
            // Loads are encoded in the I-type format and stores are S-type."
            Inst::Lb { offset, dest, base } => {
                self.translate_load(ctx, rctx, tail_idx, *base, *offset, *dest, LoadOp::I8)?;
            }

            Inst::Lh { offset, dest, base } => {
                self.translate_load(ctx, rctx, tail_idx, *base, *offset, *dest, LoadOp::I16)?;
            }

            Inst::Lw { offset, dest, base } => {
                self.translate_load(ctx, rctx, tail_idx, *base, *offset, *dest, LoadOp::I32)?;
            }

            Inst::Lbu { offset, dest, base } => {
                self.translate_load(ctx, rctx, tail_idx, *base, *offset, *dest, LoadOp::U8)?;
            }

            Inst::Lhu { offset, dest, base } => {
                self.translate_load(ctx, rctx, tail_idx, *base, *offset, *dest, LoadOp::U16)?;
            }

            // Store Instructions
            Inst::Sb { offset, src, base } => {
                self.translate_store(ctx, rctx, tail_idx, *base, *offset, *src, StoreOp::I8)?;
            }

            Inst::Sh { offset, src, base } => {
                self.translate_store(ctx, rctx, tail_idx, *base, *offset, *src, StoreOp::I16)?;
            }

            Inst::Sw { offset, src, base } => {
                self.translate_store(ctx, rctx, tail_idx, *base, *offset, *src, StoreOp::I32)?;
            }

            // Integer Computational Instructions
            // RISC-V Specification Quote:
            // "Integer computational instructions are either encoded as register-immediate operations
            // using the I-type format or as register-register operations using the R-type format."
            Inst::Addi { imm, dest, src1 } => {
                if src1.0 == 0 && dest.0 == 0 {
                    // HINT instruction: addi x0, x0, imm
                    // This has no architectural effect since x0 is hardwired to zero.
                    // In rv-corpus, this is used to mark test case boundaries.

                    let hint_info = HintInfo {
                        pc,
                        value: imm.as_i32(),
                    };

                    // Track the HINT if tracking is enabled
                    if self.track_hints {
                        self.hints.push(hint_info);
                    }

                    // Invoke callback if set
                    if let Some(ref mut callback) = self.hint_callback {
                        let mut fed = FedContext::new(rctx, tail_idx);
            let mut callback_ctx = CallbackContext::new(&mut fed);
                        callback.call(&hint_info, ctx, &mut callback_ctx);
                    }

                    // No WebAssembly code generation needed - this is a true no-op
                } else if src1.0 == 0 {
                    // li (load immediate) pseudoinstruction
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                } else if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slti { imm, dest, src1 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32LtS)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sltiu { imm, dest, src1 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64LtU)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32LtU)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Xori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    self.emit_xor(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Ori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    self.emit_or(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Andi { imm, dest, src1 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    self.emit_and(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slli { imm, dest, src1 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    self.emit_shl(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srli { imm, dest, src1 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    self.emit_shr_u(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srai { imm, dest, src1 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                    self.emit_shr_s(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Register-Register Operations
            Inst::Add { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sub { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_sub(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sll { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shl(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slt { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32LtS)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sltu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64LtU)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32LtU)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Xor { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_xor(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srl { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shr_u(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sra { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shr_s(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Or { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_or(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::And { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_and(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Fence: Memory ordering
            // RISC-V Specification Quote:
            // "The FENCE instruction is used to order device I/O and memory accesses as viewed by
            // other RISC-V harts and external devices or coprocessors."
            Inst::Fence { .. } => {
                // Under MemOrder::Relaxed, emit_fence flushes any stores that were
                // deferred by feed_lazy, committing them in program order before the
                // next instruction.  Under MemOrder::Strong the lazy buffer is always
                // empty, so this is a guaranteed no-op with zero overhead.
                emit_fence(ctx, rctx, self.mem_order, tail_idx)?;
            }

            // System calls
            Inst::Ecall => {
                let ecall_info = EcallInfo { pc };

                // Invoke callback if set
                if let Some(ref mut callback) = self.ecall_callback {
                    let mut fed = FedContext::new(rctx, tail_idx);
            let mut callback_ctx = CallbackContext::new(&mut fed);
                    callback.call(&ecall_info, ctx, &mut callback_ctx);
                } else {
                    // Default behavior: environment call - implementation specific
                    // Would need to be handled by runtime
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::Ebreak => {
                let ebreak_info = EbreakInfo { pc };

                // Invoke callback if set
                if let Some(ref mut callback) = self.ebreak_callback {
                    let mut fed = FedContext::new(rctx, tail_idx);
            let mut callback_ctx = CallbackContext::new(&mut fed);
                    callback.call(&ebreak_info, ctx, &mut callback_ctx);
                } else {
                    // Default behavior: breakpoint - implementation specific
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            // M Extension: Integer Multiplication and Division
            // RISC-V Specification Quote:
            // "This chapter describes the standard integer multiplication and division instruction-set
            // extension, which is named 'M' and contains instructions that multiply or divide values
            // held in two integer registers."
            Inst::Mul { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_mul(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Mulh { dest, src1, src2 } => {
                // Multiply high signed-signed: returns upper bits of product
                // RV32: upper 32 bits of 64-bit product
                // RV64: upper 64 bits of 128-bit product
                if dest.0 != 0 {
                    if self.enable_rv64 {
                        // For RV64: compute high 64 bits of 128-bit signed multiplication
                        self.emit_mulh_signed(ctx, rctx, tail_idx, Self::reg_to_local(*src1), Self::reg_to_local(*src2))?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply and shift
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(32))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
            }

            Inst::Mulhsu { dest, src1, src2 } => {
                // Multiply high signed-unsigned
                // RV32: upper 32 bits of 64-bit product (src1 signed, src2 unsigned)
                // RV64: upper 64 bits of 128-bit product (src1 signed, src2 unsigned)
                if dest.0 != 0 {
                    if self.enable_rv64 {
                        // For RV64: compute high 64 bits of 128-bit signed-unsigned multiplication
                        self.emit_mulh_signed_unsigned(ctx, rctx, tail_idx, Self::reg_to_local(*src1), Self::reg_to_local(*src2))?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply with mixed sign extension
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(32))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ShrS)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
            }

            Inst::Mulhu { dest, src1, src2 } => {
                // Multiply high unsigned-unsigned
                // RV32: upper 32 bits of 64-bit product
                // RV64: upper 64 bits of 128-bit product
                if dest.0 != 0 {
                    if self.enable_rv64 {
                        // For RV64: compute high 64 bits of 128-bit unsigned multiplication
                        self.emit_mulh_unsigned(ctx, rctx, tail_idx, Self::reg_to_local(*src1), Self::reg_to_local(*src2))?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply and shift
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Const(32))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ShrU)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
            }

            Inst::Div { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64DivS)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32DivS)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Divu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64DivU)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32DivU)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Rem { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64RemS)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32RemS)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Remu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64RemU)?;
                    } else {
                        rctx.feed(ctx, tail_idx, &Instruction::I32RemU)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Floating-Point Single-Precision (F Extension)
            // RISC-V Specification Quote:
            // "This chapter describes the standard instruction-set extension for single-precision
            // floating-point, which is named 'F'"
            Inst::Flw { offset, dest, base } => {
                self.translate_fload(ctx, rctx, tail_idx, *base, *offset, *dest, FLoadOp::F32)?;
            }

            Inst::Fsw { offset, src, base } => {
                self.translate_fstore(ctx, rctx, tail_idx, *base, *offset, *src, FStoreOp::F32)?;
            }

            Inst::FaddS {
                dest, src1, src2, ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Add)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubS {
                dest, src1, src2, ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Sub)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulS {
                dest, src1, src2, ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Mul)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivS {
                dest, src1, src2, ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Div)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtS { dest, src, .. } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Sqrt)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-Point Double-Precision (D Extension)
            Inst::Fld { offset, dest, base } => {
                self.translate_fload(ctx, rctx, tail_idx, *base, *offset, *dest, FLoadOp::F64)?;
            }

            Inst::Fsd { offset, src, base } => {
                self.translate_fstore(ctx, rctx, tail_idx, *base, *offset, *src, FStoreOp::F64)?;
            }

            Inst::FaddD {
                dest, src1, src2, ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Add)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubD {
                dest, src1, src2, ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Sub)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulD {
                dest, src1, src2, ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivD {
                dest, src1, src2, ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Div)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtD { dest, src, .. } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Sqrt)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point min/max operations
            Inst::FminS { dest, src1, src2 } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Min)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxS { dest, src1, src2 } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Max)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FminD { dest, src1, src2 } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Min)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxD { dest, src1, src2 } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Max)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point comparison operations
            Inst::FeqS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::F32Eq)?;
                    if self.enable_rv64 { rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?; }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::F32Lt)?;
                    if self.enable_rv64 { rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?; }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::F32Le)?;
                    if self.enable_rv64 { rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?; }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FeqD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::F64Eq)?;
                    if self.enable_rv64 { rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?; }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::F64Lt)?;
                    if self.enable_rv64 { rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?; }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::F64Le)?;
                    if self.enable_rv64 { rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?; }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Floating-point conversion operations
            // RISC-V Specification Quote:
            // "Floating-point-to-integer and integer-to-floating-point conversion instructions
            // are encoded in the OP-FP major opcode space."
            Inst::FcvtWS { dest, src, .. } => {
                // Convert single to signed 32-bit integer; sign-extend to i64 for RV64.
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32TruncF32S)?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuS { dest, src, .. } => {
                // Convert single to unsigned 32-bit integer; zero-extend to i64 for RV64.
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32TruncF32U)?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtSW { dest, src, .. } => {
                // Convert signed 32-bit integer to single; truncate i64 reg in RV64.
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::F32ConvertI32S)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to single; truncate i64 reg in RV64.
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::F32ConvertI32U)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtWD { dest, src, .. } => {
                // Convert double to signed 32-bit integer; sign-extend to i64 for RV64.
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32TruncF64S)?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuD { dest, src, .. } => {
                // Convert double to unsigned 32-bit integer; zero-extend to i64 for RV64.
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32TruncF64U)?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtDW { dest, src, .. } => {
                // Convert signed 32-bit integer to double; truncate i64 reg in RV64.
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::F64ConvertI32S)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to double; truncate i64 reg in RV64.
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::F64ConvertI32U)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSD { dest, src, .. } => {
                // RISC-V Specification Quote:
                // "FCVT.S.D converts double-precision float to single-precision float,
                // rounding according to the dynamic rounding mode."
                // Convert double to single with proper NaN-boxing
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F32DemoteF64)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDS { dest, src, .. } => {
                // RISC-V Specification Quote:
                // "FCVT.D.S converts single-precision float to double-precision float."
                // Unbox the NaN-boxed single value, then promote to double
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64PromoteF32)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point move operations
            Inst::FmvXW { dest, src } => {
                // Move bits from float register to integer register (zero-extended in RV64).
                if dest.0 != 0 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32ReinterpretF32)?;
                    if self.enable_rv64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FmvWX { dest, src } => {
                // Move bits from integer register to float register; truncate i64 in RV64.
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::F32ReinterpretI32)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Sign-injection operations for single-precision
            // RISC-V Specification Quote:
            // "FSGNJ.S, FSGNJN.S, and FSGNJX.S produce a result that takes all bits except
            // the sign bit from rs1."
            Inst::FsgnjS { dest, src1, src2 } => {
                // Result = magnitude(src1) with sign(src2)
                // We'll use a simple implementation using bit manipulation
                self.emit_fsgnj_s(ctx, rctx, tail_idx, *dest, *src1, *src2, FsgnjOp::Sgnj)?;
            }

            Inst::FsgnjnS { dest, src1, src2 } => {
                // Result = magnitude(src1) with NOT(sign(src2))
                self.emit_fsgnj_s(ctx, rctx, tail_idx, *dest, *src1, *src2, FsgnjOp::Sgnjn)?;
            }

            Inst::FsgnjxS { dest, src1, src2 } => {
                // Result = magnitude(src1) with sign(src1) XOR sign(src2)
                self.emit_fsgnj_s(ctx, rctx, tail_idx, *dest, *src1, *src2, FsgnjOp::Sgnjx)?;
            }

            Inst::FsgnjD { dest, src1, src2 } => {
                self.emit_fsgnj_d(ctx, rctx, tail_idx, *dest, *src1, *src2, FsgnjOp::Sgnj)?;
            }

            Inst::FsgnjnD { dest, src1, src2 } => {
                self.emit_fsgnj_d(ctx, rctx, tail_idx, *dest, *src1, *src2, FsgnjOp::Sgnjn)?;
            }

            Inst::FsgnjxD { dest, src1, src2 } => {
                self.emit_fsgnj_d(ctx, rctx, tail_idx, *dest, *src1, *src2, FsgnjOp::Sgnjx)?;
            }

            // Fused multiply-add operations
            // Note: WebAssembly doesn't have fused multiply-add, so we emulate with separate ops
            Inst::FmaddS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = (src1 * src2) + src3
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Add)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = (src1 * src2) - src3
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Sub)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = -(src1 * src2) + src3 = src3 - (src1 * src2)
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Sub)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = -(src1 * src2) - src3
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Neg)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::F32Sub)?;
                self.nan_box_f32(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaddD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Add)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Sub)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Sub)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Mul)?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Neg)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                rctx.feed(ctx, tail_idx, &Instruction::F64Sub)?;
                rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Atomic operations (A extension)
            // RISC-V Specification: "The atomic instruction set is divided into two
            // subsets: the standard atomic instructions (AMO) and load-reserved /
            // store-conditional (LR/SC) instructions."
            //
            // All three families lower to wasm thread-local atomics.  In the
            // single-threaded wasm model these are equivalent to plain loads/stores
            // plus the cmpxchg loop for min/max AMOs; on shared-memory wasm they
            // provide the full multi-threaded guarantees.
            Inst::LrW {
                dest,
                addr,
                order: _,
            } => {
                // LR.W: load-reserved word.  No reservation register in wasm —
                // treated as an atomic 32-bit load.  AmoOrdering is accepted for
                // API symmetry; the wasm atomic load is always seq-cst within a
                // thread.
                if dest.0 != 0 {
                    let addr_type = self.addr_val_type();
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                    // rv32+memory64: load_addr is i64, extend before tee.
                    if !self.enable_rv64 && self.use_memory64 {
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                    }
                    // Tee address for alias check in emit_lr.
                    rctx.feed(ctx, tail_idx, &Instruction::LocalTee(load_addr))?;
                    emit_lr(
                        ctx,
                        rctx,
                        RmwWidth::W32,
                        self.atomic_opts,
                        load_addr,
                        addr_type,
                        MemOrder::Strong,
                        tail_idx,
                    )?;
                    if self.enable_rv64 {
                        // Sign-extend the 32-bit loaded value to i64.
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::ScW {
                dest,
                addr,
                src,
                order: _,
            } => {
                // SC.W: store-conditional word.  Always succeeds in single-threaded
                // wasm — write 0 (success) into `dest`.
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                // RV32+memory64: address must be i64.
                if !self.enable_rv64 && self.use_memory64 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                }
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    // Truncate i64 register to the 32-bit value to store.
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                }
                emit_sc(
                    ctx,
                    rctx,
                    RmwWidth::W32,
                    self.atomic_opts,
                    self.mem_order,
                    tail_idx,
                )?;
                if dest.0 != 0 {
                    // SC always succeeds: write 0 into dest.
                    let zero = if self.enable_rv64 {
                        Instruction::I64Const(0)
                    } else {
                        Instruction::I32Const(0)
                    };
                    rctx.feed(ctx, tail_idx, &zero)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // CSR Instructions (Zicsr extension)
            // RISC-V Specification Quote:
            // "The SYSTEM major opcode is used to encode all privileged instructions, as well
            // as the ECALL and EBREAK instructions and CSR instructions."
            // Note: CSR operations are system-specific and may need special runtime support
            Inst::Csrrw { dest, src, .. }
            | Inst::Csrrs { dest, src, .. }
            | Inst::Csrrc { dest, src, .. } => {
                if dest.0 != 0 {
                    self.emit_int_const(ctx, rctx, tail_idx, 0)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
                let _ = src;
            }

            Inst::Csrrwi { dest, .. } | Inst::Csrrsi { dest, .. } | Inst::Csrrci { dest, .. } => {
                if dest.0 != 0 {
                    self.emit_int_const(ctx, rctx, tail_idx, 0)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // RV64 instructions
            // These are RV64-specific. When RV64 is disabled, we emit unreachable.
            Inst::Lwu { offset, dest, base } => {
                if self.enable_rv64 {
                    self.translate_load(ctx, rctx, tail_idx, *base, *offset, *dest, LoadOp::U32)?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::Ld { offset, dest, base } => {
                if self.enable_rv64 {
                    self.translate_load(ctx, rctx, tail_idx, *base, *offset, *dest, LoadOp::I64)?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::Sd { offset, base, src } => {
                if self.enable_rv64 {
                    self.translate_store(ctx, rctx, tail_idx, *base, *offset, *src, StoreOp::I64)?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            // RV64I: Word arithmetic instructions (operate on lower 32 bits)
            Inst::AddiW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.emit_imm(ctx, rctx, tail_idx, *imm)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        // Sign-extend lower 32 bits to 64 bits
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::SlliW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Const(imm.as_i32()))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Shl)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::SrliW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Const(imm.as_i32()))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32ShrU)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::SraiW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Const(imm.as_i32()))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32ShrS)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::AddW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Add)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::SubW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Sub)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::SllW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32Shl)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::SrlW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32ShrU)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::SraW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32ShrS)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            // RV64M: Multiplication and division word instructions
            Inst::MulW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64Mul)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::DivW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32DivS)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::DivuW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32DivU)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::RemW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32RemS)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::RemuW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I32RemU)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            // RV64F/D: Floating-point conversion instructions
            Inst::FcvtLS { dest, src, .. } => {
                // Convert single-precision float to signed 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.unbox_f32(ctx, rctx, tail_idx)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64TruncF32S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLuS { dest, src, .. } => {
                // Convert single-precision float to unsigned 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.unbox_f32(ctx, rctx, tail_idx)?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64TruncF32U)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtSL { dest, src, .. } => {
                // Convert signed 64-bit integer to single-precision float
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::F32ConvertI64S)?;
                    self.nan_box_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtSLu { dest, src, .. } => {
                // Convert unsigned 64-bit integer to single-precision float
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::F32ConvertI64U)?;
                    self.nan_box_f32(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLD { dest, src, .. } => {
                // Convert double-precision float to signed 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64TruncF64S)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLuD { dest, src, .. } => {
                // Convert double-precision float to unsigned 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64TruncF64U)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtDL { dest, src, .. } => {
                // Convert signed 64-bit integer to double-precision float
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::F64ConvertI64S)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtDLu { dest, src, .. } => {
                // Convert unsigned 64-bit integer to double-precision float
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::F64ConvertI64U)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FmvXD { dest, src } => {
                // Move bits from double-precision float register to 64-bit integer register
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        rctx.feed(ctx, tail_idx, &Instruction::I64ReinterpretF64)?;
                        rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            Inst::FmvDX { dest, src } => {
                // Move bits from 64-bit integer register to double-precision float register
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    rctx.feed(ctx, tail_idx, &Instruction::F64ReinterpretI64)?;
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                }
            }

            // AMO: atomic read-modify-write word.
            //
            // Each AMO reads the old value at [addr], applies `op(old, src)`, writes
            // the result back, and returns `old` into `dest`.  The five ops with direct
            // wasm RMW counterparts (Swap/Add/Xor/And/Or) compile to a single wasm
            // atomic instruction.  Min/Max/Minu/Maxu are synthesised with a cmpxchg
            // loop in emit_rmw — see speet-ordering for the detailed encoding.
            Inst::AmoW {
                op,
                dest,
                addr,
                src,
                order,
            } => {
                let rmw_op = match op {
                    AmoOp::Swap => RmwOp::Swap,
                    AmoOp::Add => RmwOp::Add,
                    AmoOp::Xor => RmwOp::Xor,
                    AmoOp::And => RmwOp::And,
                    AmoOp::Or => RmwOp::Or,
                    AmoOp::Min => RmwOp::Min,
                    AmoOp::Max => RmwOp::Max,
                    AmoOp::Minu => RmwOp::Minu,
                    AmoOp::Maxu => RmwOp::Maxu,
                };

                let src_local = Self::reg_to_local(*src);
                let scratch = self.amo_scratch_local(rctx.layout());
                // For memory access, always use the load-addr scratch local so
                // that the address is an i64 in memory64 mode (matching the
                // scratch local's declared type).
                let load_addr = self.load_addr_scratch_local(rctx.layout());

                // Push addr onto the stack; extend to i64 for RV32+memory64.
                rctx.feed(ctx, tail_idx, &Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                if !self.enable_rv64 && self.use_memory64 {
                    rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32U)?;
                } else if self.enable_rv64 && !self.use_memory64 {
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                }
                rctx.feed(ctx, tail_idx, &Instruction::LocalTee(load_addr))?;

                // Push src; truncate i64 register to i32 for the 32-bit AMO in RV64.
                if self.enable_rv64 {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?;
                    rctx.feed(ctx, tail_idx, &Instruction::I32WrapI64)?;
                } else {
                    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?;
                }
                // emit_rmw consumes [addr, src] from the stack and leaves [old].
                let _ = order; // guest ordering annotation; wasm atomics are seq-cst within a thread
                emit_rmw(
                    ctx,
                    rctx,
                    RmwWidth::W32,
                    rmw_op,
                    self.atomic_opts,
                    self.mem_order,
                    load_addr,
                    src_local,
                    scratch,
                    tail_idx,
                )?;

                if dest.0 != 0 {
                    if self.enable_rv64 {
                        // Sign-extend the 32-bit old value to the 64-bit register.
                        rctx.feed(ctx, tail_idx, &Instruction::I64ExtendI32S)?;
                    }
                    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                } else {
                    // Discard the returned old value.
                    rctx.feed(ctx, tail_idx, &Instruction::Drop)?;
                }
            }

            // Floating-point classify
            // These require examining the floating-point value's bit pattern
            Inst::FclassS { .. } | Inst::FclassD { .. } => {
                // FCLASS returns a 10-bit mask indicating the class of the floating-point number
                // (positive/negative infinity, normal, subnormal, zero, NaN, etc.)
                // This requires complex bit pattern analysis not yet implemented
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
            }

            // Catch-all for any other unhandled instructions
            _ => {
                // This should ideally never be reached if all instruction variants are handled.
                // If it is reached, it indicates an instruction type that was added to rv-asm
                // but not yet implemented in this recompiler.
                rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
                return Ok(()); // Don't fallthrough for unimplemented instructions
            }
        }

        // For most instructions that don't explicitly handle control flow,
        // yecta automatically handles fallthrough based on the len parameter in init_function
        Ok(())
    }

    /// Helper to translate branch instructions using yecta's ji API with custom Snippet
    pub(crate) fn translate_branch<RC: ReactorContext<Context, E> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        src1: Reg,
        src2: Reg,
        offset: Imm,
        pc: u32,
        _inst_len: u32,
        op: BranchOp,
    ) -> Result<(), E> {
        let target_pc = if self.enable_rv64 {
            // RV64: Use 64-bit PC arithmetic
            (pc as i64).wrapping_add(offset.as_i32() as i64) as u64
        } else {
            // RV32: Use 32-bit PC arithmetic
            (pc as i32).wrapping_add(offset.as_i32()) as u32 as u64
        };

        // Create a custom Snippet for the branch condition using a closure
        // The closure captures the registers and operation
        struct BranchCondition {
            src1: u32,
            src2: u32,
            op: BranchOp,
            enable_rv64: bool,
        }

        impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for BranchCondition {
            fn emit(
                &self,
                ctx: &mut Context,
                sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
            ) -> Result<(), E> {
                // Emit the same instructions as emit_instruction
                sink.instruction(ctx, &Instruction::LocalGet(self.src1))?;
                sink.instruction(ctx, &Instruction::LocalGet(self.src2))?;
                if self.enable_rv64 {
                    sink.instruction(
                        ctx,
                        &match self.op {
                            BranchOp::Eq => Instruction::I64Eq,
                            BranchOp::Ne => Instruction::I64Ne,
                            BranchOp::LtS => Instruction::I64LtS,
                            BranchOp::GeS => Instruction::I64GeS,
                            BranchOp::LtU => Instruction::I64LtU,
                            BranchOp::GeU => Instruction::I64GeU,
                        },
                    )?;
                } else {
                    sink.instruction(
                        ctx,
                        &match self.op {
                            BranchOp::Eq => Instruction::I32Eq,
                            BranchOp::Ne => Instruction::I32Ne,
                            BranchOp::LtS => Instruction::I32LtS,
                            BranchOp::GeS => Instruction::I32GeS,
                            BranchOp::LtU => Instruction::I32LtU,
                            BranchOp::GeU => Instruction::I32GeU,
                        },
                    )?;
                }
                Ok(())
            }
        }

        impl<Context, E> wax_core::build::InstructionSource<Context, E> for BranchCondition {
            fn emit_instruction(
                &self,
                ctx: &mut Context,
                sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
            ) -> Result<(), E> {
                sink.instruction(ctx, &Instruction::LocalGet(self.src1))?;
                sink.instruction(ctx, &Instruction::LocalGet(self.src2))?;
                if self.enable_rv64 {
                    sink.instruction(
                        ctx,
                        &match self.op {
                            BranchOp::Eq => Instruction::I64Eq,
                            BranchOp::Ne => Instruction::I64Ne,
                            BranchOp::LtS => Instruction::I64LtS,
                            BranchOp::GeS => Instruction::I64GeS,
                            BranchOp::LtU => Instruction::I64LtU,
                            BranchOp::GeU => Instruction::I64GeU,
                        },
                    )?;
                } else {
                    sink.instruction(
                        ctx,
                        &match self.op {
                            BranchOp::Eq => Instruction::I32Eq,
                            BranchOp::Ne => Instruction::I32Ne,
                            BranchOp::LtS => Instruction::I32LtS,
                            BranchOp::GeS => Instruction::I32GeS,
                            BranchOp::LtU => Instruction::I32LtU,
                            BranchOp::GeU => Instruction::I32GeU,
                        },
                    )?;
                }
                Ok(())
            }
        }

        let condition = BranchCondition {
            src1: Self::reg_to_local(src1),
            src2: Self::reg_to_local(src2),
            op,
            enable_rv64: self.enable_rv64,
        };

        // Jump trap: conditional branch.
        let branch_info = JumpInfo::direct(pc as u64, target_pc, JumpKind::ConditionalBranch);
        if             rctx.on_jump(&branch_info, ctx)?
            == TrapAction::Skip
        {
            return Ok(());
        }

        // Use ji with condition for branch taken path
        // When condition is true, jump to target; yecta handles else/end automatically
        let Some(target_func) = self.pc_to_func_idx(target_pc) else {
            // Target is omitted — unreachable in a correctly analyzed binary.
            rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?;
            return Ok(());
        };
        let target = yecta::Target::Static { func: target_func };

        rctx.ji(
            ctx,
            tail_idx,
            rctx.locals_mark().total_locals,
            &BTreeMap::new(),
            target,
            None,
            rctx.pool(),
            Some(&condition),
        )?;

        Ok(())
    }
}
