use crate::*;
impl<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>> RiscVRecompiler<'cb, 'ctx, Context, E, F> {
    /// Helper to translate load instructions
    pub(crate) fn translate_load(
        &mut self,
        ctx: &mut Context,
        base: Reg,
        offset: Imm,
        dest: Reg,
        op: LoadOp,
    ) -> Result<(), E> {
        if dest.0 == 0 {
            return Ok(()); // x0 is hardwired to zero
        }

        // Compute address: base + offset
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(ctx, offset)?;

        // Add instruction depends on whether we're using memory64 and RV64
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Add)?;
            // If not using memory64, wrap to 32-bit address
            if !self.use_memory64 {
                self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
            }
        } else {
            self.reactor.feed(ctx, &Instruction::I32Add)?;
        }

        // Apply address mapping if provided (for paging support)
        if let Some(mapper) = self.mapper_callback.as_mut() {
            let mut callback_ctx = CallbackContext {
                reactor: &mut self.reactor,
            };
            mapper.call(ctx, &mut callback_ctx)?;
        }

        // Load from memory
        match op {
            LoadOp::I8 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Load8S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Load8S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                    // If RV64 but not memory64, extend to i64
                    if self.enable_rv64 {
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U8 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Load8U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Load8U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                    if self.enable_rv64 {
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
                    }
                }
            }
            LoadOp::I16 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Load16S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Load16S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                    if self.enable_rv64 {
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U16 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Load16U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Load16U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                    if self.enable_rv64 {
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
                    }
                }
            }
            LoadOp::I32 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Load32S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                    if self.enable_rv64 {
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U32 => {
                // RV64 LWU instruction - load word unsigned (zero-extended)
                if self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Load32U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                    self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
                }
            }
            LoadOp::I64 => {
                // RV64 LD instruction - load double-word
                self.reactor
                    .feed(ctx, &Instruction::I64Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }))?;
            }
        }

        self.reactor
            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(dest)))?;
        Ok(())
    }

    /// Helper to translate store instructions
    pub(crate) fn translate_store(
        &mut self,
        ctx: &mut Context,
        base: Reg,
        offset: Imm,
        src: Reg,
        op: StoreOp,
    ) -> Result<(), E> {
        // Compute address: base + offset
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(ctx, offset)?;

        // Add instruction depends on whether we're using memory64 and RV64
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Add)?;
            // If not using memory64, wrap to 32-bit address
            if !self.use_memory64 {
                self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
            }
        } else {
            self.reactor.feed(ctx, &Instruction::I32Add)?;
        }

        // Apply address mapping if provided (for paging support)
        if let Some(mapper) = self.mapper_callback.as_mut() {
            let mut callback_ctx = CallbackContext {
                reactor: &mut self.reactor,
            };
            mapper.call(ctx, &mut callback_ctx)?;
        }

        // Load value to store
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(src)))?;

        // If RV64 but not memory64, need to wrap i64 value to i32 for 32-bit stores
        let need_wrap = self.enable_rv64 && !self.use_memory64 && !matches!(op, StoreOp::I64);
        if need_wrap {
            self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        }

        // Store to memory
        match op {
            StoreOp::I8 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                }
            }
            StoreOp::I16 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                }
            }
            StoreOp::I32 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(ctx, &Instruction::I64Store32(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(ctx, &Instruction::I32Store(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                }
            }
            StoreOp::I64 => {
                // RV64 SD instruction - store double-word
                self.reactor
                    .feed(ctx, &Instruction::I64Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }))?;
            }
        }

        Ok(())
    }

    /// Helper to translate floating-point load instructions
    pub(crate) fn translate_fload(
        &mut self,
        ctx: &mut Context,
        base: Reg,
        offset: Imm,
        dest: FReg,
        op: FLoadOp,
    ) -> Result<(), E> {
        // Compute address: base + offset
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(ctx, offset)?;
        self.reactor.feed(ctx, &Instruction::I32Add)?;

        // Apply address mapping if provided (for paging support)
        if let Some(mapper) = self.mapper_callback.as_mut() {
            let mut callback_ctx = CallbackContext {
                reactor: &mut self.reactor,
            };
            mapper.call(ctx, &mut callback_ctx)?;
        }

        // Load from memory
        match op {
            FLoadOp::F32 => {
                self.reactor
                    .feed(ctx, &Instruction::F32Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }))?;
                self.reactor.feed(ctx, &Instruction::F64PromoteF32)?;
            }
            FLoadOp::F64 => {
                self.reactor
                    .feed(ctx, &Instruction::F64Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }))?;
            }
        }

        self.reactor
            .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(dest)))?;
        Ok(())
    }

    /// Helper to translate floating-point store instructions
    pub(crate) fn translate_fstore(
        &mut self,
        ctx: &mut Context,
        base: Reg,
        offset: Imm,
        src: FReg,
        op: FStoreOp,
    ) -> Result<(), E> {
        // Compute address: base + offset
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(ctx, offset)?;
        self.reactor.feed(ctx, &Instruction::I32Add)?;

        // Apply address mapping if provided (for paging support)
        if let Some(mapper) = self.mapper_callback.as_mut() {
            let mut callback_ctx = CallbackContext {
                reactor: &mut self.reactor,
            };
            mapper.call(ctx, &mut callback_ctx)?;
        }

        // Load value to store
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(src)))?;

        // Store to memory
        match op {
            FStoreOp::F32 => {
                self.reactor.feed(ctx, &Instruction::F32DemoteF64)?;
                self.reactor
                    .feed(ctx, &Instruction::F32Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }))?;
            }
            FStoreOp::F64 => {
                self.reactor
                    .feed(ctx, &Instruction::F64Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }))?;
            }
        }

        Ok(())
    }

    /// Helper to emit sign-injection for single-precision floats
    pub(crate) fn emit_fsgnj_s(
        &mut self,
        ctx: &mut Context,
        dest: FReg,
        src1: FReg,
        src2: FReg,
        op: FsgnjOp,
    ) -> Result<(), E> {
        // Sign injection uses bit manipulation on the float representation
        // Get magnitude from src1, sign from src2 (possibly modified)

        // Convert src1 to i32 to manipulate bits
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(src1)))?;
        self.unbox_f32(ctx)?;
        self.reactor.feed(ctx, &Instruction::I32ReinterpretF32)?;

        // Mask to keep only magnitude (clear sign bit): 0x7FFFFFFF
        self.reactor.feed(ctx, &Instruction::I32Const(0x7FFFFFFF))?;
        self.reactor.feed(ctx, &Instruction::I32And)?;

        // Get sign bit from src2
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(src2)))?;
        self.unbox_f32(ctx)?;
        self.reactor.feed(ctx, &Instruction::I32ReinterpretF32)?;

        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly: mask with 0x80000000
                self.reactor
                    .feed(ctx, &Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(ctx, &Instruction::I32And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                self.reactor
                    .feed(ctx, &Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(ctx, &Instruction::I32And)?;
                self.reactor
                    .feed(ctx, &Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(ctx, &Instruction::I32Xor)?; // Flip the sign bit
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits of src1 and src2
                // Need original src1 sign
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::I32ReinterpretF32)?;
                self.reactor.feed(ctx, &Instruction::I32Xor)?;
                self.reactor
                    .feed(ctx, &Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(ctx, &Instruction::I32And)?;
            }
        }

        // Combine magnitude and sign
        self.reactor.feed(ctx, &Instruction::I32Or)?;
        self.reactor.feed(ctx, &Instruction::F32ReinterpretI32)?;
        self.nan_box_f32(ctx)?;
        self.reactor
            .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(dest)))?;

        Ok(())
    }

    /// Helper to emit sign-injection for double-precision floats
    pub(crate) fn emit_fsgnj_d(
        &mut self,
        ctx: &mut Context,
        dest: FReg,
        src1: FReg,
        src2: FReg,
        op: FsgnjOp,
    ) -> Result<(), E> {
        // Similar to single-precision but using i64
        // Convert src1 to i64 to manipulate bits
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(src1)))?;
        self.reactor.feed(ctx, &Instruction::I64ReinterpretF64)?;

        // Mask to keep only magnitude (clear sign bit)
        self.reactor
            .feed(ctx, &Instruction::I64Const(0x7FFFFFFFFFFFFFFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?;

        // Get sign bit from src2
        self.reactor
            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(src2)))?;
        self.reactor.feed(ctx, &Instruction::I64ReinterpretF64)?;

        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly
                self.reactor
                    .feed(ctx, &Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(ctx, &Instruction::I64And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                self.reactor
                    .feed(ctx, &Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(ctx, &Instruction::I64And)?;
                self.reactor
                    .feed(ctx, &Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(ctx, &Instruction::I64Xor)?;
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.reactor.feed(ctx, &Instruction::I64ReinterpretF64)?;
                self.reactor.feed(ctx, &Instruction::I64Xor)?;
                self.reactor
                    .feed(ctx, &Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(ctx, &Instruction::I64And)?;
            }
        }

        // Combine magnitude and sign
        self.reactor.feed(ctx, &Instruction::I64Or)?;
        self.reactor.feed(ctx, &Instruction::F64ReinterpretI64)?;
        self.reactor
            .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(dest)))?;

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
    pub fn translate_bytes(
        &mut self,
        ctx: &mut Context,
        bytes: &[u8],
        start_pc: u32,
        xlen: Xlen,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<usize, ()> {
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

            // Decode the instruction
            let (inst, is_compressed) = match Inst::decode(inst_word, xlen) {
                Ok(result) => result,
                Err(_) => break, // Stop on decode error
            };

            let pc = start_pc + offset as u32;

            // Translate the instruction
            if let Err(_) = self.translate_instruction(ctx, &inst, pc, is_compressed, f) {
                break;
            }

            // Advance by instruction size
            offset += match is_compressed {
                IsCompressed::Yes => 2,
                IsCompressed::No => 4,
            };
        }

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
    pub fn translate_instruction(
        &mut self,
        ctx: &mut Context,
        inst: &Inst,
        pc: u32,
        is_compressed: IsCompressed,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        // Calculate instruction length in 2-byte increments
        let inst_len = match is_compressed {
            IsCompressed::Yes => 1, // 2 bytes
            IsCompressed::No => 2,  // 4 bytes
        };

        // Initialize function for this instruction
        self.init_function(pc, inst_len, 8, f);
        // Update PC
        self.reactor.feed(ctx, &Instruction::I32Const(pc as i32))?;
        self.reactor
            .feed(ctx, &Instruction::LocalSet(Self::pc_local()))?;

        match inst {
            // RV32I Base Integer Instruction Set

            // Lui: Load Upper Immediate
            // RISC-V Specification Quote:
            // "LUI (load upper immediate) is used to build 32-bit constants and uses the U-type format.
            // LUI places the 32-bit U-immediate value into the destination register rd, filling in the
            // lowest 12 bits with zeros."
            Inst::Lui { uimm, dest } => {
                self.emit_imm(ctx, *uimm)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
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
                    self.reactor.feed(ctx, &Instruction::I64Const(pc as i64))?;
                    self.emit_imm(ctx, *uimm)?;
                    self.emit_add(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                } else {
                    // RV32: Use 32-bit PC
                    self.reactor.feed(ctx, &Instruction::I32Const(pc as i32))?;
                    self.emit_imm(ctx, *uimm)?;
                    self.reactor.feed(ctx, &Instruction::I32Add)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
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
                    && self.escape_tag.is_some();

                if use_speculative {
                    // Speculative call lowering: emit a native WASM call with return validation
                    // See SPECULATIVE_CALLS.md for the full pattern
                    let escape_tag = self.escape_tag.unwrap();
                    let target_func = self.pc_to_func_idx(target_pc);

                    // Save return address to ra (x1)
                    if self.enable_rv64 {
                        self.reactor
                            .feed(ctx, &Instruction::I64Const(return_addr as i64))?;
                    } else {
                        self.reactor
                            .feed(ctx, &Instruction::I32Const(return_addr as i32))?;
                    }
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;

                    // Emit the speculative call using yecta's call API
                    // This wraps the call in a try-catch block for exception-based escapes
                    self.reactor.call(
                        ctx,
                        yecta::Target::Static { func: target_func },
                        escape_tag,
                        self.pool,
                    )?;

                    // After the call returns, we need to validate the return PC
                    // The yecta Reactor's call method handles the Block/TryTable wrapper;
                    // we emit the validation logic here
                    
                    // Load the current PC (which should be the expected return address)
                    self.reactor.feed(ctx, &Instruction::LocalGet(Self::pc_local()))?;
                    if self.enable_rv64 {
                        self.reactor.feed(ctx, &Instruction::I64Const(return_addr as i64))?;
                        self.reactor.feed(ctx, &Instruction::I64Ne)?;
                    } else {
                        self.reactor.feed(ctx, &Instruction::I32Const(return_addr as i32))?;
                        self.reactor.feed(ctx, &Instruction::I32Ne)?;
                    }
                    
                    // If PC doesn't match expected return address, throw escape tag
                    self.reactor.feed(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
                    // Load all parameters for the escape
                    self.reactor.ret(ctx, 65, escape_tag)?;
                    self.reactor.feed(ctx, &Instruction::End)?;

                    // Normal return path: continue execution
                    return Ok(());
                } else {
                    // Non-speculative path: original jump-based implementation
                    if dest.0 != 0 {
                        if self.enable_rv64 {
                            self.reactor
                                .feed(ctx, &Instruction::I64Const(return_addr as i64))?;
                        } else {
                            self.reactor
                                .feed(ctx, &Instruction::I32Const(return_addr as i32))?;
                        }
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                    self.jump_to_pc(ctx, target_pc, 65)?;
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

                // Check if this is an ABI-compliant call (dest == x1/ra) and speculative calls are enabled
                let use_speculative = self.enable_speculative_calls
                    && dest.0 == 1  // x1 is the return address register (ra)
                    && self.escape_tag.is_some();

                if use_speculative {
                    // Speculative call lowering for indirect calls
                    // Since JALR is indirect, we use dynamic dispatch through the pool table
                    let escape_tag = self.escape_tag.unwrap();

                    // Save return address to ra (x1)
                    if self.enable_rv64 {
                        self.reactor
                            .feed(ctx, &Instruction::I64Const(return_addr as i64))?;
                    } else {
                        self.reactor
                            .feed(ctx, &Instruction::I32Const(return_addr as i32))?;
                    }
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;

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
                                sink.instruction(ctx, &Instruction::I64Const(0xFFFFFFFFFFFFFFFE_u64 as i64))?;
                                sink.instruction(ctx, &Instruction::I64And)?;
                                sink.instruction(ctx, &Instruction::I64Const(self.base_pc as i64))?;
                                sink.instruction(ctx, &Instruction::I64Sub)?;
                                sink.instruction(ctx, &Instruction::I64Const(1))?;
                                sink.instruction(ctx, &Instruction::I64ShrU)?;
                                sink.instruction(ctx, &Instruction::I32WrapI64)?;
                            } else {
                                sink.instruction(ctx, &Instruction::I32Const(self.offset))?;
                                sink.instruction(ctx, &Instruction::I32Add)?;
                                sink.instruction(ctx, &Instruction::I32Const(0xFFFFFFFE_u32 as i32))?;
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
                            sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
                        ) -> Result<(), E> {
                            // Compute: ((base + offset) & ~1 - base_pc) / 2
                            // This gives us the function index from the PC
                            sink.instruction(ctx, &Instruction::LocalGet(self.base_local))?;
                            if self.enable_rv64 {
                                sink.instruction(ctx, &Instruction::I64Const(self.offset as i64))?;
                                sink.instruction(ctx, &Instruction::I64Add)?;
                                sink.instruction(ctx, &Instruction::I64Const(0xFFFFFFFFFFFFFFFE_u64 as i64))?;
                                sink.instruction(ctx, &Instruction::I64And)?;
                                sink.instruction(ctx, &Instruction::I64Const(self.base_pc as i64))?;
                                sink.instruction(ctx, &Instruction::I64Sub)?;
                                sink.instruction(ctx, &Instruction::I64Const(1))?;
                                sink.instruction(ctx, &Instruction::I64ShrU)?;
                                sink.instruction(ctx, &Instruction::I32WrapI64)?;
                            } else {
                                sink.instruction(ctx, &Instruction::I32Const(self.offset))?;
                                sink.instruction(ctx, &Instruction::I32Add)?;
                                sink.instruction(ctx, &Instruction::I32Const(0xFFFFFFFE_u32 as i32))?;
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

                    // Emit the speculative call using yecta's call API with dynamic target
                    self.reactor.call(
                        ctx,
                        yecta::Target::Dynamic { idx: &target_snippet },
                        escape_tag,
                        self.pool,
                    )?;

                    // After the call returns, validate the return PC
                    self.reactor.feed(ctx, &Instruction::LocalGet(Self::pc_local()))?;
                    if self.enable_rv64 {
                        self.reactor.feed(ctx, &Instruction::I64Const(return_addr as i64))?;
                        self.reactor.feed(ctx, &Instruction::I64Ne)?;
                    } else {
                        self.reactor.feed(ctx, &Instruction::I32Const(return_addr as i32))?;
                        self.reactor.feed(ctx, &Instruction::I32Ne)?;
                    }
                    
                    // If PC doesn't match expected return address, throw escape tag
                    self.reactor.feed(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
                    self.reactor.ret(ctx, 65, escape_tag)?;
                    self.reactor.feed(ctx, &Instruction::End)?;

                    return Ok(());
                } else {
                    // Non-speculative path: original implementation
                    if dest.0 != 0 {
                        if self.enable_rv64 {
                            self.reactor
                                .feed(ctx, &Instruction::I64Const(return_addr as i64))?;
                        } else {
                            self.reactor
                                .feed(ctx, &Instruction::I32Const(return_addr as i32))?;
                        }
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                    // JALR is indirect, compute target and update PC
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*base)))?;
                    self.emit_imm(ctx, *offset)?;
                    self.emit_add(ctx)?;
                    if self.enable_rv64 {
                        self.reactor
                            .feed(ctx, &Instruction::I64Const(0xFFFFFFFFFFFFFFFE_u64 as i64))?;
                        self.reactor.feed(ctx, &Instruction::I64And)?;
                    } else {
                        self.reactor
                            .feed(ctx, &Instruction::I32Const(0xFFFFFFFE_u32 as i32))?;
                        self.reactor.feed(ctx, &Instruction::I32And)?;
                    }
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::pc_local()))?;
                    // For indirect jumps, seal with unreachable as we can't statically determine target
                    self.reactor.seal(ctx, &Instruction::Unreachable)?;
                    return Ok(());
                }
            }

            // Branch Instructions
            // RISC-V Specification Quote:
            // "All branch instructions use the B-type instruction format. The 12-bit B-immediate encodes
            // signed offsets in multiples of 2 bytes. The offset is sign-extended and added to the address
            // of the branch instruction to give the target address."
            Inst::Beq { offset, src1, src2 } => {
                self.translate_branch(ctx, *src1, *src2, *offset, pc, inst_len, BranchOp::Eq)?;
                return Ok(()); // Branch handles control flow
            }

            Inst::Bne { offset, src1, src2 } => {
                self.translate_branch(ctx, *src1, *src2, *offset, pc, inst_len, BranchOp::Ne)?;
                return Ok(());
            }

            Inst::Blt { offset, src1, src2 } => {
                self.translate_branch(ctx, *src1, *src2, *offset, pc, inst_len, BranchOp::LtS)?;
                return Ok(());
            }

            Inst::Bge { offset, src1, src2 } => {
                self.translate_branch(ctx, *src1, *src2, *offset, pc, inst_len, BranchOp::GeS)?;
                return Ok(());
            }

            Inst::Bltu { offset, src1, src2 } => {
                self.translate_branch(ctx, *src1, *src2, *offset, pc, inst_len, BranchOp::LtU)?;
                return Ok(());
            }

            Inst::Bgeu { offset, src1, src2 } => {
                self.translate_branch(ctx, *src1, *src2, *offset, pc, inst_len, BranchOp::GeU)?;
                return Ok(());
            }

            // Load Instructions
            // RISC-V Specification Quote:
            // "Load and store instructions transfer a value between the registers and memory.
            // Loads are encoded in the I-type format and stores are S-type."
            Inst::Lb { offset, dest, base } => {
                self.translate_load(ctx, *base, *offset, *dest, LoadOp::I8)?;
            }

            Inst::Lh { offset, dest, base } => {
                self.translate_load(ctx, *base, *offset, *dest, LoadOp::I16)?;
            }

            Inst::Lw { offset, dest, base } => {
                self.translate_load(ctx, *base, *offset, *dest, LoadOp::I32)?;
            }

            Inst::Lbu { offset, dest, base } => {
                self.translate_load(ctx, *base, *offset, *dest, LoadOp::U8)?;
            }

            Inst::Lhu { offset, dest, base } => {
                self.translate_load(ctx, *base, *offset, *dest, LoadOp::U16)?;
            }

            // Store Instructions
            Inst::Sb { offset, src, base } => {
                self.translate_store(ctx, *base, *offset, *src, StoreOp::I8)?;
            }

            Inst::Sh { offset, src, base } => {
                self.translate_store(ctx, *base, *offset, *src, StoreOp::I16)?;
            }

            Inst::Sw { offset, src, base } => {
                self.translate_store(ctx, *base, *offset, *src, StoreOp::I32)?;
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
                        let mut callback_ctx = CallbackContext {
                            reactor: &mut self.reactor,
                        };
                        callback.call(&hint_info, ctx, &mut callback_ctx);
                    }

                    // No WebAssembly code generation needed - this is a true no-op
                } else if src1.0 == 0 {
                    // li (load immediate) pseudoinstruction
                    self.emit_imm(ctx, *imm)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                } else if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.emit_add(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slti { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.reactor.feed(ctx, &Instruction::I32LtS)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sltiu { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.reactor.feed(ctx, &Instruction::I32LtU)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Xori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.emit_xor(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Ori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.emit_or(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Andi { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.emit_and(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slli { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.emit_shl(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srli { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.emit_shr_u(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srai { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(ctx, *imm)?;
                    self.emit_shr_s(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Register-Register Operations
            Inst::Add { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_add(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sub { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_sub(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sll { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shl(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slt { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::I32LtS)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sltu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::I32LtU)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Xor { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_xor(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srl { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shr_u(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sra { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shr_s(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Or { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_or(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::And { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_and(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Fence: Memory ordering
            // RISC-V Specification Quote:
            // "The FENCE instruction is used to order device I/O and memory accesses as viewed by
            // other RISC-V harts and external devices or coprocessors."
            Inst::Fence { .. } => {
                // WebAssembly has a different memory model; in a single-threaded environment
                // or with WebAssembly's built-in atomics, explicit fences may not be needed
                // For now, we emit a no-op
            }

            // System calls
            Inst::Ecall => {
                let ecall_info = EcallInfo { pc };

                // Invoke callback if set
                if let Some(ref mut callback) = self.ecall_callback {
                    let mut callback_ctx = CallbackContext {
                        reactor: &mut self.reactor,
                    };
                    callback.call(&ecall_info, ctx, &mut callback_ctx);
                } else {
                    // Default behavior: environment call - implementation specific
                    // Would need to be handled by runtime
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::Ebreak => {
                let ebreak_info = EbreakInfo { pc };

                // Invoke callback if set
                if let Some(ref mut callback) = self.ebreak_callback {
                    let mut callback_ctx = CallbackContext {
                        reactor: &mut self.reactor,
                    };
                    callback.call(&ebreak_info, ctx, &mut callback_ctx);
                } else {
                    // Default behavior: breakpoint - implementation specific
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            // M Extension: Integer Multiplication and Division
            // RISC-V Specification Quote:
            // "This chapter describes the standard integer multiplication and division instruction-set
            // extension, which is named 'M' and contains instructions that multiply or divide values
            // held in two integer registers."
            Inst::Mul { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_mul(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Mulh { dest, src1, src2 } => {
                // Multiply high signed-signed: returns upper bits of product
                // RV32: upper 32 bits of 64-bit product
                // RV64: upper 64 bits of 128-bit product
                if dest.0 != 0 {
                    if self.enable_rv64 {
                        // For RV64: compute high 64 bits of 128-bit signed multiplication
                        self.emit_mulh_signed(ctx, 
                            Self::reg_to_local(*src1),
                            Self::reg_to_local(*src2),
                        )?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply and shift
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor.feed(ctx, &Instruction::I64Mul)?;
                        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
                        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
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
                        self.emit_mulh_signed_unsigned(ctx, 
                            Self::reg_to_local(*src1),
                            Self::reg_to_local(*src2),
                        )?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply with mixed sign extension
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
                        self.reactor.feed(ctx, &Instruction::I64Mul)?;
                        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
                        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
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
                        self.emit_mulh_unsigned(ctx, 
                            Self::reg_to_local(*src1),
                            Self::reg_to_local(*src2),
                        )?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply and shift
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
                        self.reactor.feed(ctx, &Instruction::I64Mul)?;
                        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
                        self.reactor.feed(ctx, &Instruction::I64ShrU)?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
            }

            Inst::Div { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::I32DivS)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Divu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::I32DivU)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Rem { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::I32RemS)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Remu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::I32RemU)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Floating-Point Single-Precision (F Extension)
            // RISC-V Specification Quote:
            // "This chapter describes the standard instruction-set extension for single-precision
            // floating-point, which is named 'F'"
            Inst::Flw { offset, dest, base } => {
                self.translate_fload(ctx, *base, *offset, *dest, FLoadOp::F32)?;
            }

            Inst::Fsw { offset, src, base } => {
                self.translate_fstore(ctx, *base, *offset, *src, FStoreOp::F32)?;
            }

            Inst::FaddS {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Add)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubS {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Sub)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulS {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Mul)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivS {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Div)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtS { dest, src, .. } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Sqrt)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-Point Double-Precision (D Extension)
            Inst::Fld { offset, dest, base } => {
                self.translate_fload(ctx, *base, *offset, *dest, FLoadOp::F64)?;
            }

            Inst::Fsd { offset, src, base } => {
                self.translate_fstore(ctx, *base, *offset, *src, FStoreOp::F64)?;
            }

            Inst::FaddD {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Add)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubD {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Sub)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulD {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Mul)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivD {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Div)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtD { dest, src, .. } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(ctx, &Instruction::F64Sqrt)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point min/max operations
            Inst::FminS { dest, src1, src2 } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Min)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxS { dest, src1, src2 } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Max)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FminD { dest, src1, src2 } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Min)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxD { dest, src1, src2 } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Max)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point comparison operations
            Inst::FeqS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor.feed(ctx, &Instruction::F32Eq)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor.feed(ctx, &Instruction::F32Lt)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor.feed(ctx, &Instruction::F32Le)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FeqD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::F64Eq)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::F64Lt)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(ctx, &Instruction::F64Le)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Floating-point conversion operations
            // RISC-V Specification Quote:
            // "Floating-point-to-integer and integer-to-floating-point conversion instructions
            // are encoded in the OP-FP major opcode space."
            Inst::FcvtWS { dest, src, .. } => {
                // Convert single to signed 32-bit integer
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor.feed(ctx, &Instruction::I32TruncF32S)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuS { dest, src, .. } => {
                // Convert single to unsigned 32-bit integer
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor.feed(ctx, &Instruction::I32TruncF32U)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtSW { dest, src, .. } => {
                // Convert signed 32-bit integer to single
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(ctx, &Instruction::F32ConvertI32S)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to single
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(ctx, &Instruction::F32ConvertI32U)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtWD { dest, src, .. } => {
                // Convert double to signed 32-bit integer
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(ctx, &Instruction::I32TruncF64S)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuD { dest, src, .. } => {
                // Convert double to unsigned 32-bit integer
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(ctx, &Instruction::I32TruncF64U)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtDW { dest, src, .. } => {
                // Convert signed 32-bit integer to double
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(ctx, &Instruction::F64ConvertI32S)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to double
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(ctx, &Instruction::F64ConvertI32U)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSD { dest, src, .. } => {
                // RISC-V Specification Quote:
                // "FCVT.S.D converts double-precision float to single-precision float,
                // rounding according to the dynamic rounding mode."
                // Convert double to single with proper NaN-boxing
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(ctx, &Instruction::F32DemoteF64)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDS { dest, src, .. } => {
                // RISC-V Specification Quote:
                // "FCVT.D.S converts single-precision float to double-precision float."
                // Unbox the NaN-boxed single value, then promote to double
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F64PromoteF32)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point move operations
            Inst::FmvXW { dest, src } => {
                // Move bits from float register to integer register
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32(ctx)?;
                    self.reactor.feed(ctx, &Instruction::I32ReinterpretF32)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FmvWX { dest, src } => {
                // Move bits from integer register to float register
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(ctx, &Instruction::F32ReinterpretI32)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Sign-injection operations for single-precision
            // RISC-V Specification Quote:
            // "FSGNJ.S, FSGNJN.S, and FSGNJX.S produce a result that takes all bits except
            // the sign bit from rs1."
            Inst::FsgnjS { dest, src1, src2 } => {
                // Result = magnitude(src1) with sign(src2)
                // We'll use a simple implementation using bit manipulation
                self.emit_fsgnj_s(ctx, *dest, *src1, *src2, FsgnjOp::Sgnj)?;
            }

            Inst::FsgnjnS { dest, src1, src2 } => {
                // Result = magnitude(src1) with NOT(sign(src2))
                self.emit_fsgnj_s(ctx, *dest, *src1, *src2, FsgnjOp::Sgnjn)?;
            }

            Inst::FsgnjxS { dest, src1, src2 } => {
                // Result = magnitude(src1) with sign(src1) XOR sign(src2)
                self.emit_fsgnj_s(ctx, *dest, *src1, *src2, FsgnjOp::Sgnjx)?;
            }

            Inst::FsgnjD { dest, src1, src2 } => {
                self.emit_fsgnj_d(ctx, *dest, *src1, *src2, FsgnjOp::Sgnj)?;
            }

            Inst::FsgnjnD { dest, src1, src2 } => {
                self.emit_fsgnj_d(ctx, *dest, *src1, *src2, FsgnjOp::Sgnjn)?;
            }

            Inst::FsgnjxD { dest, src1, src2 } => {
                self.emit_fsgnj_d(ctx, *dest, *src1, *src2, FsgnjOp::Sgnjx)?;
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
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Mul)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Add)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = (src1 * src2) - src3
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Mul)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Sub)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = -(src1 * src2) + src3 = src3 - (src1 * src2)
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Mul)?;
                self.reactor.feed(ctx, &Instruction::F32Sub)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = -(src1 * src2) - src3
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Mul)?;
                self.reactor.feed(ctx, &Instruction::F32Neg)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32(ctx)?;
                self.reactor.feed(ctx, &Instruction::F32Sub)?;
                self.nan_box_f32(ctx)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaddD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Mul)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(ctx, &Instruction::F64Add)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Mul)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(ctx, &Instruction::F64Sub)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Mul)?;
                self.reactor.feed(ctx, &Instruction::F64Sub)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(ctx, &Instruction::F64Mul)?;
                self.reactor.feed(ctx, &Instruction::F64Neg)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(ctx, &Instruction::F64Sub)?;
                self.reactor
                    .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Atomic operations (A extension)
            // RISC-V Specification Quote:
            // "The atomic instruction set is divided into two subsets: the standard atomic
            // instructions (AMO) and load-reserved/store-conditional (LR/SC) instructions."
            // Note: WebAssembly atomics require special handling
            Inst::LrW { dest, addr, .. } => {
                // Load-reserved word
                // In WebAssembly, we'll implement this as a regular atomic load
                if dest.0 != 0 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                    self.reactor
                        .feed(ctx, &Instruction::I32AtomicLoad(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::ScW {
                dest, addr, src, ..
            } => {
                // Store-conditional word
                // In a simplified model, always succeed (return 0)
                // A full implementation would track reservations
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                self.reactor
                    .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor
                    .feed(ctx, &Instruction::I32AtomicStore(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }))?;
                if dest.0 != 0 {
                    self.reactor.feed(ctx, &Instruction::I32Const(0))?; // Success
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
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
                // For now, we'll stub these out as they require system support
                // A real implementation would need to call into a CSR handler
                if dest.0 != 0 {
                    // Return zero as placeholder
                    self.reactor.feed(ctx, &Instruction::I32Const(0))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
                // Silently ignore the write for now
                let _ = src;
            }

            Inst::Csrrwi { dest, .. } | Inst::Csrrsi { dest, .. } | Inst::Csrrci { dest, .. } => {
                if dest.0 != 0 {
                    self.reactor.feed(ctx, &Instruction::I32Const(0))?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // RV64 instructions
            // These are RV64-specific. When RV64 is disabled, we emit unreachable.
            Inst::Lwu { offset, dest, base } => {
                if self.enable_rv64 {
                    self.translate_load(ctx, *base, *offset, *dest, LoadOp::U32)?;
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::Ld { offset, dest, base } => {
                if self.enable_rv64 {
                    self.translate_load(ctx, *base, *offset, *dest, LoadOp::I64)?;
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::Sd { offset, base, src } => {
                if self.enable_rv64 {
                    self.translate_store(ctx, *base, *offset, *src, StoreOp::I64)?;
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            // RV64I: Word arithmetic instructions (operate on lower 32 bits)
            Inst::AddiW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.emit_imm(ctx, *imm)?;
                        self.reactor.feed(ctx, &Instruction::I64Add)?;
                        // Sign-extend lower 32 bits to 64 bits
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::SlliW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32Const(imm.as_i32()))?;
                        self.reactor.feed(ctx, &Instruction::I32Shl)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::SrliW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32Const(imm.as_i32()))?;
                        self.reactor.feed(ctx, &Instruction::I32ShrU)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::SraiW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32Const(imm.as_i32()))?;
                        self.reactor.feed(ctx, &Instruction::I32ShrS)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::AddW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I64Add)?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::SubW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I64Sub)?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::SllW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32Shl)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::SrlW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32ShrU)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::SraW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32ShrS)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            // RV64M: Multiplication and division word instructions
            Inst::MulW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I64Mul)?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::DivW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32DivS)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::DivuW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32DivU)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::RemW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32RemS)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::RemuW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
                        self.reactor.feed(ctx, &Instruction::I32RemU)?;
                        self.reactor.feed(ctx, &Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            // RV64F/D: Floating-point conversion instructions
            Inst::FcvtLS { dest, src, .. } => {
                // Convert single-precision float to signed 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.unbox_f32(ctx)?;
                        self.reactor.feed(ctx, &Instruction::I64TruncF32S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLuS { dest, src, .. } => {
                // Convert single-precision float to unsigned 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.unbox_f32(ctx)?;
                        self.reactor.feed(ctx, &Instruction::I64TruncF32U)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtSL { dest, src, .. } => {
                // Convert signed 64-bit integer to single-precision float
                if self.enable_rv64 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(ctx, &Instruction::F32ConvertI64S)?;
                    self.nan_box_f32(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtSLu { dest, src, .. } => {
                // Convert unsigned 64-bit integer to single-precision float
                if self.enable_rv64 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(ctx, &Instruction::F32ConvertI64U)?;
                    self.nan_box_f32(ctx)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLD { dest, src, .. } => {
                // Convert double-precision float to signed 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.reactor.feed(ctx, &Instruction::I64TruncF64S)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLuD { dest, src, .. } => {
                // Convert double-precision float to unsigned 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.reactor.feed(ctx, &Instruction::I64TruncF64U)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtDL { dest, src, .. } => {
                // Convert signed 64-bit integer to double-precision float
                if self.enable_rv64 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(ctx, &Instruction::F64ConvertI64S)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FcvtDLu { dest, src, .. } => {
                // Convert unsigned 64-bit integer to double-precision float
                if self.enable_rv64 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(ctx, &Instruction::F64ConvertI64U)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FmvXD { dest, src } => {
                // Move bits from double-precision float register to 64-bit integer register
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(ctx, &Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.reactor.feed(ctx, &Instruction::I64ReinterpretF64)?;
                        self.reactor
                            .feed(ctx, &Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            Inst::FmvDX { dest, src } => {
                // Move bits from 64-bit integer register to double-precision float register
                if self.enable_rv64 {
                    self.reactor
                        .feed(ctx, &Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(ctx, &Instruction::F64ReinterpretI64)?;
                    self.reactor
                        .feed(ctx, &Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(ctx, &Instruction::Unreachable)?;
                }
            }

            // Advanced atomic memory operations
            // These require more sophisticated atomic support than simple LR/SC
            Inst::AmoW { .. } => {
                // AMO operations (AMOSWAP, AMOADD, etc.) need WebAssembly atomic RMW operations
                // Future implementation should map these to appropriate wasm atomic instructions
                self.reactor.feed(ctx, &Instruction::Unreachable)?;
            }

            // Floating-point classify
            // These require examining the floating-point value's bit pattern
            Inst::FclassS { .. } | Inst::FclassD { .. } => {
                // FCLASS returns a 10-bit mask indicating the class of the floating-point number
                // (positive/negative infinity, normal, subnormal, zero, NaN, etc.)
                // This requires complex bit pattern analysis not yet implemented
                self.reactor.feed(ctx, &Instruction::Unreachable)?;
            }

            // Catch-all for any other unhandled instructions
            _ => {
                // This should ideally never be reached if all instruction variants are handled.
                // If it is reached, it indicates an instruction type that was added to rv-asm
                // but not yet implemented in this recompiler.
                self.reactor.feed(ctx, &Instruction::Unreachable)?;
                return Ok(()); // Don't fallthrough for unimplemented instructions
            }
        }

        // For most instructions that don't explicitly handle control flow,
        // yecta automatically handles fallthrough based on the len parameter in init_function
        Ok(())
    }

    /// Helper to translate branch instructions using yecta's ji API with custom Snippet
    pub(crate) fn translate_branch(
        &mut self,
        ctx: &mut Context,
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
                ctx: &mut Context, sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
            ) -> Result<(), E> {
                // Emit the same instructions as emit_instruction
                sink.instruction(ctx, &Instruction::LocalGet(self.src1))?;
                sink.instruction(ctx, &Instruction::LocalGet(self.src2))?;
                if self.enable_rv64 {
                    sink.instruction(ctx, &match self.op {
                        BranchOp::Eq => Instruction::I64Eq,
                        BranchOp::Ne => Instruction::I64Ne,
                        BranchOp::LtS => Instruction::I64LtS,
                        BranchOp::GeS => Instruction::I64GeS,
                        BranchOp::LtU => Instruction::I64LtU,
                        BranchOp::GeU => Instruction::I64GeU,
                    })?;
                } else {
                    sink.instruction(ctx, &match self.op {
                        BranchOp::Eq => Instruction::I32Eq,
                        BranchOp::Ne => Instruction::I32Ne,
                        BranchOp::LtS => Instruction::I32LtS,
                        BranchOp::GeS => Instruction::I32GeS,
                        BranchOp::LtU => Instruction::I32LtU,
                        BranchOp::GeU => Instruction::I32GeU,
                    })?;
                }
                Ok(())
            }
        }

        impl<Context, E> wax_core::build::InstructionSource<Context, E> for BranchCondition {
            fn emit_instruction(
                &self,
                ctx: &mut Context, sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
            ) -> Result<(), E> {
                sink.instruction(ctx, &Instruction::LocalGet(self.src1))?;
                sink.instruction(ctx, &Instruction::LocalGet(self.src2))?;
                if self.enable_rv64 {
                    sink.instruction(ctx, &match self.op {
                        BranchOp::Eq => Instruction::I64Eq,
                        BranchOp::Ne => Instruction::I64Ne,
                        BranchOp::LtS => Instruction::I64LtS,
                        BranchOp::GeS => Instruction::I64GeS,
                        BranchOp::LtU => Instruction::I64LtU,
                        BranchOp::GeU => Instruction::I64GeU,
                    })?;
                } else {
                    sink.instruction(ctx, &match self.op {
                        BranchOp::Eq => Instruction::I32Eq,
                        BranchOp::Ne => Instruction::I32Ne,
                        BranchOp::LtS => Instruction::I32LtS,
                        BranchOp::GeS => Instruction::I32GeS,
                        BranchOp::LtU => Instruction::I32LtU,
                        BranchOp::GeU => Instruction::I32GeU,
                    })?;
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

        // Use ji with condition for branch taken path
        // When condition is true, jump to target; yecta handles else/end automatically
        let target_func = self.pc_to_func_idx(target_pc);
        let target = yecta::Target::Static { func: target_func };

        self.reactor.ji(
            ctx,
            65,               // params: pass all registers
            &BTreeMap::new(), // fixups: none needed
            target,           // target: branch target
            None,             // call: not an escape call
            self.pool,        // pool: for indirect calls
            Some(&condition), // condition: branch condition
        )?;

        Ok(())
    }
}
