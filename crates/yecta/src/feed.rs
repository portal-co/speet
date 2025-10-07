use crate::*;
pub struct FeedState {
    functions: Vec<(Function, Option<u32>)>,
    counters: VecDeque<Option<u32>>,
    opts: Opts,
}
impl FeedState {
    pub fn new(opts: Opts) -> Self {
        Self {
            functions: Default::default(),
            counters: Default::default(),
            opts,
        }
    }
    pub fn id_for_offset(&self, offset: i32) -> u32 {
        return (self.functions.len() as u32)
            .wrapping_add_signed(offset)
            .wrapping_add(self.opts.offset);
    }
    pub fn begin_inst(&mut self, len: u32) {
        while self.counters.len() <= len as usize {
            self.counters.push_back(None);
        }
        *self.counters.iter_mut().nth(len as usize).unwrap() = Some(self.functions.len() as u32);
        self.functions.push((
            Function::new(self.opts.locals.clone()),
            self.counters.pop_front().flatten(),
        ));
    }
    pub fn end(mut self) -> (Opts, Vec<Function>) {
        for (f, g) in self.functions.iter_mut() {
            f.instruction(&Instruction::Unreachable);
        }
        // self.functions.reverse();
        return (self.opts, self.functions.into_iter().map(|a| a.0).collect());
    }
    pub fn instr<'a>(&'a mut self, i: &Instruction<'_>) -> &'a mut Self {
        let mut fi = self.functions.len() - 1;
        loop {
            self.functions[fi].0.instruction(i);
            let Some(a) = &self.functions[fi].1 else {
                return self;
            };
            fi = *a as usize;
        }
    }
    pub fn instrs<'a>(&'a mut self, i: &[Instruction<'_>]) -> &'a mut Self {
        let mut fi = self.functions.len() - 1;
        loop {
            for i in i.iter() {
                self.functions[fi].0.instruction(i);
            }
            let Some(a) = &self.functions[fi].1 else {
                return self;
            };
            fi = *a as usize;
        }
    }
    pub fn jmp(&mut self, offset: i32, lcall: Option<Link>, conditional: bool) {
        let mut fi = self.functions.len() - 1;
        loop {
            self.fi_jmp(fi, offset, lcall);
            if conditional || lcall.is_some() {
                let Some(a) = &self.functions[fi].1 else {
                    return;
                };
                fi = *a as usize;
            } else {
                let Some(a) = self.functions[fi].1.take() else {
                    return;
                };
                fi = a as usize;
            }
        }
    }
    fn fi_jmp(&mut self, fi: usize, offset: i32, lcall: Option<Link>) {
        let next = self.id_for_offset(offset);
        let off = lcall.as_ref().map(|l| self.id_for_offset(l.last_len));
        if let Some(off) = off {
            self.opts.pinned.flag(
                off.wrapping_sub(self.opts.offset)
                    .wrapping_add(self.opts.table_offset) as usize,
            );
        }
        let f = &mut self.functions[fi].0;

        if let Some(l) = lcall.as_ref() {
            let off = off.unwrap() as u64;
            let off = off
                .wrapping_sub(self.opts.offset.into())
                .wrapping_add(self.opts.code_offset.into());
            f.instruction(&match self.opts.xlen {
                xLen::_64 => Instruction::I64Const(off as i64),
                xLen::_32 => Instruction::I32Const((off & 0xffff_ffff) as u32 as i32),
            });

            f.instruction(&Instruction::LocalSet(l.reg));
            if let Some(fc) = self.opts.fastcall.as_ref()
                && fc.lr == l.reg
            {
                for mut a in 0..self.opts.params {
                    if a == fc.lr_backup {
                        a = fc.lr
                    }
                    f.instruction(&Instruction::LocalGet(a));
                }
                f.instruction(&Instruction::Call(next));
                for a in (0..self.opts.params).rev() {
                    if a == fc.lr_backup {
                        f.instruction(&Instruction::Drop);
                    } else {
                        f.instruction(&Instruction::LocalSet(a));
                    }
                }

                return;
            }
        }
        for a in 0..self.opts.params {
            f.instruction(&Instruction::LocalGet(a));
        }
        f.instruction(&Instruction::ReturnCall(next));
    }
    pub fn jr(&mut self, idx: u32, lcall: Option<Link>, conditional: bool) {
        let mut fi = self.functions.len() - 1;
        loop {
            self.fi_jr(fi, idx, lcall);
            if conditional || lcall.is_some() {
                let Some(a) = &self.functions[fi].1 else {
                    return;
                };
                fi = *a as usize;
            } else {
                let Some(a) = self.functions[fi].1.take() else {
                    return;
                };
                fi = a as usize;
            }
        }
    }
    fn fi_jr(&mut self, fi: usize, idx: u32, lcall: Option<Link>) {
        let off = lcall.as_ref().map(|l| self.id_for_offset(l.last_len));
        if let Some(off) = off {
            self.opts.pinned.flag(
                off.wrapping_sub(self.opts.offset)
                    .wrapping_add(self.opts.table_offset) as usize,
            );
        }
        let f = &mut self.functions[fi].0;
        macro_rules! table_index {
            () => {
                f.instruction(&Instruction::LocalGet(idx))
                    .instruction(&match self.opts.xlen {
                        xLen::_64 => Instruction::I64Const(
                            (self
                                .opts
                                .code_offset
                                .wrapping_sub(self.opts.table_offset.into()))
                                as i64,
                        ),
                        xLen::_32 => Instruction::I32Const(
                            (self
                                .opts
                                .code_offset
                                .wrapping_sub(self.opts.table_offset.into())
                                & 0xffff_ffff) as u32 as i32,
                        ),
                    })
                    .instruction(&match self.opts.xlen {
                        xLen::_64 => Instruction::I64Sub,
                        xLen::_32 => Instruction::I32Sub,
                    });
            };
        }
        let mut peg = false;
        if let Some(l) = lcall.as_ref() {
            let off = off.unwrap() as u64;
            let off = off
                .wrapping_sub(self.opts.offset.into())
                .wrapping_add(self.opts.code_offset.into());
            f.instruction(&match self.opts.xlen {
                xLen::_64 => Instruction::I64Const(off as i64),
                xLen::_32 => Instruction::I32Const((off & 0xffff_ffff) as u32 as i32),
            });
            f.instruction(&Instruction::LocalSet(l.reg));
            if let Some(fc) = self.opts.fastcall.as_ref()
                && fc.lr == l.reg
            {
                if fc.lr == idx {
                    peg = true;
                } else {
                    for mut a in 0..self.opts.params {
                        if a == fc.lr_backup {
                            a = fc.lr
                        }
                        f.instruction(&Instruction::LocalGet(a));
                    }
                    table_index!();
                    f.instruction(&Instruction::CallIndirect {
                        type_index: self.opts.function_ty,
                        table_index: self.opts.table,
                    });
                    for a in (0..self.opts.params).rev() {
                        if a == fc.lr_backup {
                            f.instruction(&Instruction::Drop);
                        } else {
                            f.instruction(&Instruction::LocalSet(a));
                        }
                    }
                    return;
                }
            }
        }
        let mut needs_end = false;
        if let Some(fc) = self.opts.fastcall.as_ref()
            && fc.lr == idx
            && !peg
        {
            f.instruction(&Instruction::LocalGet(fc.lr));
            f.instruction(&Instruction::LocalGet(fc.lr_backup));
            f.instruction(&match self.opts.xlen {
                xLen::_32 => Instruction::I32Eq,
                xLen::_64 => Instruction::I64Eq,
            })
            .instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
            f.instruction(&Instruction::Drop);
            for a in 0..self.opts.params {
                f.instruction(&Instruction::LocalGet(a));
            }
            f.instruction(&Instruction::Return);
            f.instruction(&Instruction::Else);

            needs_end = true;
        }
        for a in 0..self.opts.params {
            f.instruction(&Instruction::LocalGet(a));
        }
        table_index!();
        f.instruction(&Instruction::ReturnCallIndirect {
            type_index: self.opts.function_ty,
            table_index: self.opts.table,
        });
        if needs_end {
            f.instruction(&Instruction::End);
        }
    }
}
