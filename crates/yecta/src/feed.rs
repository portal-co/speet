use crate::*;
pub struct FeedState {
    functions: Vec<(Function, Option<u32>)>,
    counters: VecDeque<Option<u32>>,
    opts: Opts,
}
impl FeedState {
    pub fn env(&self) -> Env {
        self.opts.env.clone()
    }
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
            .wrapping_add(self.opts.env.offset);
    }
    pub fn begin_inst(&mut self, len: u32) {
        while self.counters.len() <= len as usize {
            self.counters.push_back(None);
        }
        *self.counters.iter_mut().nth(len as usize).unwrap() = Some(self.functions.len() as u32);
        self.functions.push((
            match Function::new(self.opts.locals.clone()) {
                mut f => f,
            },
            self.counters.pop_front().flatten(),
        ));
        if let Some(i) = self.opts.env.inst_start.as_ref().cloned() {
            for p in 0..self.opts.env.params {
                self.instr(&Instruction::LocalGet(p));
            }
            self.instr(&Instruction::I32Const(self.functions.len() as u32 as i32));
            self.instr(&Instruction::Call(i));

            self.instr(&Instruction::If(wasm_encoder::BlockType::Empty));
            if self.opts.env.tail_calls_disabled {
                self.instr(&Instruction::Return);
            } else {
                self.instr(&Instruction::ReturnCallIndirect {
                    type_index: self.opts.env.function_ty,
                    table_index: self.opts.env.table,
                });
            }
            self.instr(&Instruction::Else);
            for p in (0..self.opts.env.params).rev() {
                self.instr(&Instruction::LocalSet(p));
            }
            self.instr(&Instruction::Drop);
            self.instr(&Instruction::End);
        }
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
        let soff = self.id_for_offset(-(self.functions.len() as u32 as i32));
        let sidx = self.functions.len();
        if let Some(off) = off {
            self.opts.pinned.flag(
                off.wrapping_sub(self.opts.env.offset)
                    .wrapping_add(self.opts.env.table_offset) as usize,
            );
        }
        let f = &mut self.functions[fi].0;
        let mut flr = false;
        if let Some(l) = lcall.as_ref() {
            let off = off.unwrap() as u64;
            let off = off
                .wrapping_sub(self.opts.env.offset.into())
                .wrapping_add(self.opts.env.code_offset.into());
            f.instruction(&match self.opts.env.xlen {
                xLen::_64 => Instruction::I64Const(off as i64),
                xLen::_32 => Instruction::I32Const((off & 0xffff_ffff) as u32 as i32),
            });

            f.instruction(&Instruction::LocalSet(l.reg));
            if let Some(fc) = self.opts.fastcall.as_ref()
                && fc.lr == l.reg
            {
                let fl = match self.opts.env.feat_flags.as_ref().cloned() {
                    None => true,
                    Some(fl) => {
                        f.instruction(&Instruction::Call(fl))
                            .instruction(&Instruction::I32Const(1))
                            .instruction(&Instruction::I32And)
                            .instruction(&Instruction::I32Eqz)
                            .instruction(&Instruction::I32Eqz)
                            .instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
                        flr = true;
                        false
                    }
                };
                for mut a in 0..self.opts.env.params {
                    if self.opts.non_arg_params.contains(&a) {
                        f.instruction(&match self.opts.env.xlen {
                            xLen::_32 => Instruction::I32Const(0),
                            xLen::_64 => Instruction::I64Const(0),
                        });
                        continue;
                    }
                    if a == fc.lr_backup {
                        a = fc.lr
                    }
                    f.instruction(&Instruction::LocalGet(a));
                }
                apply_env_tco(&self.opts.env, f);
                f.instruction(&Instruction::Call(next));
                apply_env_tco_end(&self.opts.env, f);
                for a in (0..self.opts.env.params).rev() {
                    if a == fc.lr_backup || self.opts.non_arg_params.contains(&a) {
                        f.instruction(&Instruction::Drop);
                    } else {
                        f.instruction(&Instruction::LocalSet(a));
                    }
                }
                if fl {
                    return;
                } else {
                    f.instruction(&Instruction::Else);
                }
            }
        }
        for a in 0..self.opts.env.params {
            f.instruction(&Instruction::LocalGet(a));
        }
        if self.opts.env.tail_calls_disabled {
            self.opts.pinned.flag(sidx);
            f.instruction(&match self.opts.env.xlen {
                xLen::_64 => Instruction::I64Const(soff as u64 as i64),
                xLen::_32 => Instruction::I32Const(soff as i32),
            })
            .instruction(&Instruction::Return);
        } else {
            f.instruction(&Instruction::ReturnCall(next));
        }
        if flr {
            f.instruction(&Instruction::End);
        }
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
                off.wrapping_sub(self.opts.env.offset)
                    .wrapping_add(self.opts.env.table_offset) as usize,
            );
        }
        let f = &mut self.functions[fi].0;
        let mut flr = false;
        macro_rules! table_index {
            () => {
                f.instruction(&Instruction::LocalGet(idx))
                    .instruction(&match self.opts.env.xlen {
                        xLen::_64 => Instruction::I64Const(
                            (self
                                .opts
                                .env
                                .code_offset
                                .wrapping_sub(self.opts.env.table_offset.into()))
                                as i64,
                        ),
                        xLen::_32 => Instruction::I32Const(
                            (self
                                .opts
                                .env
                                .code_offset
                                .wrapping_sub(self.opts.env.table_offset.into())
                                & 0xffff_ffff) as u32 as i32,
                        ),
                    })
                    .instruction(&match self.opts.env.xlen {
                        xLen::_64 => Instruction::I64Sub,
                        xLen::_32 => Instruction::I32Sub,
                    });
            };
        }
        let mut peg = false;
        if let Some(l) = lcall.as_ref() {
            let off = off.unwrap() as u64;
            let off = off
                .wrapping_sub(self.opts.env.offset.into())
                .wrapping_add(self.opts.env.code_offset.into());
            f.instruction(&match self.opts.env.xlen {
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
                    let fl = match self.opts.env.feat_flags.as_ref().cloned() {
                        None => true,
                        Some(fl) => {
                            f.instruction(&Instruction::Call(fl))
                                .instruction(&Instruction::I32Const(1))
                                .instruction(&Instruction::I32And)
                                .instruction(&Instruction::I32Eqz)
                                .instruction(&Instruction::I32Eqz)
                                .instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
                            flr = true;
                            false
                        }
                    };
                    for mut a in 0..self.opts.env.params {
                        if self.opts.non_arg_params.contains(&a) {
                            f.instruction(&match self.opts.env.xlen {
                                xLen::_32 => Instruction::I32Const(0),
                                xLen::_64 => Instruction::I64Const(0),
                            });
                            continue;
                        }
                        if a == fc.lr_backup {
                            a = fc.lr
                        }
                        f.instruction(&Instruction::LocalGet(a));
                    }
                    table_index!();
                    apply_env_tco(&self.opts.env, f);
                    f.instruction(&Instruction::CallIndirect {
                        type_index: self.opts.env.function_ty,
                        table_index: self.opts.env.table,
                    });
                    apply_env_tco_end(&self.opts.env, f);
                    for a in (0..self.opts.env.params).rev() {
                        if a == fc.lr_backup || self.opts.non_arg_params.contains(&a) {
                            f.instruction(&Instruction::Drop);
                        } else {
                            f.instruction(&Instruction::LocalSet(a));
                        }
                    }
                    if fl {
                        return;
                    } else {
                        f.instruction(&Instruction::Else);
                    }
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
            f.instruction(&match self.opts.env.xlen {
                xLen::_32 => Instruction::I32Eq,
                xLen::_64 => Instruction::I64Eq,
            });
            if let Some(fl) = self.opts.env.feat_flags.as_ref().cloned() {
                f.instruction(&Instruction::Call(fl))
                    .instruction(&Instruction::I32Const(1))
                    .instruction(&Instruction::I32And)
                    .instruction(&Instruction::I32Eqz)
                    .instruction(&Instruction::I32Eqz)
                    .instruction(&Instruction::I32And);
            }
            f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
            f.instruction(&Instruction::Drop);
            for a in 0..self.opts.env.params {
                f.instruction(&Instruction::LocalGet(a));
            }
            if self.opts.env.tail_calls_disabled {
                f.instruction(&match self.opts.env.xlen {
                    xLen::_64 => Instruction::I64Const(0),
                    xLen::_32 => Instruction::I32Const(0),
                });
            }
            f.instruction(&Instruction::Return);
            f.instruction(&Instruction::Else);

            needs_end = true;
        }
        for a in 0..self.opts.env.params {
            f.instruction(&Instruction::LocalGet(a));
        }
        table_index!();
        if self.opts.env.tail_calls_disabled {
            f.instruction(&Instruction::Return);
        } else {
            f.instruction(&Instruction::ReturnCallIndirect {
                type_index: self.opts.env.function_ty,
                table_index: self.opts.env.table,
            });
        }
        if needs_end {
            f.instruction(&Instruction::End);
        }
        if flr {
            f.instruction(&Instruction::End);
        }
    }
}
fn apply_env_tco(env: &Env, f: &mut Function) {
    if env.tail_calls_disabled {
        f.instruction(&Instruction::Loop(wasm_encoder::BlockType::FunctionType(
            env.function_ty,
        )));
    }
}
fn apply_env_tco_end(env: &Env, f: &mut Function) {
    if env.tail_calls_disabled {
        f.instruction(&Instruction::LocalTee(env.params))
            .instruction(&match env.xlen {
                xLen::_32 => Instruction::I32Eqz,
                xLen::_64 => Instruction::I64Eqz,
            })
            .instruction(&Instruction::I32Eqz)
            .instruction(&Instruction::If(wasm_encoder::BlockType::Empty))
            .instruction(&Instruction::LocalGet(env.params))
            .instruction(&Instruction::CallIndirect {
                type_index: env.function_ty,
                table_index: env.table,
            })
            .instruction(&Instruction::Br(0))
            .instruction(&Instruction::Else)
            .instruction(&Instruction::End)
            .instruction(&Instruction::LocalGet(env.params))
            .instruction(&Instruction::End)
            .instruction(&Instruction::Drop);
    }
}
