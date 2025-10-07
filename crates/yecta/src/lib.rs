#![no_std]

use alloc::{collections::vec_deque::VecDeque, vec::Vec};
use wasm_encoder::{Function, Instruction, ValType};
extern crate alloc;
pub struct Opts {
    pub size: u32,
    pub locals: Vec<(u32, ValType)>,
    pub params: u32,
    pub table: u32,
    pub function_ty: u32,
    pub fastcall: Option<FastCall>,
    pub pinned: Vec<bool>,
}
pub struct FastCall {
    pub lr: u32,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Link {
    pub last_len: i32,
    pub reg: u32,
}
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
        return (self.functions.len() as u32).wrapping_add_signed(offset);
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
            while self.opts.pinned.len() <= off as usize {
                self.opts.pinned.push(false);
            }
            self.opts.pinned[off as usize] = true
        }
        let f = &mut self.functions[fi].0;

        for a in 0..self.opts.params {
            f.instruction(&Instruction::LocalGet(a));
        }
        if let Some(l) = lcall.as_ref() {
            f.instruction(&Instruction::I32Const(off.unwrap() as i32))
                .instruction(&Instruction::LocalSet(l.reg));
            if let Some(fc) = self.opts.fastcall.as_ref()
                && fc.lr == l.reg
            {
                f.instruction(&Instruction::Call(next));
                for a in (0..self.opts.params).rev() {
                    f.instruction(&Instruction::LocalSet(a));
                }
                return;
            }
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
            while self.opts.pinned.len() <= off as usize {
                self.opts.pinned.push(false);
            }
            self.opts.pinned[off as usize] = true
        }
        let f = &mut self.functions[fi].0;
        for a in 0..self.opts.params {
            f.instruction(&Instruction::LocalGet(a));
        }
        f.instruction(&Instruction::LocalGet(idx));
        let mut peg = false;
        if let Some(l) = lcall.as_ref() {
            f.instruction(&Instruction::I32Const(off.unwrap() as i32))
                .instruction(&Instruction::LocalSet(l.reg));
            if let Some(fc) = self.opts.fastcall.as_ref()
                && fc.lr == l.reg
            {
                if fc.lr == idx {
                    peg = true;
                } else {
                    f.instruction(&Instruction::CallIndirect {
                        type_index: self.opts.function_ty,
                        table_index: self.opts.table,
                    });
                    for a in (0..self.opts.params).rev() {
                        f.instruction(&Instruction::LocalSet(a));
                    }
                    return;
                }
            }
        }
        if let Some(fc) = self.opts.fastcall.as_ref()
            && fc.lr == idx
            && !peg
        {
            f.instruction(&Instruction::Drop);
            f.instruction(&Instruction::Return);
            return;
        }
        f.instruction(&Instruction::ReturnCallIndirect {
            type_index: self.opts.function_ty,
            table_index: self.opts.table,
        });
    }
}
