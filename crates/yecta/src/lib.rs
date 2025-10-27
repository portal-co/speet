#![no_std]

use core::mem::take;

use alloc::{
    collections::{btree_map::BTreeMap, btree_set::BTreeSet, vec_deque::VecDeque},
    vec::Vec,
};
use wasm_encoder::{Catch, Function, Instruction, ValType};

extern crate alloc;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct EscapeTag {
    pub tag: u32,
    pub ty: u32,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Pool {
    pub table: u32,
    pub ty: u32,
}
#[derive(Clone, Copy)]
pub enum Target<'a> {
    Static { func: u32 },
    Dynamic { idx: &'a (dyn Snippet + 'a) },
}
pub trait Snippet {
    fn emit(&self, go: &mut (dyn FnMut(&Instruction<'_>) + '_));
}
#[derive(Default)]
pub struct Reactor {
    fns: Vec<Entry>,
    lens: VecDeque<BTreeSet<u32>>,
}
struct Entry {
    function: Function,
    preds: BTreeSet<u32>,
    if_stmts: usize,
}
impl Reactor {
    pub fn next(
        &mut self,
        locals: impl IntoIterator<Item = (u32, ValType), IntoIter: ExactSizeIterator>,
        len: u32,
    ) {
        while self.lens.len() != len as usize + 1 {
            self.lens.push_back(Default::default());
        }
        self.lens
            .iter_mut()
            .nth(len as usize)
            .unwrap()
            .insert(self.fns.len() as u32);
        self.fns.push(Entry {
            function: Function::new(locals),
            preds: self.lens.pop_front().into_iter().flatten().collect(),
            if_stmts: 0,
        });
    }
    fn add_pred(&mut self, succ: u32, pred: u32) {
        match self.fns.get_mut(succ as usize) {
            Some(a) => {
                a.preds.insert(pred);
            }
            None => {
                let len = (self.fns.len() - succ as usize) as u32;
                while self.lens.len() != len as usize + 1 {
                    self.lens.push_back(Default::default());
                }
                self.lens.iter_mut().nth(len as usize).unwrap().insert(pred);
            }
        }
    }
    fn add_pred_checked(&mut self, succ: u32, pred: u32, params: u32) {
        let ifs = self.total_ifs(pred);
        let mut queue = VecDeque::new();
        queue.push_back(pred);
        let mut cache = BTreeSet::new();

        while let Some(q) = queue.pop_front() {
            if cache.contains(&q) {
                continue;
            };
            cache.insert(q);
            // self.fns[q as usize].function.instruction(instruction);
            for p in self.fns[q as usize].preds.iter().cloned() {
                queue.push_back(p);
            }
        }
        if cache.contains(&succ) {
            for k in cache {
                let f = &mut self.fns[k as usize];
                _ = take(&mut f.preds);
                for p in 0..params {
                    f.function.instruction(&Instruction::LocalGet(p));
                }
                f.function.instruction(&Instruction::ReturnCall(succ));
                for _ in 0..ifs {
                    f.function.instruction(&Instruction::End);
                }
            }
        } else {
            self.add_pred(succ, pred);
        }
    }
    pub fn call(&mut self, target: Target, tag: EscapeTag, pool: Pool) {
        let EscapeTag { tag, ty } = tag;
        self.feed(&Instruction::Block(wasm_encoder::BlockType::FunctionType(
            ty,
        )));
        self.feed(&Instruction::TryTable(
            wasm_encoder::BlockType::FunctionType(ty),
            [Catch::One { tag, label: 0 }].into_iter().collect(),
        ));
        match target {
            Target::Static { func } => {
                self.feed(&Instruction::Call(func));
            }
            Target::Dynamic { idx } => {
                idx.emit(&mut |a| self.feed(a));
                self.feed(&Instruction::CallIndirect {
                    type_index: pool.ty,
                    table_index: pool.table,
                });
            }
        }
        self.feed(&Instruction::Return);
        self.feed(&Instruction::End);
        self.feed(&Instruction::End);
    }
    pub fn ret(&mut self, params: u32, tag: EscapeTag) {
        let EscapeTag { tag, ty } = tag;
        for p in 0..params {
            self.feed(&Instruction::LocalGet(p));
        }
        self.feed(&Instruction::Throw(tag));
    }
    pub fn ji(
        &mut self,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet + '_)>,
        target: Target,
        call: Option<EscapeTag>,
        pool: Pool,
        condition: Option<&(dyn Snippet + '_)>,
    ) {
        if let Some(c) = condition.as_deref() {
            let mut queue = VecDeque::new();
            queue.push_back((self.fns.len() - 1) as u32);
            let mut cache = BTreeSet::new();
            while let Some(q) = queue.pop_front() {
                if cache.contains(&q) {
                    continue;
                };
                cache.insert(q);
                // self.fns[q as usize].function.instruction(instruction);
                for p in self.fns[q as usize].preds.iter().cloned() {
                    queue.push_back(p);
                }
            }
            for c in cache {
                self.fns[c as usize].if_stmts += 1;
            }
        }
        match call {
            Some(tag) => {
                if let Some(s) = condition.as_deref() {
                    s.emit(&mut |a| self.feed(a));
                    self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
                }
                for p in 0..params {
                    if let Some(f) = fixups.get(&p) {
                        f.emit(&mut |a| self.feed(a));
                    } else {
                        self.feed(&Instruction::LocalGet(p));
                    }
                }
                self.call(target, tag, pool);
                for p in (0..params).rev() {
                    if let Some(f) = fixups.get(&p) {
                        self.feed(&Instruction::Drop);
                    } else {
                        self.feed(&Instruction::LocalSet(p));
                    }
                }
                if let Some(s) = condition.as_deref() {
                    self.feed(&Instruction::Else);
                }
            }
            None => match target {
                Target::Static { func } => {
                    if let Some(s) = condition.as_deref() {
                        s.emit(&mut |a| self.feed(a));
                        self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
                    }

                    if let Some(s) = condition.as_deref() {
                        for p in 0..params {
                            if let Some(f) = fixups.get(&p) {
                                f.emit(&mut |a| self.feed(a));
                            } else {
                                self.feed(&Instruction::LocalGet(p));
                            }
                        }
                        self.feed(&Instruction::ReturnCall(func));
                        self.feed(&Instruction::Else);
                    } else {
                        for (i, f) in fixups.iter() {
                            f.emit(&mut |a| self.feed(a));
                            self.feed(&Instruction::LocalSet(*i));
                        }
                        self.jmp(func, params)
                    }
                }
                Target::Dynamic { idx } => {
                    if let Some(s) = condition.as_deref() {
                        s.emit(&mut |a| self.feed(a));
                        self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
                    }
                    for p in 0..params {
                        if let Some(f) = fixups.get(&p) {
                            f.emit(&mut |a| self.feed(a));
                        } else {
                            self.feed(&Instruction::LocalGet(p));
                        }
                    }
                    idx.emit(&mut |a| self.feed(a));
                    if let Some(s) = condition.as_deref() {
                        self.feed(&Instruction::ReturnCallIndirect {
                            type_index: pool.ty,
                            table_index: pool.table,
                        });
                        self.feed(&Instruction::Else);
                    } else {
                        self.seal(&Instruction::ReturnCallIndirect {
                            type_index: pool.ty,
                            table_index: pool.table,
                        });
                    }
                }
            },
        }
    }
    pub fn jmp(&mut self, target: u32, params: u32) {
        let mut queue = VecDeque::new();
        queue.push_back((self.fns.len() - 1) as u32);
        let mut cache = BTreeSet::new();
        while let Some(q) = queue.pop_front() {
            if cache.contains(&q) {
                continue;
            };
            cache.insert(q);
            // self.fns[q as usize].function.instruction(instruction);
            for p in self.fns[q as usize].preds.iter().cloned() {
                queue.push_back(p);
            }
        }
        for x in cache {
            self.add_pred_checked(target, x, params);
        }
    }
    pub fn feed(&mut self, instruction: &Instruction<'_>) {
        let mut queue = VecDeque::new();
        queue.push_back((self.fns.len() - 1) as u32);
        let mut cache = BTreeSet::new();
        while let Some(q) = queue.pop_front() {
            if cache.contains(&q) {
                continue;
            };
            cache.insert(q);
            self.fns[q as usize].function.instruction(instruction);
            for p in self.fns[q as usize].preds.iter().cloned() {
                queue.push_back(p);
            }
        }
    }
    pub fn seal(&mut self, instruction: &Instruction<'_>) {
        let ifs = self.total_ifs((self.fns.len() - 1) as u32);
        let mut queue = VecDeque::new();
        queue.push_back((self.fns.len() - 1) as u32);
        let mut cache = BTreeSet::new();
        while let Some(q) = queue.pop_front() {
            if cache.contains(&q) {
                continue;
            };
            cache.insert(q);
            self.fns[q as usize].function.instruction(instruction);
            for _ in 0..ifs {
                self.fns[q as usize].function.instruction(&Instruction::End);
            }
            for p in take(&mut self.fns[q as usize].preds) {
                queue.push_back(p);
            }
        }
    }
    fn total_ifs(&self, p: u32) -> usize {
        return self.fns[p as usize].if_stmts;
    }
}
