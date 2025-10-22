#![no_std]
use crate::pin::PinTracker;
use alloc::{
    collections::{btree_set::BTreeSet, vec_deque::VecDeque},
    sync::Arc,
    vec::Vec,
};
use wasm_encoder::{Function, Instruction, ValType};
extern crate alloc;
pub mod feed;
pub mod opts;
pub mod pin;
pub struct FastCall {
    pub lr: u32,
    pub lr_backup: u32,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Link {
    pub last_len: i32,
    pub reg: u32,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum xLen {
    _32,
    _64,
}
pub trait InstFeed{
    fn instr(&mut self, i: &Instruction<'_>);
}
pub trait InstFeedExt: InstFeed{
    fn instruction(&mut self, i: &Instruction<'_>) -> &mut Self{
        self.instr(i);
        self
    }
}
impl<T: InstFeed + ?Sized> InstFeedExt for T{

}
impl InstFeed for Function{
    fn instr(&mut self, i: &Instruction<'_>) {
        self.instruction(i);
    }
}