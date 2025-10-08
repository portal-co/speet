#![no_std]

use alloc::{
    collections::{btree_set::BTreeSet, vec_deque::VecDeque},
    sync::Arc,
    vec::Vec,
};
use wasm_encoder::{Function, Instruction, ValType};

use crate::pin::PinTracker;
extern crate alloc;
pub mod feed;
pub mod pin;
pub mod opts;
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
