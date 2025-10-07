#![no_std]

use alloc::{
    collections::{btree_set::BTreeSet, vec_deque::VecDeque},
    vec::Vec,
};
use wasm_encoder::{Function, Instruction, ValType};

use crate::pin::PinTracker;
extern crate alloc;
pub mod feed;
pub mod pin;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Env {
    pub size: u32,
    pub offset: u32,
    pub table_offset: u32,
    pub code_offset: u64,
    pub xlen: xLen,
    pub params: u32,
    pub table: u32,
    pub function_ty: u32,
}
#[non_exhaustive]
pub struct Opts {
    pub env: Env,
    pub locals: Vec<(u32, ValType)>,
    pub fastcall: Option<FastCall>,
    pub pinned: PinTracker,
    pub non_arg_params: BTreeSet<u32>,
}
impl From<Env> for Opts {
    fn from(value: Env) -> Self {
        Self {
            env: value,
            locals: Default::default(),
            fastcall: None,
            pinned: Default::default(),
            non_arg_params: Default::default(),
        }
    }
}
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
