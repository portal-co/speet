#![no_std]

use alloc::{collections::vec_deque::VecDeque, vec::Vec};
use wasm_encoder::{Function, Instruction, ValType};

use crate::pin::PinTracker;
extern crate alloc;
pub mod pin;
pub mod feed;
pub struct Opts {
    pub size: u32,
    pub offset: u32,
    pub table_offset: u32,
    pub code_offset: u64,
    pub xlen: xLen,
    pub locals: Vec<(u32, ValType)>,
    pub params: u32,
    pub table: u32,
    pub function_ty: u32,
    pub fastcall: Option<FastCall>,
    pub pinned: PinTracker,
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
