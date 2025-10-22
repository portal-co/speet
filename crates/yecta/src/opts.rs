use crate::*;
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
    pub inst_start: Option<u32>,
    pub feat_flags: Option<u32>,
    pub tail_calls_disabled: bool,
    pub exception_mode: Option<ExceptionMode>,
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ExceptionMode {
    pub reentrancy: Glocal,
    pub exn_flag: Glocal,
    pub kind: ExceptionModeKind,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ExceptionModeKind {
    Wasm { tag: u32 },
    ReturnBased,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Glocal {
    Global(u32),
    Local(u32),
}

impl Glocal {
    pub fn get<'a>(&self) -> Instruction<'a> {
        match self {
            Glocal::Global(a) => Instruction::GlobalGet(*a),
            Glocal::Local(a) => Instruction::LocalGet(*a),
        }
    }
    pub fn set<'a>(&self) -> Instruction<'a> {
        match self {
            Glocal::Global(a) => Instruction::GlobalSet(*a),
            Glocal::Local(a) => Instruction::LocalSet(*a),
        }
    }
}
