//! Linux x86_64 syscall numbers used by the thin runtime.

use super::{NativeSyscallImports, ParamSource, SyscallEntry, SyscallTable};

pub const SYS_WRITE: u64 = 1;
pub const SYS_EXIT: u64 = 60;

/// x86_64 Linux GPR locals in the speet-x86_64 layout: RAX=0, RDI=7, RSI=6, RDX=2.
pub const RAX: u32 = 0;
pub const RDI: u32 = 7;
pub const RSI: u32 = 6;
pub const RDX: u32 = 2;

/// Build the Linux x86_64 → native-host syscall table.
pub fn linux_x86_64_table(imports: &NativeSyscallImports) -> SyscallTable {
    let exit = SyscallEntry {
        func_idx: imports.exit,
        param_map: vec![ParamSource::LocalI64AsI32(RDI)],
        saves: vec![],
        result_local: None,
        negate_nonzero_result: false,
        has_return: false,
        terminates: true,
        memory_stores: vec![],
        load_mem_on_success: None,
    };
    let write = SyscallEntry {
        func_idx: imports.write,
        param_map: vec![
            ParamSource::LocalI64AsI32(RDI),
            ParamSource::LocalI64AsI32(RSI),
            ParamSource::LocalI64AsI32(RDX),
        ],
        saves: vec![],
        result_local: Some(RAX),
        negate_nonzero_result: false,
        has_return: true,
        terminates: false,
        memory_stores: vec![],
        load_mem_on_success: None,
    };
    SyscallTable::new(vec![(SYS_EXIT, exit), (SYS_WRITE, write)])
}
