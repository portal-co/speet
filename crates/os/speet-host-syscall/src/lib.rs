//! Host-ambient syscall tables for the full-binary recompiler.
//!
//! Unlike `speet-linux-wasi` (which maps Linux syscalls to WASI imports), this
//! crate maps Linux RISC-V64 syscalls to **native host imports** that the
//! recompiled binary's runtime shim provides (and which ultimately call the host
//! libc / kernel). The recompiler renders these imports as `env__{name}` symbols
//! that the system linker resolves against the shim.
//!
//! v1 scope: `exit` and `write` — enough for a recompiled guest to
//! terminate with a code and emit output.

mod x86_64;

pub use x86_64::{linux_x86_64_table, RAX, RDI, RDX, RSI, SYS_EXIT, SYS_WRITE};

use speet_syscall::{ParamSource, SyscallEntry, SyscallTable};

/// RISC-V64 Linux argument registers: `a0..a2` = `x10..x12`, syscall number in
/// `a7` = `x17`.
pub const A0: u32 = 10;
pub const A1: u32 = 11;
pub const A2: u32 = 12;
pub const A7: u32 = 17;

/// WASM import indices the recompiled module reserves for the host shim.
#[derive(Clone, Copy, Debug)]
pub struct NativeSyscallImports {
    /// `env.exit : (i32 code) -> ()`
    pub exit: u32,
    /// `env.write : (i32 fd, i32 guest_ptr, i32 len) -> i32` — the shim adds the
    /// `__wasm_mem` base to `guest_ptr` before calling the host `write`.
    pub write: u32,
}

/// Build the Linux RISC-V64 → native-host syscall table.
pub fn linux_rv64_table(imports: &NativeSyscallImports) -> SyscallTable {
    // exit(code): pass a0 as the i32 exit code; never returns.
    let exit = SyscallEntry {
        func_idx: imports.exit,
        param_map: vec![ParamSource::LocalI64AsI32(A0)],
        saves: vec![],
        result_local: None,
        negate_nonzero_result: false,
        has_return: false,
        terminates: true,
        memory_stores: vec![],
        load_mem_on_success: None,
    };
    // write(fd, buf, len): pass a0/a1/a2; store the returned count back into a0.
    let write = SyscallEntry {
        func_idx: imports.write,
        param_map: vec![
            ParamSource::LocalI64AsI32(A0),
            ParamSource::LocalI64AsI32(A1),
            ParamSource::LocalI64AsI32(A2),
        ],
        saves: vec![],
        result_local: Some(A0),
        negate_nonzero_result: false,
        has_return: true,
        terminates: false,
        memory_stores: vec![],
        load_mem_on_success: None,
    };
    SyscallTable::new(vec![(93, exit), (64, write)])
}
