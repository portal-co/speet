#![no_std]

extern crate alloc;

use speet_link_core::EntityIndexSpace;
use speet_module_target::ModuleTarget;
use speet_syscall::{ParamSource, SyscallEntry, SyscallTable};
use wasm_encoder::ValType;

/// The 16-byte memory scratch area offset used for iovec marshalling.
pub const IOVEC_SCRATCH_OFFSET: u32 = 0x200;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasiImports {
    pub fd_write: u32,  // (fd i32, iovs i32, iovs_len i32, nwritten i32) -> i32
    pub fd_read: u32,   // (fd i32, iovs i32, iovs_len i32, nread i32) -> i32
    pub fd_close: u32,  // (fd i32) -> i32
    pub proc_exit: u32, // (code i32) -> ()
}

impl WasiImports {
    /// Declare all needed WASI preview1 imports in `entity_space` (pass 1).
    pub fn register(entity_space: &mut EntityIndexSpace) -> Self {
        let slot = entity_space.functions.append(4);
        let base = entity_space.functions.base(slot);
        Self {
            fd_write: base + 0,
            fd_read: base + 1,
            fd_close: base + 2,
            proc_exit: base + 3,
        }
    }

    /// Declare all needed WASI preview1 imports to the module target (pass 2).
    pub fn declare<Ctx, Err>(
        &self,
        target: &mut dyn ModuleTarget<Ctx, Err>,
        ctx: &mut Ctx,
    ) -> Result<(), Err> {
        let fd_write = target.declare_func_import(
            ctx,
            "wasi_snapshot_preview1",
            "fd_write",
            &[ValType::I32, ValType::I32, ValType::I32, ValType::I32],
            &[ValType::I32],
        )?;
        assert_eq!(fd_write, self.fd_write);

        let fd_read = target.declare_func_import(
            ctx,
            "wasi_snapshot_preview1",
            "fd_read",
            &[ValType::I32, ValType::I32, ValType::I32, ValType::I32],
            &[ValType::I32],
        )?;
        assert_eq!(fd_read, self.fd_read);

        let fd_close = target.declare_func_import(
            ctx,
            "wasi_snapshot_preview1",
            "fd_close",
            &[ValType::I32],
            &[ValType::I32],
        )?;
        assert_eq!(fd_close, self.fd_close);

        let proc_exit = target.declare_func_import(
            ctx,
            "wasi_snapshot_preview1",
            "proc_exit",
            &[ValType::I32],
            &[],
        )?;
        assert_eq!(proc_exit, self.proc_exit);

        Ok(())
    }
}

pub struct LinuxToWasi {
    pub imports: WasiImports,
    pub iovec_scratch_offset: u32,
}

impl LinuxToWasi {
    pub fn new(imports: WasiImports) -> Self {
        Self {
            imports,
            iovec_scratch_offset: IOVEC_SCRATCH_OFFSET,
        }
    }

    /// Build the `SyscallTable` for the RV64 Linux ABI.
    ///
    /// `xn_local(n)` maps RISC-V register index to WASM local index.
    /// For the default layout (locals 0-31 = x0-x31): `|n| n as u32`.
    pub fn build_table(&self, xn_local: impl Fn(u8) -> u32) -> SyscallTable {
        use speet_syscall::MemoryStore;

        let a0_local = xn_local(10);
        let a1_local = xn_local(11);
        let a2_local = xn_local(12);

        let mut entries = alloc::vec::Vec::new();

        // 1. Read (syscall 63)
        // WASI: fd_read(fd, iovs, iovs_len, nread) -> wasi_errno
        // We write the buffer ptr (a1) to SCRATCH_OFFSET + 0 (4 bytes),
        // and the length (a2) to SCRATCH_OFFSET + 4 (4 bytes).
        // Then call fd_read(a0, SCRATCH_OFFSET, 1, SCRATCH_OFFSET + 8).
        // On success (result = 0), load read bytes from SCRATCH_OFFSET + 8.
        entries.push((
            63,
            SyscallEntry {
                func_idx: self.imports.fd_read,
                param_map: alloc::vec![
                    ParamSource::LocalI64AsI32(a0_local),
                    ParamSource::ConstI32(self.iovec_scratch_offset as i32),
                    ParamSource::ConstI32(1),
                    ParamSource::ConstI32((self.iovec_scratch_offset + 8) as i32),
                ],
                saves: alloc::vec![],
                result_local: Some(a0_local),
                negate_nonzero_result: true,
                has_return: true,
                memory_stores: alloc::vec![
                    MemoryStore {
                        addr: self.iovec_scratch_offset,
                        value_local: a1_local,
                        value_is_i64: true,
                    },
                    MemoryStore {
                        addr: self.iovec_scratch_offset + 4,
                        value_local: a2_local,
                        value_is_i64: true,
                    },
                ],
                load_mem_on_success: Some(self.iovec_scratch_offset + 8),
            },
        ));

        // 2. Write (syscall 64)
        // WASI: fd_write(fd, iovs, iovs_len, nwritten) -> wasi_errno
        // We write the buffer ptr (a1) to SCRATCH_OFFSET + 0 (4 bytes),
        // and the length (a2) to SCRATCH_OFFSET + 4 (4 bytes).
        // Then call fd_write(a0, SCRATCH_OFFSET, 1, SCRATCH_OFFSET + 8).
        // On success (result = 0), load written bytes from SCRATCH_OFFSET + 8.
        entries.push((
            64,
            SyscallEntry {
                func_idx: self.imports.fd_write,
                param_map: alloc::vec![
                    ParamSource::LocalI64AsI32(a0_local),
                    ParamSource::ConstI32(self.iovec_scratch_offset as i32),
                    ParamSource::ConstI32(1),
                    ParamSource::ConstI32((self.iovec_scratch_offset + 8) as i32),
                ],
                saves: alloc::vec![],
                result_local: Some(a0_local),
                negate_nonzero_result: true,
                has_return: true,
                memory_stores: alloc::vec![
                    MemoryStore {
                        addr: self.iovec_scratch_offset,
                        value_local: a1_local,
                        value_is_i64: true,
                    },
                    MemoryStore {
                        addr: self.iovec_scratch_offset + 4,
                        value_local: a2_local,
                        value_is_i64: true,
                    },
                ],
                load_mem_on_success: Some(self.iovec_scratch_offset + 8),
            },
        ));

        // 3. Close (syscall 57)
        // WASI: fd_close(fd) -> wasi_errno
        entries.push((
            57,
            SyscallEntry {
                func_idx: self.imports.fd_close,
                param_map: alloc::vec![ParamSource::LocalI64AsI32(a0_local)],
                saves: alloc::vec![],
                result_local: Some(a0_local),
                negate_nonzero_result: true,
                has_return: true,
                memory_stores: alloc::vec![],
                load_mem_on_success: None,
            },
        ));

        // 4. Exit (syscall 93)
        // WASI: proc_exit(code) -> (never returns)
        entries.push((
            93,
            SyscallEntry {
                func_idx: self.imports.proc_exit,
                param_map: alloc::vec![ParamSource::LocalI64AsI32(a0_local)],
                saves: alloc::vec![],
                result_local: None,
                negate_nonzero_result: false,
                has_return: false,
                memory_stores: alloc::vec![],
                load_mem_on_success: None,
            },
        ));

        // 5. ExitGroup (syscall 94)
        // WASI: proc_exit(code) -> (never returns)
        entries.push((
            94,
            SyscallEntry {
                func_idx: self.imports.proc_exit,
                param_map: alloc::vec![ParamSource::LocalI64AsI32(a0_local)],
                saves: alloc::vec![],
                result_local: None,
                negate_nonzero_result: false,
                has_return: false,
                memory_stores: alloc::vec![],
                load_mem_on_success: None,
            },
        ));

        SyscallTable::new(entries)
    }
}
