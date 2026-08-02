//! Compatibility shim that re-exports the generic `os-linux-wasi` crate and
//! adds `@speet`-specific index-space hooks (`register` / `declare`) that live
//! on this side of the dependency boundary.
//!
//! Phase 2 migration: encapsulate ecall handlers as linked module functions
//! (not inline `br_table` at guest sites) and route scratch copies through
//! `__speet_host_mem_*` imports pre-shimming — see `docs/guides/dual-backends.md`.

#![no_std]
extern crate alloc;

mod handler_module;

pub use handler_module::{emit_handler_function, HandlerModulePlan, HostMemImportIndices};
pub use os_linux_wasi::{IOVEC_SCRATCH_OFFSET, LinuxToWasi, WasiImports};

use speet_link_core::EntityIndexSpace;
use speet_module_target::ModuleTarget;
use wasm_encoder::ValType;

/// `@speet`-specific extension methods for declaring the four WASI preview1
/// imports that `os-linux-wasi` requires.
pub trait WasiImportsExt {
    /// Reserve four consecutive function indices in `entity_space`.
    fn register(entity_space: &mut EntityIndexSpace) -> Self;
    /// Declare the four WASI preview1 imports on a `ModuleTarget`.
    fn declare<Ctx, Err>(
        &self,
        target: &mut dyn ModuleTarget<Ctx, Err>,
        ctx: &mut Ctx,
    ) -> Result<(), Err>;
}

impl WasiImportsExt for WasiImports {
    fn register(entity_space: &mut EntityIndexSpace) -> Self {
        let slot = entity_space.functions.append(4);
        let base = entity_space.functions.base(slot);
        Self {
            fd_write: base + 0,
            fd_read: base + 1,
            fd_close: base + 2,
            proc_exit: base + 3,
        }
    }

    fn declare<Ctx, Err>(
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