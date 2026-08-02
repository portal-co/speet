//! Compatibility shim that re-exports the generic `os-linux-wasi` crate and
//! adds `@speet`-specific index-space hooks (`register` / `declare`) that live
//! on this side of the dependency boundary.
//!
//! Linux→WASI handler bodies live in [`speet-linux-wasi-guest`] (Rust → WASM),
//! are host-mem lowered at build time, and exposed as [`CANONICAL_GUEST_WASM`].

#![no_std]
extern crate alloc;

mod assemble;
mod ecall;
mod guest_module;
mod manifest;
mod merge;
mod translate;

pub use assemble::{assemble_wasi_module, recompile_rv64_wasi_to_wasm, HOST_MEMORY_INDEX};
pub use ecall::{A0_LOCAL, A1_LOCAL, A2_LOCAL, A7_LOCAL, LinuxWasiEcall};
pub use guest_module::{
    guest_export_names, handler_export_name, validate_canonical_guest, HandlerModulePlan,
    CANONICAL_GUEST_WASM, EXPORT_HANDLER_CLOSE, EXPORT_HANDLER_EXIT, EXPORT_HANDLER_READ,
    EXPORT_HANDLER_WRITE, EXPORT_SYSCALL_DISPATCH,
};
pub use manifest::wasi_preview1_manifest;
pub use merge::{extract_guest_handlers, GuestHandlerFunctions};
pub use translate::{translate_rv64_wasi, WasiTranslation};
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
