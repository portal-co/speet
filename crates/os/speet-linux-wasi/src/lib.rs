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
mod link;
mod manifest;
mod merge;
mod translate;

pub use assemble::recompile_rv64_wasi_to_wasm;
pub use link::HOST_MEMORY_INDEX;
pub use ecall::{A0_LOCAL, A1_LOCAL, A2_LOCAL, A7_LOCAL, LinuxWasiEcall};
pub use guest_module::{
    guest_export_names, guest_func_imports, guest_defined_fn_count, guest_import_count,
    handler_export_name, validate_canonical_guest, wasi_imports_for_guest_layout,
    wasi_imports_from_guest_imports, GuestFuncImport, HandlerModulePlan,
    CANONICAL_GUEST_WASM, EXPORT_HANDLER_CLOSE, EXPORT_HANDLER_EXIT, EXPORT_HANDLER_READ,
    EXPORT_HANDLER_WRITE, EXPORT_SYSCALL_DISPATCH,
};
pub use link::{
    func_export_index, link_canonical_guest_wasm, link_wasi_megabinary,
    link_wasi_megabinary_with_plan, WasiLinkPlan,
};
pub use manifest::wasi_preview1_manifest;
pub use merge::{extract_guest_defined_module, extract_guest_handlers, GuestDefinedModule, GuestHandlerFunctions};
pub use translate::{translate_rv64_wasi, WasiTranslation};
pub use os_linux_wasi::{IOVEC_SCRATCH_OFFSET, LinuxToWasi, WasiImports};

use speet_link_core::EntityIndexSpace;
use speet_module_target::ModuleTarget;

/// `@speet`-specific extension methods for declaring guest-module imports on a megabinary.
pub trait WasiImportsExt {
    /// Reserve import indices in `entity_space` matching [`guest_module::guest_func_imports`].
    fn register(entity_space: &mut EntityIndexSpace) -> Self;
    /// Declare guest-module imports (order + signatures) on a `ModuleTarget`.
    fn declare<Ctx, Err>(
        &self,
        target: &mut dyn ModuleTarget<Ctx, Err>,
        ctx: &mut Ctx,
    ) -> Result<(), Err>;
}

impl WasiImportsExt for WasiImports {
    fn register(entity_space: &mut EntityIndexSpace) -> Self {
        let imports = guest_module::guest_func_imports(guest_module::CANONICAL_GUEST_WASM);
        let slot = entity_space.functions.append(imports.len() as u32);
        let base = entity_space.functions.base(slot);
        guest_module::wasi_imports_from_guest_imports(base, &imports)
    }

    fn declare<Ctx, Err>(
        &self,
        target: &mut dyn ModuleTarget<Ctx, Err>,
        ctx: &mut Ctx,
    ) -> Result<(), Err> {
        for imp in guest_module::guest_func_imports(guest_module::CANONICAL_GUEST_WASM) {
            let idx = target.declare_func_import(
                ctx,
                &imp.module,
                &imp.name,
                &imp.params,
                &imp.results,
            )?;
            let expected = match imp.name.as_str() {
                "fd_write" => self.fd_write,
                "fd_read" => self.fd_read,
                "fd_close" => self.fd_close,
                "proc_exit" => self.proc_exit,
                other => panic!("unexpected WASI import in guest module: {other}"),
            };
            assert_eq!(idx, expected);
        }
        Ok(())
    }
}
