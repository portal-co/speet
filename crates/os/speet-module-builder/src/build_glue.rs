//! Basic BuildGlue wiring for `MegabinaryBuilder`.
//!
//! This is intentionally a minimal, functionally-correct shape implementation:
//! every required method emits the corresponding `OsOp` stack operations rather
//! than producing backend-specific handles.  It is intended to exercise the
//! `os-build` trait contract and compile against every `Backend`; replacing
//! these stubs with real target-aware emission is future work as `os-page-codegen`
//! and `os-syscall-emit` mature.

use os_build::{
    AbiSpec, BuildGlue, DispatchEntry, GlueError, GuestMemory, MemoryAccessOp,
    MemoryCodegen, MemorySpec, PltRedirect, RedirectCodegen, SyscallCodegen,
    SyscallTable,
};
use os_target_core::{Backend, GuestAddr, MemWidth, OsOp};

use crate::MegabinaryBuilder;

// ----------------------------------------------------------------------------
// Compile-time memory view
// ----------------------------------------------------------------------------

impl<F, B: Backend> GuestMemory<B> for MegabinaryBuilder<F> {
    fn emit_page_size(&self) -> u64 {
        // WASM page size: 64 KiB. This is the compile-time default used while
        // emitting memory helpers, regardless of the runtime backing.
        65536
    }

    fn emit_load(&mut self, backend: &mut B, width: MemWidth, signed: bool) {
        backend.op(OsOp::Load { width, signed });
    }

    fn emit_store(&mut self, backend: &mut B, width: MemWidth) {
        backend.op(OsOp::Store { width });
    }
}

// ----------------------------------------------------------------------------
// Codegen supertraits
// ----------------------------------------------------------------------------

impl<F, B: Backend> MemoryCodegen<B> for MegabinaryBuilder<F> {
    fn emit_memory_access(&mut self, backend: &mut B, op: &MemoryAccessOp) {
        if op.write {
            backend.op(OsOp::Store { width: op.width });
        } else {
            backend.op(OsOp::Load {
                width: op.width,
                signed: op.signed,
            });
        }
    }

    fn emit_page_table_glue(&mut self, _backend: &mut B, _spec: &MemorySpec) {
        // Placeholder: real page-table emission will come from os-page-codegen.
    }
}

impl<F, B: Backend> SyscallCodegen<B> for MegabinaryBuilder<F> {
    fn emit_syscall_dispatch(&mut self, backend: &mut B, _table: &SyscallTable) {
        // Placeholder: a real implementation would emit a br_table/dispatch over
        // `_table.entries`. For now we emit a single ecall token so the trait
        // is satisfied and code can compile.
        backend.op(OsOp::Ecall { may_await: false });
    }

    fn emit_osfuncall_stub(&mut self, backend: &mut B, _spec: &AbiSpec, symbol: &str) {
        backend.op(OsOp::TailCall {
            helper: symbol.into(),
        });
    }
}

impl<F, B: Backend> RedirectCodegen<B> for MegabinaryBuilder<F> {
    fn emit_redirect_stub(&mut self, backend: &mut B, redirect: &PltRedirect) {
        backend.op(OsOp::TailCall {
            helper: redirect.import_name.clone(),
        });
    }
}

// ----------------------------------------------------------------------------
// BuildGlue
// ----------------------------------------------------------------------------

impl<F, B: Backend> BuildGlue<B> for MegabinaryBuilder<F> {
    fn emit_jump_to_address(&mut self, backend: &mut B, target: GuestAddr) {
        backend.op(OsOp::Jump { target });
    }

    fn reserve_os_glue(
        &mut self,
        _backend: &mut B,
        _spec: &os_build::OsGlueSpec,
    ) -> Result<(), GlueError> {
        Ok(())
    }

    fn emit_dispatch_entry(&mut self, backend: &mut B, entries: &[DispatchEntry]) {
        for entry in entries {
            backend.op(OsOp::PushU64(entry.hash_id));
            backend.op(OsOp::PushU64(entry.guest_address));
            backend.op(OsOp::Jump { target: entry.guest_address });
        }
    }

    fn emit_memory_glue(&mut self, _backend: &mut B, _spec: &MemorySpec) -> Result<(), GlueError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use os_build::{MemoryAccessOp, OsGlueSpec};
    use os_target_core::MemWidth;

    #[test]
    fn build_glue_emits_ops() {
        let mut builder: MegabinaryBuilder<u32> = MegabinaryBuilder::new();
        let mut ops: Vec<OsOp> = Vec::new();

        assert_eq!(<MegabinaryBuilder<u32> as GuestMemory<Vec<OsOp>>>::emit_page_size(&builder), 65536);

        builder.emit_jump_to_address(&mut ops, 0x1000);
        builder.emit_memory_access(
            &mut ops,
            &MemoryAccessOp {
                width: MemWidth::W32,
                write: false,
                signed: true,
            },
        );
        builder.emit_redirect_stub(
            &mut ops,
            &PltRedirect {
                guest_symbol: "write".into(),
                import_name: "wasi_snapshot_preview1::fd_write".into(),
                import_module: "wasi_snapshot_preview1".into(),
            },
        );
        builder
            .reserve_os_glue(&mut ops, &OsGlueSpec::default())
            .unwrap();
        builder.emit_memory_glue(&mut ops, &MemorySpec::default()).unwrap();

        assert!(ops.iter().any(|op| matches!(op, OsOp::Jump { .. })));
        assert!(ops.iter().any(|op| matches!(op, OsOp::Load { .. })));
        assert!(ops.iter().any(|op| matches!(op, OsOp::TailCall { .. })));
    }
}