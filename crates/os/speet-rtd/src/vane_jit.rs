//! Daemon-side JIT backends (Phase 4 of the dynamic-JIT plan).
//!
//! Swap-in replacements for the in-process compile step Phase 2a/2b already
//! proved (`speet-dynamic-jit`, `speet-native-jit`) — same codegen, invoked
//! from the daemon over the wire instead of from a background thread inside
//! the guest's own process. Registered only when this crate is built with
//! the `jit` feature (see `Cargo.toml`'s comment).

use os_jit_core::{JitArtifact, JitBackend, JitBackendId, JitError, JitRequest};
use vane_riscv::Mem;

/// The container-megabinary/WASM-engine track: compiles a guest region to a
/// standalone WASM module via vane's real RV64 frontend. See
/// `speet-dynamic-jit::compile_pc`, proved end-to-end against a live
/// dynamic dispatch table in that crate's tests.
pub struct VaneWasmJitBackend;

impl JitBackend for VaneWasmJitBackend {
    fn id(&self) -> JitBackendId {
        JitBackendId("vane-wasm")
    }

    fn compile(&mut self, req: &JitRequest) -> Result<JitArtifact, JitError> {
        if req.guest_bytes.is_empty() {
            return Err(JitError::Decode("empty guest_bytes".into()));
        }
        let mut mem = Mem::default();
        for (i, b) in req.guest_bytes.iter().enumerate() {
            mem.write_byte(req.guest_pc + i as u64, *b);
        }
        let wasm = speet_dynamic_jit::compile_pc(&mem, req.guest_pc, req.layout.num_regs);
        Ok(JitArtifact::Wasm(wasm))
    }
}

/// AArch64 sibling of [`VaneWasmJitBackend`]: compiles a guest region to a
/// standalone WASM module via vane's real AArch64 frontend (unblocked by
/// Phase 1's `state_slots` lowering). Registered under a distinct id —
/// `"vane-wasm"` stays RISC-V-only — so a client selects the ISA explicitly
/// rather than this backend guessing from `guest_bytes` alone. See
/// `speet-dynamic-jit::compile_pc_aarch64`, proved end-to-end against a live
/// dynamic dispatch table (both `TableIndirect` and, feature-gated,
/// `FunctionRef` dispatch modes) in that crate's tests.
pub struct VaneWasmAarch64JitBackend;

impl JitBackend for VaneWasmAarch64JitBackend {
    fn id(&self) -> JitBackendId {
        JitBackendId("vane-wasm-aarch64")
    }

    fn compile(&mut self, req: &JitRequest) -> Result<JitArtifact, JitError> {
        if req.guest_bytes.is_empty() {
            return Err(JitError::Decode("empty guest_bytes".into()));
        }
        // AArch64's register file is architecturally fixed (73 named state
        // slots — x0-x30, NZCV, V0-V31 — see `compile_pc_aarch64`'s doc
        // comment), unlike RISC-V's caller-adjustable `num_regs`, so a
        // mismatched request is a genuine caller error rather than
        // something to silently reinterpret.
        if req.layout.num_regs != 73 {
            return Err(JitError::Unsupported(format!(
                "vane-wasm-aarch64: num_regs must be 73 (AArch64's full architectural state), got {}",
                req.layout.num_regs
            )));
        }
        let mut mem = vane_aarch64::Mem::default();
        for (i, b) in req.guest_bytes.iter().enumerate() {
            mem.write_byte(req.guest_pc + i as u64, *b);
        }
        let wasm = speet_dynamic_jit::compile_pc_aarch64(&mem, req.guest_pc);
        Ok(JitArtifact::Wasm(wasm))
    }
}

/// The thin-runtime/native track: compiles the same guest region to raw
/// AArch64 machine code via wasm-blitz, reusing `VaneWasmJitBackend`'s vane
/// output purely as an intermediate representation (never executed as
/// WASM). See `speet-native-jit`, proved end-to-end for import-free modules.
///
/// **Not yet functional for real requests**: `speet-native-jit::
/// compile_native_arm64` only handles import-free WASM, but
/// `speet_dynamic_jit::compile_pc`'s output always imports `ecall`/
/// `jit_invalidate`/`lookup_stub`/`guest_memory` — resolving those without
/// an external linker is genuine dynamic-relocation work, a separate design
/// question `speet-native-jit`'s crate doc calls out and this backend does
/// not attempt. Registered anyway so the daemon's dispatch/registry wiring
/// is complete and a client asking for `"vane-blitz"` gets a clear,
/// structured [`JitError::Unsupported`] rather than "unknown backend".
pub struct VaneBlitzJitBackend;

impl JitBackend for VaneBlitzJitBackend {
    fn id(&self) -> JitBackendId {
        JitBackendId("vane-blitz")
    }

    fn compile(&mut self, _req: &JitRequest) -> Result<JitArtifact, JitError> {
        Err(JitError::Unsupported(
            "vane-blitz: import resolution for compiled traces (ecall/jit_invalidate/lookup_stub/\
             guest_memory) is not yet implemented -- see speet-native-jit's crate doc"
                .into(),
        ))
    }
}
