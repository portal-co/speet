//! Phase 2b of the dynamic-JIT plan: in-process native (SysV ABI) compilation
//! via wasm-blitz, producing raw machine code executed directly through a
//! function pointer — the "effectively-native-to-native JIT" path, for the
//! thin-runtime (linked native executable, no WASM engine present at guest
//! runtime) deployment mode.
//!
//! **Scope note**: this compiles a standalone, **import-free** WASM function
//! to native code. wasm-blitz's production AOT pipeline
//! (`speet-recompile::drive`) resolves `Call`/`ReturnCall` to imported host
//! functions via relocations, patched in by an external linker (clang/lld);
//! reproducing that without a linker (true dynamic-linking-style relocation
//! resolution against live Rust function pointers) is a real, separate
//! design question, not attempted here — see the crate-root section of the
//! dynamic-JIT plan doc for why. This proves the more fundamental, novel
//! risk instead: that wasm-blitz's in-process codegen produces genuinely
//! executable native machine code for the *current host*, callable via a
//! raw function pointer with no object file or external tool involved.
//!
//! Also — per `speet-recompile/STATUS.md` — wasm-blitz's `ReturnCall` lowers
//! to `call`+`return`, not a true tail call, growing the native stack per
//! hop. Fine for the single-hop case this crate demonstrates; a blocking
//! prerequisite (tracked upstream in wasm-blitz, not fixed here) before this
//! track can handle long OOB dispatch chains.

use portal_solutions_blitz_aarch64::{AArch64Arch, AArch64Label, naive, sysv};
use portal_solutions_blitz_common::{HandleOpError, dce_pass, ops::mach_operators};
use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
use wasm_encoder::reencode::RoundtripReencoder;
use wasmparser::{FuncType as WpFuncType, Payload};

/// Parse a WASM module's type section and function-to-type-index mapping.
/// Import-free, single-function modules only (matches [`compile_native_arm64`]'s scope).
fn parse_sigs(wasm: &[u8]) -> (Vec<WpFuncType>, Vec<u32>) {
    let mut sigs_wp: Vec<WpFuncType> = Vec::new();
    let mut fsigs: Vec<u32> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        match payload {
            Payload::TypeSection(reader) => {
                for group in reader.into_iter().flatten() {
                    for subtype in group.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) = subtype.composite_type.inner {
                            sigs_wp.push(ft);
                        }
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                fsigs.extend(reader.into_iter().flatten());
            }
            _ => {}
        }
    }
    (sigs_wp, fsigs)
}

/// Compile an import-free, single-function WASM module to raw AArch64 SysV
/// machine code bytes — no object file, no external linker. Mirrors
/// `wasm-blitz/crates/blitz-tests/tests/e2e.rs`'s `compile_native_binary`
/// reference shape (the same in-process codegen call chain
/// `speet-recompile::drive` uses for AOT builds), narrowed to the
/// AArch64+SysV target this host actually runs.
pub fn compile_native_arm64(wasm: &[u8]) -> Vec<u8> {
    let (sigs_wp, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut reencoder = RoundtripReencoder;
    let mut ctx = ();

    let mut out = AArch64Writer::<AArch64Label>::new();
    let mut state = naive::State::default();
    for op in ops {
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out,
            &mut ctx,
            AArch64Arch::default(),
            &mut state,
            &[],
            &op.unwrap(),
            &mut reencoder,
            0,
        )
        .unwrap();
    }
    out.into_bytes()
}

/// An executable mapping of raw machine code, made with `mmap` (RWX — no
/// separate write-then-remap step; fine for a one-shot proof of concept, not
/// a production W^X policy).
pub struct ExecutableCode {
    ptr: *mut libc::c_void,
    len: usize,
}

impl ExecutableCode {
    /// Map `code` as executable memory: `mmap` RW, copy the bytes in, then
    /// `mprotect` to RX. A simultaneous-RWX `mmap` is refused on hardened
    /// runtimes (macOS in particular, without the `com.apple.security.cs.
    /// allow-jit`/`allow-unsigned-executable-memory` entitlements
    /// `os-codesign-macho` already models for the AOT thin-runtime path) —
    /// the W^X-toggle sequence used here is the portable way to get
    /// executable memory without those entitlements, and is also just the
    /// more correct default regardless of platform.
    pub fn new(code: &[u8]) -> Self {
        let len = code.len().max(1);
        // SAFETY: mmap with MAP_ANON|MAP_PRIVATE over no existing mapping;
        // the returned pointer is checked against MAP_FAILED before use.
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap failed");
        // SAFETY: `ptr` was just mapped RW for exactly `len` bytes.
        unsafe { core::ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, code.len()) };
        // SAFETY: same mapping, dropping W and adding X.
        let rc = unsafe { libc::mprotect(ptr, len, libc::PROT_READ | libc::PROT_EXEC) };
        assert_eq!(rc, 0, "mprotect(PROT_READ|PROT_EXEC) failed");
        Self { ptr, len }
    }

    /// Call the mapped code as `extern "C" fn(i64, i64) -> i64` (AArch64
    /// SysV: `x0,x1` in, `x0` out) — the calling convention
    /// `compile_native_arm64`'s SysV backend targets.
    ///
    /// # Safety
    /// The mapped bytes must actually implement this exact signature.
    pub unsafe fn call_i64_i64_to_i64(&self, a: i64, b: i64) -> i64 {
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(self.ptr) };
        f(a, b)
    }
}

impl Drop for ExecutableCode {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}
