//! Backend stage: lower a WASM megabinary to native machine code via wasm-blitz,
//! then emit a relocatable object via `binary-io`.
//!
//! Generalizes wasm-blitz's `compile_native_binary`
//! (`crates/blitz-tests/tests/e2e.rs`), with one required change: use the new
//! `into_parts_with_relocs()` (added to the asm-arch binary emitters) instead of
//! `into_bytes()`, so unresolved `External`/`Ambient` labels become
//! undefined-symbol relocations rather than being dropped.
//!
//! Plan:
//! 1. Encode the MegabinaryOutput to WASM bytes, then `mach_operators` per fn.
//! 2. Emit ALL functions into ONE asm-arch binary writer (IcedWriter<X64Label> /
//!    AArch64Writer<AArch64Label>) so cross-function `Func`/`Indexed` labels bind
//!    internally as PC-relative; `place_label(Func{fn})` at each function start.
//!    The naive/SysV `State` MUST set `mem_base = MemBase::WasmMemSymbol` so loads
//!    and stores reference the `__wasm_mem` base provided by `speet-rt` (rather
//!    than treating the WASM address as a raw host pointer). See STATUS.md #1.
//! 3. `let (text, _labels, unresolved) = writer.into_parts_with_relocs();`
//! 4. Map each unresolved reloc to a `binary_io::ObjReloc`:
//!      External{name}  -> symbol `name`
//!      Ambient{name}   -> symbol `__ambient_{name}`
//!    and the asm `AsmRelocKind` -> `binary_io::RelocKind`
//!    (x86 Branch32 -> X86Plt32, Pcrel32 -> X86Pc32;
//!     aarch64 Call26 -> A64Call26, Jump26 -> A64Jump26, AdrPrel21 -> A64AdrPrel21).
//! 5. Hand (text, data, defined syms incl. `__guest_entry`, relocs) to
//!    `binary_io::{ElfObjectWriter, MachOObjectWriter}::write_object`.

use binary_io::RelocKind;

/// Map an asm-x86-64 unresolved-reloc kind to the neutral [`RelocKind`].
pub fn map_x86_reloc_kind_branch() -> RelocKind {
    // CALL/JMP/Jcc external target.
    RelocKind::X86Plt32
}

/// Map an asm-x86-64 LEA (RIP-relative address) external reference.
pub fn map_x86_reloc_kind_pcrel() -> RelocKind {
    RelocKind::X86Pc32
}

/// Render an `Ambient { name }` label to its emitted object symbol.
pub fn ambient_symbol(name: &str) -> String {
    format!("__ambient_{name}")
}

// TODO(M1+): the concrete blitz drive loop. It needs the wasm-blitz backend
// entry (`Sink::start_fn` / `mach_operators` + `handle_op`) wired against a
// single IcedWriter/AArch64Writer; built up in M1 once the frontend produces a
// MegabinaryOutput.
