//! Address-to-label plugin resource: guest-address → symbolic external-call
//! label, for both syscall-style platforms (a single trap instruction
//! dispatches by register value) and library-call-style platforms (a PLT
//! stub, import table, or vtable slot is reached by *address*).
//!
//! This module is the plugin-accessible generalization of what used to be
//! two separately-maintained, arch-specific `BTreeMap<u64, String>` +
//! `BTreeMap<String, u32>` field pairs duplicated inside
//! `speet-x86_64`/`speet-aarch64`'s recompiler structs. See
//! `docs/guides/thin-runtime-genericity.md` principle 1 and
//! `docs/guides/plugin-api.md` §2 for the resource-kind pattern this
//! extends.
//!
//! Kept data-in/data-out (no `Context`/`E` generics), per this crate's core
//! invariant — see the crate-level docs.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Identifies a distinct library/image an address-to-label table describes,
/// so a plugin can report entries from more than one independently
/// versioned library without address collisions. [`LibraryId::MAIN_IMAGE`]
/// is the default for today's single-binary, same-platform ELF/Mach-O case
/// (one guest binary, one flat address space).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LibraryId(pub u32);

impl LibraryId {
    pub const MAIN_IMAGE: LibraryId = LibraryId(0);
}

/// Minimal calling-convention description needed to marshal arguments/results
/// around a redirected call, without hand-matching an instruction shape at
/// the call site. Each native arch crate supplies symbol conventions via
/// its own `plt_calling_convention` helper — see `speet-x86_64` /
/// `speet-aarch64` and `speet_abi_stubs`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CallingConvention {
    /// Local indices (register slots) holding the callee's arguments, in
    /// order.
    pub arg_locals: Vec<u32>,
    /// Per-argument: whether the register's (64-bit) value must be narrowed
    /// with `i32.wrap_i64` before the call — set when the target WASM
    /// import declares an `i32` param but the source register slot is
    /// natively 64-bit (true of every general-purpose register on both
    /// aarch64 and x86_64). Empty means no argument needs narrowing (all
    /// pass through as-is, e.g. an import declaring `i64` params like
    /// `__speet_execve`). Parallel to `arg_locals`; a missing/absent entry
    /// for an index is treated as `false`.
    pub arg_wrap_i32: Vec<bool>,
    /// Local index the return value should be stored into, if any.
    pub result_local: Option<u32>,
    /// Whether the call's WASM result is `i32` and must be widened with
    /// `i64.extend_i32_u` before storing into `result_local` (a 64-bit
    /// register slot). Ignored when `result_local` is `None`.
    pub result_extend_i32: bool,
}

impl CallingConvention {
    /// Whether argument index `i` (into `arg_locals`) needs narrowing.
    pub fn wraps_i32(&self, i: usize) -> bool {
        self.arg_wrap_i32.get(i).copied().unwrap_or(false)
    }
}

/// One guest-address → symbolic-label hook, keyed by `(library, address)`
/// rather than a bare `u64` so a plugin can describe multiple
/// independently-versioned libraries without collision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalTargetEntry {
    pub library: LibraryId,
    pub address: u64,
    pub label: String,
}

/// A table of address-to-label entries for one or more libraries.
#[derive(Debug, Clone, Default)]
pub struct ExternalTargetTable {
    entries: BTreeMap<(LibraryId, u64), String>,
}

impl ExternalTargetTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, library: LibraryId, address: u64, label: impl Into<String>) {
        self.entries.insert((library, address), label.into());
    }

    pub fn lookup(&self, library: LibraryId, address: u64) -> Option<&str> {
        self.entries.get(&(library, address)).map(String::as_str)
    }

    /// Convenience for the common single-image case.
    pub fn lookup_main(&self, address: u64) -> Option<&str> {
        self.lookup(LibraryId::MAIN_IMAGE, address)
    }

    pub fn iter(&self) -> impl Iterator<Item = ExternalTargetEntry> + '_ {
        self.entries.iter().map(|(&(library, address), label)| ExternalTargetEntry {
            library,
            address,
            label: label.clone(),
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A plugin-supplied source of address-to-label entries for one library —
/// e.g. a specific libc/libSystem build, or a guest binary's own PLT.
///
/// Wiring into `speet-plugin-host*` as a sixth `PluginKind` is follow-up
/// work — see `docs/plugin-api.md`. In-process consumers can use
/// [`ExternalTargetTable`] / [`PltHookTable`] directly today.
pub trait ExternalTargetPlugin: Send + Sync {
    fn library_name(&self) -> &str;
    fn entries(&self) -> ExternalTargetTable;
}

/// How a PC-check hook realizes the redirect at recompile time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PltHookTarget {
    /// Call a WASM import (host tunnel / integrated hook).
    WasmImport { import_idx: u32 },
    /// Link-time ambient alias only — **not** emitted by the interim
    /// PC-check path; see `docs/future/redirect-shim-got.md`.
    Ambient,
}

/// A resolved redirect hook for one guest address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PltHook {
    pub label: String,
    pub target: PltHookTarget,
    pub convention: CallingConvention,
}

impl PltHook {
    pub fn import_idx(&self) -> Option<u32> {
        match self.target {
            PltHookTarget::WasmImport { import_idx } => Some(import_idx),
            PltHookTarget::Ambient => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PltHookTable {
    hooks: BTreeMap<(LibraryId, u64), PltHook>,
}

impl PltHookTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, library: LibraryId, address: u64, hook: PltHook) {
        self.hooks.insert((library, address), hook);
    }

    pub fn lookup(&self, library: LibraryId, address: u64) -> Option<&PltHook> {
        self.hooks.get(&(library, address))
    }

    pub fn lookup_main(&self, address: u64) -> Option<&PltHook> {
        self.lookup(LibraryId::MAIN_IMAGE, address)
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    pub fn len(&self) -> usize {
        self.hooks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_keys_by_library_and_address() {
        let mut t = ExternalTargetTable::new();
        t.insert(LibraryId::MAIN_IMAGE, 0x1000, "execve");
        t.insert(LibraryId(1), 0x1000, "other_lib_fn");
        assert_eq!(t.lookup_main(0x1000), Some("execve"));
        assert_eq!(t.lookup(LibraryId(1), 0x1000), Some("other_lib_fn"));
        assert_eq!(t.lookup(LibraryId(2), 0x1000), None);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn hook_table_keys_by_library_and_address() {
        let mut hooks = PltHookTable::new();
        hooks.insert(
            LibraryId::MAIN_IMAGE,
            0x2000,
            PltHook {
                label: "execve".into(),
                target: PltHookTarget::WasmImport { import_idx: 4 },
                convention: CallingConvention {
                    arg_locals: alloc::vec![7, 6, 2],
                    result_local: Some(0),
                    ..Default::default()
                },
            },
        );
        assert!(hooks.lookup(LibraryId(1), 0x2000).is_none());
        let hook = hooks.lookup_main(0x2000).expect("hook present");
        assert_eq!(hook.import_idx(), Some(4));
    }
}
