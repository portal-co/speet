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
/// the call site. A plugin arch supplies its own via
/// [`crate::arch::ArchPlugin::calling_convention_for`] instead of requiring
/// a new hand-written match arm in `speet-recompile` every time a new
/// architecture is added.
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
/// Generalizes the old flat "one address space, one `env` module"
/// assumption in `speet_recompile::frontend::ExternalTargets`.
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
/// This is the trait a future out-of-process transport (subprocess/dylib/
/// WASM) would adapt; wiring it into `speet-plugin-host*`/`speet-plugin-adapter`
/// as a sixth `PluginKind` is intentionally left as follow-up work (adding a
/// sixth kind is a documented version bump — see `docs/plugin-api.md`), not
/// implemented in this pass. In-process consumers (native arch recompilers)
/// can use [`ExternalTargetTable`]/[`CallingConvention`] directly today.
pub trait ExternalTargetPlugin: Send + Sync {
    /// Human-readable identity for diagnostics (e.g. `"libSystem.B.dylib"`).
    fn library_name(&self) -> &str;
    /// All address-to-label entries this plugin knows about.
    fn entries(&self) -> ExternalTargetTable;
}

/// A resolved redirect hook: which WASM import (or other host-capability
/// slot) to call for a hooked guest address, and what calling convention to
/// marshal arguments/results with.
///
/// This is the *single shared type* `speet-x86_64` and `speet-aarch64` hold
/// by composition (`Option<PltHookTable>`) instead of each independently
/// declaring `plt_by_addr: BTreeMap<u64, String>` +
/// `plt_imports: BTreeMap<String, u32>` and a duplicated `lookup_plt_import`
/// method. See `docs/guides/thin-runtime-genericity.md` principle 1 — one
/// arch adding its own copy of this pair was exactly the drift this
/// consolidation prevents for a third/future arch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PltHook {
    pub label: String,
    pub import_idx: u32,
    pub convention: CallingConvention,
}

#[derive(Debug, Clone, Default)]
pub struct PltHookTable {
    hooks: BTreeMap<u64, PltHook>,
}

impl PltHookTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, address: u64, hook: PltHook) {
        self.hooks.insert(address, hook);
    }

    pub fn lookup(&self, address: u64) -> Option<&PltHook> {
        self.hooks.get(&address)
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
        t.insert(LibraryId(1), 0x1000, "other_lib_fn"); // same address, different library
        assert_eq!(t.lookup_main(0x1000), Some("execve"));
        assert_eq!(t.lookup(LibraryId(1), 0x1000), Some("other_lib_fn"));
        assert_eq!(t.lookup(LibraryId(2), 0x1000), None);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn hook_table_roundtrip() {
        let mut hooks = PltHookTable::new();
        hooks.insert(
            0x2000,
            PltHook {
                label: "execve".into(),
                import_idx: 4,
                convention: CallingConvention {
                    arg_locals: alloc::vec![7, 6, 2],
                    result_local: Some(0),
                    ..Default::default()
                },
            },
        );
        let hook = hooks.lookup(0x2000).expect("hook present");
        assert_eq!(hook.import_idx, 4);
        assert_eq!(hook.convention.arg_locals, alloc::vec![7, 6, 2]);
        assert!(hooks.lookup(0x3000).is_none());
    }
}
