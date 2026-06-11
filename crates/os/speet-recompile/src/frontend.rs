//! Frontend stage: load a host binary and drive speet to a WASM megabinary.
//!
//! Responsibilities (M1 wires the minimal path; M3 adds PLT tunneling):
//! 1. Load via [`binary_io::load_auto`]; assert OS+arch == host (v1 same-platform).
//! 2. Build an external-target table from `imports` (undefined dyn syms + PLT
//!    addrs) and undefined-symbol relocations, mapping `plt_addr -> guest_name`.
//!    During speet translation, a call/jmp whose target is in this table is
//!    lowered via `AmbientSink::call_ambient(name, sig)` instead of an internal
//!    tail-call.
//! 3. Feed non-text sections as passive WASM data segments + a `data_init_fn`.
//! 4. Register the page-table memory model (`speet_memory::VirtualMemory`).
//! 5. Resolve the entry point to a megabinary export (`__guest_entry`).

use binary_io::{BinArch, BinOs, ImportSym, LoadedBinary};

/// External call targets discovered in the loaded binary: guest address (PLT
/// stub or relocation site target) -> external symbol name to tunnel.
#[derive(Debug, Default, Clone)]
pub struct ExternalTargets {
    pub by_plt_addr: std::collections::BTreeMap<u64, String>,
}

impl ExternalTargets {
    /// Build the table from a loaded binary's import list (PLT addresses).
    pub fn from_imports(imports: &[ImportSym]) -> Self {
        let mut by_plt_addr = std::collections::BTreeMap::new();
        for imp in imports {
            if let Some(addr) = imp.plt_addr {
                by_plt_addr.insert(addr, imp.name.clone());
            }
        }
        Self { by_plt_addr }
    }

    /// Look up an external symbol name for a call/jmp target address.
    pub fn lookup(&self, target: u64) -> Option<&str> {
        self.by_plt_addr.get(&target).map(|s| s.as_str())
    }
}

/// Returns the current host (OS, arch) so the driver can enforce same-platform.
pub fn host_platform() -> (BinOs, BinArch) {
    let os = if cfg!(target_os = "macos") {
        BinOs::MacOs
    } else {
        BinOs::Linux
    };
    let arch = if cfg!(target_arch = "aarch64") {
        BinArch::AArch64
    } else {
        BinArch::X86_64
    };
    (os, arch)
}

/// Assert the loaded binary matches the host platform (v1 constraint).
pub fn assert_same_platform(bin: &LoadedBinary) -> Result<(), String> {
    let (os, arch) = host_platform();
    if !matches!((&bin.os, os), (BinOs::Linux, BinOs::Linux) | (BinOs::MacOs, BinOs::MacOs)) {
        return Err(format!("input OS {:?} != host {:?} (v1 is same-platform)", bin.os, os));
    }
    if !matches!((&bin.arch, arch), (BinArch::X86_64, BinArch::X86_64) | (BinArch::AArch64, BinArch::AArch64)) {
        return Err(format!("input arch {:?} != host {:?} (v1 is same-platform)", bin.arch, arch));
    }
    Ok(())
}

// TODO(M1+): the actual speet translation driver:
//   - construct EntityIndexSpace, register VirtualMemory + host-syscall table,
//   - call speet_x86_64 / speet_aarch64 `translate_bytes` over the `.text`
//     section, passing `&ExternalTargets` so external call sites become ambient,
//   - accumulate BinaryUnits into a MegabinaryBuilder, attach data segments and
//     the `__guest_entry` export.
// This requires threading the speet ReactorContext; built up in M1.
