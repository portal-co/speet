//! LLVM link orchestration.

use crate::execve_hook;
use crate::toolchain::{compile_c, link_executable, LlvmToolchain};
use binary_io::{BinArch, BinOs};
use speet_host_api::HostApi;
use speet_rt::{entry_bridge_c, generate_memory_tu, generate_shim};
use std::path::{Path, PathBuf};

/// Materialize and link guest + runtime objects into an executable.
///
/// `entry_param_count`/`sp_param_index` describe `__guest_entry`'s real WASM
/// signature (guest registers are WASM params — see
/// `speet_recompile::drive::{entry_param_count, sp_param_index}` for the
/// common same-arch case; a guest translated by a different frontend than
/// `arch`, e.g. RISC-V recompiled to an AArch64/x86-64 *output*, must instead
/// use that frontend's own constant, e.g. `speet_riscv::RiscVRecompiler::
/// SP_PARAM_INDEX` — the guest register layout is a property of whichever
/// frontend built the WASM, not of `arch`, which is only the output/link
/// target here). `halt_addr`/`lr_param_index` seed the halt-sentinel return
/// address (see `speet_recompile::frontend::halt_addr`,
/// `speet_recompile::drive::lr_param_index`, and
/// `docs/guides/thin-runtime-genericity.md` principle 4) — a normal-feature
/// contract, not a test-only convenience: real guest programs may return
/// out of `main`. See `docs/guides/thin-runtime-genericity.md`.
#[allow(clippy::too_many_arguments)]
pub fn link_guest(
    tc: &LlvmToolchain,
    host: &dyn HostApi,
    guest_obj: &[u8],
    arch: BinArch,
    os: BinOs,
    entry_param_count: u32,
    sp_param_index: u32,
    halt_addr: u64,
    lr_param_index: Option<u32>,
    work_dir: &Path,
    out_exe: &Path,
) -> Result<(), String> {
    link_guest_integrated(
        tc, host, guest_obj, arch, os, entry_param_count, sp_param_index, halt_addr,
        lr_param_index, work_dir, out_exe,
    )
}

/// Integrated link: shim + entry bridge + execve hook.
#[allow(clippy::too_many_arguments)]
pub fn link_guest_integrated(
    tc: &LlvmToolchain,
    host: &dyn HostApi,
    guest_obj: &[u8],
    arch: BinArch,
    os: BinOs,
    entry_param_count: u32,
    sp_param_index: u32,
    halt_addr: u64,
    lr_param_index: Option<u32>,
    work_dir: &Path,
    out_exe: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(work_dir).map_err(|e| e.to_string())?;
    let guest_path = work_dir.join("guest.o");
    std::fs::write(&guest_path, guest_obj).map_err(|e| e.to_string())?;

    let mem_path = work_dir.join("mem.o");
    compile_c(tc, &generate_memory_tu(), &mem_path, arch, os)?;

    let shim_path = work_dir.join("shim.o");
    let shim_src = generate_shim_integrated(&host.import_manifest());
    compile_c(tc, &shim_src, &shim_path, arch, os)?;

    let bridge_path = work_dir.join("entry_bridge.o");
    let bridge_src = entry_bridge_c(entry_param_count, sp_param_index, halt_addr, lr_param_index);
    compile_c(tc, &bridge_src, &bridge_path, arch, os)?;

    let hook_path = work_dir.join("execve_hook.o");
    compile_c(tc, &execve_hook::generate_execve_hook_c(), &hook_path, arch, os)?;

    let recipe = host.link_recipe();
    let extra = dylib_flags_for_os(os, &recipe.dylib_flags);

    link_executable(
        tc,
        &[&guest_path, &mem_path, &shim_path, &bridge_path, &hook_path],
        out_exe,
        arch,
        os,
        &extra,
    )
}

fn generate_shim_integrated(manifest: &speet_host_api::ImportManifest) -> String {
    let mut src = generate_shim(manifest);
    if src.contains("int main(void)") {
        src = src.replace(
            "int main(void) {",
            "int main(int argc, char **argv) {",
        );
        src = src.replace(
            "    speet_rt_bootstrap();\n    __guest_entry();",
            "    speet_rt_bootstrap();\n    __speet_start(argc, argv);",
        );
    }
    src
}

fn dylib_flags_for_os(os: BinOs, recipe_flags: &[String]) -> Vec<String> {
    if !recipe_flags.is_empty() {
        return recipe_flags.to_vec();
    }
    match os {
        BinOs::Linux => vec!["-lc".to_string()],
        BinOs::MacOs => vec!["-lSystem".to_string()],
    }
}

/// Convenience: write guest bytes to `work_dir/guest.o` path.
pub fn write_guest_object(work_dir: &Path, guest_obj: &[u8]) -> Result<PathBuf, String> {
    let p = work_dir.join("guest.o");
    std::fs::write(&p, guest_obj).map_err(|e| e.to_string())?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrated_shim_uses_speet_start() {
        let src = generate_shim_integrated(&speet_host_api::ImportManifest::integrated_native());
        assert!(src.contains("int main(int argc, char **argv)"));
        assert!(src.contains("__speet_start(argc, argv)"));
        assert!(src.contains("__speet_execve_hook"));
    }
}
