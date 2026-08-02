//! LLVM link orchestration.

use crate::toolchain::{compile_c, compile_c_path, link_executable, LlvmToolchain};
use binary_io::{BinArch, BinOs};
use speet_host_api::HostApi;
use speet_recompile::frontend::DataSegment;
use speet_rt::{
    entry_bridge_c, entry_bridge_direct_c, entry_stub_symbol, generate_data_segments_c,
    generate_guest_stubs_c, generate_memory_tu, generate_shim, DataSegmentBytes, GuestStubEntry,
};
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
/// `halt_addr` is retained for callers that still derive guest layout from
/// the binary; link uses catalog stub symbols for halt seeding instead.
#[allow(clippy::too_many_arguments)]
pub fn link_guest(
    tc: &LlvmToolchain,
    host: &dyn HostApi,
    guest_obj: &[u8],
    arch: BinArch,
    os: BinOs,
    entry_param_count: u32,
    sp_param_index: u32,
    _halt_addr: u64,
    lr_param_index: Option<u32>,
    work_dir: &Path,
    out_exe: &Path,
) -> Result<(), String> {
    link_guest_integrated(
        tc,
        host,
        guest_obj,
        arch,
        os,
        entry_param_count,
        sp_param_index,
        lr_param_index,
        &[],
        0,
        0,
        Some(_halt_addr),
        &[],
        work_dir,
        out_exe,
    )
}

/// Integrated link: shim + entry bridge + execve hook + optional per-guest-function stubs.
#[allow(clippy::too_many_arguments)]
pub fn link_guest_integrated(
    tc: &LlvmToolchain,
    host: &dyn HostApi,
    guest_obj: &[u8],
    arch: BinArch,
    os: BinOs,
    entry_param_count: u32,
    sp_param_index: u32,
    lr_param_index: Option<u32>,
    stub_entries: &[GuestStubEntry],
    entry_local_idx: u32,
    halt_local_idx: u32,
    legacy_halt_addr: Option<u64>,
    data_segments: &[DataSegment],
    work_dir: &Path,
    out_exe: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(work_dir).map_err(|e| e.to_string())?;
    let guest_path = work_dir.join("guest.o");
    std::fs::write(&guest_path, guest_obj).map_err(|e| e.to_string())?;

    let mem_path = work_dir.join("mem.o");
    compile_c(tc, &generate_memory_tu(), &mem_path, arch, os)?;

    let abi_pad_args: u32 = match arch {
        BinArch::X86_64 => 6,
        BinArch::AArch64 => 0,
    };

    let bridge_path = work_dir.join("entry_bridge.o");
    let bridge_src = if stub_entries.is_empty() {
        entry_bridge_direct_c(
            entry_param_count,
            sp_param_index,
            legacy_halt_addr.unwrap_or(0),
            lr_param_index,
            abi_pad_args,
        )
    } else {
        let halt_guest_pc = stub_entries
            .iter()
            .find(|e| e.local_func_idx == halt_local_idx)
            .map(|e| e.guest_pc)
            .unwrap_or_else(|| legacy_halt_addr.unwrap_or(0));
        entry_bridge_c(
            entry_param_count,
            sp_param_index,
            lr_param_index,
            &entry_stub_symbol(entry_local_idx),
            halt_guest_pc,
        )
    };
    compile_c(tc, &bridge_src, &bridge_path, arch, os)?;

    let shim_path = work_dir.join("shim.o");
    let shim_src = generate_shim_integrated(&host.import_manifest());
    compile_c_with_shim_include(tc, &shim_src, &shim_path, arch, os)?;

    let mut owned_objs: Vec<std::path::PathBuf> = vec![
        guest_path.clone(),
        mem_path.clone(),
        shim_path.clone(),
        bridge_path.clone(),
    ];

    for (i, core_src) in os_shim_core::core_source_paths().iter().enumerate() {
        let core_obj = work_dir.join(format!("os_shim_core_{i}.o"));
        compile_c_path(
            tc,
            core_src,
            &core_obj,
            arch,
            os,
            &[os_shim_core::include_dir().as_path()],
        )?;
        owned_objs.push(core_obj);
    }
    if let Some(daemon_src) = os_shim_core::write_daemon_execve_override(work_dir)? {
        let daemon_obj = work_dir.join("os_shim_execve_daemon.o");
        compile_c_path(
            tc,
            &daemon_src,
            &daemon_obj,
            arch,
            os,
            &[os_shim_core::include_dir().as_path()],
        )?;
        owned_objs.push(daemon_obj);
    }

    let stubs_path = work_dir.join("guest_stubs.o");
    if !stub_entries.is_empty() {
        let stubs_src = generate_guest_stubs_c(stub_entries, entry_param_count, abi_pad_args);
        compile_c(tc, &stubs_src, &stubs_path, arch, os)?;
        owned_objs.push(stubs_path);
    }

    // Always linked: defines `__wasm_data_seg_N`/`__wasm_memory_init_copy`
    // when there are segments, or a no-op `__speet_data_init` stub when
    // there aren't — `entry_bridge`'s `__speet_start` always calls
    // `__speet_data_init` unconditionally, so exactly one definition (this
    // one, or `guest.o`'s real one when segments are present) must exist.
    let data_segs_path = work_dir.join("data_segments.o");
    let segs: Vec<DataSegmentBytes> = data_segments
        .iter()
        .map(|s| DataSegmentBytes { bytes: &s.bytes })
        .collect();
    let data_segs_src = generate_data_segments_c(&segs);
    compile_c(tc, &data_segs_src, &data_segs_path, arch, os)?;
    owned_objs.push(data_segs_path);

    let link_objs: Vec<&Path> = owned_objs.iter().map(|p| p.as_path()).collect();

    let recipe = host.link_recipe();
    let extra = dylib_flags_for_os(os, &recipe.dylib_flags);

    link_executable(tc, &link_objs, out_exe, arch, os, &extra)
}

fn compile_c_with_shim_include(
    tc: &LlvmToolchain,
    src: &str,
    out_obj: &Path,
    arch: BinArch,
    os: BinOs,
) -> Result<(), String> {
    std::fs::write(out_obj.with_extension("c"), src).map_err(|e| e.to_string())?;
    compile_c_path(
        tc,
        &out_obj.with_extension("c"),
        out_obj,
        arch,
        os,
        &[os_shim_core::include_dir().as_path()],
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
        assert!(src.contains("os_shim_execve"));
        assert!(!src.contains("__speet_execve_hook"));
    }
}
