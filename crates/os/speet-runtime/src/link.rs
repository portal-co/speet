//! LLVM link orchestration.

use crate::toolchain::{compile_c, link_executable, LlvmToolchain};
use binary_io::{BinArch, BinOs};
use speet_host_api::HostApi;
use speet_rt::{generate_memory_tu, generate_shim};
use std::path::{Path, PathBuf};

/// Materialize and link guest + runtime objects into an executable.
pub fn link_guest(
    tc: &LlvmToolchain,
    host: &dyn HostApi,
    guest_obj: &[u8],
    arch: BinArch,
    os: BinOs,
    work_dir: &Path,
    out_exe: &Path,
) -> Result<(), String> {
    let recipe = host.link_recipe();
    let guest_path = work_dir.join("guest.o");
    std::fs::write(&guest_path, guest_obj).map_err(|e| e.to_string())?;

    let mem_path = work_dir.join("mem.o");
    compile_c(tc, &generate_memory_tu(), &mem_path, arch, os)?;

    let shim_path = work_dir.join("shim.o");
    let shim_src = generate_shim(&host.import_manifest());
    compile_c(tc, &shim_src, &shim_path, arch, os)?;

    let extra: Vec<String> = dylib_flags_for_os(os, &recipe.dylib_flags);

    link_executable(
        tc,
        &[&guest_path, &mem_path, &shim_path],
        out_exe,
        arch,
        os,
        &extra,
    )
}

/// Pick dylib flags appropriate for the link target OS.
fn dylib_flags_for_os(os: BinOs, _recipe_flags: &[String]) -> Vec<String> {
    match os {
        BinOs::Linux => vec!["-lc".to_string()],
        BinOs::MacOs => vec!["-lSystem".to_string()],
    }
}

fn mangle_macho(name: &str) -> String {
    if name.starts_with('_') {
        name.to_string()
    } else {
        format!("_{name}")
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
    fn macho_alias_mangles() {
        assert_eq!(mangle_macho("write"), "_write");
        assert_eq!(mangle_macho("_write"), "_write");
    }
}
