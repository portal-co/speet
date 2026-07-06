//! LLVM link orchestration.

use crate::execve_hook;
use crate::toolchain::{compile_c, link_executable, LlvmToolchain};
use binary_io::{BinArch, BinOs};
use speet_host_api::HostApi;
use speet_rt::{generate_memory_tu, generate_shim, ENTRY_BRIDGE_C};
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
    link_guest_integrated(tc, host, guest_obj, arch, os, work_dir, out_exe)
}

/// Integrated link: shim + entry bridge + execve hook + ambient aliases.
pub fn link_guest_integrated(
    tc: &LlvmToolchain,
    host: &dyn HostApi,
    guest_obj: &[u8],
    arch: BinArch,
    os: BinOs,
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
    compile_c(tc, ENTRY_BRIDGE_C, &bridge_path, arch, os)?;

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

fn alias_link_flags(os: BinOs, aliases: &[(String, String)]) -> Vec<String> {
    let mut flags = Vec::new();
    for (alias, real) in aliases {
        flags.push(alias_flag(os, alias, real));
    }
    flags
}

fn execve_alias_flags(os: BinOs) -> Vec<String> {
    vec![
        alias_flag(os, "execve", "__speet_execve_hook"),
        alias_flag(os, "_execve", "__speet_execve_hook"),
    ]
}

fn alias_flag(os: BinOs, alias: &str, real: &str) -> String {
    match os {
        BinOs::MacOs => {
            format!(
                "-Wl,-alias,{},{}",
                mangle_macho(alias),
                mangle_macho(real)
            )
        }
        BinOs::Linux => format!("-Wl,--defsym,{alias}={real}"),
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

    #[test]
    fn integrated_shim_uses_speet_start() {
        let src = generate_shim_integrated(&speet_host_api::ImportManifest::integrated_native());
        assert!(src.contains("int main(int argc, char **argv)"));
        assert!(src.contains("__speet_start(argc, argv)"));
    }
}
