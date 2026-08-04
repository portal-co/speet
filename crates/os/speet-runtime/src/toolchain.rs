//! LLVM `clang` + `lld` detection for the link step.

use std::path::Path;
use std::process::Command;

/// Paths to LLVM link tools discovered at build time.
#[derive(Debug, Clone)]
pub struct LlvmToolchain {
    pub clang: String,
    pub lld: Option<String>,
}

impl LlvmToolchain {
    /// Load toolchain from `build.rs` env (`SPEET_RT_CLANG`, `SPEET_RT_LLD`).
    pub fn from_build_env() -> Option<Self> {
        let clang = option_env_nonempty("SPEET_RT_CLANG")?;
        let lld = option_env_nonempty("SPEET_RT_LLD");
        Some(Self { clang, lld })
    }

    pub fn is_available(&self) -> bool {
        Command::new(&self.clang)
            .arg("--version")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// `-fuse-ld=...` argument when an explicit lld binary was found.
    pub fn fuse_ld_arg(&self) -> Option<String> {
        self.lld.as_ref().map(|lld| format!("-fuse-ld={lld}"))
    }
}

fn option_env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

/// Map `(arch, os)` to an LLVM `--target=` triple for compile/link.
pub fn llvm_target(arch: binary_io::BinArch, os: binary_io::BinOs) -> &'static str {
    match (arch, os) {
        (binary_io::BinArch::X86_64, binary_io::BinOs::Linux) => "x86_64-unknown-linux-gnu",
        (binary_io::BinArch::AArch64, binary_io::BinOs::Linux) => "aarch64-unknown-linux-gnu",
        (binary_io::BinArch::X86_64, binary_io::BinOs::MacOs) => "x86_64-apple-macosx",
        (binary_io::BinArch::AArch64, binary_io::BinOs::MacOs) => "aarch64-apple-darwin",
        // Host-RV thin link is not wired; triple kept for completeness.
        (binary_io::BinArch::RiscV64, binary_io::BinOs::Linux) => "riscv64-unknown-linux-gnu",
        (binary_io::BinArch::RiscV64, binary_io::BinOs::MacOs) => "riscv64-apple-darwin",
        (binary_io::BinArch::RiscV32, binary_io::BinOs::Linux) => "riscv32-unknown-linux-gnu",
        (binary_io::BinArch::Arm, binary_io::BinOs::Linux) => "arm-linux-gnueabihf",
        (binary_io::BinArch::X86, binary_io::BinOs::Linux) => "i686-unknown-linux-gnu",
        // ILP32 Mach-O is unsupported; stub triples satisfy exhaustiveness.
        (binary_io::BinArch::RiscV32, binary_io::BinOs::MacOs) => "riscv32-apple-darwin",
        (binary_io::BinArch::Arm, binary_io::BinOs::MacOs) => "armv7-apple-darwin",
        (binary_io::BinArch::X86, binary_io::BinOs::MacOs) => "i386-apple-darwin",
    }
}

/// Architecture flag for Mach-O links on Apple hosts (`-arch arm64` etc.).
pub fn clang_arch_flag(arch: binary_io::BinArch) -> &'static str {
    match arch {
        binary_io::BinArch::X86_64 => "x86_64",
        binary_io::BinArch::AArch64 => "arm64",
        binary_io::BinArch::RiscV64 => "riscv64",
        binary_io::BinArch::RiscV32 => "riscv32",
        binary_io::BinArch::Arm => "armv7",
        binary_io::BinArch::X86 => "i386",
    }
}

/// Compile an on-disk `.c` file to `out_obj`.
pub fn compile_c_path(
    tc: &LlvmToolchain,
    src_path: &Path,
    out_obj: &Path,
    arch: binary_io::BinArch,
    os: binary_io::BinOs,
    include_dirs: &[&Path],
) -> Result<(), String> {
    let mut cmd = Command::new(&tc.clang);
    cmd.args(["-c", "-O0", "-Wno-implicit-function-declaration"]);
    cmd.arg(format!("--target={}", llvm_target(arch, os)));
    for inc in include_dirs {
        cmd.arg(format!("-I{}", inc.display()));
    }
    if matches!(os, binary_io::BinOs::MacOs) {
        cmd.args(["-arch", clang_arch_flag(arch), "-mmacosx-version-min=11.0"]);
        if let Some(sdk) = macos_sdk_path() {
            cmd.args(["-isysroot", &sdk]);
        }
    } else {
        cmd.arg("-ffreestanding");
    }
    if let Some(fuse) = tc.fuse_ld_arg() {
        cmd.arg(fuse);
    }
    cmd.arg(src_path);
    cmd.arg("-o").arg(out_obj);
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "clang compile failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

/// Write `src` C to `path` and compile to `out_obj`.
pub fn compile_c(
    tc: &LlvmToolchain,
    src: &str,
    out_obj: &Path,
    arch: binary_io::BinArch,
    os: binary_io::BinOs,
) -> Result<(), String> {
    std::fs::write(out_obj.with_extension("c"), src).map_err(|e| e.to_string())?;
    compile_c_path(
        tc,
        &out_obj.with_extension("c"),
        out_obj,
        arch,
        os,
        &[],
    )
}

/// Link object files into an executable.
pub fn link_executable(
    tc: &LlvmToolchain,
    objects: &[&Path],
    out_exe: &Path,
    arch: binary_io::BinArch,
    os: binary_io::BinOs,
    extra_flags: &[String],
) -> Result<(), String> {
    let mut cmd = Command::new(&tc.clang);
    cmd.arg(format!("--target={}", llvm_target(arch, os)));
    if matches!(os, binary_io::BinOs::MacOs) {
        cmd.args(["-arch", clang_arch_flag(arch), "-mmacosx-version-min=11.0"]);
        if let Some(sdk) = macos_sdk_path() {
            cmd.args(["-isysroot", &sdk]);
        }
    }
    if let Some(fuse) = tc.fuse_ld_arg() {
        cmd.arg(fuse);
    }
    for obj in objects {
        cmd.arg(obj);
    }
    for flag in extra_flags {
        cmd.arg(flag);
    }
    cmd.arg("-o").arg(out_exe);
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "clang link failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn macos_sdk_path() -> Option<String> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
