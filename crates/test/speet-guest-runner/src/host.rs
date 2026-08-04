//! Host environment probe.

use binary_io::{BinArch, BinOs};
use std::env;

#[derive(Debug, Clone)]
pub struct HostInfo {
    pub arch: BinArch,
    pub os: BinOs,
    pub is_vm: bool,
    pub nested_vm_available: bool,
}

impl HostInfo {
    pub fn detect() -> Self {
        let arch = host_arch();
        let os = host_os();
        let nested_vm_available = probe_nested_vm();
        let is_vm = env::var("SPEET_HOST_IS_VM")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(!nested_vm_available);
        Self {
            arch,
            os,
            is_vm,
            nested_vm_available,
        }
    }
}

fn host_arch() -> BinArch {
    match env::consts::ARCH {
        "x86_64" => BinArch::X86_64,
        "aarch64" => BinArch::AArch64,
        "riscv64" => BinArch::RiscV64,
        other => panic!("unsupported host arch: {other}"),
    }
}

fn host_os() -> BinOs {
    match env::consts::OS {
        "linux" => BinOs::Linux,
        "macos" => BinOs::MacOs,
        other => panic!("unsupported host os: {other}"),
    }
}

fn probe_nested_vm() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/dev/kvm").exists()
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "kern.hv_support"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "1")
            .unwrap_or(false)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        false
    }
}
