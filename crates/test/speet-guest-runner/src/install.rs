//! Emulator cache + auto-install policy.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::guest::GuestArch;
use crate::path::{RunnerPath, RunnerStep};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallPolicy {
    /// CI + agentic dev default: install missing tools automatically.
    Auto,
    /// Only use env/PATH/cache; fail ensure if absent.
    RequirePresent,
    /// Never download; for hermetic offline runs.
    Offline,
}

impl InstallPolicy {
    pub fn from_env() -> Self {
        match std::env::var("SPEET_AUTO_INSTALL_EMULATORS")
            .ok()
            .as_deref()
        {
            Some("0") | Some("false") => Self::Offline,
            Some("require") => Self::RequirePresent,
            _ if std::env::var("CI").ok().as_deref() == Some("true") => Self::Auto,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmulatorStore {
    root: PathBuf,
}

impl EmulatorStore {
    pub fn default_root() -> PathBuf {
        std::env::var("SPEET_EMULATOR_CACHE")
            .map(PathBuf::from)
            .or_else(|_| {
                std::env::var("RUNNER_TEMP")
                    .map(|t| PathBuf::from(t).join("speet-emulators"))
                    .or_else(|_| {
                        dirs_home().map(|h| h.join(".cache/speet/emulators"))
                    })
            })
            .unwrap_or_else(|_| PathBuf::from(".speet-emulators"))
    }

    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blink_path(&self) -> PathBuf {
        self.root.join("blink")
    }

    pub fn qemu_user_path(&self, arch: GuestArch) -> PathBuf {
        let name = match arch {
            GuestArch::X86_64 => "qemu-x86_64",
            GuestArch::AArch64 => "qemu-aarch64",
            GuestArch::Riscv64 => "qemu-riscv64",
            GuestArch::Riscv32 => "qemu-riscv32",
        };
        self.root.join(name)
    }

    pub fn prepend_to_path(&self, ctx_env: &mut std::collections::HashMap<String, String>) {
        let root = self.root.display().to_string();
        let path = ctx_env
            .get("PATH")
            .map(|p| format!("{root}:{p}"))
            .unwrap_or(root);
        ctx_env.insert("PATH".to_string(), path);
    }
}

pub struct EmulatorInstaller {
    pub store: EmulatorStore,
    pub policy: InstallPolicy,
}

impl EmulatorInstaller {
    pub fn new(store: EmulatorStore, policy: InstallPolicy) -> Self {
        Self { store, policy }
    }

    pub fn from_env() -> Self {
        Self::new(EmulatorStore::new(EmulatorStore::default_root()), InstallPolicy::from_env())
    }

    pub fn ensure_blink(&self) -> Result<PathBuf, String> {
        if let Ok(p) = std::env::var("BLINK") {
            let pb = PathBuf::from(&p);
            if pb.is_file() {
                return Ok(pb);
            }
        }
        if let Ok(p) = which("blink") {
            return Ok(p);
        }
        let cached = self.store.blink_path();
        if cached.is_file() {
            return Ok(cached);
        }
        match self.policy {
            InstallPolicy::Offline | InstallPolicy::RequirePresent => {
                Err("blink not found (auto-install disabled)".into())
            }
            InstallPolicy::Auto => self.install_blink(),
        }
    }

    pub fn ensure_qemu_user(&self, arch: GuestArch) -> Result<PathBuf, String> {
        let env_key = match arch {
            GuestArch::Riscv64 => "SPEET_QEMU_RISCV64",
            GuestArch::Riscv32 => "SPEET_QEMU_RISCV32",
            GuestArch::AArch64 => "SPEET_QEMU_AARCH64",
            GuestArch::X86_64 => "SPEET_QEMU_X86_64",
        };
        if let Ok(p) = std::env::var(env_key) {
            let pb = PathBuf::from(&p);
            if pb.is_file() {
                return Ok(pb);
            }
        }
        let names = qemu_names(arch);
        for name in names {
            if let Ok(p) = which(name) {
                return Ok(p);
            }
        }
        let cached = self.store.qemu_user_path(arch);
        if cached.is_file() {
            return Ok(cached);
        }
        match self.policy {
            InstallPolicy::Offline | InstallPolicy::RequirePresent => {
                Err(format!("qemu-user for {arch:?} not found (auto-install disabled)"))
            }
            InstallPolicy::Auto => self.install_qemu_user(arch),
        }
    }

    pub fn ensure_for_path(&self, path: &RunnerPath) -> Result<(), String> {
        for step in &path.0 {
            match step {
                RunnerStep::Blink => {
                    self.ensure_blink()?;
                }
                RunnerStep::QemuUser { arch } => {
                    self.ensure_qemu_user(*arch)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn install_blink(&self) -> Result<PathBuf, String> {
        std::fs::create_dir_all(self.store.root()).map_err(|e| e.to_string())?;
        if let Some(script) = install_script_path() {
            let status = Command::new("bash")
                .arg(&script)
                .arg("blink")
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                return Err("install-corpus-emulators.sh blink failed".into());
            }
        }
        let cached = self.store.blink_path();
        if cached.is_file() {
            return Ok(cached);
        }
        Err("blink install did not produce cache binary".into())
    }

    fn install_qemu_user(&self, arch: GuestArch) -> Result<PathBuf, String> {
        std::fs::create_dir_all(self.store.root()).map_err(|e| e.to_string())?;
        if let Some(script) = install_script_path() {
            let arch_arg = match arch {
                GuestArch::Riscv64 => "qemu-riscv64",
                GuestArch::Riscv32 => "qemu-riscv32",
                GuestArch::AArch64 => "qemu-aarch64",
                GuestArch::X86_64 => "qemu-x86_64",
            };
            let status = Command::new("bash")
                .arg(&script)
                .arg(arch_arg)
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                return Err(format!("install-corpus-emulators.sh {arch_arg} failed"));
            }
        }
        let cached = self.store.qemu_user_path(arch);
        if cached.is_file() {
            return Ok(cached);
        }
        for name in qemu_names(arch) {
            if let Ok(p) = which(name) {
                return Ok(p);
            }
        }
        Err(format!("qemu-user for {arch:?} install failed"))
    }
}

fn qemu_names(arch: GuestArch) -> &'static [&'static str] {
    match arch {
        GuestArch::Riscv64 => &["qemu-riscv64", "qemu-riscv64-static"],
        GuestArch::Riscv32 => &["qemu-riscv32", "qemu-riscv32-static"],
        GuestArch::AArch64 => &["qemu-aarch64", "qemu-aarch64-static"],
        GuestArch::X86_64 => &["qemu-x86_64", "qemu-x86_64-static"],
    }
}

fn which(name: &str) -> Result<PathBuf, String> {
    let path = std::env::var("PATH").unwrap_or_default();
    for dir in path.split(':') {
        let candidate = PathBuf::from(dir).join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(format!("{name} not in PATH"))
}

fn dirs_home() -> Result<PathBuf, String> {
    std::env::var("HOME")
        .map(PathBuf::from)
        .map_err(|e| e.to_string())
}

fn install_script_path() -> Option<PathBuf> {
    std::env::var("CARGO_MANIFEST_DIR")
        .ok()
        .map(|m| PathBuf::from(m).join("../../../scripts/install-corpus-emulators.sh"))
        .filter(|p| p.is_file())
}
