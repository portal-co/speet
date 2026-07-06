//! Integrated thin runtime: suitability gate, instrumented recompile, disk cache.

use crate::cache::{load_binary, ArtifactCache};
use crate::link::link_guest_integrated;
use crate::runtime::validate_wasm_public;
use crate::suitability::analyze_imports;
use crate::toolchain::LlvmToolchain;
use binary_io::{BinArch, BinOs};
use speet_host_api::{HostApi, ImportManifest};
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::{
    assert_same_platform, host_platform, recompile_to_wasm_instrumented_plt, ExternalTargets,
};
use speet_recompile::plt::PltCallPlan;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::Arc;

/// Import-based suitability report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuitabilityReport {
    pub suitable: bool,
    pub unresolved_deps: Vec<String>,
    pub fn_ptr_deps: Vec<String>,
}

/// Failure to obtain a recompiled executable.
#[derive(Debug, Clone)]
pub enum ObtainError {
    Unsuitable(SuitabilityReport),
    RecompileFailed(String),
}

impl std::fmt::Display for ObtainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObtainError::Unsuitable(r) => {
                write!(
                    f,
                    "unsuitable: unresolved={:?} fn_ptr={:?}",
                    r.unresolved_deps, r.fn_ptr_deps
                )
            }
            ObtainError::RecompileFailed(e) => write!(f, "recompile failed: {e}"),
        }
    }
}

/// Native runtime abstraction for the integrated thin-runtime path.
pub trait NativeRuntime: Send {
    fn analyze(&self, path: &Path) -> Result<SuitabilityReport, String>;
    fn obtain_executable(&mut self, path: &Path) -> Result<PathBuf, ObtainError>;
    fn spawn(
        &self,
        exe: &Path,
        argv: &[&OsStr],
        env: Option<&[(&OsStr, &OsStr)]>,
    ) -> Result<ExitStatus, String>;
}

/// Integrated driver with unreachable instrumentation, suitability, and exe cache.
pub struct IntegratedNativeRuntime {
    pub host: Arc<dyn HostApi>,
    pub toolchain: Option<LlvmToolchain>,
    pub cache: ArtifactCache,
    pub out_arch: BinArch,
    pub out_os: BinOs,
}

impl IntegratedNativeRuntime {
    pub fn new(host: Arc<dyn HostApi>) -> Self {
        let (mut out_os, mut out_arch) = host_platform();
        if cfg!(target_os = "macos") {
            out_arch = BinArch::X86_64;
            out_os = BinOs::MacOs;
        }
        Self {
            host,
            toolchain: LlvmToolchain::from_build_env(),
            cache: ArtifactCache::new(),
            out_arch,
            out_os,
        }
    }

    pub fn with_disk_cache(mut self, root: PathBuf) -> Self {
        self.cache = ArtifactCache::with_disk_root(root);
        self
    }

    pub fn with_output_target(mut self, arch: BinArch, os: BinOs) -> Self {
        self.out_arch = arch;
        self.out_os = os;
        self
    }

    pub fn llvm_available(&self) -> bool {
        self.toolchain
            .as_ref()
            .map(|t| t.is_available())
            .unwrap_or(false)
    }

    fn host_id(&self) -> &'static str {
        "integrated"
    }

    fn file_hash(path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        Ok(ArtifactCache::hash_input(&bytes))
    }

    fn manifest(&self) -> ImportManifest {
        ImportManifest::integrated_native()
    }

    pub fn analyze_binary(&self, bin: &binary_io::LoadedBinary) -> SuitabilityReport {
        let (unresolved_deps, fn_ptr_deps) = analyze_imports(self.host.as_ref(), bin);
        let suitable = unresolved_deps.is_empty() && fn_ptr_deps.is_empty();
        SuitabilityReport {
            suitable,
            unresolved_deps,
            fn_ptr_deps,
        }
    }

    fn recompile_wasm(&mut self, path: &Path, guest_arch: BinArch) -> Result<Vec<u8>, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let input_hash = ArtifactCache::hash_input(&bytes);
        if let Some(w) = self.cache.get_wasm(&input_hash) {
            return Ok(w);
        }

        let bin = load_binary(path)?;
        assert_same_platform(&bin)?;
        let text = bin
            .sections
            .iter()
            .find(|s| {
                s.name == ".text"
                    || s.name == "__TEXT,__text"
                    || s.name == "__text"
                    || matches!(s.kind, binary_io::SectionKind::Text)
            })
            .ok_or_else(|| "no .text section".to_string())?;
        let start = text.addr;
        let targets = ExternalTargets::from_imports(&bin.imports);
        let plt_plan = PltCallPlan::from_targets(&targets, self.host.as_ref());

        let (wasm, _unsupported) = match guest_arch {
            BinArch::X86_64 | BinArch::AArch64 => recompile_to_wasm_instrumented_plt(
                &text.data,
                start,
                guest_arch,
                Some(&plt_plan),
            ),
        };
        self.cache.put_wasm(&input_hash, wasm.clone());
        Ok(wasm)
    }

    fn compile_and_link(
        &mut self,
        path: &Path,
        guest_arch: BinArch,
    ) -> Result<PathBuf, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let input_hash = ArtifactCache::hash_input(&bytes);
        let arch_os = format!(
            "{}-{}",
            arch_label(self.out_arch),
            os_label(self.out_os)
        );
        let cache_key = format!("{input_hash}:{arch_os}:{}", self.host_id());

        if let Some(exe) = self.cache.get_executable(&cache_key) {
            return Ok(exe);
        }

        let wasm = self.recompile_wasm(path, guest_arch)?;
        validate_wasm_public(&wasm)?;

        let obj = if let Some(o) = self.cache.get_object(&input_hash, &arch_os) {
            o
        } else {
            let o = compile_wasm_to_object(&wasm, self.out_arch, self.out_os)?;
            self.cache.put_object(&input_hash, &arch_os, o.clone());
            o
        };

        let tc = self
            .toolchain
            .as_ref()
            .ok_or_else(|| "LLVM toolchain not available".to_string())?;

        let out_dir = self
            .cache
            .executable_dir()
            .unwrap_or_else(|| std::env::temp_dir().join("speet_rt_integrated"));
        std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        let exe = out_dir.join(format!("{cache_key}.exe"));

        let host = self.host.clone();
        link_guest_integrated(
            tc,
            host.as_ref(),
            &obj,
            self.out_arch,
            self.out_os,
            &out_dir.join(format!("{cache_key}.work")),
            &exe,
        )?;

        self.cache.put_executable(&cache_key, exe.clone());
        Ok(exe)
    }
}

impl NativeRuntime for IntegratedNativeRuntime {
    fn analyze(&self, path: &Path) -> Result<SuitabilityReport, String> {
        let bin = load_binary(path)?;
        assert_same_platform(&bin)?;
        Ok(self.analyze_binary(&bin))
    }

    fn obtain_executable(&mut self, path: &Path) -> Result<PathBuf, ObtainError> {
        let report = self.analyze(path).map_err(ObtainError::RecompileFailed)?;
        if !report.suitable {
            return Err(ObtainError::Unsuitable(report));
        }
        let guest_arch = guest_arch_from_path(path).map_err(ObtainError::RecompileFailed)?;
        self.compile_and_link(path, guest_arch)
            .map_err(ObtainError::RecompileFailed)
    }

    fn spawn(
        &self,
        exe: &Path,
        argv: &[&OsStr],
        env: Option<&[(&OsStr, &OsStr)]>,
    ) -> Result<ExitStatus, String> {
        let mut cmd = Command::new(exe);
        if !argv.is_empty() {
            cmd.args(argv);
        }
        if let Some(pairs) = env {
            cmd.envs(pairs.iter().copied());
        }
        cmd.status().map_err(|e| e.to_string())
    }
}

fn arch_label(a: BinArch) -> &'static str {
    match a {
        BinArch::X86_64 => "x86_64",
        BinArch::AArch64 => "aarch64",
    }
}

fn os_label(o: BinOs) -> &'static str {
    match o {
        BinOs::Linux => "linux",
        BinOs::MacOs => "macos",
    }
}

fn guest_arch_from_path(path: &Path) -> Result<BinArch, String> {
    let bin = load_binary(path)?;
    Ok(bin.arch)
}
