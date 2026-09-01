//! Integrated thin runtime: suitability gate, instrumented recompile, disk cache.

use crate::cache::{load_binary, ArtifactCache};
use crate::link::link_guest_integrated;
use crate::report::{GuestInfo, PluginHashes, RecompileReport};
use crate::runtime::validate_wasm_public;
use crate::suitability::analyze_imports_with_model_and_wired;
use crate::toolchain::LlvmToolchain;
use binary_io::{BinArch, BinOs};
use speet_host_api::{HostApi, ImportManifest};
use speet_link_core::GuestImageLayout;
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::{
    assert_same_platform, external_targets_from_imports, host_platform,
    recompile_to_wasm_instrumented_plt_with_layout, DataSegment,
};
use speet_recompile::plt::PltCallPlan;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::{Arc, Mutex};

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

/// Optional hot-pluggable frontend (wasmi guest). When set, `recompile_wasm`
/// calls this instead of the statically linked native match arm.
pub trait HotFrontend: Send + Sync {
    fn recompile(
        &self,
        text: &[u8],
        layout: &GuestImageLayout,
        arch: BinArch,
        plt_plan: Option<&PltCallPlan>,
        entry_addr: Option<u64>,
        manifest: &ImportManifest,
    ) -> Result<(Vec<u8>, Vec<String>), String>;
}

struct HotState {
    hashes: PluginHashes,
    extra_wired: Vec<String>,
    frontend: Option<Arc<dyn HotFrontend>>,
    last_report: Option<RecompileReport>,
}

impl Default for HotState {
    fn default() -> Self {
        Self {
            hashes: PluginHashes::static_link(),
            extra_wired: Vec::new(),
            frontend: None,
            last_report: None,
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
    hot: Mutex<HotState>,
}

impl IntegratedNativeRuntime {
    pub fn new(host: Arc<dyn HostApi>) -> Self {
        // Same-platform host output: aarch64 Mach-O on Apple Silicon (blitz →
        // asm-arch `AArch64Writer`), x86_64 elsewhere. The old macOS→x86_64
        // Rosetta override is gone — guest-stack seeding inside `__wasm_mem`
        // fixed the aarch64 native fault that motivated it (see
        // speet-recompile/STATUS.md).
        let (out_os, out_arch) = host_platform();
        Self {
            host,
            toolchain: LlvmToolchain::from_build_env(),
            cache: ArtifactCache::new(),
            out_arch,
            out_os,
            hot: Mutex::new(HotState::default()),
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

    /// Install (or clear) a hot-pluggable frontend and stub overlay. Called
    /// by `speet-rtd` after `ensure_fresh()`.
    pub fn set_hot_plugins(
        &self,
        hashes: PluginHashes,
        extra_wired: Vec<String>,
        frontend: Option<Arc<dyn HotFrontend>>,
    ) {
        let mut hot = self.hot.lock().unwrap();
        hot.hashes = hashes;
        hot.extra_wired = extra_wired;
        hot.frontend = frontend;
    }

    pub fn last_report(&self) -> Option<RecompileReport> {
        self.hot.lock().unwrap().last_report.clone()
    }

    pub fn plugin_hashes(&self) -> PluginHashes {
        self.hot.lock().unwrap().hashes.clone()
    }

    /// Guest coordinates for `path` (best-effort if the file cannot be parsed).
    pub fn guest_info(&self, path: &Path) -> GuestInfo {
        self.guest_info_from_path(path)
    }

    /// Cache key `compile_and_link` would use, including hot-plugin hashes.
    pub fn artifact_cache_key(&self, path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let input_hash = format!(
            "{}:layout-plt{}",
            ArtifactCache::hash_input(&bytes),
            self.plugin_key_suffix()
        );
        let arch_os = format!("{}-{}", arch_label(self.out_arch), os_label(self.out_os));
        Ok(format!("{input_hash}:{arch_os}:{}", self.host_id()))
    }

    /// Analyze and return the typed report (also stored as `last_report`).
    pub fn analyze_report(&self, path: &Path) -> Result<RecompileReport, String> {
        let _ = NativeRuntime::analyze(self, path)?;
        Ok(self.last_report().unwrap_or_default())
    }

    /// Translate, optionally link. Always records `last_report`.
    pub fn recompile_report(&mut self, path: &Path, link: bool) -> RecompileReport {
        if link {
            let _ = NativeRuntime::obtain_executable(self, path);
            return self.last_report().unwrap_or_default();
        }
        let guest = self.guest_info_from_path(path);
        let plugins = self.plugin_hashes();
        let suit = match NativeRuntime::analyze(self, path) {
            Ok(s) => s,
            Err(e) => {
                let mut full = RecompileReport {
                    guest,
                    plugins,
                    translate_error: Some(e),
                    ..RecompileReport::default()
                };
                self.record_report(full.clone());
                return full;
            }
        };
        let guest_arch = match guest_arch_from_path(path) {
            Ok(a) => a,
            Err(e) => {
                let mut full = RecompileReport::from_suitability(path, &suit, plugins);
                full.guest = guest;
                full.translate_error = Some(e);
                self.record_report(full.clone());
                return full;
            }
        };
        match self.recompile_wasm(path, guest_arch) {
            Ok((_wasm, unsupported)) => {
                let mut full = RecompileReport::from_suitability(path, &suit, plugins);
                full.guest = guest;
                full.unsupported_insns = unsupported;
                self.record_report(full.clone());
                full
            }
            Err(e) => {
                let mut full = RecompileReport::from_suitability(path, &suit, plugins);
                full.guest = guest;
                if e.contains("LLVM") || e.contains("link") || e.contains("clang") {
                    full.link_error = Some(e);
                } else {
                    full.translate_error = Some(e);
                }
                self.record_report(full.clone());
                full
            }
        }
    }

    /// Obtain + spawn, capturing truncated stdout/stderr.
    pub fn run_guest(
        &mut self,
        path: &Path,
        argv: &[String],
    ) -> Result<(i32, String, String), String> {
        let exe = NativeRuntime::obtain_executable(self, path).map_err(|e| e.to_string())?;
        let output = Command::new(&exe)
            .args(argv)
            .output()
            .map_err(|e| e.to_string())?;
        Ok((
            output.status.code().unwrap_or(-1),
            truncate_utf8(&output.stdout),
            truncate_utf8(&output.stderr),
        ))
    }

    fn record_report(&self, report: RecompileReport) {
        self.hot.lock().unwrap().last_report = Some(report);
    }

    fn plugin_key_suffix(&self) -> String {
        let h = self.plugin_hashes();
        if h.recompiler == PluginHashes::NATIVE && h.stubs == PluginHashes::LINKED_STUBS {
            String::new()
        } else {
            format!(":rec={}:stubs={}", h.recompiler, h.stubs)
        }
    }

    fn extra_wired(&self) -> Vec<String> {
        self.hot.lock().unwrap().extra_wired.clone()
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

    /// The manifest to translate/assemble/link against — always the actual
    /// host's own `import_manifest()`, never a hardcoded
    /// `ImportManifest::integrated_native()` assumption, so a differently
    /// configured `HostApi` (e.g. one with a different host-capability set)
    /// stays self-consistent across recompile, PLT-index resolution, and
    /// link-shim generation. See `docs/guides/thin-runtime-genericity.md`.
    fn manifest(&self) -> ImportManifest {
        self.host.import_manifest()
    }

    pub fn analyze_binary(&self, bin: &binary_io::LoadedBinary) -> SuitabilityReport {
        let model = if bin
            .sections
            .iter()
            .any(|s| matches!(s.kind, binary_io::SectionKind::Text))
        {
            GuestImageLayout::from_loaded_binary(bin).memory_model
        } else {
            speet_link_core::MemoryModel::OwnedLinear
        };
        let (unresolved_deps, fn_ptr_deps) = analyze_imports_with_model_and_wired(
            self.host.as_ref(),
            bin,
            model,
            &self.extra_wired(),
        );
        let suitable = unresolved_deps.is_empty() && fn_ptr_deps.is_empty();
        SuitabilityReport {
            suitable,
            unresolved_deps,
            fn_ptr_deps,
        }
    }

    fn recompile_wasm(
        &mut self,
        path: &Path,
        guest_arch: BinArch,
    ) -> Result<(Vec<u8>, Vec<String>), String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        // Suffix invalidates caches from the prior text-only megabinary shape
        // and from a different hot-plugged frontend/stubs pair.
        let input_hash = format!(
            "{}:layout-plt{}",
            ArtifactCache::hash_input(&bytes),
            self.plugin_key_suffix()
        );
        if let Some(w) = self.cache.get_wasm(&input_hash) {
            return Ok((w, Vec::new()));
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
        let targets = external_targets_from_imports(&bin.imports);
        let plt_plan = PltCallPlan::from_targets(&targets, self.host.as_ref());
        let manifest = self.manifest();
        let layout = GuestImageLayout::from_loaded_binary(&bin);
        let frontend = self.hot.lock().unwrap().frontend.clone();
        let (wasm, unsupported) = if let Some(fe) = frontend {
            fe.recompile(
                &text.data,
                &layout,
                guest_arch,
                Some(&plt_plan),
                Some(bin.entry),
                &manifest,
            )?
        } else {
            match guest_arch {
                BinArch::X86_64
                | BinArch::AArch64
                | BinArch::RiscV64
                | BinArch::RiscV32
                | BinArch::Arm
                | BinArch::X86 => recompile_to_wasm_instrumented_plt_with_layout(
                    &text.data,
                    &layout,
                    guest_arch,
                    Some(&plt_plan),
                    Some(bin.entry),
                    &manifest,
                ),
            }
        };
        self.cache.put_wasm(&input_hash, wasm.clone());
        Ok((wasm, unsupported))
    }

    /// Halt-sentinel guest address (see
    /// `speet_recompile::frontend::halt_addr` and
    /// `docs/guides/thin-runtime-genericity.md` principle 4) for `path`'s
    /// `.text` span. Re-reads/re-parses the binary rather than threading
    /// this through the wasm disk cache — mirrors [`Self::recompile_wasm`]'s
    /// own section lookup; cheap relative to the recompile/compile/link
    /// steps this always runs alongside.
    fn halt_addr_for(&self, path: &Path) -> Result<u64, String> {
        let bin = load_binary(path)?;
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
        Ok(speet_recompile::frontend::halt_addr(
            text.addr,
            text.data.len(),
        ))
    }

    fn compile_and_link(
        &mut self,
        path: &Path,
        guest_arch: BinArch,
    ) -> Result<(PathBuf, Vec<String>), String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let input_hash = format!(
            "{}:layout-plt{}",
            ArtifactCache::hash_input(&bytes),
            self.plugin_key_suffix()
        );
        let arch_os = format!("{}-{}", arch_label(self.out_arch), os_label(self.out_os));
        let cache_key = format!("{input_hash}:{arch_os}:{}", self.host_id());

        if let Some(exe) = self.cache.get_executable(&cache_key) {
            return Ok((exe, Vec::new()));
        }

        let (wasm, unsupported) = self.recompile_wasm(path, guest_arch)?;
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
        let entry_param_count = speet_recompile::drive::entry_param_count(&wasm);
        // `sp_param_index`/`lr_param_index` describe the WASM's own
        // parameter layout, which is a property of whichever frontend
        // translated the guest (`guest_arch`) — NOT of `self.out_arch`,
        // which is only the native encoding/link target and can differ from
        // `guest_arch` when an embedder overrides `with_output_target` for a
        // cross-arch pairing (see `link.rs`'s `link_guest_integrated` doc
        // comment). Default construction keeps guest_arch == out_arch via
        // `host_platform()`.
        let sp_idx = speet_recompile::drive::sp_param_index(guest_arch);
        let lr_idx = speet_recompile::drive::lr_param_index(guest_arch);
        // Halt-sentinel layout is derived inside the guest-function catalog.
        let bin = load_binary(path)?;
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
        let targets = external_targets_from_imports(&bin.imports);
        let plt_plan = PltCallPlan::from_targets(&targets, self.host.as_ref());
        let n_redirect_shims = plt_plan.wasm_import_by_addr.len() as u32;
        let catalog = speet_recompile::guest_func_catalog::GuestFuncCatalog::from_wasm(
            &wasm,
            text.addr,
            guest_arch,
            text.data.len(),
            n_redirect_shims,
        );
        let entry_local = speet_recompile::drive::entry_local_func_idx(&wasm);
        let halt_local = catalog.halt_entry().local_func_idx;
        let layout = GuestImageLayout::from_loaded_binary(&bin);
        let data_segments = data_segments_from_layout(&layout);
        let text_hole = layout
            .memory_model
            .text_unmapped()
            .then_some((layout.text_base, layout.text_len as u64));
        link_guest_integrated(
            tc,
            host.as_ref(),
            &obj,
            self.out_arch,
            self.out_os,
            entry_param_count,
            sp_idx,
            lr_idx,
            &catalog.to_stub_entries(),
            entry_local,
            halt_local,
            None,
            &data_segments,
            text_hole,
            &out_dir.join(format!("{cache_key}.work")),
            &exe,
        )?;

        self.cache.put_executable(&cache_key, exe.clone());
        Ok((exe, unsupported))
    }

    fn guest_info_from_path(&self, path: &Path) -> GuestInfo {
        let Ok(bin) = load_binary(path) else {
            return GuestInfo {
                path: path.to_path_buf(),
                ..GuestInfo::default()
            };
        };
        let text = bin.sections.iter().find(|s| {
            s.name == ".text"
                || s.name == "__TEXT,__text"
                || s.name == "__text"
                || matches!(s.kind, binary_io::SectionKind::Text)
        });
        GuestInfo {
            path: path.to_path_buf(),
            arch: arch_label(bin.arch).to_string(),
            text_addr: text.map(|s| s.addr).unwrap_or(0),
            text_len: text.map(|s| s.data.len() as u64).unwrap_or(0),
            entry: bin.entry,
            imports: bin.imports.iter().map(|i| i.name.clone()).collect(),
        }
    }
}

impl NativeRuntime for IntegratedNativeRuntime {
    fn analyze(&self, path: &Path) -> Result<SuitabilityReport, String> {
        let bin = load_binary(path)?;
        assert_same_platform(&bin)?;
        let report = self.analyze_binary(&bin);
        let mut full = RecompileReport::from_suitability(path, &report, self.plugin_hashes());
        full.guest = self.guest_info_from_path(path);
        self.record_report(full);
        Ok(report)
    }

    fn obtain_executable(&mut self, path: &Path) -> Result<PathBuf, ObtainError> {
        let report = self.analyze(path).map_err(ObtainError::RecompileFailed)?;
        if !report.suitable {
            return Err(ObtainError::Unsuitable(report));
        }
        let guest_arch = guest_arch_from_path(path).map_err(ObtainError::RecompileFailed)?;
        match self.compile_and_link(path, guest_arch) {
            Ok((exe, unsupported)) => {
                let mut full =
                    RecompileReport::from_suitability(path, &report, self.plugin_hashes());
                full.guest = self.guest_info_from_path(path);
                full.unsupported_insns = unsupported;
                full.exe_path = Some(exe.clone());
                self.record_report(full);
                Ok(exe)
            }
            Err(e) => {
                let mut full =
                    RecompileReport::from_suitability(path, &report, self.plugin_hashes());
                full.guest = self.guest_info_from_path(path);
                if e.contains("LLVM") || e.contains("link") || e.contains("clang") {
                    full.link_error = Some(e.clone());
                } else {
                    full.translate_error = Some(e.clone());
                }
                self.record_report(full);
                Err(ObtainError::RecompileFailed(e))
            }
        }
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

/// Extract initialized data sections from a [`GuestImageLayout`].
/// Under [`MemoryModel::ZeroOffset`], sections overlapping the unrecompiled
/// text hole are dropped (`.text` must never appear in the host mirror).
fn data_segments_from_layout(layout: &GuestImageLayout) -> Vec<DataSegment> {
    layout
        .data_sections
        .iter()
        .filter(|s| {
            if !layout.memory_model.text_unmapped() {
                return true;
            }
            let end = s.addr.saturating_add(s.bytes.len() as u64);
            end <= layout.text_base || s.addr >= layout.text_end()
        })
        .map(|s| DataSegment {
            addr: s.addr,
            bytes: s.bytes.clone(),
        })
        .collect()
}

/// Extract initialized data (`.data`/`.rodata`) sections as WASM passive data
/// segments. `.bss` is skipped: `__wasm_mem` already starts zeroed, so it
/// needs no `memory.init` copy. Section addresses are the guest's own virtual
/// addresses, matching the identity address mapper wired into every
/// architecture frontend (see `speet_recompile::frontend::translate_with_plt`).
#[allow(dead_code)]
fn data_segments_from_binary(bin: &binary_io::LoadedBinary) -> Vec<DataSegment> {
    data_segments_from_layout(&GuestImageLayout::from_loaded_binary(bin))
}

fn arch_label(a: BinArch) -> &'static str {
    match a {
        BinArch::X86_64 => "x86_64",
        BinArch::AArch64 => "aarch64",
        BinArch::RiscV64 => "riscv64",
        BinArch::RiscV32 => "riscv32",
        BinArch::Arm => "arm",
        BinArch::X86 => "i686",
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

const IO_TRUNCATE: usize = 8 * 1024;

fn truncate_utf8(bytes: &[u8]) -> String {
    let n = bytes.len().min(IO_TRUNCATE);
    let mut s = String::from_utf8_lossy(&bytes[..n]).into_owned();
    if bytes.len() > IO_TRUNCATE {
        s.push_str("…(truncated)");
    }
    s
}
