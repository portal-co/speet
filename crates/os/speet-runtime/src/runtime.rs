//! Thin runtime driver.

use crate::cache::{load_binary, load_text_from_object, ArtifactCache};
use crate::link::link_guest;
use crate::toolchain::LlvmToolchain;
use binary_io::{BinArch, BinOs};
use object::Object;
use speet_host_api::HostApi;
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::{
    assert_same_platform, recompile_rv64_to_wasm, recompile_to_wasm, recompile_to_wasm_instrumented_plt,
    external_targets_from_imports,
};
use speet_recompile::plt::PltCallPlan;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::Arc;

/// Recompilation + link + spawn driver.
pub struct Runtime {
    pub host: Arc<dyn HostApi>,
    pub toolchain: Option<LlvmToolchain>,
    pub cache: ArtifactCache,
}

impl Runtime {
    pub fn new(host: Arc<dyn HostApi>) -> Self {
        Self {
            host,
            toolchain: LlvmToolchain::from_build_env(),
            cache: ArtifactCache::new(),
        }
    }

    pub fn with_cache(mut self, cache: ArtifactCache) -> Self {
        self.cache = cache;
        self
    }

    pub fn llvm_available(&self) -> bool {
        self.toolchain
            .as_ref()
            .map(|t| t.is_available())
            .unwrap_or(false)
    }

    /// Recompile RV64 guest `.text` to WASM (syscall path).
    pub fn recompile_rv64_text(&mut self, text: &[u8], start_addr: u64) -> Vec<u8> {
        self.recompile_rv64_text_with_escape(text, start_addr, yecta::SpeculativeEscape::JUMP)
    }

    /// Like [`recompile_rv64_text`], with an explicit speculative-call escape policy.
    pub fn recompile_rv64_text_with_escape(
        &mut self,
        text: &[u8],
        start_addr: u64,
        speculative: yecta::SpeculativeEscape,
    ) -> Vec<u8> {
        let key = format!(
            "{}:{start_addr:x}:{:?}",
            ArtifactCache::hash_input(text),
            speculative
        );
        if let Some(w) = self.cache.get_wasm(&key) {
            return w;
        }
        let wasm = speet_recompile::frontend::recompile_rv64_to_wasm_with_escape(
            text,
            start_addr,
            speculative,
        );
        self.cache.put_wasm(&key, wasm.clone());
        wasm
    }

    /// Lower WASM to a native relocatable object for `arch`/`os`.
    pub fn compile_to_object(
        &mut self,
        wasm: &[u8],
        arch: BinArch,
        os: BinOs,
    ) -> Result<Vec<u8>, String> {
        let key = format!("{}:{}:{}", ArtifactCache::hash_input(wasm), arch_label(arch), os_label(os));
        let input_hash = ArtifactCache::hash_input(wasm);
        let arch_os = format!("{}-{}", arch_label(arch), os_label(os));
        if let Some(o) = self.cache.get_object(&input_hash, &arch_os) {
            return Ok(o);
        }
        let obj = compile_wasm_to_object(wasm, arch, os)?;
        self.cache.put_object(&input_hash, &arch_os, obj.clone());
        let _ = key; // reserved for richer cache keys
        Ok(obj)
    }

    /// Full pipeline: RV64 corpus text → native object → link → run.
    pub fn recompile_rv64_and_run(
        &mut self,
        text: &[u8],
        start_addr: u64,
        arch: BinArch,
        os: BinOs,
    ) -> Result<ExitStatus, String> {
        self.recompile_rv64_and_run_with_escape(
            text,
            start_addr,
            arch,
            os,
            yecta::SpeculativeEscape::JUMP,
        )
    }

    /// Like [`recompile_rv64_and_run`], with an explicit speculative-call escape policy.
    pub fn recompile_rv64_and_run_with_escape(
        &mut self,
        text: &[u8],
        start_addr: u64,
        arch: BinArch,
        os: BinOs,
        speculative: yecta::SpeculativeEscape,
    ) -> Result<ExitStatus, String> {
        let wasm = self.recompile_rv64_text_with_escape(text, start_addr, speculative);
        validate_wasm(&wasm)?;
        let obj = self.compile_to_object(&wasm, arch, os)?;
        // Guest is RV64 regardless of `arch` (the native *output* target) —
        // its SP/RA positions come from the RV64 frontend, not from `arch`.
        self.link_and_run(
            &obj,
            speet_recompile::drive::entry_param_count(&wasm),
            speet_riscv::RV64_SP_PARAM_INDEX,
            speet_recompile::frontend::halt_addr(start_addr, text.len()),
            Some(speet_riscv::RV64_RA_PARAM_INDEX),
            arch,
            os,
        )
    }

    /// Load a host binary from disk, recompile its `.text`, and run.
    ///
    /// Prefers the full [`load_binary`] path (which surfaces
    /// `bin.imports`, and therefore lets PLT/external calls — any
    /// dynamically-linked libc symbol, e.g. `exit`, whose real target
    /// address lies outside the recompiled `.text` — actually resolve via
    /// [`PltCallPlan`]/[`HostApi::resolve_plt_redirect`]) and only falls
    /// back to the raw `.text`-only [`load_text_from_object`] extraction
    /// for inputs `load_binary` can't parse as a complete binary (e.g. a
    /// bare relocatable corpus object with no dynamic-import metadata at
    /// all). See `docs/guides/thin-runtime-genericity.md` principle 2 —
    /// skipping the PLT plan here silently drops every guest external
    /// call, which previously surfaced as a guest crash with no exit
    /// status rather than a translation error.
    pub fn recompile_binary_and_run(
        &mut self,
        path: &Path,
        arch: BinArch,
        os: BinOs,
    ) -> Result<ExitStatus, String> {
        if let Ok(bin) = load_binary(path) {
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
            let targets = external_targets_from_imports(&bin.imports);
            let plt_plan = PltCallPlan::from_targets(&targets, self.host.as_ref());
            let manifest = self.host.import_manifest();
            let wasm = match bin.arch {
                BinArch::X86_64 | BinArch::AArch64 => {
                    let (w, unsupported) = recompile_to_wasm_instrumented_plt(
                        &text.data,
                        start,
                        bin.arch,
                        Some(&plt_plan),
                        Some(bin.entry),
                        &manifest,
                    );
                    if !unsupported.is_empty() {
                        eprintln!("recompile unsupported: {:?}", unsupported);
                    }
                    w
                }
            };
            validate_wasm(&wasm)?;
            let obj = self.compile_to_object(&wasm, arch, os)?;
            let entry_param_count = speet_recompile::drive::entry_param_count(&wasm);
            let sp_idx = speet_recompile::drive::sp_param_index(bin.arch);
            let halt_addr = speet_recompile::frontend::halt_addr(start, text.data.len());
            let lr_idx = speet_recompile::drive::lr_param_index(bin.arch);
            return self.link_and_run(&obj, entry_param_count, sp_idx, halt_addr, lr_idx, arch, os);
        }

        let (text, start) = load_text_from_object(path)?;
        let guest_arch = guest_arch_from_object_path(path)?;
        let wasm = match guest_arch {
            BinArch::X86_64 | BinArch::AArch64 => {
                let (w, unsupported) = recompile_to_wasm(&text, start, guest_arch);
                if !unsupported.is_empty() {
                    eprintln!("recompile unsupported: {:?}", unsupported);
                }
                w
            }
        };
        validate_wasm(&wasm)?;
        let obj = self.compile_to_object(&wasm, arch, os)?;
        self.link_and_run(
            &obj,
            speet_recompile::drive::entry_param_count(&wasm),
            speet_recompile::drive::sp_param_index(guest_arch),
            speet_recompile::frontend::halt_addr(start, text.len()),
            speet_recompile::drive::lr_param_index(guest_arch),
            arch,
            os,
        )
    }

    /// Corpus helper: load `.elf`/`.macho` object, extract `.text`, run RV64 path.
    pub fn run_corpus_object(&mut self, guest_obj: &Path, arch: BinArch, os: BinOs) -> Result<ExitStatus, String> {
        let (text, addr) = load_text_from_object(guest_obj)?;
        self.recompile_rv64_and_run(&text, addr, arch, os)
    }

    /// `entry_param_count`/`sp_param_index` describe `__guest_entry`'s real
    /// WASM signature (guest registers are WASM params — see
    /// `speet_recompile::drive::{entry_param_count, sp_param_index}` for the
    /// same-arch case, or the matching frontend's own constant otherwise),
    /// needed so the entry bridge calls it with a C-ABI-correct argument
    /// list instead of leaving guest registers (SP in particular) as
    /// caller-side garbage. See `docs/guides/thin-runtime-genericity.md`.
    #[allow(clippy::too_many_arguments)]
    pub fn link_and_run(
        &self,
        guest_obj: &[u8],
        entry_param_count: u32,
        sp_param_index: u32,
        halt_addr: u64,
        lr_param_index: Option<u32>,
        arch: BinArch,
        os: BinOs,
    ) -> Result<ExitStatus, String> {
        let tc = self
            .toolchain
            .as_ref()
            .ok_or_else(|| "LLVM toolchain not available (set CLANG or install LLVM)".to_string())?;
        // `std::process::id()` alone is shared by every test in this binary — cargo
        // test runs tests in parallel threads within one process, so two guests
        // linking/spawning concurrently would fight over the exact same `guest_exe`
        // path, corrupting or truncating whichever one was mid-write when the other
        // called `Command::new(&exe).status()` (observed as a signal-killed child,
        // `ExitStatus::code() == None`, rather than a real guest exit code). Fold in
        // the thread id and a hash of the guest object so concurrent calls, even
        // from the same process, never share a directory.
        let unique = format!(
            "{:?}-{}",
            std::thread::current().id(),
            ArtifactCache::hash_input(guest_obj)
        );
        let dir = std::env::temp_dir().join(format!("speet_rt_{}_{unique}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let exe = dir.join("guest_exe");
        link_guest(
            tc, self.host.as_ref(), guest_obj, arch, os, entry_param_count, sp_param_index,
            halt_addr, lr_param_index, &dir, &exe,
        )?;
        let status = Command::new(&exe)
            .status()
            .map_err(|e| e.to_string())?;
        if std::env::var("SPEET_RT_KEEP_TMP").is_err() {
            let _ = std::fs::remove_dir_all(&dir);
        } else {
            eprintln!("kept: {}", dir.display());
        }
        Ok(status)
    }

    /// Link only (no spawn) — for tests that assert link success.
    ///
    /// See [`Self::link_and_run`] for what `entry_param_count`/
    /// `sp_param_index`/`halt_addr`/`lr_param_index` mean.
    #[allow(clippy::too_many_arguments)]
    pub fn link_guest_object(
        &self,
        guest_obj: &[u8],
        entry_param_count: u32,
        sp_param_index: u32,
        halt_addr: u64,
        lr_param_index: Option<u32>,
        arch: BinArch,
        os: BinOs,
        out_exe: &Path,
    ) -> Result<(), String> {
        let tc = self
            .toolchain
            .as_ref()
            .ok_or_else(|| "LLVM toolchain not available".to_string())?;
        let dir = out_exe.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        link_guest(
            tc, self.host.as_ref(), guest_obj, arch, os, entry_param_count, sp_param_index,
            halt_addr, lr_param_index, dir, out_exe,
        )
    }
}

fn validate_wasm(wasm: &[u8]) -> Result<(), String> {
    validate_wasm_public(wasm)
}

/// Validate a WASM module (public for integrated runtime).
pub fn validate_wasm_public(wasm: &[u8]) -> Result<(), String> {
    let mut v = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all());
    v.validate_all(wasm).map(|_| ()).map_err(|e| e.to_string())
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

fn guest_arch_from_object_path(path: &Path) -> Result<BinArch, String> {
    let s = path.to_string_lossy();
    if s.contains("aarch64") || s.contains("arm64") {
        return Ok(BinArch::AArch64);
    }
    if s.contains("x86_64") {
        return Ok(BinArch::X86_64);
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let obj = object::File::parse(&*bytes).map_err(|e| e.to_string())?;
    match obj.architecture() {
        object::Architecture::Aarch64 => Ok(BinArch::AArch64),
        object::Architecture::X86_64 => Ok(BinArch::X86_64),
        other => Err(format!("unsupported guest arch in {}: {other:?}", path.display())),
    }
}
