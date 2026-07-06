//! Thin runtime driver.

use crate::cache::{load_binary, load_text_from_object, ArtifactCache};
use crate::link::link_guest;
use crate::toolchain::LlvmToolchain;
use binary_io::{BinArch, BinOs};
use object::Object;
use speet_host_api::HostApi;
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::{assert_same_platform, recompile_rv64_to_wasm, recompile_to_wasm};
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
        let key = ArtifactCache::hash_input(text);
        if let Some(w) = self.cache.get_wasm(&key) {
            return w;
        }
        let wasm = recompile_rv64_to_wasm(text, start_addr);
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
        let wasm = self.recompile_rv64_text(text, start_addr);
        validate_wasm(&wasm)?;
        let obj = self.compile_to_object(&wasm, arch, os)?;
        self.link_and_run(&obj, arch, os)
    }

    /// Load a host binary from disk, recompile its `.text`, and run.
    pub fn recompile_binary_and_run(
        &mut self,
        path: &Path,
        arch: BinArch,
        os: BinOs,
    ) -> Result<ExitStatus, String> {
        if let Ok((text, start)) = load_text_from_object(path) {
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
            return self.link_and_run(&obj, arch, os);
        }

        let bin = load_binary(path)?;
        assert_same_platform(&bin)?;
        let text = bin
            .sections
            .iter()
            .find(|s| s.name == ".text" || s.name == "__TEXT,__text")
            .ok_or_else(|| "no .text section".to_string())?;
        let start = text.addr;
        let wasm = match bin.arch {
            BinArch::X86_64 | BinArch::AArch64 => {
                let (w, unsupported) = recompile_to_wasm(&text.data, start, bin.arch);
                if !unsupported.is_empty() {
                    eprintln!("recompile unsupported: {:?}", unsupported);
                }
                w
            }
        };
        validate_wasm(&wasm)?;
        let obj = self.compile_to_object(&wasm, arch, os)?;
        self.link_and_run(&obj, arch, os)
    }

    /// Corpus helper: load `.elf`/`.macho` object, extract `.text`, run RV64 path.
    pub fn run_corpus_object(&mut self, guest_obj: &Path, arch: BinArch, os: BinOs) -> Result<ExitStatus, String> {
        let (text, addr) = load_text_from_object(guest_obj)?;
        self.recompile_rv64_and_run(&text, addr, arch, os)
    }

    pub fn link_and_run(&self, guest_obj: &[u8], arch: BinArch, os: BinOs) -> Result<ExitStatus, String> {
        let tc = self
            .toolchain
            .as_ref()
            .ok_or_else(|| "LLVM toolchain not available (set CLANG or install LLVM)".to_string())?;
        let dir = std::env::temp_dir().join(format!("speet_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let exe = dir.join("guest_exe");
        link_guest(tc, self.host.as_ref(), guest_obj, arch, os, &dir, &exe)?;
        let status = Command::new(&exe)
            .status()
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_dir_all(&dir);
        Ok(status)
    }

    /// Link only (no spawn) — for tests that assert link success.
    pub fn link_guest_object(
        &self,
        guest_obj: &[u8],
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
        link_guest(tc, self.host.as_ref(), guest_obj, arch, os, dir, out_exe)
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
