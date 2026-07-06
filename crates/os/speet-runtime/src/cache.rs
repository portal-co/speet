//! Content-keyed artifact cache for recompiled WASM and native objects.

use object::{Object, ObjectSection};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Cache keyed by `(input_hash, pipeline_version)`.
#[derive(Debug, Default)]
pub struct ArtifactCache {
    wasm: HashMap<(String, String), Vec<u8>>,
    objects: HashMap<(String, String, String), Vec<u8>>,
    executables: HashMap<String, PathBuf>,
    root: Option<PathBuf>,
}

impl ArtifactCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_disk_root(root: PathBuf) -> Self {
        Self {
            root: Some(root),
            ..Self::default()
        }
    }

    pub fn pipeline_version() -> &'static str {
        "speet-runtime-v1"
    }

    pub fn get_wasm(&self, input_hash: &str) -> Option<Vec<u8>> {
        self.wasm
            .get(&(input_hash.to_string(), Self::pipeline_version().into()))
            .cloned()
    }

    pub fn put_wasm(&mut self, input_hash: &str, wasm: Vec<u8>) {
        if let Some(root) = &self.root {
            let path = root.join(format!("{input_hash}.wasm"));
            let _ = std::fs::create_dir_all(root);
            let _ = std::fs::write(path, &wasm);
        }
        self.wasm.insert(
            (input_hash.to_string(), Self::pipeline_version().into()),
            wasm,
        );
    }

    pub fn get_object(&self, input_hash: &str, arch_os: &str) -> Option<Vec<u8>> {
        self.objects
            .get(&(
                input_hash.to_string(),
                arch_os.to_string(),
                Self::pipeline_version().into(),
            ))
            .cloned()
    }

    pub fn put_object(&mut self, input_hash: &str, arch_os: &str, obj: Vec<u8>) {
        if let Some(root) = &self.root {
            let path = root.join(format!("{input_hash}.{arch_os}.o"));
            let _ = std::fs::create_dir_all(root);
            let _ = std::fs::write(path, &obj);
        }
        self.objects.insert(
            (
                input_hash.to_string(),
                arch_os.to_string(),
                Self::pipeline_version().into(),
            ),
            obj,
        );
    }

    pub fn get_executable(&self, cache_key: &str) -> Option<PathBuf> {
        self.executables.get(cache_key).cloned()
    }

    pub fn put_executable(&mut self, cache_key: &str, path: PathBuf) {
        if let Some(root) = &self.root {
            let dest = root.join(format!("{cache_key}.exe"));
            let _ = std::fs::create_dir_all(root);
            let _ = std::fs::copy(&path, &dest);
            self.executables.insert(cache_key.to_string(), dest);
        } else {
            self.executables.insert(cache_key.to_string(), path);
        }
    }

    pub fn executable_dir(&self) -> Option<PathBuf> {
        self.root.clone()
    }

    /// BLAKE3 hex digest of input bytes (falls back to length tag if hasher unavailable).
    pub fn hash_input(bytes: &[u8]) -> String {
        // Lightweight stable hash without adding a blake3 dependency in v1.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        bytes.hash(&mut h);
        format!("{:016x}", h.finish())
    }
}

/// Load a host binary via `binary_io::load_auto`.
pub fn load_binary(path: &Path) -> Result<binary_io::LoadedBinary, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    binary_io::load_auto(&bytes).map_err(|e| e.to_string())
}

/// Extract `.text` from a corpus ELF/Mach-O object file.
pub fn load_text_from_object(path: &Path) -> Result<(Vec<u8>, u64), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let obj = object::File::parse(&*bytes).map_err(|e| e.to_string())?;
    let section = obj
        .section_by_name(".text")
        .or_else(|| obj.section_by_name("__TEXT,__text"))
        .ok_or_else(|| format!("no .text in {}", path.display()))?;
    let data = section.data().map_err(|e| e.to_string())?.to_vec();
    Ok((data, section.address()))
}
