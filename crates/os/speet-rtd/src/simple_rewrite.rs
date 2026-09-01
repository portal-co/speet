//! Speet-side glue wiring the OS-neutral simple-rewrite primitives
//! (`os-rewrite-macho`/`os-rewrite-elf`/`os-codesign-macho`) behind
//! `os_transform_core::TransformBackend`, registered as the
//! `"simple-rewrite"` backend alongside speet's existing AOT recompiler.
//!
//! Unlike the AOT recompiler, this backend never recompiles guest code — it
//! only patches the target's dylib/so load requirements (via
//! `os-rewrite-macho`/`os-rewrite-elf`) and, on macOS, re-signs the result
//! (`os-codesign-macho`), always producing a new cached executable that is
//! then run in place via `execve`, exactly like the AOT path.

use os_transform_core::{BackendId, ObtainError, RunAs, Suitability, TransformBackend};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Rewrite-format version bump whenever the rewrite algorithm or signing
/// policy changes, per the schema doc's cache-invariant requirement.
const REWRITE_FORMAT_VERSION: &str = "simple-rewrite-v1";

/// Where to inject the shim and (on macOS) what identity to sign with.
pub struct SimpleRewriteConfig {
    /// Path to the shim dylib (macOS) or shared object (Linux/BSD) to
    /// inject beside every rewritten executable.
    pub shim_path: PathBuf,
    /// Private staging root; each rewrite gets its own subdirectory.
    pub cache_root: PathBuf,
    #[cfg(target_os = "macos")]
    pub identity: os_codesign_macho::SigningIdentity,
    #[cfg(target_os = "macos")]
    pub keychain: Option<String>,
    #[cfg(target_os = "macos")]
    pub allow_get_task_allow: bool,
}

/// Cache keyed on the invariants the schema doc requires for the macOS
/// path (original bytes hash, shim bytes hash, signing identity,
/// entitlement policy, rewrite-format version) — a sibling type to
/// `speet_runtime::ArtifactCache` rather than an extension of it, since
/// signing identity/entitlement policy aren't concepts that cache models.
/// On Linux/BSD there is no signing identity or entitlement policy, so the
/// key there is just `(original hash, shim hash, format version)`.
#[derive(Default)]
struct RewriteArtifactCache {
    entries: HashMap<String, PathBuf>,
}

impl RewriteArtifactCache {
    fn get(&self, key: &str) -> Option<PathBuf> {
        self.entries.get(key).cloned()
    }
    fn put(&mut self, key: String, path: PathBuf) {
        self.entries.insert(key, path);
    }
}

pub struct SimpleRewriteBackend {
    config: SimpleRewriteConfig,
    cache: RewriteArtifactCache,
}

impl SimpleRewriteBackend {
    pub fn new(config: SimpleRewriteConfig) -> Self {
        Self {
            config,
            cache: RewriteArtifactCache::default(),
        }
    }

    fn stage_dir(&self, key: &str) -> std::io::Result<PathBuf> {
        let name = speet_runtime::ArtifactCache::hash_input(key.as_bytes());
        let dir = self.config.cache_root.join(name);
        std::fs::create_dir_all(&dir)?;
        set_private_dir_perms(&dir)?;
        Ok(dir)
    }
}

#[cfg(unix)]
fn set_private_dir_perms(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
}

#[cfg(unix)]
fn set_executable_perms(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
}

#[cfg(target_os = "macos")]
mod macho_impl {
    use super::*;
    use os_codesign_macho::{
        read_original_entitlements, sign_rewritten_executable, sign_shim, verify_signed,
        EntitlementPolicy, SigningIdentity,
    };
    use os_rewrite_macho::{rewrite_macho, MachORewriteInput};

    fn identity_key(identity: &SigningIdentity) -> String {
        match identity {
            SigningIdentity::Real(s) => s.clone(),
            SigningIdentity::AdHoc => "-".to_string(),
        }
    }

    fn cache_key(
        original_hash: &str,
        shim_hash: &str,
        identity: &SigningIdentity,
        ent: &EntitlementPolicy,
    ) -> String {
        format!(
            "{original_hash}:{shim_hash}:{}:{ent:?}:{REWRITE_FORMAT_VERSION}",
            identity_key(identity)
        )
    }

    impl TransformBackend for SimpleRewriteBackend {
        fn id(&self) -> BackendId {
            BackendId::SIMPLE_REWRITE
        }

        fn analyze(&self, path: &Path) -> Result<Suitability, String> {
            let original = std::fs::read(path).map_err(|e| e.to_string())?;
            match rewrite_macho(&MachORewriteInput {
                original: &original,
                dylib_load_path: "@executable_path/x".into(),
            }) {
                Ok(_) => Ok(Suitability {
                    suitable: true,
                    reasons: vec![],
                }),
                Err(e) => Ok(Suitability {
                    suitable: false,
                    reasons: vec![e.to_string()],
                }),
            }
        }

        fn obtain(&mut self, path: &Path) -> Result<RunAs, ObtainError> {
            let original =
                std::fs::read(path).map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            let shim_bytes = std::fs::read(&self.config.shim_path)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            let original_hash = speet_runtime::ArtifactCache::hash_input(&original);
            let shim_hash = speet_runtime::ArtifactCache::hash_input(&shim_bytes);
            let orig_entitlements = read_original_entitlements(path).unwrap_or_default();

            let key = cache_key(
                &original_hash,
                &shim_hash,
                &self.config.identity,
                &orig_entitlements,
            );
            if let Some(cached) = self.cache.get(&key) {
                if verify_signed(&cached) {
                    return Ok(RunAs::Exec(cached));
                }
            }

            let rewritten = rewrite_macho(&MachORewriteInput {
                original: &original,
                dylib_load_path: "@executable_path/x".into(),
            })
            .map_err(|e| {
                ObtainError::Unsuitable(Suitability {
                    suitable: false,
                    reasons: vec![e.to_string()],
                })
            })?;

            let dir = self
                .stage_dir(&key)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            let program_path = dir.join("program");
            let shim_out = dir.join("x");

            std::fs::copy(&self.config.shim_path, &shim_out)
                .map_err(|e| ObtainError::TransformFailed(format!("stage shim: {e}")))?;
            std::fs::write(&program_path, &rewritten)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            set_executable_perms(&shim_out)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            set_executable_perms(&program_path)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;

            let shim_info = sign_shim(
                &shim_out,
                &self.config.identity,
                self.config.keychain.as_deref(),
            )
            .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            sign_rewritten_executable(
                &program_path,
                &shim_info,
                &self.config.identity,
                &orig_entitlements,
                self.config.allow_get_task_allow,
                &dir,
                self.config.keychain.as_deref(),
            )
            .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;

            self.cache.put(key, program_path.clone());
            Ok(RunAs::Exec(program_path))
        }
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
))]
mod elf_impl {
    use super::*;
    use os_rewrite_elf::{rewrite_elf, ElfRewriteInput};

    fn cache_key(original_hash: &str, shim_hash: &str) -> String {
        format!("{original_hash}:{shim_hash}:{REWRITE_FORMAT_VERSION}")
    }

    impl TransformBackend for SimpleRewriteBackend {
        fn id(&self) -> BackendId {
            BackendId::SIMPLE_REWRITE
        }

        fn analyze(&self, path: &Path) -> Result<Suitability, String> {
            let original = std::fs::read(path).map_err(|e| e.to_string())?;
            match rewrite_elf(&ElfRewriteInput {
                original: &original,
                needed_name: "x".to_string(),
                rpath_entry: Some("$ORIGIN/x".to_string()),
            }) {
                Ok(_) => Ok(Suitability {
                    suitable: true,
                    reasons: vec![],
                }),
                Err(e) => Ok(Suitability {
                    suitable: false,
                    reasons: vec![e.to_string()],
                }),
            }
        }

        fn obtain(&mut self, path: &Path) -> Result<RunAs, ObtainError> {
            let original =
                std::fs::read(path).map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            let shim_bytes = std::fs::read(&self.config.shim_path)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            let original_hash = speet_runtime::ArtifactCache::hash_input(&original);
            let shim_hash = speet_runtime::ArtifactCache::hash_input(&shim_bytes);

            let key = cache_key(&original_hash, &shim_hash);
            if let Some(cached) = self.cache.get(&key) {
                return Ok(RunAs::Exec(cached));
            }

            let rewritten = rewrite_elf(&ElfRewriteInput {
                original: &original,
                needed_name: "x".to_string(),
                rpath_entry: Some("$ORIGIN/x".to_string()),
            })
            .map_err(|e| {
                ObtainError::Unsuitable(Suitability {
                    suitable: false,
                    reasons: vec![e.to_string()],
                })
            })?;

            let dir = self
                .stage_dir(&key)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            let program_path = dir.join("program");
            let shim_out = dir.join("x");

            std::fs::copy(&self.config.shim_path, &shim_out)
                .map_err(|e| ObtainError::TransformFailed(format!("stage shim: {e}")))?;
            std::fs::write(&program_path, &rewritten)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            set_executable_perms(&shim_out)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;
            set_executable_perms(&program_path)
                .map_err(|e| ObtainError::TransformFailed(e.to_string()))?;

            self.cache.put(key, program_path.clone());
            Ok(RunAs::Exec(program_path))
        }
    }
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
)))]
impl TransformBackend for SimpleRewriteBackend {
    fn id(&self) -> BackendId {
        BackendId::SIMPLE_REWRITE
    }

    fn analyze(&self, _path: &Path) -> Result<Suitability, String> {
        Ok(Suitability {
            suitable: false,
            reasons: vec![
                "simple-rewrite is only implemented for macOS, Linux, and BSD".to_string(),
            ],
        })
    }

    fn obtain(&mut self, _path: &Path) -> Result<RunAs, ObtainError> {
        Err(ObtainError::Unsuitable(Suitability {
            suitable: false,
            reasons: vec![
                "simple-rewrite is only implemented for macOS, Linux, and BSD".to_string(),
            ],
        }))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::process::Command;

    fn build_shim(dir: &Path) -> PathBuf {
        let c_src = dir.join("shim.c");
        std::fs::write(
            &c_src,
            "__attribute__((constructor)) static void i(void) {}\n",
        )
        .unwrap();
        let dylib = dir.join("shim.dylib");
        let status = Command::new("cc")
            .args(["-dynamiclib", "-O2", "-o"])
            .arg(&dylib)
            .arg(&c_src)
            .status()
            .expect("run cc");
        assert!(status.success(), "cc failed to build the test shim");
        dylib
    }

    /// Full pipeline: rewrite a real linked Mach-O corpus binary, ad-hoc
    /// sign it and its shim, execute it, and confirm a second `obtain` hits
    /// the cache and returns the same path.
    #[test]
    fn obtain_rewrites_signs_caches_and_runs_a_real_binary() {
        if !cfg!(target_arch = "aarch64") {
            eprintln!("SKIP: aarch64-only corpus fixture");
            return;
        }
        let guest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../test-data/c-corpus/aarch64-macos/exit42.linked.macho");
        if !guest.is_file() {
            eprintln!("SKIP: missing corpus fixture {}", guest.display());
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let shim = build_shim(dir.path());
        let mut backend = SimpleRewriteBackend::new(SimpleRewriteConfig {
            shim_path: shim,
            cache_root: dir.path().join("cache"),
            identity: os_codesign_macho::SigningIdentity::AdHoc,
            keychain: None,
            allow_get_task_allow: false,
        });

        let suitability = backend.analyze(&guest).expect("analyze should not error");
        assert!(
            suitability.suitable,
            "expected suitable: {:?}",
            suitability.reasons
        );

        let RunAs::Exec(exe) = backend.obtain(&guest).expect("obtain should succeed");
        assert!(
            os_codesign_macho::verify_signed(&exe),
            "rewritten executable must be validly signed"
        );

        let status = Command::new(&exe)
            .status()
            .expect("run rewritten executable");
        assert_eq!(status.code(), Some(42));

        let RunAs::Exec(exe2) = backend
            .obtain(&guest)
            .expect("second obtain should succeed");
        assert_eq!(
            exe, exe2,
            "second obtain should be a cache hit returning the same path"
        );
    }
}
