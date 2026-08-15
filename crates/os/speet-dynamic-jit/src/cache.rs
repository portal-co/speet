//! On-disk JIT artifact cache (Phase 5 of the dynamic-JIT plan).
//!
//! Mirrors `speet-runtime::cache::ArtifactCache`'s pattern (in-memory
//! `HashMap` plus an optional on-disk mirror, same lightweight
//! non-cryptographic `DefaultHasher`-based `hash_input` — "without adding a
//! blake3 dependency", matching that crate's own stated reasoning) rather
//! than reusing that type directly: `speet-runtime` is stable-toolchain,
//! this crate requires nightly (transitively, from vane-arch), so the two
//! caches stay independent rather than forcing a toolchain dependency
//! either direction.
//!
//! Cache key: `(binary_hash, guest_pc, encoding_hash)`. `encoding_hash` is
//! computed over a bounded prefix of the actual guest bytes at `guest_pc`
//! (see [`ENCODING_WINDOW`]), not just `guest_pc` alone — if
//! self-modifying code changes the bytes at a previously-compiled PC, the
//! key changes with it, so a stale on-disk entry simply becomes
//! unreachable rather than needing active eviction. (Active eviction of the
//! *runtime* dynamic-dispatch-table entry is a separate concern — see
//! `speet_interp::emit_jit_lookup_stub`'s `CheckCode` handling — this cache
//! only avoids redundant recompilation across runs.)
//!
//! **Scope note**: bounding the encoding hash to a fixed-size window is the
//! same simplification `JitRequest::guest_bytes` already accepts elsewhere
//! ("callers unsure how much to send should send a generous prefix") — a
//! trace that reads further than `ENCODING_WINDOW` bytes from `guest_pc`
//! isn't fully captured by the cache key. `ENCODING_WINDOW` is comfortably
//! larger than any short trace exercised so far; revisit if long traces
//! become common.

use crate::compile_pc;
use std::collections::HashMap;
use std::path::PathBuf;
use vane_riscv::Mem;

/// Bytes read from `guest_pc` to compute a cache entry's `encoding_hash`.
pub const ENCODING_WINDOW: usize = 64;

#[derive(Debug, Default)]
pub struct JitArtifactCache {
    memory: HashMap<(String, u64, String), Vec<u8>>,
    root: Option<PathBuf>,
}

impl JitArtifactCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// `root` is typically `speet-runtime::cache_root().join("jit")` in a
    /// real deployment; entries land at `root/<binary_hash>/<pc>-<encoding_hash>.wasm`.
    pub fn with_disk_root(root: PathBuf) -> Self {
        Self { root: Some(root), ..Self::default() }
    }

    /// Same non-cryptographic, dependency-free hash as
    /// `speet-runtime::cache::ArtifactCache::hash_input`.
    pub fn hash_input(bytes: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        bytes.hash(&mut h);
        format!("{:016x}", h.finish())
    }

    fn read_window(mem: &Mem, guest_pc: u64) -> Vec<u8> {
        (0..ENCODING_WINDOW as u64).map(|i| mem.read_byte(guest_pc + i)).collect()
    }

    fn disk_path(&self, binary_hash: &str, guest_pc: u64, encoding_hash: &str) -> Option<PathBuf> {
        self.root.as_ref().map(|root| root.join(binary_hash).join(format!("{guest_pc:x}-{encoding_hash}.wasm")))
    }

    pub fn get(&self, binary_hash: &str, guest_pc: u64, encoding_hash: &str) -> Option<Vec<u8>> {
        let key = (binary_hash.to_string(), guest_pc, encoding_hash.to_string());
        if let Some(cached) = self.memory.get(&key) {
            return Some(cached.clone());
        }
        let path = self.disk_path(binary_hash, guest_pc, encoding_hash)?;
        std::fs::read(path).ok()
    }

    pub fn put(&mut self, binary_hash: &str, guest_pc: u64, encoding_hash: &str, wasm: Vec<u8>) {
        if let Some(path) = self.disk_path(binary_hash, guest_pc, encoding_hash) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&path, &wasm);
        }
        self.memory.insert((binary_hash.to_string(), guest_pc, encoding_hash.to_string()), wasm);
    }

    /// Compile `guest_pc` via [`compile_pc`], reusing a cached artifact
    /// (in-memory, then on-disk) when the guest bytes at `guest_pc` match a
    /// previous compile exactly.
    pub fn compile_or_cached(&mut self, mem: &Mem, binary_hash: &str, guest_pc: u64, num_regs: u32) -> Vec<u8> {
        let window = Self::read_window(mem, guest_pc);
        let encoding_hash = Self::hash_input(&window);
        if let Some(cached) = self.get(binary_hash, guest_pc, &encoding_hash) {
            return cached;
        }
        let wasm = compile_pc(mem, guest_pc, num_regs);
        self.put(binary_hash, guest_pc, &encoding_hash, wasm.clone());
        wasm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_with_ret_at(pc: u64) -> Mem {
        let mut mem = Mem::default();
        for (i, b) in 0x0000_8067u32.to_le_bytes().into_iter().enumerate() {
            mem.write_byte(pc + i as u64, b);
        }
        mem
    }

    #[test]
    fn compile_or_cached_returns_consistent_bytes() {
        let mem = mem_with_ret_at(0x1000);
        let mut cache = JitArtifactCache::new();
        let a = cache.compile_or_cached(&mem, "bin-hash", 0x1000, 2);
        let b = cache.compile_or_cached(&mem, "bin-hash", 0x1000, 2);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn different_guest_bytes_at_the_same_pc_get_different_cache_entries() {
        let mem_a = mem_with_ret_at(0x1000);
        let mut mem_b = Mem::default();
        // A different (nonsense, never decoded) instruction word at the
        // same PC -- simulates self-modifying code between two compiles.
        for (i, b) in 0x0000_0013u32.to_le_bytes().into_iter().enumerate() {
            mem_b.write_byte(0x1000 + i as u64, b);
        }

        let mut cache = JitArtifactCache::new();
        cache.compile_or_cached(&mem_a, "bin-hash", 0x1000, 2);
        cache.compile_or_cached(&mem_b, "bin-hash", 0x1000, 2);

        // Both entries must exist independently under the same
        // (binary_hash, guest_pc) -- neither overwrote the other.
        let hash_a = JitArtifactCache::hash_input(&JitArtifactCache::read_window(&mem_a, 0x1000));
        let hash_b = JitArtifactCache::hash_input(&JitArtifactCache::read_window(&mem_b, 0x1000));
        assert_ne!(hash_a, hash_b);
        assert!(cache.get("bin-hash", 0x1000, &hash_a).is_some());
        assert!(cache.get("bin-hash", 0x1000, &hash_b).is_some());
    }

    #[test]
    fn disk_cache_persists_across_cache_instances() {
        let dir = std::env::temp_dir().join(format!(
            "speet-dynamic-jit-cache-test-{}",
            JitArtifactCache::hash_input(&std::process::id().to_le_bytes())
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let mem = mem_with_ret_at(0x2000);
        let first_bytes = {
            let mut cache = JitArtifactCache::with_disk_root(dir.clone());
            cache.compile_or_cached(&mem, "bin-hash", 0x2000, 2)
        };

        // Fresh cache instance, same disk root, no in-memory state: must
        // still find the entry on disk.
        let window = JitArtifactCache::read_window(&mem, 0x2000);
        let encoding_hash = JitArtifactCache::hash_input(&window);
        let cache2 = JitArtifactCache::with_disk_root(dir.clone());
        let from_disk = cache2.get("bin-hash", 0x2000, &encoding_hash).expect("should be on disk");
        assert_eq!(from_disk, first_bytes);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
