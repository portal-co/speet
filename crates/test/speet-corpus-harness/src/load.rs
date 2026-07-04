//! Load `.text.elf` blobs and companion `.entry` sidecars.

use object::{Object, ObjectSection};
use std::path::Path;

pub fn load_text_blob(path: &Path) -> (Vec<u8>, u64) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let obj = object::File::parse(&*bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let sec = obj
        .section_by_name(".text")
        .unwrap_or_else(|| panic!("no .text in {}", path.display()));
    let data = sec
        .data()
        .unwrap_or_else(|e| panic!("read .text from {}: {e}", path.display()))
        .to_vec();
    assert!(!data.is_empty(), ".text empty in {}", path.display());
    (data, sec.address())
}

/// Main symbol offset within `.text` (from `compile_corpus.sh` sidecar).
pub fn load_entry_offset(text_elf: &Path) -> u64 {
    let stem = text_elf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| panic!("bad corpus path {}", text_elf.display()));
    let base = stem.strip_suffix(".text").unwrap_or(stem);
    let entry_path = text_elf.with_file_name(format!("{base}.entry"));
    let s = std::fs::read_to_string(&entry_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", entry_path.display()));
    let s = s.trim();
    u64::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16)
        .unwrap_or_else(|e| panic!("parse entry offset {s:?}: {e}"))
}

pub fn entry_pc(text_elf: &Path) -> (Vec<u8>, u64, u64) {
    let (text, base) = load_text_blob(text_elf);
    let off = load_entry_offset(text_elf);
    (text, base, base.wrapping_add(off))
}
