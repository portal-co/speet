//! Optional declarative manifest format — a convenience layer over
//! [`crate::PluginRegistry`]'s programmatic API, not required for it.
//!
//! Hand-rolled `key = value` text, one plugin per blank-line-separated
//! block, `#`-comment lines skipped — no TOML/JSON dependency, consistent
//! with the wire codec's "no serde" rule (see `docs/guides/plugin-api.md`).
//!
//! ```text
//! name    = my-custom-mmu
//! version = 0.1.0
//! kind    = memory          # arch | memory | table | object-model | target
//! host    = wasm             # inproc | wasm | subprocess
//! path    = ./plugins/my-mmu.wasm
//! imports = table:host-table # optional, comma-separated kind:name pairs
//! ```

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use speet_plugin_api::PluginKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestError(pub String);

impl core::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "manifest error: {}", self.0)
    }
}

/// One parsed plugin declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestEntry {
    pub name: String,
    pub version: String,
    pub kind: PluginKind,
    /// The transport scheme this plugin loads through (`"inproc"`, `"wasm"`,
    /// `"subprocess"`, or a 4th host's own scheme name) — looked up in
    /// `PluginRegistry`'s registered transports, not interpreted here.
    pub host: String,
    pub path: String,
    /// `(kind, name)` pairs this plugin is granted as host-entity imports —
    /// see `PluginRegistry::restricted_view`.
    pub imports: Vec<(PluginKind, String)>,
    /// Any other `key = value` pairs in the block (e.g. subprocess `args`),
    /// for host-specific extension without a manifest format change.
    pub extra: Vec<(String, String)>,
}

fn parse_block(block: &str) -> Result<ManifestEntry, ManifestError> {
    let mut fields: Vec<(String, String)> = Vec::new();
    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Strip a trailing `# comment` (only outside of the value's content —
        // simplest correct rule for this format: comments only ever follow
        // real content on the same line, never appear inside a value).
        let line = match line.find('#') {
            Some(idx) => line[..idx].trim_end(),
            None => line,
        };
        let Some((key, value)) = line.split_once('=') else {
            return Err(ManifestError(alloc::format!("malformed line: {line:?}")));
        };
        fields.push((key.trim().to_string(), value.trim().to_string()));
    }

    let get = |key: &str| {
        fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    };

    let name = get("name").ok_or_else(|| ManifestError("missing `name`".to_string()))?;
    let version = get("version").ok_or_else(|| ManifestError("missing `version`".to_string()))?;
    let kind_str = get("kind").ok_or_else(|| ManifestError("missing `kind`".to_string()))?;
    let kind = PluginKind::parse(&kind_str)
        .ok_or_else(|| ManifestError(alloc::format!("unknown kind: {kind_str:?}")))?;
    let host = get("host").ok_or_else(|| ManifestError("missing `host`".to_string()))?;
    let path = get("path").ok_or_else(|| ManifestError("missing `path`".to_string()))?;

    let mut imports = Vec::new();
    if let Some(raw) = get("imports") {
        for entry in raw.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let Some((kind_str, name)) = entry.split_once(':') else {
                return Err(ManifestError(alloc::format!(
                    "malformed import (expected kind:name): {entry:?}"
                )));
            };
            let import_kind = PluginKind::parse(kind_str.trim()).ok_or_else(|| {
                ManifestError(alloc::format!("unknown import kind: {kind_str:?}"))
            })?;
            imports.push((import_kind, name.trim().to_string()));
        }
    }

    let extra: Vec<(String, String)> = fields
        .into_iter()
        .filter(|(k, _)| !matches!(k.as_str(), "name" | "version" | "kind" | "host" | "path" | "imports"))
        .collect();

    Ok(ManifestEntry {
        name,
        version,
        kind,
        host,
        path,
        imports,
        extra,
    })
}

/// Parse one or more blank-line-separated plugin blocks.
pub fn parse_manifest(input: &str) -> Result<Vec<ManifestEntry>, ManifestError> {
    let mut entries = Vec::new();
    let mut block = String::new();
    for line in input.lines() {
        if line.trim().is_empty() {
            if !block.trim().is_empty() {
                entries.push(parse_block(&block)?);
                block.clear();
            }
            continue;
        }
        block.push_str(line);
        block.push('\n');
    }
    if !block.trim().is_empty() {
        entries.push(parse_block(&block)?);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_entry_with_imports() {
        let input = "\
name    = my-arch
version = 0.1.0
kind    = arch
host    = wasm
path    = ./plugins/my-arch.wasm
imports = memory:host-mmu, table:host-table
";
        let entries = parse_manifest(input).unwrap();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.name, "my-arch");
        assert_eq!(e.kind, PluginKind::Arch);
        assert_eq!(e.host, "wasm");
        assert_eq!(
            e.imports,
            alloc::vec![
                (PluginKind::Memory, "host-mmu".to_string()),
                (PluginKind::Table, "host-table".to_string()),
            ]
        );
    }

    #[test]
    fn parses_multiple_blocks_and_comments() {
        let input = "\
# first plugin
name    = a
version = 0.1.0
kind    = target
host    = inproc
path    = n/a

# second plugin
name    = b
version = 0.2.0
kind    = memory          # inline comment
host    = subprocess
path    = ./b.sh
";
        let entries = parse_manifest(input).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "a");
        assert_eq!(entries[1].name, "b");
        assert_eq!(entries[1].kind, PluginKind::Memory);
    }

    #[test]
    fn missing_field_errs() {
        let input = "name = a\nkind = target\nhost = inproc\npath = n/a\n";
        assert!(parse_manifest(input).is_err());
    }
}
