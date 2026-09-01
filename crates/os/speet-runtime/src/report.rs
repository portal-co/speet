//! Structured recompile/suitability report for the debug MCP and `speet-rtdctl`.

use crate::integrated::SuitabilityReport;
use std::path::{Path, PathBuf};

/// Content hashes of the currently loaded hot-pluggable WASM guests, or the
/// static-link sentinels when the `hot-recompiler` feature is off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginHashes {
    pub recompiler: String,
    pub stubs: String,
}

impl PluginHashes {
    pub const NATIVE: &'static str = "native";
    pub const LINKED_STUBS: &'static str = "linked-stubs";

    pub fn static_link() -> Self {
        Self {
            recompiler: Self::NATIVE.into(),
            stubs: Self::LINKED_STUBS.into(),
        }
    }
}

impl Default for PluginHashes {
    fn default() -> Self {
        Self::static_link()
    }
}

/// Guest binary coordinates captured alongside a report.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GuestInfo {
    pub path: PathBuf,
    pub arch: String,
    pub text_addr: u64,
    pub text_len: u64,
    pub entry: u64,
    pub imports: Vec<String>,
}

impl GuestInfo {
    pub fn to_json(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        let imports = self
            .imports
            .iter()
            .map(|s| format!("\"{}\"", esc(s)))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"path\":\"{}\",\"arch\":\"{}\",\"text_addr\":{},\"text_len\":{},\"entry\":{},\"imports\":[{}]}}",
            esc(&self.path.display().to_string()),
            esc(&self.arch),
            self.text_addr,
            self.text_len,
            self.entry,
            imports
        )
    }
}

/// One analyze/recompile/obtain attempt. Typed buckets stay typed — MCP and
/// the CLI print them as JSON; the generic daemon wire still flattens
/// suitability into `unresolved-dep:` / `fn-ptr-dep:` / `unsupported-insn:`
/// tags so existing clients keep working.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RecompileReport {
    pub guest: GuestInfo,
    pub suitable: bool,
    pub unresolved_deps: Vec<String>,
    pub fn_ptr_deps: Vec<String>,
    pub unsupported_insns: Vec<String>,
    pub translate_error: Option<String>,
    pub link_error: Option<String>,
    pub plugins: PluginHashes,
    pub exe_path: Option<PathBuf>,
}

impl RecompileReport {
    pub fn from_suitability(path: &Path, r: &SuitabilityReport, plugins: PluginHashes) -> Self {
        Self {
            guest: GuestInfo {
                path: path.to_path_buf(),
                ..GuestInfo::default()
            },
            suitable: r.suitable,
            unresolved_deps: r.unresolved_deps.clone(),
            fn_ptr_deps: r.fn_ptr_deps.clone(),
            plugins,
            ..Self::default()
        }
    }

    pub fn to_json(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        fn arr(items: &[String]) -> String {
            let inner = items
                .iter()
                .map(|s| format!("\"{}\"", esc(s)))
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        format!(
            "{{\
\"guest\":{{\"path\":\"{}\",\"arch\":\"{}\",\"text_addr\":{},\"text_len\":{},\"entry\":{},\"imports\":{}}},\
\"suitable\":{},\
\"unresolved_deps\":{},\
\"fn_ptr_deps\":{},\
\"unsupported_insns\":{},\
\"translate_error\":{},\
\"link_error\":{},\
\"plugins\":{{\"recompiler\":\"{}\",\"stubs\":\"{}\"}},\
\"exe_path\":{}\
}}",
            esc(&self.guest.path.display().to_string()),
            esc(&self.guest.arch),
            self.guest.text_addr,
            self.guest.text_len,
            self.guest.entry,
            arr(&self.guest.imports),
            self.suitable,
            arr(&self.unresolved_deps),
            arr(&self.fn_ptr_deps),
            arr(&self.unsupported_insns),
            opt_str(&self.translate_error),
            opt_str(&self.link_error),
            esc(&self.plugins.recompiler),
            esc(&self.plugins.stubs),
            match &self.exe_path {
                Some(p) => format!("\"{}\"", esc(&p.display().to_string())),
                None => "null".into(),
            },
        )
    }
}

fn opt_str(v: &Option<String>) -> String {
    match v {
        Some(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        None => "null".into(),
    }
}

impl From<&RecompileReport> for SuitabilityReport {
    fn from(r: &RecompileReport) -> Self {
        SuitabilityReport {
            suitable: r.suitable,
            unresolved_deps: r.unresolved_deps.clone(),
            fn_ptr_deps: r.fn_ptr_deps.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_json_keeps_typed_buckets() {
        let r = RecompileReport {
            unresolved_deps: vec!["foo".into()],
            fn_ptr_deps: vec!["bar".into()],
            unsupported_insns: vec!["undef:ffffffff".into()],
            plugins: PluginHashes {
                recompiler: "abc".into(),
                stubs: "def".into(),
            },
            ..RecompileReport::default()
        };
        let j = r.to_json();
        assert!(j.contains("\"unresolved_deps\":[\"foo\"]"), "{j}");
        assert!(j.contains("\"fn_ptr_deps\":[\"bar\"]"), "{j}");
        assert!(
            j.contains("\"unsupported_insns\":[\"undef:ffffffff\"]"),
            "{j}"
        );
        assert!(j.contains("\"recompiler\":\"abc\""), "{j}");
        assert!(j.contains("\"stubs\":\"def\""), "{j}");
    }

    #[test]
    fn static_link_hashes_are_sentinels() {
        let h = PluginHashes::static_link();
        assert_eq!(h.recompiler, PluginHashes::NATIVE);
        assert_eq!(h.stubs, PluginHashes::LINKED_STUBS);
    }
}
