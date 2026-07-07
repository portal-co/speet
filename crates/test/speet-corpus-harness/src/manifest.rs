//! Parse committed `manifest.toml` artifact index.

use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Artifact {
    pub program: String,
    pub triple: String,
    pub linked: String,
    pub text: String,
    pub entry: String,
    pub arch: String,
    pub os: String,
}

pub fn load_manifest(path: &Path) -> Result<Vec<Artifact>, String> {
    let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse_manifest(&raw)
}

pub fn parse_manifest(raw: &str) -> Result<Vec<Artifact>, String> {
    let mut out = Vec::new();
    let mut cur: Option<Artifact> = None;

    let flush = |out: &mut Vec<Artifact>, cur: &mut Option<Artifact>| {
        if let Some(a) = cur.take() {
            out.push(a);
        }
    };

    for line in raw.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[artifact]]" {
            flush(&mut out, &mut cur);
            cur = Some(Artifact {
                program: String::new(),
                triple: String::new(),
                linked: String::new(),
                text: String::new(),
                entry: String::new(),
                arch: String::new(),
                os: String::new(),
            });
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim().trim_matches('"');
        let Some(ref mut a) = cur else { continue };
        match key {
            "program" => a.program = val.to_string(),
            "triple" => a.triple = val.to_string(),
            "linked" => a.linked = val.to_string(),
            "text" => a.text = val.to_string(),
            "entry" => a.entry = val.to_string(),
            "arch" => a.arch = val.to_string(),
            "os" => a.os = val.to_string(),
            _ => {}
        }
    }
    flush(&mut out, &mut cur);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_one_artifact() {
        let raw = r#"
[[artifact]]
program = "arith"
triple = "x86_64-linux-gnu"
linked = "x86_64-linux/arith.linked.elf"
text = "x86_64-linux/arith.text.elf"
entry = "x86_64-linux/arith.entry"
arch = "x86_64"
os = "linux"
"#;
        let arts = parse_manifest(raw).unwrap();
        assert_eq!(arts.len(), 1);
        assert_eq!(arts[0].program, "arith");
    }
}
