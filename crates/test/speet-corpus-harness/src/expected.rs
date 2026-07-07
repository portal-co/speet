//! Parse `expected.toml` for per-triple semantic assertions.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct TripleExpectation {
    pub exit_code: Option<i32>,
    pub main_return: Option<i32>,
    pub stdout_sha256: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ProgramExpectations {
    pub by_triple: HashMap<String, TripleExpectation>,
}

/// Minimal TOML parser for the corpus `expected.toml` shape (no external dep).
pub fn load_expected(path: &Path) -> Result<ProgramExpectations, String> {
    let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse_expected(&raw)
}

pub fn parse_expected(raw: &str) -> Result<ProgramExpectations, String> {
    let mut out = ProgramExpectations::default();
    let mut current_triple: Option<String> = None;
    let mut current = TripleExpectation::default();

    let flush = |out: &mut ProgramExpectations, triple: &mut Option<String>, cur: &mut TripleExpectation| {
        if let Some(t) = triple.take() {
            out.by_triple.insert(t, cur.clone());
            *cur = TripleExpectation::default();
        }
    };

    for line in raw.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[triple]]" {
            flush(&mut out, &mut current_triple, &mut current);
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim().trim_matches('"');
        match key {
            "name" => {
                flush(&mut out, &mut current_triple, &mut current);
                current_triple = Some(val.to_string());
            }
            "exit_code" => current.exit_code = Some(val.parse().map_err(|e| format!("exit_code: {e}"))?),
            "main_return" => {
                current.main_return = Some(val.parse().map_err(|e| format!("main_return: {e}"))?)
            }
            "stdout_sha256" => current.stdout_sha256 = Some(val.to_string()),
            _ => {}
        }
    }
    flush(&mut out, &mut current_triple, &mut current);
    Ok(out)
}

/// Resolve expectation for a build triple (e.g. `x86_64-linux-gnu`).
pub fn expectation_for_triple<'a>(
    exp: &'a ProgramExpectations,
    triple: &str,
) -> Option<&'a TripleExpectation> {
    exp.by_triple.get(triple)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_arith_expected() {
        let raw = r#"
[[triple]]
name = "x86_64-linux-gnu"
main_return = 40
"#;
        let exp = parse_expected(raw).unwrap();
        let t = exp.by_triple.get("x86_64-linux-gnu").unwrap();
        assert_eq!(t.main_return, Some(40));
    }
}
