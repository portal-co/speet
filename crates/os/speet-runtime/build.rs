use std::{env, path::Path, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let clang = find_clang();
    let lld = find_lld(&clang);

    set_env("SPEET_RT_CLANG", &clang);
    set_env("SPEET_RT_LLD", &lld);
}

fn set_env(key: &str, value: &Option<String>) {
    match value {
        Some(v) => println!("cargo:rustc-env={key}={v}"),
        None => println!("cargo:rustc-env={key}="),
    }
}

fn find_clang() -> Option<String> {
    let candidates = [
        env::var("CLANG").ok(),
        Some("/opt/homebrew/opt/llvm/bin/clang".into()),
        Some("/usr/local/opt/llvm/bin/clang".into()),
        Some("clang".into()),
    ];
    candidates
        .iter()
        .filter_map(|c| c.as_deref())
        .find(|c| tool_works(c, &["--version"]))
        .map(|s| s.to_string())
}

fn find_lld(clang: &Option<String>) -> Option<String> {
    let mut candidates = vec![
        env::var("LLD").ok(),
        Some("/opt/homebrew/opt/llvm/bin/ld.lld".into()),
        Some("/opt/homebrew/opt/llvm/bin/lld".into()),
        Some("/usr/local/opt/llvm/bin/ld.lld".into()),
        Some("ld.lld".into()),
        Some("lld".into()),
    ];
    if let Some(c) = clang {
        if let Some(parent) = Path::new(c).parent() {
            candidates.push(Some(parent.join("ld.lld").display().to_string()));
        }
    }
    candidates
        .into_iter()
        .flatten()
        .find(|c| tool_works(c, &["--version"]) || tool_works(c, &[]))
}

fn tool_works(tool: &str, args: &[&str]) -> bool {
    Command::new(tool)
        .args(args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
