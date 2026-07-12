//! CLI for `speet-abi-codegen` — emits ABI redirect stub Rust sources.

use std::fs;
use std::path::PathBuf;

use speet_abi_codegen::{generate, CodegenConfig, StubArch};
use speet_abi_spec::parse_bridgesupport;

fn main() {
    if let Err(e) = run() {
        eprintln!("speet-abi-codegen: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut symbols: Vec<String> = Vec::new();
    let mut archs: Vec<StubArch> = Vec::new();
    let mut all_symbols = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" | "-i" => {
                input = Some(PathBuf::from(args.next().ok_or("--input requires a path")?));
            }
            "--output" | "-o" => {
                output = Some(PathBuf::from(args.next().ok_or("--output requires a path")?));
            }
            "--symbols" | "-s" => {
                let list = args.next().ok_or("--symbols requires a comma-separated list")?;
                symbols = list
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--archs" | "-a" => {
                let list = args.next().ok_or("--archs requires a comma-separated list")?;
                for part in list.split(',') {
                    let part = part.trim();
                    archs.push(
                        StubArch::parse(part).ok_or_else(|| {
                            format!("unknown arch {part:?} (expected x86_64 or aarch64)")
                        })?,
                    );
                }
            }
            "--all-symbols" => {
                all_symbols = true;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    let input = input.ok_or("--input is required")?;
    let output = output.ok_or("--output is required")?;
    if !all_symbols && symbols.is_empty() {
        symbols = vec!["write".into(), "exit".into()];
    }
    if archs.is_empty() {
        archs = vec![StubArch::X86_64, StubArch::Aarch64];
    }

    let xml = fs::read_to_string(&input).map_err(|e| e.to_string())?;
    let spec = parse_bridgesupport(&xml).map_err(|e| e.to_string())?;
    let config = CodegenConfig {
        symbols,
        all_symbols,
        archs,
        source_label: input.display().to_string(),
    };
    let files = generate(&spec, &config)?;

    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    for file in &files {
        let path = output.join(&file.path);
        fs::write(&path, &file.contents).map_err(|e| e.to_string())?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        r#"speet-abi-codegen — emit ABI redirect stub Rust sources

Usage:
  speet-abi-codegen -i INPUT.bridgesupport.xml -o OUTPUT_DIR [options]

Options:
  -i, --input PATH       BridgeSupport XML input (required)
  -o, --output PATH      Output directory for generated Rust modules (required)
  -s, --symbols LIST     Comma-separated checked-in / wired subset (default: write,exit)
  -a, --archs LIST       Comma-separated arch list: x86_64,aarch64 (default: both)
      --all-symbols      Emit every function in the spec (libc/libSystem baseline)
  -h, --help             Show this help

Scope (see AGENTS.md §9 and docs/future/abi-spec-redirects.md):
  - Checked-in / runtime-wired stubs: -s … -o speet-abi-stubs/src/generated
  - Full libc + libSystem baseline (gitignored): --all-symbols -o speet-abi-stubs/.generated/libc
  - Do not check in non-baseline Apple frameworks or GPLed SDK surfaces (licensing).
"#
    );
}
