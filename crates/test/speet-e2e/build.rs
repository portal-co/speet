use std::{env, path::Path, process::Command};

fn main() {
    let out = env::var("OUT_DIR").unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/e2e/c");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", root.display());

    // ── Detect clang ──────────────────────────────────────────────────────────
    // Prefer `CLANG` env var, then try known brew paths, then plain `clang`.
    let clang_candidates = [
        env::var("CLANG").ok(),
        Some("/opt/homebrew/opt/llvm/bin/clang".into()),
        Some("/usr/local/opt/llvm/bin/clang".into()),
        Some("clang".into()),
    ];

    let clang = clang_candidates
        .iter()
        .filter_map(|c| c.as_deref())
        .find(|c| Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false))
        .map(|s| s.to_string());

    // ── RV32 targets ─────────────────────────────────────────────────────────
    let rv32_ok = try_compile(
        &clang,
        &root.join("riscv/arith.c"),
        Path::new(&out).join("rv32_arith.o"),
        &[
            "--target=riscv32-unknown-elf",
            "-march=rv32im",
            "-mabi=ilp32",
            "-nostdlib",
            "-ffreestanding",
            "-O0",
            "-c",
        ],
    );

    // ── RV64 targets ─────────────────────────────────────────────────────────
    let rv64_ok = try_compile(
        &clang,
        &root.join("riscv/arith.c"),
        Path::new(&out).join("rv64_arith.o"),
        &[
            "--target=riscv64-unknown-elf",
            "-march=rv64im",
            "-mabi=lp64",
            "-nostdlib",
            "-ffreestanding",
            "-O0",
            "-c",
        ],
    );

    // ── x86_64 targets ───────────────────────────────────────────────────────
    let x86_ok = try_compile(
        &clang,
        &root.join("x86_64/arith.c"),
        Path::new(&out).join("x86_arith.o"),
        &[
            "--target=x86_64-unknown-none",
            "-ffreestanding",
            "-nostdlib",
            "-O0",
            "-c",
        ],
    );

    // Emit flags so tests can detect which programs are available.
    println!("cargo:rustc-env=E2E_RV32_ARITH={}", if rv32_ok { format!("{}/rv32_arith.o", out) } else { String::new() });
    println!("cargo:rustc-env=E2E_RV64_ARITH={}", if rv64_ok { format!("{}/rv64_arith.o", out) } else { String::new() });
    println!("cargo:rustc-env=E2E_X86_ARITH={}", if x86_ok { format!("{}/x86_arith.o", out) } else { String::new() });
}

/// Compile `src` → `dst` using `clang` with extra `flags`.
/// Returns `true` on success.  Silent on failure — tests will skip.
fn try_compile(
    clang: &Option<String>,
    src: &Path,
    dst: std::path::PathBuf,
    flags: &[&str],
) -> bool {
    let Some(clang) = clang else { return false };
    if !src.exists() { return false; }

    println!("cargo:rerun-if-changed={}", src.display());

    let status = Command::new(clang)
        .args(flags)
        .arg(src)
        .arg("-o")
        .arg(&dst)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=compiled {} → {}", src.file_name().unwrap().to_string_lossy(), dst.display());
            true
        }
        Ok(s) => {
            println!("cargo:warning=skipping {}: clang exited {}", src.display(), s);
            false
        }
        Err(e) => {
            println!("cargo:warning=skipping {}: {e}", src.display());
            false
        }
    }
}
