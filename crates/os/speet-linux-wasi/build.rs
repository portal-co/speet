//! Compile [`speet-linux-wasi-guest`] to WASM, lower host-mem imports, embed.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.join("../../..");
    let guest_src = workspace_root.join("crates/os/speet-linux-wasi-guest/src");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let canonical = out_dir.join("canonical_guest.wasm");
    let fallback = manifest_dir.join("guest/canonical.wasm");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", guest_src.join("lib.rs").display());

    if try_build_and_lower(&workspace_root, &canonical) {
        if let Some(parent) = fallback.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::copy(&canonical, &fallback);
    } else if fallback.is_file() {
        std::fs::copy(&fallback, &canonical).expect("copy fallback guest wasm");
    } else {
        panic!(
            "speet-linux-wasi-guest build failed and no fallback at {}",
            fallback.display()
        );
    }
}

fn try_build_and_lower(workspace_root: &Path, out: &Path) -> bool {
    let target = "wasm32-unknown-unknown";
    let status = Command::new("rustup")
        .args([
            "run",
            "stable",
            "cargo",
            "build",
            "-p",
            "speet-linux-wasi-guest",
            "--target",
            target,
            "--release",
        ])
        .current_dir(workspace_root)
        .status();

    let Ok(status) = status else {
        return false;
    };
    if !status.success() {
        return false;
    }

    let raw_wasm = guest_wasm_path(workspace_root, target);
    let Ok(wasm) = std::fs::read(&raw_wasm) else {
        eprintln!("guest wasm missing at {}", raw_wasm.display());
        return false;
    };

    let canonical =
        speet_host_mem_shim::lower_host_mem_imports(&wasm).expect("lower guest wasm");
    std::fs::write(out, &canonical).expect("write canonical guest wasm");
    true
}

fn guest_wasm_path(workspace_root: &Path, target: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        let p = PathBuf::from(dir)
            .join(target)
            .join("release")
            .join("speet_linux_wasi_guest.wasm");
        if p.is_file() {
            return p;
        }
    }
    workspace_root
        .join("target")
        .join(target)
        .join("release")
        .join("speet_linux_wasi_guest.wasm")
}
