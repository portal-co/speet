//! Original vs recompiled outcome comparison.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use binary_io::{BinArch, BinOs};
use speet_corpus_harness::{
    digest::sha256_hex, expectation_for_triple, expected::load_expected, manifest::Artifact,
};
use speet_guest_runner::{GuestArch, GuestOs, PathPlanner, RunOutcome};
use speet_host_api::default_host_api;
use speet_runtime::Runtime;

pub fn assert_original_matches_recompiled(
    corpus_root: &Path,
    artifact: &Artifact,
    planner: &PathPlanner,
    rt: &mut Runtime,
) -> Result<(), String> {
    let linked = corpus_root.join(&artifact.linked);
    if !linked.is_file() {
        return Err(format!("missing linked guest {}", linked.display()));
    }

    let guest_arch = GuestArch::from_manifest(&artifact.arch)
        .ok_or_else(|| format!("unknown arch {}", artifact.arch))?;
    let guest_os = GuestOs::from_manifest(&artifact.os)
        .ok_or_else(|| format!("unknown os {}", artifact.os))?;

    let (original, path) = planner.run_guest(&linked, guest_arch, guest_os, &[])?;

    let out_arch = guest_arch
        .to_bin_arch()
        .unwrap_or(host_bin_arch()?);
    let out_os = guest_os.to_bin_os();
    let recompiled = run_recompiled(rt, &linked, out_arch, out_os)?;

    compare_outcomes(corpus_root, artifact, &original, &recompiled)?;
    eprintln!(
        "  ✓ equiv {} ({}) via {path}",
        artifact.program, artifact.triple
    );
    Ok(())
}

fn run_recompiled(
    rt: &mut Runtime,
    linked: &Path,
    arch: BinArch,
    os: BinOs,
) -> Result<RunOutcome, String> {
    let status = rt
        .recompile_binary_and_run(linked, arch, os)
        .map_err(|e| e.to_string())?;
    Ok(RunOutcome {
        exit_code: status.code().unwrap_or(-1),
        stdout: Vec::new(),
        stderr: Vec::new(),
    })
}

fn compare_outcomes(
    corpus_root: &Path,
    artifact: &Artifact,
    original: &RunOutcome,
    recompiled: &RunOutcome,
) -> Result<(), String> {
    let expected_path = corpus_root
        .join("programs")
        .join(&artifact.program)
        .join("expected.toml");
    let expectations = load_expected(&expected_path)?;

    let exp = expectation_for_triple(&expectations, &artifact.triple).ok_or_else(|| {
        format!(
            "no [[triple]] name={} in {}",
            artifact.triple,
            expected_path.display()
        )
    })?;

    let expected_code = exp
        .exit_code
        .or(exp.main_return)
        .unwrap_or(original.exit_code);

    if original.exit_code != recompiled.exit_code {
        return Err(format!(
            "exit mismatch for {}: original={} recompiled={}",
            artifact.program, original.exit_code, recompiled.exit_code
        ));
    }

    if original.exit_code != expected_code {
        return Err(format!(
            "original exit {} != expected {} for {}",
            original.exit_code, expected_code, artifact.program
        ));
    }

    if let Some(ref digest) = exp.stdout_sha256 {
        let orig_digest = sha256_hex(&original.stdout);
        if orig_digest != *digest {
            return Err(format!(
                "original stdout digest mismatch for {}",
                artifact.program
            ));
        }
    }

    Ok(())
}

fn host_bin_arch() -> Result<BinArch, String> {
    match std::env::consts::ARCH {
        "x86_64" => Ok(BinArch::X86_64),
        "aarch64" => Ok(BinArch::AArch64),
        other => Err(format!("unsupported host arch for recompiled run: {other}")),
    }
}

pub fn default_runtime() -> Runtime {
    Runtime::new(Arc::new(default_host_api()))
}

pub fn linked_exists(corpus_root: &Path, artifact: &Artifact) -> bool {
    corpus_root.join(&artifact.linked).is_file()
}

pub fn corpus_roots() -> Vec<PathBuf> {
    vec![
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/rv-c-corpus"),
    ]
}
