//! Thin-runtime B-native path wrapper (macOS + LLVM).

use super::capabilities::{soft_skip, PathKind};
use super::env_preview1::RunOutcome;

/// Soft-skip reason for thin-native on this host, if any.
pub fn thin_soft_skip() -> Option<&'static str> {
    soft_skip(PathKind::ThinNative)
}

/// Run RV64 Linux guest text through the thin-runtime pipeline when available.
///
/// Returns `Err` with a skip message when the host cannot run B-native.
#[cfg(target_os = "macos")]
pub fn run_thin_rv64(text: &[u8], start_addr: u64) -> Result<RunOutcome, String> {
    run_thin_rv64_with_escape(text, start_addr, yecta::SpeculativeEscape::JUMP)
}

/// Thin-runtime path with an explicit speculative-call escape policy.
#[cfg(target_os = "macos")]
pub fn run_thin_rv64_with_escape(
    text: &[u8],
    start_addr: u64,
    speculative: yecta::SpeculativeEscape,
) -> Result<RunOutcome, String> {
    use speet_recompile::binary_io::{BinArch, BinOs};
    use speet_runtime::{default_host_api, Runtime};
    use std::sync::Arc;

    if let Some(reason) = thin_soft_skip() {
        return Err(reason.to_string());
    }
    let mut rt = Runtime::new(Arc::new(default_host_api()));
    if !rt.llvm_available() {
        return Err("LLVM not available".into());
    }
    let arch = if cfg!(target_arch = "aarch64") {
        BinArch::AArch64
    } else {
        BinArch::X86_64
    };
    let status = rt
        .recompile_rv64_and_run_with_escape(text, start_addr, arch, BinOs::MacOs, speculative)
        .map_err(|e| e.to_string())?;
    Ok(RunOutcome {
        stdout: Vec::new(),
        exit_code: status.code(),
    })
}

#[cfg(not(target_os = "macos"))]
pub fn run_thin_rv64(_text: &[u8], _start_addr: u64) -> Result<RunOutcome, String> {
    Err(thin_soft_skip()
        .unwrap_or("thin-runtime unavailable")
        .to_string())
}

#[cfg(not(target_os = "macos"))]
pub fn run_thin_rv64_with_escape(
    text: &[u8],
    start_addr: u64,
    _speculative: yecta::SpeculativeEscape,
) -> Result<RunOutcome, String> {
    run_thin_rv64(text, start_addr)
}
