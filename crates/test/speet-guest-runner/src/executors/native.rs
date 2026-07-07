//! Native same-arch guest execution.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use binary_io::{BinArch, BinOs};

use crate::context::ExecutionContext;
use crate::executor::StepExecutor;
use crate::host::HostInfo;
use crate::outcome::RunOutcome;
use crate::path::RunnerStep;

pub struct NativeExecutor {
    host: HostInfo,
}

impl NativeExecutor {
    pub fn new(host: HostInfo) -> Self {
        Self { host }
    }
}

impl StepExecutor for NativeExecutor {
    fn step(&self) -> RunnerStep {
        RunnerStep::Native
    }

    fn enter(&self, parent: &ExecutionContext) -> Result<ExecutionContext, String> {
        Ok(parent.child("native"))
    }

    fn run_guest(
        &self,
        ctx: &ExecutionContext,
        guest: &Path,
        stdin: &[u8],
        timeout: Duration,
    ) -> Result<RunOutcome, String> {
        run_subprocess(ctx, guest, stdin, timeout)
    }
}

pub(crate) fn run_subprocess(
    ctx: &ExecutionContext,
    guest: &Path,
    stdin: &[u8],
    timeout: Duration,
) -> Result<RunOutcome, String> {
    let mut cmd = Command::new(guest);
    cmd.current_dir(&ctx.cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in &ctx.env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().map_err(|e| format!("spawn {}: {e}", guest.display()))?;
    if !stdin.is_empty() {
        use std::io::Write;
        if let Some(mut inp) = child.stdin.take() {
            let _ = inp.write_all(stdin);
        }
    }
    let start = std::time::Instant::now();
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            let stdout = read_child_pipe(&mut child.stdout);
            let stderr = read_child_pipe(&mut child.stderr);
            return Ok(RunOutcome::from_status(status, stdout, stderr));
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            return Err(format!("timeout running {}", guest.display()));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn read_child_pipe(pipe: &mut Option<std::process::ChildStdout>) -> Vec<u8> {
    use std::io::Read;
    let mut buf = Vec::new();
    if let Some(p) = pipe {
        let _ = p.read_to_end(&mut buf);
    }
    buf
}

pub(crate) fn guest_os_from_path(guest: &Path) -> Option<BinOs> {
    let ext = guest.extension()?.to_str()?;
    match ext {
        "elf" => Some(BinOs::Linux),
        "macho" => Some(BinOs::MacOs),
        _ => None,
    }
}

pub(crate) fn guest_arch_hint(name: &str) -> Option<BinArch> {
    if name.contains("x86_64") || name.contains("amd64") {
        Some(BinArch::X86_64)
    } else if name.contains("aarch64") || name.contains("arm64") {
        Some(BinArch::AArch64)
    } else if name.contains("riscv64") || name.contains("rv64") {
        Some(BinArch::Riscv64)
    } else {
        None
    }
}
