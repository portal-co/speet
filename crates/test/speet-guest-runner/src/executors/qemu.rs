//! QEMU user-mode guest execution.

use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use crate::context::ExecutionContext;
use crate::executor::StepExecutor;
use crate::guest::GuestArch;
use crate::install::EmulatorInstaller;
use crate::outcome::RunOutcome;
use crate::path::RunnerStep;

pub struct QemuUserExecutor {
    installer: Arc<EmulatorInstaller>,
}

impl QemuUserExecutor {
    pub fn new(installer: Arc<EmulatorInstaller>) -> Self {
        Self { installer }
    }
}

impl StepExecutor for QemuUserExecutor {
    fn step(&self) -> RunnerStep {
        RunnerStep::QemuUser {
            arch: GuestArch::Riscv64,
        }
    }

    fn enter(&self, _parent: &ExecutionContext) -> Result<ExecutionContext, String> {
        Err("QemuUser enter requires arch — use planner composite enter".into())
    }

    fn run_guest(
        &self,
        ctx: &ExecutionContext,
        guest: &Path,
        stdin: &[u8],
        timeout: Duration,
    ) -> Result<RunOutcome, String> {
        let arch = ctx
            .env
            .get("SPEET_QEMU_ARCH")
            .and_then(|s| parse_arch(s))
            .ok_or_else(|| "SPEET_QEMU_ARCH not set in context".to_string())?;
        let qemu = self.installer.ensure_qemu_user(arch)?;
        run_qemu(&qemu, guest, stdin, timeout)
    }
}

pub(crate) fn enter_qemu(
    installer: &EmulatorInstaller,
    parent: &ExecutionContext,
    arch: GuestArch,
) -> Result<ExecutionContext, String> {
    let _qemu = installer.ensure_qemu_user(arch)?;
    let mut ctx = parent.child(&format!("qemu-{arch:?}"));
    ctx.env
        .insert("SPEET_QEMU_ARCH".to_string(), format!("{arch:?}"));
    installer.store.prepend_to_path(&mut ctx.env);
    Ok(ctx)
}

fn parse_arch(s: &str) -> Option<GuestArch> {
    match s {
        "X86_64" => Some(GuestArch::X86_64),
        "AArch64" => Some(GuestArch::AArch64),
        "Riscv64" => Some(GuestArch::Riscv64),
        "Riscv32" => Some(GuestArch::Riscv32),
        _ => None,
    }
}

fn run_qemu(
    qemu: &std::path::Path,
    guest: &Path,
    stdin: &[u8],
    timeout: Duration,
) -> Result<RunOutcome, String> {
    let mut cmd = Command::new(qemu);
    cmd.arg(guest)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("qemu spawn {}: {e}", guest.display()))?;
    if !stdin.is_empty() {
        use std::io::Write;
        if let Some(mut inp) = child.stdin.take() {
            let _ = inp.write_all(stdin);
        }
    }
    let start = std::time::Instant::now();
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            use std::io::Read;
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            if let Some(mut o) = child.stdout.take() {
                let _ = o.read_to_end(&mut stdout);
            }
            if let Some(mut e) = child.stderr.take() {
                let _ = e.read_to_end(&mut stderr);
            }
            return Ok(RunOutcome::from_status(status, stdout, stderr));
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            return Err(format!("qemu timeout: {}", guest.display()));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
