//! Blink userspace Linux x86_64 substrate.

use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use crate::context::ExecutionContext;
use crate::executor::StepExecutor;
use crate::install::EmulatorInstaller;
use crate::outcome::RunOutcome;
use crate::path::RunnerStep;

pub struct BlinkExecutor {
    installer: Arc<EmulatorInstaller>,
}

impl BlinkExecutor {
    pub fn new(installer: Arc<EmulatorInstaller>) -> Self {
        Self { installer }
    }
}

impl StepExecutor for BlinkExecutor {
    fn step(&self) -> RunnerStep {
        RunnerStep::Blink
    }

    fn enter(&self, parent: &ExecutionContext) -> Result<ExecutionContext, String> {
        let blink = self.installer.ensure_blink()?;
        let mut ctx = parent.child("blink");
        ctx.env
            .insert("BLINK_BIN".to_string(), blink.display().to_string());
        self.installer.store.prepend_to_path(&mut ctx.env);
        Ok(ctx)
    }

    fn run_guest(
        &self,
        ctx: &ExecutionContext,
        guest: &Path,
        stdin: &[u8],
        timeout: Duration,
    ) -> Result<RunOutcome, String> {
        let blink = ctx
            .env
            .get("BLINK_BIN")
            .ok_or_else(|| "blink context missing BLINK_BIN".to_string())?;
        run_blink(blink, guest, stdin, timeout)
    }
}

fn run_blink(
    blink: &str,
    guest: &Path,
    stdin: &[u8],
    timeout: Duration,
) -> Result<RunOutcome, String> {
    let mut cmd = Command::new(blink);
    cmd.arg(guest)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("blink spawn {}: {e}", guest.display()))?;
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
            return Err(format!("blink timeout: {}", guest.display()));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
