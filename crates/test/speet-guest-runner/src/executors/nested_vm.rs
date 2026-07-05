//! Hardware-assisted nested VM substrate (probe only).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::context::ExecutionContext;
use crate::executor::StepExecutor;
use crate::host::HostInfo;
use crate::install::EmulatorInstaller;
use crate::outcome::RunOutcome;
use crate::path::RunnerStep;

pub struct NestedVmExecutor {
    host: HostInfo,
    #[allow(dead_code)]
    installer: Arc<EmulatorInstaller>,
}

impl NestedVmExecutor {
    pub fn new(host: HostInfo, installer: Arc<EmulatorInstaller>) -> Self {
        Self { host, installer }
    }
}

impl StepExecutor for NestedVmExecutor {
    fn step(&self) -> RunnerStep {
        RunnerStep::NestedVm
    }

    fn enter(&self, parent: &ExecutionContext) -> Result<ExecutionContext, String> {
        if !self.host.nested_vm_available {
            return Err("nested VM not available on this host".into());
        }
        Ok(parent.child("nested-vm"))
    }

    fn run_guest(
        &self,
        _ctx: &ExecutionContext,
        guest: &Path,
        _stdin: &[u8],
        _timeout: Duration,
    ) -> Result<RunOutcome, String> {
        Err(format!(
            "NestedVmExecutor cannot run terminal guest directly: {}",
            guest.display()
        ))
    }
}
