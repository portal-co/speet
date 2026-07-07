//! Step executor trait.

use crate::context::ExecutionContext;
use crate::outcome::RunOutcome;
use crate::path::RunnerStep;
use std::path::Path;
use std::time::Duration;

pub trait StepExecutor: Send + Sync {
    fn step(&self) -> RunnerStep;

    /// Enter this layer from `parent`. Returns child context or incompatibility reason.
    fn enter(&self, parent: &ExecutionContext) -> Result<ExecutionContext, String>;

    /// Run guest binary inside the innermost active context.
    fn run_guest(
        &self,
        ctx: &ExecutionContext,
        guest: &Path,
        stdin: &[u8],
        timeout: Duration,
    ) -> Result<RunOutcome, String>;
}
