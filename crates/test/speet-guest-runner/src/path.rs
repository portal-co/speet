//! Runner path step types.

use crate::guest::GuestArch;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunnerStep {
    Native,
    NestedVm,
    Blink,
    QemuUser { arch: GuestArch },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunnerPath(pub Vec<RunnerStep>);

impl RunnerPath {
    pub fn new(steps: Vec<RunnerStep>) -> Self {
        Self(steps)
    }

    pub fn first_step(&self) -> Option<&RunnerStep> {
        self.0.first()
    }
}

impl fmt::Display for RunnerPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.0.iter().map(|s| format!("{s:?}")).collect();
        write!(f, "[{}]", parts.join(", "))
    }
}

/// First-step priority (lower = try earlier).
pub fn first_step_rank(step: &RunnerStep) -> u8 {
    match step {
        RunnerStep::Native => 1,
        RunnerStep::NestedVm => 2,
        RunnerStep::Blink => 3,
        RunnerStep::QemuUser { .. } => 4,
    }
}

pub fn path_rank(path: &RunnerPath) -> (u8, usize) {
    (
        path.first_step().map(first_step_rank).unwrap_or(99),
        path.0.len(),
    )
}
