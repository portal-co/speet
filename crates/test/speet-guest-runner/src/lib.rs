//! Nestable runner paths for executing original C corpus guests.

pub mod context;
pub mod executor;
pub mod executors;
pub mod guest;
pub mod host;
pub mod install;
pub mod outcome;
pub mod path;
pub mod planner;

pub use context::ExecutionContext;
pub use executor::StepExecutor;
pub use guest::{GuestArch, GuestOs};
pub use host::HostInfo;
pub use install::{EmulatorInstaller, EmulatorStore, InstallPolicy};
pub use outcome::RunOutcome;
pub use path::{first_step_rank, RunnerPath, RunnerStep};
pub use planner::PathPlanner;
