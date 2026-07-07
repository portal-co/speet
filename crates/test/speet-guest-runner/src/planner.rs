//! Path enumeration and attempt orchestration.

use std::path::Path;
use std::sync::Arc;

use binary_io::{BinArch, BinOs};

use crate::context::ExecutionContext;
use crate::executor::StepExecutor;
use crate::executors::qemu::enter_qemu;
use crate::guest::{GuestArch, GuestOs};
use crate::host::HostInfo;
use crate::install::EmulatorInstaller;
use crate::outcome::RunOutcome;
use crate::path::{path_rank, RunnerPath, RunnerStep};

pub struct PathPlanner {
    executors: Vec<Box<dyn StepExecutor>>,
    host: HostInfo,
    installer: Arc<EmulatorInstaller>,
}

impl PathPlanner {
    pub fn new(
        host: HostInfo,
        executors: Vec<Box<dyn StepExecutor>>,
        installer: Arc<EmulatorInstaller>,
    ) -> Self {
        Self {
            executors,
            host,
            installer,
        }
    }

    pub fn from_env() -> Self {
        let host = HostInfo::detect();
        let installer = Arc::new(EmulatorInstaller::from_env());
        let executors = crate::executors::default_executors(&host, installer.clone());
        Self::new(host, executors, installer)
    }

    pub fn host(&self) -> &HostInfo {
        &self.host
    }

    pub fn installer(&self) -> &EmulatorInstaller {
        &self.installer
    }

    pub fn candidate_paths(&self, guest_arch: GuestArch, guest_os: GuestOs) -> Vec<RunnerPath> {
        let mut paths = Vec::new();
        let host_bin_arch = Some(self.host.arch);
        let guest_bin = guest_arch.to_bin_arch();

        if guest_bin == host_bin_arch && guest_os.to_bin_os() == self.host.os {
            paths.push(RunnerPath::new(vec![RunnerStep::Native]));
        }

        if guest_os == GuestOs::Linux {
            paths.push(RunnerPath::new(vec![
                RunnerStep::NestedVm,
                RunnerStep::QemuUser { arch: guest_arch },
            ]));
            if guest_arch == GuestArch::X86_64 {
                paths.push(RunnerPath::new(vec![RunnerStep::Blink]));
            }
            paths.push(RunnerPath::new(vec![RunnerStep::QemuUser { arch: guest_arch }]));
            if guest_arch != GuestArch::X86_64 {
                paths.push(RunnerPath::new(vec![
                    RunnerStep::Blink,
                    RunnerStep::QemuUser { arch: guest_arch },
                ]));
            }
        }

        if guest_os == GuestOs::MacOs
            && guest_bin == host_bin_arch
            && self.host.os == BinOs::MacOs
        {
            // Native Mach-O covered above when arch matches.
        }

        paths.sort_by_key(path_rank);
        paths.dedup();
        paths
    }

    pub fn ensure_toolchain(&self, path: &RunnerPath) -> Result<(), String> {
        self.installer.ensure_for_path(path)
    }

    pub fn run_guest(
        &self,
        guest: &Path,
        guest_arch: GuestArch,
        guest_os: GuestOs,
        stdin: &[u8],
    ) -> Result<(RunOutcome, RunnerPath), String> {
        let paths = self.candidate_paths(guest_arch, guest_os);
        let mut attempts = Vec::new();

        for path in paths {
            if let Err(e) = self.ensure_toolchain(&path) {
                attempts.push(format!("{path}: ensure_toolchain: {e}"));
                continue;
            }
            match self.try_path(&path, guest, stdin) {
                Ok(outcome) => {
                    eprintln!("original runner path: {path}");
                    return Ok((outcome, path));
                }
                Err(e) => attempts.push(format!("{path}: {e}")),
            }
        }

        Err(format!(
            "all runner paths exhausted for {} (arch={guest_arch:?} os={guest_os:?}):\n{}",
            guest.display(),
            attempts.join("\n")
        ))
    }

    fn try_path(&self, path: &RunnerPath, guest: &Path, stdin: &[u8]) -> Result<RunOutcome, String> {
        let mut ctx = ExecutionContext::host_root();
        self.installer.store.prepend_to_path(&mut ctx.env);

        let steps = &path.0;
        if steps.is_empty() {
            return Err("empty path".into());
        }

        for (i, step) in steps.iter().enumerate() {
            let is_last = i + 1 == steps.len();
            ctx = match step {
                RunnerStep::Native => self.enter_step(RunnerStep::Native, &ctx)?,
                RunnerStep::NestedVm => self.enter_step(RunnerStep::NestedVm, &ctx)?,
                RunnerStep::Blink => self.enter_step(RunnerStep::Blink, &ctx)?,
                RunnerStep::QemuUser { arch } => enter_qemu(&self.installer, &ctx, *arch)?,
            };
            if is_last {
                return self.run_terminal(step, &ctx, guest, stdin);
            }
        }
        Err("unreachable".into())
    }

    fn enter_step(
        &self,
        want: RunnerStep,
        parent: &ExecutionContext,
    ) -> Result<ExecutionContext, String> {
        for ex in &self.executors {
            if ex.step() == want
                || matches!(
                    (&want, ex.step()),
                    (RunnerStep::QemuUser { .. }, RunnerStep::QemuUser { .. })
                )
            {
                if let RunnerStep::QemuUser { arch } = want {
                    return enter_qemu(&self.installer, parent, arch);
                }
                return ex.enter(parent);
            }
        }
        Err(format!("no executor for {want:?}"))
    }

    fn run_terminal(
        &self,
        step: &RunnerStep,
        ctx: &ExecutionContext,
        guest: &Path,
        stdin: &[u8],
    ) -> Result<RunOutcome, String> {
        let timeout = ctx.timeout;
        match step {
            RunnerStep::Native => self
                .find_executor(RunnerStep::Native)?
                .run_guest(ctx, guest, stdin, timeout),
            RunnerStep::Blink => self
                .find_executor(RunnerStep::Blink)?
                .run_guest(ctx, guest, stdin, timeout),
            RunnerStep::QemuUser { .. } => self
                .find_executor(RunnerStep::QemuUser {
                    arch: GuestArch::Riscv64,
                })?
                .run_guest(ctx, guest, stdin, timeout),
            RunnerStep::NestedVm => Err("NestedVm is not a terminal step".into()),
        }
    }

    fn find_executor(&self, step: RunnerStep) -> Result<&dyn StepExecutor, String> {
        self.executors
            .iter()
            .find(|e| {
                e.step() == step
                    || matches!(
                        (&step, e.step()),
                        (RunnerStep::QemuUser { .. }, RunnerStep::QemuUser { .. })
                    )
            })
            .map(|b| b.as_ref())
            .ok_or_else(|| format!("missing executor for {step:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_path_ranked_first_on_linux_x86_host() {
        let host = HostInfo {
            arch: BinArch::X86_64,
            os: BinOs::Linux,
            is_vm: false,
            nested_vm_available: true,
        };
        let installer = Arc::new(EmulatorInstaller::from_env());
        let planner = PathPlanner::new(host, vec![], installer);
        let paths = planner.candidate_paths(GuestArch::X86_64, GuestOs::Linux);
        assert_eq!(paths.first().unwrap().0[0], RunnerStep::Native);
    }

    #[test]
    fn rv64_linux_includes_blink_qemu_on_vm_host() {
        let host = HostInfo {
            arch: BinArch::AArch64,
            os: BinOs::MacOs,
            is_vm: true,
            nested_vm_available: false,
        };
        let installer = Arc::new(EmulatorInstaller::from_env());
        let planner = PathPlanner::new(host, vec![], installer);
        let paths = planner.candidate_paths(GuestArch::Riscv64, GuestOs::Linux);
        assert!(paths.iter().any(|p| {
            p.0
                == vec![
                    RunnerStep::Blink,
                    RunnerStep::QemuUser {
                        arch: GuestArch::Riscv64
                    }
                ]
        }));
    }
}
