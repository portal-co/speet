//! Step executor implementations.

pub mod blink;
pub mod native;
pub mod nested_vm;
pub mod qemu;

use std::sync::Arc;

use crate::executor::StepExecutor;
use crate::host::HostInfo;
use crate::install::EmulatorInstaller;

pub fn default_executors(host: &HostInfo, installer: Arc<EmulatorInstaller>) -> Vec<Box<dyn StepExecutor>> {
    vec![
        Box::new(native::NativeExecutor::new(host.clone())),
        Box::new(nested_vm::NestedVmExecutor::new(host.clone(), installer.clone())),
        Box::new(blink::BlinkExecutor::new(installer.clone())),
        Box::new(qemu::QemuUserExecutor::new(installer)),
    ]
}
