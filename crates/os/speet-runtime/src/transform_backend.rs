//! `os_transform_core::TransformBackend` adapter for the existing
//! AOT-recompile pipeline, so `speet-rtd` can register
//! [`IntegratedNativeRuntime`] into the generic daemon's backend registry
//! alongside future backends (e.g. the dylib/so rewriter) instead of the
//! daemon hardcoding one transformation strategy.

use crate::integrated::{
    IntegratedNativeRuntime, NativeRuntime, ObtainError as LocalObtainError, SuitabilityReport,
};
use os_transform_core::{BackendId, ObtainError, RunAs, Suitability, TransformBackend};
use std::path::Path;

/// `Suitability.reasons` is a flat `Vec<String>` (see `os-transform-core`),
/// so the two typed dependency buckets are tagged rather than dropped.
fn to_generic_suitability(r: &SuitabilityReport) -> Suitability {
    let reasons = r
        .unresolved_deps
        .iter()
        .map(|d| format!("unresolved-dep:{d}"))
        .chain(r.fn_ptr_deps.iter().map(|d| format!("fn-ptr-dep:{d}")))
        .collect();
    Suitability {
        suitable: r.suitable,
        reasons,
    }
}

impl TransformBackend for IntegratedNativeRuntime {
    fn id(&self) -> BackendId {
        BackendId::INTEGRATED_RECOMPILE
    }

    fn analyze(&self, path: &Path) -> Result<Suitability, String> {
        NativeRuntime::analyze(self, path).map(|r| to_generic_suitability(&r))
    }

    fn obtain(&mut self, path: &Path) -> Result<RunAs, ObtainError> {
        match self.obtain_executable(path) {
            Ok(exe) => Ok(RunAs::Exec(exe)),
            Err(LocalObtainError::Unsuitable(r)) => {
                Err(ObtainError::Unsuitable(to_generic_suitability(&r)))
            }
            Err(LocalObtainError::RecompileFailed(e)) => Err(ObtainError::TransformFailed(e)),
        }
    }
}
