//! Supported-cell filter for the combinatorial e2e matrix.

use super::config::EscapeConfig;
use super::Arch;

/// Execution / compile path in the dual-lane matrix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathKind {
    /// Backend A: wasmi (wasmtime fallback for exception opcodes).
    Wasmi,
    /// Backend B-wasm: wasm-blitz compile-check.
    Blitz,
    /// Backend B-native: thin runtime link+spawn (macOS + LLVM).
    ThinNative,
    /// Lane A: linux-wasi → wasmi preview1.
    LinuxWasi,
    /// Lane A: darwin-wasi → wasmi preview1.
    DarwinWasi,
}

/// Fixture capability tags used by the generator and soft-skip logic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FixtureClass {
    /// Generic e2e corpus (hint/write/exit native syscall imports).
    GenericE2e,
    /// Linux RV64 ecall guest for WASI / thin-runtime.
    LinuxEcall,
    /// aarch64 Darwin svc / GOT guest for darwin-wasi.
    DarwinSvc,
    /// Managed WASM frontend fixture.
    WasmFixture,
}

pub fn arch_supports_speculative(arch: Arch) -> bool {
    matches!(arch, Arch::Rv32 | Arch::Rv64 | Arch::X86_64)
}

/// Whether `(arch, path, config)` is a legal matrix cell.
pub fn cell_supported(arch: Arch, path: PathKind, config: EscapeConfig, class: FixtureClass) -> bool {
    if config.speculative() && !arch_supports_speculative(arch) {
        return false;
    }
    match (path, class) {
        (PathKind::Wasmi | PathKind::Blitz, FixtureClass::GenericE2e | FixtureClass::WasmFixture) => {
            true
        }
        (PathKind::Wasmi | PathKind::Blitz, FixtureClass::LinuxEcall | FixtureClass::DarwinSvc) => {
            false
        }
        (PathKind::LinuxWasi, FixtureClass::LinuxEcall) => {
            // Exception TagSection not yet declared in WASI megabinary assemble;
            // FlagSpec + Jump are fully wired.
            matches!(config, EscapeConfig::None | EscapeConfig::FlagSpec)
        }
        (PathKind::LinuxWasi, _) => false,
        (PathKind::DarwinWasi, FixtureClass::DarwinSvc) => {
            // No aarch64 speculative yet — Jump only.
            matches!(config, EscapeConfig::None)
        }
        (PathKind::DarwinWasi, _) => false,
        (PathKind::ThinNative, FixtureClass::LinuxEcall) => {
            // Thin-runtime assemble supports Jump + Flag; ExceptionSpec needs tags.
            matches!(arch, Arch::Rv64)
                && matches!(config, EscapeConfig::None | EscapeConfig::FlagSpec)
        }
        (PathKind::ThinNative, _) => false,
    }
}

pub fn soft_skip(path: PathKind) -> Option<&'static str> {
    match path {
        PathKind::ThinNative => {
            if cfg!(not(target_os = "macos")) {
                Some("thin-runtime B-native requires macOS")
            } else {
                None
            }
        }
        _ => None,
    }
}
