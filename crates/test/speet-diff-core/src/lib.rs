//! Comparison (differential) fuzzing core: random x86-64 cases executed under
//! the Unicorn oracle and under speet's recompiled output (wasmi Lane A),
//! with exact architectural-state comparison.
//!
//! Scope rules (docs/comparison-fuzzing-plan.md §1):
//! - RX memory model (code region executable + readable, data/stack RW).
//! - Unsupported instructions are skipped **at execution only** — translation
//!   never gates on `unsupported_insns` (frontends overtranslate); an
//!   executed-unsupported instruction surfaces as `unreachable`, which counts
//!   as a skip, never a failure.
//! - Traps are skipped.
//! - Stores to read-only pages fault/trap → skip (for now).

pub mod branchfix;
pub mod case;
pub mod compare;
pub mod generator;
pub mod gen_32;
pub mod gen_a64;
pub mod gen_rv64;
pub mod minimize;
pub mod report;
pub mod recompiled;
#[cfg(feature = "oracle")]
pub mod oracle;

pub use case::{ExecOutcome, ExitKind, FuzzCase, RegState};
pub use compare::{compare_outcomes, Comparison};
pub use generator::{generate_case, SeedRng};
pub use minimize::{minimize, with_code_public};
pub use report::SkipReason;
pub use report::{record_divergence, record_skip};
#[cfg(feature = "oracle")]
pub use oracle::run_oracle;
pub use recompiled::{build_case_module, run_recompiled, unsupported_for, RecompiledError};

/// Parse an `--arch`/argv arch name (`x86_64`, `aarch64`, `riscv64`,
/// `arm`, `x86_32`, `riscv32`, `mips`) — default x86_64.
pub fn parse_arch_name(s: Option<&str>) -> crate::case::Arch {
    match s {
        Some("aarch64") => crate::case::Arch::AArch64,
        Some("riscv64") => crate::case::Arch::RiscV64,
        Some("arm") => crate::case::Arch::Arm,
        Some("x86_32") => crate::case::Arch::X86_32,
        Some("riscv32") => crate::case::Arch::RiscV32,
        Some("mips") => crate::case::Arch::Mips,
        _ => crate::case::Arch::X86_64,
    }
}

/// Debug helper: report the recompiled error for one seed (diagnostics only).
pub fn debug_seed(seed: u64) -> String {
    let case = generator::generate_case(case::Arch::X86_64, seed);
    let o = crate::oracle::run_oracle(&case);
    let r = recompiled::run_recompiled(&case);
    format!(
        "seed={seed} code_len={} insns~{} oracle={:?} recompiled={:?}",
        case.code.len(),
        case.code.len() / 4,
        o.map(|x| x.exit).map_err(|e| e),
        r.map(|x| x.exit).map_err(|e| format!("{e:?}"))
    )
}

/// Testing/example support: build the recompiled module for a raw code blob
/// (exposed so examples/tests can inspect the WASM).
pub mod tests_common {
    use crate::case::{FuzzCase, RegState};
    pub fn make_case(code: Vec<u8>, seed: u64) -> FuzzCase {
        make_case_arch(crate::case::Arch::X86_64, code, seed)
    }
    pub fn make_case_arch(arch: crate::case::Arch, code: Vec<u8>, seed: u64) -> FuzzCase {
        let mut regs = RegState::default();
        regs.rip = crate::generator::CODE_BASE;
        match arch {
            crate::case::Arch::X86_64 => {
                regs.gprs[3] = crate::generator::DATA_BASE;
                regs.gprs[7] = 0x1234;
                regs.gprs[15] = 0xAAAA;
            }
            crate::case::Arch::AArch64 => {
                regs.gprs[20] = crate::generator::DATA_BASE; // X20 anchor
                regs.gprs[7] = 0x1234;
                regs.gprs[15] = 0xAAAA;
            }
            crate::case::Arch::RiscV64 => {
                regs.gprs[5] = crate::generator::DATA_BASE; // x5/t0 anchor
                regs.gprs[7] = 0x1234;
                regs.gprs[15] = 0xAAAA;
            }
            _ => {}
        }
        regs.gprs[3] = crate::generator::DATA_BASE;
        let sp = crate::generator::STACK_BASE
            + ((crate::generator::STACK_SIZE as u64) & !0xF)
            - 8;
        regs.gprs[4] = sp;
        regs.gprs[7] = 0x1234;
        regs.gprs[15] = 0xAAAA;
        regs.rip = crate::generator::CODE_BASE;
        let data = vec![0u8; crate::generator::DATA_SIZE];
        let mut stack = vec![0u8; crate::generator::STACK_SIZE];
        if arch.stack_based_ret() {
            let off = (sp - crate::generator::STACK_BASE) as usize;
            stack[off..off + 8].copy_from_slice(
                &((crate::generator::CODE_BASE + code.len() as u64).to_le_bytes()),
            );
        }
        FuzzCase {
            arch,
            code,
            entry_pc: crate::generator::CODE_BASE,
            regs,
            data,
            data_base: crate::generator::DATA_BASE,
            stack,
            stack_base: crate::generator::STACK_BASE,
            read_only: vec![(crate::generator::RO_DATA_OFFSET, crate::generator::RO_DATA_SIZE)],
            step_budget: 4000,
            seed,
        }
    }
}
