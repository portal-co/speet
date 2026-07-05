//! Guest architecture vocabulary (includes RISC-V not yet in binary-io).

use binary_io::{BinArch, BinOs};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuestArch {
    X86_64,
    AArch64,
    Riscv32,
    Riscv64,
}

impl GuestArch {
    pub fn from_manifest(s: &str) -> Option<Self> {
        match s {
            "x86_64" => Some(Self::X86_64),
            "aarch64" => Some(Self::AArch64),
            "rv32" | "riscv32" => Some(Self::Riscv32),
            "rv64" | "riscv64" => Some(Self::Riscv64),
            _ => None,
        }
    }

    pub fn to_bin_arch(self) -> Option<BinArch> {
        match self {
            Self::X86_64 => Some(BinArch::X86_64),
            Self::AArch64 => Some(BinArch::AArch64),
            Self::Riscv32 | Self::Riscv64 => None,
        }
    }

    pub fn qemu_arch(self) -> Self {
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuestOs {
    Linux,
    MacOs,
}

impl GuestOs {
    pub fn from_manifest(s: &str) -> Option<Self> {
        match s {
            "linux" => Some(Self::Linux),
            "macos" => Some(Self::MacOs),
            _ => None,
        }
    }

    pub fn to_bin_os(self) -> BinOs {
        match self {
            Self::Linux => BinOs::Linux,
            Self::MacOs => BinOs::MacOs,
        }
    }
}
