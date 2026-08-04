//! Recompiler escape / speculative-call configuration for e2e tests.

use yecta::{CallEscape, EscapeTag, TagIdx, TypeIdx};

/// Canonical recompiler config axis for the combinatorial matrix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EscapeConfig {
    /// `return_call` only — no native-stack speculative calls, no tags.
    None,
    /// TagSection present; speculative calls off.
    Exception,
    /// Native-stack calls + exception-tag escape on RA mismatch.
    ExceptionSpec,
    /// Native-stack calls + trailing i32 flag escape (no TagSection).
    FlagSpec,
}

impl EscapeConfig {
    pub fn needs_exception_tags(self) -> bool {
        matches!(self, Self::Exception | Self::ExceptionSpec)
    }

    pub fn speculative(self) -> bool {
        matches!(self, Self::ExceptionSpec | Self::FlagSpec)
    }

    /// Function results include a trailing `i32` escape flag.
    pub fn needs_flag_result(self) -> bool {
        matches!(self, Self::FlagSpec)
    }

    /// True when the module uses WASM exception instructions (`try_table`/`throw`).
    pub fn uses_exception_opcodes(self) -> bool {
        matches!(self, Self::ExceptionSpec)
    }

    /// Map to yecta's [`CallEscape`] for a register-file type index.
    pub fn call_escape(self, type_idx: TypeIdx) -> CallEscape {
        match self {
            Self::None => CallEscape::Jump,
            Self::Exception | Self::ExceptionSpec => CallEscape::Exception(EscapeTag {
                tag: TagIdx(type_idx.0),
                ty: type_idx,
            }),
            Self::FlagSpec => CallEscape::Flag,
        }
    }
}

/// Legacy EH × speculative pair used by older macros; prefer [`EscapeConfig`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Eh {
    None,
    With,
}

impl EscapeConfig {
    pub fn from_eh_spec(eh: Eh, speculative: bool) -> Self {
        match (eh, speculative) {
            (Eh::None, _) => Self::None,
            (Eh::With, false) => Self::Exception,
            (Eh::With, true) => Self::ExceptionSpec,
        }
    }

    /// Map to yecta's [`yecta::SpeculativeEscape`] for OS / thin-runtime frontends.
    ///
    /// Exception modes use a placeholder tag (`TagIdx(0)` / `TypeIdx(0)`). The
    /// harness assemble path for wasmi/blitz declares real tags; linux-wasi /
    /// thin-native cells that need exception opcodes are filtered out of the
    /// matrix until those assemblers declare TagSection entries.
    pub fn speculative_escape(self) -> yecta::SpeculativeEscape {
        use yecta::{CallEscape, EscapeTag, SpeculativeEscape, TagIdx, TypeIdx};
        match self {
            Self::None => SpeculativeEscape::JUMP,
            Self::Exception => SpeculativeEscape {
                escape: CallEscape::Exception(EscapeTag {
                    tag: TagIdx(0),
                    ty: TypeIdx(0),
                }),
                enable: false,
            },
            Self::ExceptionSpec => SpeculativeEscape::exception_spec(EscapeTag {
                tag: TagIdx(0),
                ty: TypeIdx(0),
            }),
            Self::FlagSpec => SpeculativeEscape::FLAG_SPEC,
        }
    }
}
