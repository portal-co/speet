//! [`CodeSnippet`] — the cross-boundary "emit code" carrier.
//!
//! A snippet is a bare WASM instruction-stream byte sequence (no function
//! wrapper, no leading locals vector, no implicit trailing `End`) built with
//! `wasm-encoder`. The host-side adapter decodes it with `wasmparser` and
//! forwards each instruction into the real `dyn InstructionSink<Context, E>`.
//! See `docs/guides/plugin-api.md` §1.

use alloc::vec::Vec;
use wasm_encoder::{Encode, Instruction};

use crate::wire::{WireDecode, WireEncode, WireError};

/// Encoded WASM instruction bytes for one snippet of code.
///
/// Local/param indices referenced inside the snippet are host-relative
/// absolute indices, handed to the plugin as plain `u32` arguments to the
/// same call that produced this snippet — the plugin never needs to know how
/// many total locals exist elsewhere.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodeSnippet {
    pub wasm: Vec<u8>,
}

impl CodeSnippet {
    pub fn empty() -> Self {
        Self { wasm: Vec::new() }
    }

    /// Convenience constructor for in-process (or WASM-guest-written-in-Rust)
    /// plugin authors: encode a slice of `wasm-encoder` instructions into a
    /// snippet directly, without hand-writing bytes.
    pub fn from_instructions(instrs: &[Instruction<'_>]) -> Self {
        let mut wasm = Vec::new();
        for instr in instrs {
            instr.encode(&mut wasm);
        }
        Self { wasm }
    }

    /// Append another snippet's bytes — the composition primitive a plugin
    /// uses to splice an imported entity's snippet into its own (§2.7 of the
    /// plan / "host entity imports" in the guide).
    pub fn extend(&mut self, other: &CodeSnippet) {
        self.wasm.extend_from_slice(&other.wasm);
    }
}

impl WireEncode for CodeSnippet {
    fn encode(&self, out: &mut Vec<u8>) {
        self.wasm.encode(out);
    }
}
impl WireDecode for CodeSnippet {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (wasm, rest) = Vec::<u8>::decode(input)?;
        Ok((CodeSnippet { wasm }, rest))
    }
}

/// The subset of `wasm_encoder::ValType` meaningful across the plugin
/// boundary. Defined fresh here (not re-exported) so plugin authors never
/// need `wasm-encoder` as a dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginValType {
    I32,
    I64,
    F32,
    F64,
}

impl From<PluginValType> for wasm_encoder::ValType {
    fn from(v: PluginValType) -> Self {
        match v {
            PluginValType::I32 => wasm_encoder::ValType::I32,
            PluginValType::I64 => wasm_encoder::ValType::I64,
            PluginValType::F32 => wasm_encoder::ValType::F32,
            PluginValType::F64 => wasm_encoder::ValType::F64,
        }
    }
}

/// Fails for `ValType::Ref(_)` — outside the plugin-representable subset
/// (see this type's own doc comment). Used by the reverse adapters
/// (`speet-plugin-adapter::reverse`) when reading back a wrapped built-in's
/// declared local types. The error is the original `ValType`, for context.
impl TryFrom<wasm_encoder::ValType> for PluginValType {
    type Error = wasm_encoder::ValType;

    fn try_from(v: wasm_encoder::ValType) -> Result<Self, Self::Error> {
        match v {
            wasm_encoder::ValType::I32 => Ok(PluginValType::I32),
            wasm_encoder::ValType::I64 => Ok(PluginValType::I64),
            wasm_encoder::ValType::F32 => Ok(PluginValType::F32),
            wasm_encoder::ValType::F64 => Ok(PluginValType::F64),
            other => Err(other),
        }
    }
}

impl WireEncode for PluginValType {
    fn encode(&self, out: &mut Vec<u8>) {
        let tag: u8 = match self {
            PluginValType::I32 => 0,
            PluginValType::I64 => 1,
            PluginValType::F32 => 2,
            PluginValType::F64 => 3,
        };
        tag.encode(out);
    }
}
impl WireDecode for PluginValType {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        let v = match tag {
            0 => PluginValType::I32,
            1 => PluginValType::I64,
            2 => PluginValType::F32,
            3 => PluginValType::F64,
            _ => return Err(WireError),
        };
        Ok((v, rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_roundtrip() {
        let snip = CodeSnippet::from_instructions(&[
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::I64Add,
        ]);
        let mut out = Vec::new();
        snip.encode(&mut out);
        let (decoded, rest) = CodeSnippet::decode(&out).unwrap();
        assert_eq!(decoded, snip);
        assert!(rest.is_empty());
    }

    #[test]
    fn valtype_roundtrip() {
        for v in [
            PluginValType::I32,
            PluginValType::I64,
            PluginValType::F32,
            PluginValType::F64,
        ] {
            let mut out = Vec::new();
            v.encode(&mut out);
            let (decoded, rest) = PluginValType::decode(&out).unwrap();
            assert_eq!(decoded, v);
            assert!(rest.is_empty());
        }
    }
}
