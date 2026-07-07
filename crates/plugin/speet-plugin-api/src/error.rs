//! [`PluginError`] — the one fixed error representation every plugin trait
//! uses, since a plugin cannot share the host's own `E` type parameter.

use alloc::string::String;
use alloc::vec::Vec;

use crate::wire::{WireDecode, WireEncode, WireError};

/// A plugin-reported failure. `code` is plugin-defined (`0` reserved for
/// "unspecified"); `message` is for logs/diagnostics only, never parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginError {
    pub code: u32,
    pub message: String,
}

impl PluginError {
    pub fn new(code: u32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// A method this plugin deliberately does not implement (e.g. the
    /// optional `emit_memory_size`/`emit_memory_grow` methods).
    pub fn unsupported(method: &str) -> Self {
        Self::new(0, alloc::format!("unsupported: {method}"))
    }

    /// A plugin-returned [`CodeSnippet`](crate::snippet::CodeSnippet) failed
    /// to decode as a well-formed WASM instruction stream — always possible
    /// for the WASM/subprocess hosts, where the bytes cross an untrusted
    /// boundary. Used by `speet-plugin-adapter`'s `replay_snippet`.
    pub fn decode_failure(message: impl Into<String>) -> Self {
        Self::new(0, message)
    }
}

impl core::fmt::Display for PluginError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "plugin error {}: {}", self.code, self.message)
    }
}

pub type PResult<T> = Result<T, PluginError>;

impl WireEncode for PluginError {
    fn encode(&self, out: &mut Vec<u8>) {
        self.code.encode(out);
        self.message.encode(out);
    }
}
impl WireDecode for PluginError {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (code, rest) = u32::decode(input)?;
        let (message, rest) = String::decode(rest)?;
        Ok((PluginError { code, message }, rest))
    }
}

impl<T: WireEncode> WireEncode for PResult<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Ok(v) => {
                out.push(0);
                v.encode(out);
            }
            Err(e) => {
                out.push(1);
                e.encode(out);
            }
        }
    }
}
impl<T: WireDecode> WireDecode for PResult<T> {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        match tag {
            0 => {
                let (v, rest) = T::decode(rest)?;
                Ok((Ok(v), rest))
            }
            1 => {
                let (e, rest) = PluginError::decode(rest)?;
                Ok((Err(e), rest))
            }
            _ => Err(WireError),
        }
    }
}
