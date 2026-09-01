//! Reloadable recompiler frontend for `speet-rtd`'s `hot-recompiler` feature.
//!
//! Custom `speet_plugin_call` ABI (not `ArchOp`): one `Translate` request
//! returns a complete WASM megabinary plus `unsupported_insns`. In-guest yecta
//! runs the real native frontends via `speet-recompile` without the blitz
//! backend.

use binary_io::BinArch;
use speet_plugin_api::wire::{WireDecode, WireEncode, WireError};
use speet_plugin_guest::{bump_alloc, pack, wasm_alloc};
use speet_recompile::frontend::recompile_to_wasm;

/// `BinArch` wire tag — keep in sync with `binary_io::BinArch` discriminants
/// used by this crate only (not the plugin PROTOCOL_VERSION).
fn arch_to_u8(a: BinArch) -> u8 {
    match a {
        BinArch::X86_64 => 0,
        BinArch::AArch64 => 1,
        BinArch::RiscV64 => 2,
        BinArch::RiscV32 => 3,
        BinArch::Arm => 4,
        BinArch::X86 => 5,
    }
}

fn arch_from_u8(v: u8) -> Result<BinArch, WireError> {
    Ok(match v {
        0 => BinArch::X86_64,
        1 => BinArch::AArch64,
        2 => BinArch::RiscV64,
        3 => BinArch::RiscV32,
        4 => BinArch::Arm,
        5 => BinArch::X86,
        _ => return Err(WireError),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslateRequest {
    pub arch: BinArch,
    pub start_addr: u64,
    pub entry: u64,
    pub text: Vec<u8>,
}

impl WireEncode for TranslateRequest {
    fn encode(&self, out: &mut Vec<u8>) {
        0u8.encode(out); // tag: Translate
        arch_to_u8(self.arch).encode(out);
        self.start_addr.encode(out);
        self.entry.encode(out);
        self.text.encode(out);
    }
}
impl WireDecode for TranslateRequest {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        if tag != 0 {
            return Err(WireError);
        }
        let (arch_tag, rest) = u8::decode(rest)?;
        let arch = arch_from_u8(arch_tag)?;
        let (start_addr, rest) = u64::decode(rest)?;
        let (entry, rest) = u64::decode(rest)?;
        let (text, rest) = Vec::<u8>::decode(rest)?;
        Ok((
            TranslateRequest {
                arch,
                start_addr,
                entry,
                text,
            },
            rest,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslateResponse {
    pub wasm: Vec<u8>,
    pub unsupported: Vec<String>,
    pub error: String,
}

impl WireEncode for TranslateResponse {
    fn encode(&self, out: &mut Vec<u8>) {
        self.wasm.encode(out);
        self.unsupported.encode(out);
        self.error.encode(out);
    }
}
impl WireDecode for TranslateResponse {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (wasm, rest) = Vec::<u8>::decode(input)?;
        let (unsupported, rest) = Vec::<String>::decode(rest)?;
        let (error, rest) = String::decode(rest)?;
        Ok((
            TranslateResponse {
                wasm,
                unsupported,
                error,
            },
            rest,
        ))
    }
}

pub fn handle_translate(req: &TranslateRequest) -> TranslateResponse {
    let _ = req.entry;
    let text = req.text.clone();
    let start_addr = req.start_addr;
    let arch = req.arch;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        recompile_to_wasm(&text, start_addr, arch)
    })) {
        Ok((wasm, unsupported)) => TranslateResponse {
            wasm,
            unsupported,
            error: String::new(),
        },
        Err(_) => TranslateResponse {
            wasm: Vec::new(),
            unsupported: Vec::new(),
            error: String::from("translate panicked"),
        },
    }
}

fn dispatch(req: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    match TranslateRequest::decode(req) {
        Ok((tr, _)) => handle_translate(&tr).encode(&mut out),
        Err(_) => TranslateResponse {
            wasm: Vec::new(),
            unsupported: Vec::new(),
            error: String::from("malformed TranslateRequest"),
        }
        .encode(&mut out),
    }
    out
}

fn call_raw(req_ptr: i32, req_len: i32) -> i64 {
    if req_ptr < 0 || req_len < 0 {
        return pack(0, 0);
    }
    let req = unsafe { core::slice::from_raw_parts(req_ptr as *const u8, req_len as usize) };
    let resp = dispatch(req);
    let ptr = bump_alloc(resp.len());
    if ptr.is_null() {
        return pack(0, 0);
    }
    unsafe {
        core::ptr::copy_nonoverlapping(resp.as_ptr(), ptr, resp.len());
    }
    pack(ptr as u32, resp.len() as u32)
}

#[no_mangle]
pub extern "C" fn speet_plugin_alloc(len: i32) -> i32 {
    wasm_alloc(len)
}

#[no_mangle]
pub extern "C" fn speet_plugin_call(req_ptr: i32, req_len: i32) -> i64 {
    call_raw(req_ptr, req_len)
}

#[no_mangle]
pub extern "C" fn speet_plugin_dealloc(_ptr: i32, _len: i32) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_req_roundtrip() {
        let req = TranslateRequest {
            arch: BinArch::X86_64,
            start_addr: 0x1000,
            entry: 0x1000,
            text: vec![0x90],
        };
        let mut buf = Vec::new();
        req.encode(&mut buf);
        let (decoded, rest) = TranslateRequest::decode(&buf).unwrap();
        assert!(rest.is_empty());
        assert_eq!(decoded, req);
    }
}
