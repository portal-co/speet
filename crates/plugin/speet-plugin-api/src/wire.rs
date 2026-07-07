//! Hand-rolled binary wire codec.
//!
//! Used identically by `speet-plugin-host-wasm` (guest-memory marshalling),
//! `speet-plugin-host-subprocess` (stdio framing), and
//! `speet-plugin-host-inproc`'s dylib mode (direct `extern "C"` calls) — one
//! payload codec, three transports.
//!
//! Deliberately not `serde`/`bincode`/`rkyv`: see `docs/guides/plugin-api.md`.
//! Fields are encoded positionally, in declared order, with no
//! self-description — simple to hand-implement in any language, at the cost
//! of not being forward-compatible field-by-field. Breaking changes bump
//! [`PROTOCOL_VERSION`].

use alloc::string::String;
use alloc::vec::Vec;

/// Wire protocol version. Bump on any field add/remove/reorder or
/// enum-variant change in any type that crosses a plugin boundary.
pub const PROTOCOL_VERSION: u8 = 1;

/// A decode failure: truncated input, invalid UTF-8, or an unrecognized enum
/// discriminant. Carries no payload — wire decode errors are not meant to be
/// diagnosed in detail, only rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireError;

/// Encode `self` by appending bytes to `out`.
pub trait WireEncode {
    fn encode(&self, out: &mut Vec<u8>);
}

/// Decode a `Self` from the front of `input`, returning the value and the
/// unconsumed remainder.
pub trait WireDecode: Sized {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError>;
}

macro_rules! impl_wire_int {
    ($($t:ty),* $(,)?) => {$(
        impl WireEncode for $t {
            fn encode(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
        impl WireDecode for $t {
            fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
                const N: usize = core::mem::size_of::<$t>();
                if input.len() < N {
                    return Err(WireError);
                }
                let (head, tail) = input.split_at(N);
                let mut buf = [0u8; N];
                buf.copy_from_slice(head);
                Ok((<$t>::from_le_bytes(buf), tail))
            }
        }
    )*};
}
impl_wire_int!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl WireEncode for bool {
    fn encode(&self, out: &mut Vec<u8>) {
        out.push(if *self { 1 } else { 0 });
    }
}
impl WireDecode for bool {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (b, rest) = u8::decode(input)?;
        Ok((b != 0, rest))
    }
}

impl WireEncode for [u8] {
    fn encode(&self, out: &mut Vec<u8>) {
        (self.len() as u32).encode(out);
        out.extend_from_slice(self);
    }
}
impl WireEncode for str {
    fn encode(&self, out: &mut Vec<u8>) {
        self.as_bytes().encode(out);
    }
}
impl WireEncode for String {
    fn encode(&self, out: &mut Vec<u8>) {
        self.as_str().encode(out);
    }
}
impl WireDecode for String {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (bytes, rest) = Vec::<u8>::decode(input)?;
        let s = String::from_utf8(bytes).map_err(|_| WireError)?;
        Ok((s, rest))
    }
}

impl<T: WireEncode> WireEncode for Vec<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        (self.len() as u32).encode(out);
        for item in self {
            item.encode(out);
        }
    }
}
impl<T: WireDecode> WireDecode for Vec<T> {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (len, mut rest) = u32::decode(input)?;
        let mut items = Vec::with_capacity(len as usize);
        for _ in 0..len {
            let (item, r) = T::decode(rest)?;
            items.push(item);
            rest = r;
        }
        Ok((items, rest))
    }
}

impl<T: WireEncode> WireEncode for Option<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            None => out.push(0),
            Some(v) => {
                out.push(1);
                v.encode(out);
            }
        }
    }
}
impl<T: WireDecode> WireDecode for Option<T> {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        match tag {
            0 => Ok((None, rest)),
            1 => {
                let (v, rest) = T::decode(rest)?;
                Ok((Some(v), rest))
            }
            _ => Err(WireError),
        }
    }
}

impl<A: WireEncode, B: WireEncode> WireEncode for (A, B) {
    fn encode(&self, out: &mut Vec<u8>) {
        self.0.encode(out);
        self.1.encode(out);
    }
}
impl<A: WireDecode, B: WireDecode> WireDecode for (A, B) {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (a, rest) = A::decode(input)?;
        let (b, rest) = B::decode(rest)?;
        Ok(((a, b), rest))
    }
}

/// A single host↔plugin request or response frame, as used by WASM
/// guest-memory marshalling and by subprocess stdio (wrapped there with an
/// extra direction tag + request id for the host-entity-import case — see
/// `speet-plugin-host-subprocess`).
///
/// Wire shape: `[1 byte version][1 byte method_tag][4 bytes LE len][payload]`.
pub struct Envelope<'a> {
    pub version: u8,
    pub method_tag: u8,
    pub payload: &'a [u8],
}

impl<'a> Envelope<'a> {
    pub fn write(method_tag: u8, payload: &[u8], out: &mut Vec<u8>) {
        PROTOCOL_VERSION.encode(out);
        method_tag.encode(out);
        payload.encode(out);
    }

    pub fn parse(input: &'a [u8]) -> Result<(Envelope<'a>, &'a [u8]), WireError> {
        let (version, rest) = u8::decode(input)?;
        let (method_tag, rest) = u8::decode(rest)?;
        let (len, rest) = u32::decode(rest)?;
        let len = len as usize;
        if rest.len() < len {
            return Err(WireError);
        }
        let (payload, tail) = rest.split_at(len);
        Ok((
            Envelope {
                version,
                method_tag,
                payload,
            },
            tail,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn roundtrip_primitives() {
        let mut out = Vec::new();
        42u32.encode(&mut out);
        (-7i64).encode(&mut out);
        true.encode(&mut out);
        let (a, rest) = u32::decode(&out).unwrap();
        assert_eq!(a, 42);
        let (b, rest) = i64::decode(rest).unwrap();
        assert_eq!(b, -7);
        let (c, rest) = bool::decode(rest).unwrap();
        assert!(c);
        assert!(rest.is_empty());
    }

    #[test]
    fn roundtrip_collections() {
        let mut out = Vec::new();
        let v: Vec<u32> = vec![1, 2, 3];
        v.encode(&mut out);
        let s = String::from("hello");
        s.encode(&mut out);
        let o: Option<u8> = None;
        o.encode(&mut out);

        let (v2, rest) = Vec::<u32>::decode(&out).unwrap();
        assert_eq!(v2, v);
        let (s2, rest) = String::decode(rest).unwrap();
        assert_eq!(s2, s);
        let (o2, rest) = Option::<u8>::decode(rest).unwrap();
        assert_eq!(o2, None);
        assert!(rest.is_empty());
    }

    #[test]
    fn truncated_input_errs() {
        let buf = [1u8, 2u8];
        assert!(u32::decode(&buf).is_err());
    }

    #[test]
    fn envelope_roundtrip() {
        let mut out = Vec::new();
        Envelope::write(5, b"payload", &mut out);
        let (env, rest) = Envelope::parse(&out).unwrap();
        assert_eq!(env.version, PROTOCOL_VERSION);
        assert_eq!(env.method_tag, 5);
        assert_eq!(env.payload, b"payload");
        assert!(rest.is_empty());
    }
}
