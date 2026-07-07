//! The bidirectional frame protocol spoken over a subprocess plugin's
//! `stdin`/`stdout` pipes. `stderr` is left for the plugin's own logs and is
//! never parsed as protocol data. See `docs/guides/plugin-api.md` §5.3.
//!
//! Wire shape: `[1 byte version][1 byte kind][4 bytes LE request_id][4 bytes
//! LE payload_len][payload]`. The payload for [`FrameKind::MainRequest`] and
//! [`FrameKind::ImportResponse`] is a `speet_plugin_api::remote::XResponse`/
//! `XRequest` encoding (self-describing — no separate method tag needed, the
//! request's own leading byte carries it). [`FrameKind::ImportRequest`]'s
//! payload is `[role: u8][name: wire-encoded String][request bytes]` — see
//! `process` module docs.
//!
//! Every call is strictly sequential (the host issues one `MainRequest` and
//! does not issue another until it has seen that request's `MainResponse`,
//! interleaved with at most one outstanding `ImportRequest`/`ImportResponse`
//! pair at a time), so `request_id` is not load-bearing for routing in this
//! MVP — `kind` alone disambiguates. It is still threaded through and
//! echoed back, both for diagnostics and so a future concurrent extension
//! (pipelining multiple outstanding `MainRequest`s) doesn't need a wire
//! format change.

use std::io::{self, Read, Write};

pub const PROTOCOL_VERSION: u8 = 1;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// host → plugin: "run this call".
    MainRequest = 0,
    /// plugin → host: the result of the most recent `MainRequest`.
    MainResponse = 1,
    /// plugin → host: "resolve this host-entity import for me" (§2.7).
    ImportRequest = 2,
    /// host → plugin: the result of an `ImportRequest`.
    ImportResponse = 3,
}

impl FrameKind {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(FrameKind::MainRequest),
            1 => Some(FrameKind::MainResponse),
            2 => Some(FrameKind::ImportRequest),
            3 => Some(FrameKind::ImportResponse),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub kind: FrameKind,
    pub request_id: u32,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn write(&self, out: &mut impl Write) -> io::Result<()> {
        out.write_all(&[PROTOCOL_VERSION, self.kind as u8])?;
        out.write_all(&self.request_id.to_le_bytes())?;
        out.write_all(&(self.payload.len() as u32).to_le_bytes())?;
        out.write_all(&self.payload)?;
        out.flush()
    }

    pub fn read(input: &mut impl Read) -> io::Result<Frame> {
        let mut header = [0u8; 2];
        input.read_exact(&mut header)?;
        let kind = FrameKind::from_u8(header[1])
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown frame kind"))?;
        let mut request_id_buf = [0u8; 4];
        input.read_exact(&mut request_id_buf)?;
        let request_id = u32::from_le_bytes(request_id_buf);
        let mut len_buf = [0u8; 4];
        input.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; len];
        input.read_exact(&mut payload)?;
        Ok(Frame {
            kind,
            request_id,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip() {
        let frame = Frame {
            kind: FrameKind::ImportRequest,
            request_id: 7,
            payload: vec![1, 2, 3, 4],
        };
        let mut buf = Vec::new();
        frame.write(&mut buf).unwrap();
        let decoded = Frame::read(&mut &buf[..]).unwrap();
        assert_eq!(decoded.kind, FrameKind::ImportRequest);
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.payload, vec![1, 2, 3, 4]);
    }

    #[test]
    fn empty_payload_roundtrip() {
        let frame = Frame {
            kind: FrameKind::ImportResponse,
            request_id: 0,
            payload: vec![],
        };
        let mut buf = Vec::new();
        frame.write(&mut buf).unwrap();
        let decoded = Frame::read(&mut &buf[..]).unwrap();
        assert!(decoded.payload.is_empty());
    }
}
