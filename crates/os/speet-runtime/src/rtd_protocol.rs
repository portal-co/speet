//! Hand-rolled binary wire codec for `speet-rtd` IPC.
//!
//! Frame shape: `[u8 version][u8 tag][u32 LE payload_len][payload…]`.

use std::io::{Read, Write};

pub const PROTOCOL_VERSION: u8 = 1;

pub const OP_PING: u8 = 0;
pub const OP_ANALYZE: u8 = 1;
pub const OP_OBTAIN: u8 = 2;

pub const STATUS_PONG: u8 = 0;
pub const STATUS_SUITABLE: u8 = 1;
pub const STATUS_UNSUITABLE: u8 = 2;
pub const STATUS_READY: u8 = 3;
pub const STATUS_ERROR: u8 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Ping,
    Analyze { path: String },
    Obtain { path: String, host_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Response {
    Pong,
    Suitable,
    Unsuitable {
        unresolved_deps: Vec<String>,
        fn_ptr_deps: Vec<String>,
    },
    Ready {
        exe_path: String,
        cache_hit: bool,
    },
    Error { message: String },
}

pub fn encode_request(req: &Request) -> Vec<u8> {
    let mut payload = Vec::new();
    let tag = match req {
        Request::Ping => OP_PING,
        Request::Analyze { path } => {
            encode_string(path, &mut payload);
            OP_ANALYZE
        }
        Request::Obtain { path, host_id } => {
            encode_string(path, &mut payload);
            encode_string(host_id, &mut payload);
            OP_OBTAIN
        }
    };
    encode_frame(tag, &payload)
}

pub fn decode_request(buf: &[u8]) -> Result<Request, WireError> {
    let (tag, payload) = decode_frame(buf)?;
    match tag {
        OP_PING if payload.is_empty() => Ok(Request::Ping),
        OP_ANALYZE => {
            let (path, rest) = decode_string(payload)?;
            if !rest.is_empty() {
                return Err(WireError);
            }
            Ok(Request::Analyze { path })
        }
        OP_OBTAIN => {
            let (path, rest) = decode_string(payload)?;
            let (host_id, rest) = decode_string(rest)?;
            if !rest.is_empty() {
                return Err(WireError);
            }
            Ok(Request::Obtain { path, host_id })
        }
        _ => Err(WireError),
    }
}

pub fn encode_response(resp: &Response) -> Vec<u8> {
    let mut payload = Vec::new();
    let tag = match resp {
        Response::Pong => STATUS_PONG,
        Response::Suitable => STATUS_SUITABLE,
        Response::Unsuitable {
            unresolved_deps,
            fn_ptr_deps,
        } => {
            encode_string_vec(unresolved_deps, &mut payload);
            encode_string_vec(fn_ptr_deps, &mut payload);
            STATUS_UNSUITABLE
        }
        Response::Ready { exe_path, cache_hit } => {
            encode_string(exe_path, &mut payload);
            payload.push(u8::from(*cache_hit));
            STATUS_READY
        }
        Response::Error { message } => {
            encode_string(message, &mut payload);
            STATUS_ERROR
        }
    };
    encode_frame(tag, &payload)
}

pub fn decode_response(buf: &[u8]) -> Result<Response, WireError> {
    let (tag, payload) = decode_frame(buf)?;
    match tag {
        STATUS_PONG if payload.is_empty() => Ok(Response::Pong),
        STATUS_SUITABLE if payload.is_empty() => Ok(Response::Suitable),
        STATUS_UNSUITABLE => {
            let (unresolved_deps, rest) = decode_string_vec(payload)?;
            let (fn_ptr_deps, rest) = decode_string_vec(rest)?;
            if !rest.is_empty() {
                return Err(WireError);
            }
            Ok(Response::Unsuitable {
                unresolved_deps,
                fn_ptr_deps,
            })
        }
        STATUS_READY => {
            let (exe_path, rest) = decode_string(payload)?;
            if rest.is_empty() {
                return Err(WireError);
            }
            Ok(Response::Ready {
                exe_path,
                cache_hit: rest[0] != 0,
            })
        }
        STATUS_ERROR => {
            let (message, rest) = decode_string(payload)?;
            if !rest.is_empty() {
                return Err(WireError);
            }
            Ok(Response::Error { message })
        }
        _ => Err(WireError),
    }
}

pub fn read_frame<R: Read>(r: &mut R) -> Result<Vec<u8>, String> {
    let mut hdr = [0u8; 6];
    r.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    let len = u32::from_le_bytes(hdr[2..6].try_into().unwrap()) as usize;
    let mut frame = Vec::with_capacity(6 + len);
    frame.extend_from_slice(&hdr);
    if len > 0 {
        let mut payload = vec![0u8; len];
        r.read_exact(&mut payload).map_err(|e| e.to_string())?;
        frame.extend_from_slice(&payload);
    }
    Ok(frame)
}

pub fn write_frame<W: Write>(w: &mut W, frame: &[u8]) -> Result<(), String> {
    w.write_all(frame).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())
}

fn encode_frame(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(6 + payload.len());
    out.push(PROTOCOL_VERSION);
    out.push(tag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn decode_frame(buf: &[u8]) -> Result<(u8, &[u8]), WireError> {
    if buf.len() < 6 || buf[0] != PROTOCOL_VERSION {
        return Err(WireError);
    }
    let tag = buf[1];
    let len = u32::from_le_bytes(buf[2..6].try_into().unwrap()) as usize;
    if buf.len() != 6 + len {
        return Err(WireError);
    }
    Ok((tag, &buf[6..]))
}

fn encode_string(s: &str, out: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn decode_string(input: &[u8]) -> Result<(String, &[u8]), WireError> {
    if input.len() < 4 {
        return Err(WireError);
    }
    let len = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    let rest = &input[4..];
    if rest.len() < len {
        return Err(WireError);
    }
    let s = std::str::from_utf8(&rest[..len]).map_err(|_| WireError)?;
    Ok((s.to_string(), &rest[len..]))
}

fn encode_string_vec(items: &[String], out: &mut Vec<u8>) {
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for item in items {
        encode_string(item, out);
    }
}

fn decode_string_vec(input: &[u8]) -> Result<(Vec<String>, &[u8]), WireError> {
    if input.len() < 4 {
        return Err(WireError);
    }
    let len = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    let mut rest = &input[4..];
    let mut items = Vec::with_capacity(len);
    for _ in 0..len {
        let (s, r) = decode_string(rest)?;
        items.push(s);
        rest = r;
    }
    Ok((items, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_roundtrip() {
        let frame = encode_request(&Request::Ping);
        assert_eq!(decode_request(&frame).unwrap(), Request::Ping);
        let resp = encode_response(&Response::Pong);
        assert_eq!(decode_response(&resp).unwrap(), Response::Pong);
    }

    #[test]
    fn obtain_roundtrip() {
        let frame = encode_request(&Request::Obtain {
            path: "/bin/ls".into(),
            host_id: "integrated".into(),
        });
        match decode_request(&frame).unwrap() {
            Request::Obtain { path, host_id } => {
                assert_eq!(path, "/bin/ls");
                assert_eq!(host_id, "integrated");
            }
            _ => panic!("expected obtain"),
        }
    }

    #[test]
    fn ready_roundtrip() {
        let resp = encode_response(&Response::Ready {
            exe_path: "/tmp/x".into(),
            cache_hit: true,
        });
        match decode_response(&resp).unwrap() {
            Response::Ready { exe_path, cache_hit } => {
                assert_eq!(exe_path, "/tmp/x");
                assert!(cache_hit);
            }
            _ => panic!("expected ready"),
        }
    }
}
