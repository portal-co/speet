//! [`SubprocessPlugin`] — one spawned plugin process, speaking the
//! [`crate::frame`] protocol over its `stdin`/`stdout` pipes.

use std::io::BufReader;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use speet_plugin_api::remote::{dispatch_import, ImportRole};
use speet_plugin_api::wire::WireDecode;
use speet_plugin_api::HostImports;

use crate::frame::{Frame, FrameKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubprocessError(pub String);

impl core::fmt::Display for SubprocessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "subprocess plugin error: {}", self.0)
    }
}
impl std::error::Error for SubprocessError {}

impl From<std::io::Error> for SubprocessError {
    fn from(e: std::io::Error) -> Self {
        SubprocessError(format!("io: {e}"))
    }
}

struct Inner {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_request_id: u32,
    imports: Arc<dyn HostImports>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One spawned plugin process — one per [`speet_plugin_host::PluginHandle`],
/// spawned at load and killed on `Drop` (no pooling/reuse in this MVP). See
/// `docs/guides/plugin-api.md` §5.3.
pub struct SubprocessPlugin {
    inner: Mutex<Inner>,
}

impl SubprocessPlugin {
    /// Spawn `command` with `args`, granting it exactly `imports` (already a
    /// restricted view — see `speet_plugin_host::PluginRegistry::restricted_view`
    /// and §2.7's trust note) as its nested `ImportRequest` resolution table.
    pub fn spawn(
        command: &str,
        args: &[String],
        imports: Arc<dyn HostImports>,
    ) -> Result<Self, SubprocessError> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| SubprocessError(format!("spawn {command}: {e}")))?;
        let stdin = child.stdin.take().ok_or_else(|| SubprocessError("no stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| SubprocessError("no stdout".into()))?;
        Ok(Self {
            inner: Mutex::new(Inner {
                child,
                stdin,
                stdout: BufReader::new(stdout),
                next_request_id: 0,
                imports,
            }),
        })
    }

    /// Send `payload` (a `speet_plugin_api::remote::XRequest` encoding) as a
    /// [`FrameKind::MainRequest`], servicing any interleaved
    /// [`FrameKind::ImportRequest`] frames (§2.7) until the matching
    /// [`FrameKind::MainResponse`] arrives.
    pub fn call_raw(&self, payload: &[u8]) -> Result<Vec<u8>, SubprocessError> {
        let mut guard = self.inner.lock().unwrap();
        let request_id = guard.next_request_id;
        guard.next_request_id = guard.next_request_id.wrapping_add(1);

        Frame {
            kind: FrameKind::MainRequest,
            request_id,
            payload: payload.to_vec(),
        }
        .write(&mut guard.stdin)?;

        loop {
            let frame = Frame::read(&mut guard.stdout)?;
            match frame.kind {
                FrameKind::MainResponse => return Ok(frame.payload),
                FrameKind::ImportRequest => {
                    let resp_payload = dispatch_import_request(&*guard.imports, &frame.payload)
                        .unwrap_or_default();
                    Frame {
                        kind: FrameKind::ImportResponse,
                        request_id: frame.request_id,
                        payload: resp_payload,
                    }
                    .write(&mut guard.stdin)?;
                }
                other => {
                    return Err(SubprocessError(format!(
                        "unexpected frame kind from plugin: {other:?}"
                    )))
                }
            }
        }
    }
}

/// Decode an `ImportRequest` payload (`[role: u8][name: String][request
/// bytes]`) and delegate to the shared `speet_plugin_api::remote::dispatch_import`
/// — the same "resolve, decode, dispatch, encode" logic the WASM and dylib
/// hosts use for the identical nested-import-call shape. `None` means "not
/// granted, unknown name, or malformed request" — the caller sends back an
/// empty `ImportResponse` payload, the documented denied/failed sentinel a
/// well-behaved plugin must check for (mirrors `speet-plugin-host-wasm`'s
/// `(0, 0)` pack sentinel for the same case).
fn dispatch_import_request(imports: &dyn HostImports, payload: &[u8]) -> Option<Vec<u8>> {
    let (role, rest) = ImportRole::decode(payload).ok()?;
    let (name, req) = String::decode(rest).ok()?;
    dispatch_import(imports, role, &name, req)
}
