//! MCP server (`mcp` feature): thin-mcp tools/resources over stdio and an
//! optional Unix socket. Same ops as `speet-rtdctl`; no reload tool.

use crate::Daemon;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use thin_mcp::{
    McpError, McpServer, McpServerConfig, Resource, ResourcesHandler, ServerInfo, Tool,
    ToolsHandler,
};

fn text_result(body: String) -> Value {
    json!({
        "content": [{ "type": "text", "text": body }]
    })
}

struct SpeetTools {
    daemon: Arc<Daemon>,
}

#[async_trait]
impl ToolsHandler for SpeetTools {
    async fn list(&self, _params: Value) -> Result<Value, McpError> {
        let tools = vec![
            Tool {
                name: "analyze_binary".into(),
                description: Some("Typed suitability report for a guest binary.".into()),
                input_schema: Some(json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                })),
            },
            Tool {
                name: "recompile".into(),
                description: Some(
                    "Translate a guest binary (+ optional obtain/link). Returns RecompileReport."
                        .into(),
                ),
                input_schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "link": { "type": "boolean", "default": true }
                    },
                    "required": ["path"]
                })),
            },
            Tool {
                name: "last_report".into(),
                description: Some(
                    "Last failure/success RecompileReport for the pinned guest.".into(),
                ),
                input_schema: Some(json!({ "type": "object", "properties": {} })),
            },
            Tool {
                name: "guest_info".into(),
                description: Some("Path, arch, .text span, imports, entry.".into()),
                input_schema: Some(json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                })),
            },
            Tool {
                name: "run_guest".into(),
                description: Some(
                    "Obtain + spawn; exit status and truncated stdout/stderr.".into(),
                ),
                input_schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "argv": { "type": "array", "items": { "type": "string" } }
                    },
                    "required": ["path"]
                })),
            },
        ];
        Ok(json!({ "tools": tools }))
    }

    async fn call(&self, name: &str, args: Value) -> Result<Value, McpError> {
        match name {
            "analyze_binary" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| McpError::invalid_params("missing path"))?;
                let report = self
                    .daemon
                    .analyze_report(std::path::Path::new(path))
                    .map_err(McpError::internal)?;
                Ok(text_result(report.to_json()))
            }
            "recompile" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| McpError::invalid_params("missing path"))?;
                let link = args["link"].as_bool().unwrap_or(true);
                let report = self
                    .daemon
                    .recompile_report(std::path::Path::new(path), link);
                Ok(text_result(report.to_json()))
            }
            "last_report" => {
                let json = self
                    .daemon
                    .last_report()
                    .map(|r| r.to_json())
                    .unwrap_or_else(|| "{}".into());
                Ok(text_result(json))
            }
            "guest_info" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| McpError::invalid_params("missing path"))?;
                Ok(text_result(
                    self.daemon.guest_info(std::path::Path::new(path)).to_json(),
                ))
            }
            "run_guest" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| McpError::invalid_params("missing path"))?;
                let argv: Vec<String> = args["argv"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                match self.daemon.run_guest(std::path::Path::new(path), &argv) {
                    Ok((exit_code, stdout, stderr)) => Ok(text_result(
                        json!({"exit_code": exit_code, "stdout": stdout, "stderr": stderr})
                            .to_string(),
                    )),
                    Err(e) => {
                        if let Some(r) = self.daemon.last_report() {
                            Ok(text_result(r.to_json()))
                        } else {
                            Err(McpError::internal(e))
                        }
                    }
                }
            }
            other => Err(McpError::method_not_found(other)),
        }
    }
}

struct SpeetResources {
    daemon: Arc<Daemon>,
}

#[async_trait]
impl ResourcesHandler for SpeetResources {
    async fn list(&self, _params: Value) -> Result<Value, McpError> {
        Ok(json!({
            "resources": [
                Resource {
                    uri: "speet://report/latest".into(),
                    name: Some("latest RecompileReport".into()),
                    description: Some("Last analyze/recompile/obtain report.".into()),
                    mime_type: Some("application/json".into()),
                },
                Resource {
                    uri: "speet://guest/imports".into(),
                    name: Some("pinned guest imports".into()),
                    description: Some("Imports of the last reported guest.".into()),
                    mime_type: Some("application/json".into()),
                }
            ]
        }))
    }

    async fn read(&self, params: Value) -> Result<Value, McpError> {
        let uri = params["uri"]
            .as_str()
            .ok_or_else(|| McpError::invalid_params("missing uri"))?;
        match uri {
            "speet://report/latest" => {
                let text = self
                    .daemon
                    .last_report()
                    .map(|r| r.to_json())
                    .unwrap_or_else(|| "{}".into());
                Ok(json!({
                    "contents": [{ "uri": uri, "mimeType": "application/json", "text": text }]
                }))
            }
            "speet://guest/imports" => {
                let text = self
                    .daemon
                    .last_report()
                    .map(|r| {
                        let items = r
                            .guest
                            .imports
                            .iter()
                            .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
                            .collect::<Vec<_>>()
                            .join(",");
                        format!("[{items}]")
                    })
                    .unwrap_or_else(|| "[]".into());
                Ok(json!({
                    "contents": [{ "uri": uri, "mimeType": "application/json", "text": text }]
                }))
            }
            other => Err(McpError::invalid_params(format!(
                "unknown resource: {other}"
            ))),
        }
    }
}

pub fn build_mcp_server(daemon: Arc<Daemon>) -> McpServer {
    let config = McpServerConfig::builder()
        .server_info(ServerInfo {
            name: "speet-recompiler-debug".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        })
        .instructions(
            "Debug speet recompiles. Rebuild recompiler/stubs WASM then rerun; reload is automatic.",
        )
        .tools_handler(SpeetTools {
            daemon: daemon.clone(),
        })
        .resources_handler(SpeetResources { daemon })
        .build();
    McpServer::new(config)
}

/// One JSON-RPC frame → optional response JSON. For tests.
pub fn handle_mcp_message(server: &McpServer, session: &str, raw: &str) -> Option<String> {
    block_on(server.handle_message(session, raw))
}

fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn dummy_raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker {
            dummy_raw_waker()
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(core::ptr::null(), &VTABLE)
    }

    let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = core::pin::pin!(fut);
    loop {
        match Pin::as_mut(&mut fut).poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// Stdio JSON-RPC loop (newline-delimited). Optional `SPEET_RTD_MCP_SOCK`
/// accepts the same frames on a Unix stream.
pub fn run_stdio(daemon: Arc<Daemon>) -> Result<(), String> {
    let server = Arc::new(build_mcp_server(daemon));
    if let Ok(path) = std::env::var("SPEET_RTD_MCP_SOCK") {
        let srv = server.clone();
        std::thread::spawn(move || {
            if let Err(e) = serve_mcp_sock(&srv, &path) {
                eprintln!("mcp sock: {e}");
            }
        });
    }
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let sid = server.new_session();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(resp) = handle_mcp_message(&server, &sid, line) {
            writeln!(stdout, "{resp}").map_err(|e| e.to_string())?;
            stdout.flush().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn serve_mcp_sock(server: &McpServer, path: &str) -> Result<(), String> {
    use std::os::unix::net::UnixListener;
    let p = std::path::Path::new(path);
    if p.exists() {
        let _ = std::fs::remove_file(p);
    }
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let listener = UnixListener::bind(p).map_err(|e| e.to_string())?;
    for stream in listener.incoming() {
        let stream = stream.map_err(|e| e.to_string())?;
        let sid = server.new_session();
        let reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
        let mut writer = stream;
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(resp) = handle_mcp_message(server, &sid, line) {
                let _ = writeln!(writer, "{resp}");
                let _ = writer.flush();
            }
        }
        server.remove_session(&sid);
    }
    Ok(())
}
