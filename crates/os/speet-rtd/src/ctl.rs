//! Unix-socket client for `speet-rtd` (`speet-rtdctl` / `speet-rtd --ctl`).

use os_daemon_protocol::{
    decode_response, encode_request, read_frame, write_frame, Request, Response,
};
use speet_runtime::default_socket_path;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

const INTEGRATED: &str = "integrated";

fn sock_path() -> PathBuf {
    std::env::var_os("SPEET_RTD_SOCK")
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}

fn send(req: &Request) -> Result<Response, String> {
    let path = sock_path();
    let mut sock =
        UnixStream::connect(&path).map_err(|e| format!("connect {}: {e}", path.display()))?;
    write_frame(&mut sock, &encode_request(req))?;
    let frame = read_frame(&mut sock)?;
    decode_response(&frame).map_err(|_| "invalid rtd response".to_string())
}

fn print_response(resp: Response) -> i32 {
    match resp {
        Response::Report { json } | Response::GuestInfo { json } => {
            println!("{json}");
            0
        }
        Response::Suitable { backend } => {
            // Typed buckets live on LastReport; fetch them so the CLI
            // never flattens away unresolved/fn-ptr types.
            match send(&Request::LastReport) {
                Ok(Response::Report { json }) => {
                    println!("{json}");
                    0
                }
                Ok(other) => {
                    println!(
                        "{{\"suitable\":true,\"backend\":\"{backend}\",\"note\":\"{other:?}\"}}"
                    );
                    0
                }
                Err(e) => {
                    eprintln!("{e}");
                    1
                }
            }
        }
        Response::Unsuitable { backend, reasons } => match send(&Request::LastReport) {
            Ok(Response::Report { json }) => {
                println!("{json}");
                1
            }
            _ => {
                println!(
                    "{{\"suitable\":false,\"backend\":\"{backend}\",\"reasons\":{reasons:?}}}"
                );
                1
            }
        },
        Response::Ready { exe_path, .. } => match send(&Request::LastReport) {
            Ok(Response::Report { json }) => {
                println!("{json}");
                0
            }
            _ => {
                println!("{{\"exe_path\":\"{exe_path}\"}}");
                0
            }
        },
        Response::Run {
            exit_code,
            stdout,
            stderr,
        } => {
            println!("{{\"exit_code\":{exit_code},\"stdout\":{stdout:?},\"stderr\":{stderr:?}}}");
            if exit_code == 0 {
                0
            } else {
                1
            }
        }
        Response::Error { message } => {
            match send(&Request::LastReport) {
                Ok(Response::Report { json }) if json != "{}" => {
                    eprintln!("{json}");
                }
                _ => eprintln!("{message}"),
            }
            1
        }
        other => {
            eprintln!("{other:?}");
            1
        }
    }
}

/// `speet-rtdctl <verb> [args…]` / `speet-rtd --ctl <verb> [args…]`.
pub fn run(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("usage: speet-rtdctl <analyze|recompile|last-report|info|run> [path] [-- argv…]");
        return 2;
    }
    let verb = args[0].as_str();
    match verb {
        "analyze" => {
            let path = match args.get(1) {
                Some(p) => p.clone(),
                None => {
                    eprintln!("usage: speet-rtdctl analyze <path>");
                    return 2;
                }
            };
            match send(&Request::Analyze {
                path,
                backend: Some(INTEGRATED.into()),
            }) {
                Ok(r) => print_response(r),
                Err(e) => {
                    eprintln!("{e}");
                    1
                }
            }
        }
        "recompile" => {
            let mut link = true;
            let mut path = None;
            for a in &args[1..] {
                if a == "--no-link" {
                    link = false;
                } else if !a.starts_with('-') {
                    path = Some(a.clone());
                }
            }
            let Some(path) = path else {
                eprintln!("usage: speet-rtdctl recompile [--no-link] <path>");
                return 2;
            };
            let _ = link; // Unix Obtain always links; --no-link is MCP-in-process only.
            match send(&Request::Obtain {
                path,
                backend: INTEGRATED.into(),
            }) {
                Ok(r) => print_response(r),
                Err(e) => {
                    eprintln!("{e}");
                    1
                }
            }
        }
        "last-report" | "last_report" => match send(&Request::LastReport) {
            Ok(r) => print_response(r),
            Err(e) => {
                eprintln!("{e}");
                1
            }
        },
        "info" | "guest_info" | "guest-info" => {
            let path = match args.get(1) {
                Some(p) => p.clone(),
                None => {
                    eprintln!("usage: speet-rtdctl info <path>");
                    return 2;
                }
            };
            match send(&Request::GuestInfo { path }) {
                Ok(r) => print_response(r),
                Err(e) => {
                    eprintln!("{e}");
                    1
                }
            }
        }
        "run" => {
            let path = match args.get(1) {
                Some(p) => p.clone(),
                None => {
                    eprintln!("usage: speet-rtdctl run <path> [-- args…]");
                    return 2;
                }
            };
            let argv = if let Some(i) = args.iter().position(|a| a == "--") {
                args[i + 1..].to_vec()
            } else {
                args[2..].to_vec()
            };
            match send(&Request::Run { path, argv }) {
                Ok(r) => print_response(r),
                Err(e) => {
                    eprintln!("{e}");
                    1
                }
            }
        }
        other => {
            eprintln!("unknown verb: {other}");
            2
        }
    }
}
