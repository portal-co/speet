//! Recompile daemon for the integrated thin runtime.

use speet_rtd::{bind_socket, default_listen_path, Daemon};
use std::env;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--ctl") {
        std::process::exit(speet_rtd::ctl::run(&args[1..]));
    }

    let path = env::var_os("SPEET_RTD_SOCK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_listen_path);

    let mcp = args.iter().any(|a| a == "--mcp");
    if mcp {
        #[cfg(feature = "mcp")]
        {
            eprintln!("speet-rtd listening on {} (mcp stdio)", path.display());
            let listener = bind_socket(&path).unwrap_or_else(|e| {
                eprintln!("bind failed: {e}");
                std::process::exit(1);
            });
            let daemon = Arc::new(Daemon::new());
            let d = daemon.clone();
            std::thread::spawn(move || {
                if let Err(e) = d.serve(listener) {
                    eprintln!("daemon error: {e}");
                }
            });
            if let Err(e) = speet_rtd::mcp::run_stdio(daemon) {
                eprintln!("mcp error: {e}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "mcp"))]
        {
            eprintln!("speet-rtd was not built with the `mcp` feature");
            std::process::exit(2);
        }
    }

    eprintln!("speet-rtd listening on {}", path.display());
    let listener = bind_socket(&path).unwrap_or_else(|e| {
        eprintln!("bind failed: {e}");
        std::process::exit(1);
    });
    if let Err(e) = Daemon::run_on_listener(listener) {
        eprintln!("daemon error: {e}");
        std::process::exit(1);
    }
}
