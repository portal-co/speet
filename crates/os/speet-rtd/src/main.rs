//! Recompile daemon for the integrated thin runtime.

use speet_rtd::{bind_socket, default_listen_path, Daemon};
use std::env;

fn main() {
    let path = env::var_os("SPEET_RTD_SOCK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_listen_path);
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
