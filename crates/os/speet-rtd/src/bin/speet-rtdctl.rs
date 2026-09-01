//! `speet-rtdctl` — Unix-socket client for speet-rtd (skill-only / MCP-absent path).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(speet_rtd::ctl::run(&args));
}
