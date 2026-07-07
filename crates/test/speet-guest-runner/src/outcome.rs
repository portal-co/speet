//! Guest execution outcome.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl RunOutcome {
    pub fn from_status(status: std::process::ExitStatus, stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        let code = status.code().unwrap_or(-1);
        Self {
            exit_code: code,
            stdout,
            stderr,
        }
    }
}
