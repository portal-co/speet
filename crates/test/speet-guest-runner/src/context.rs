//! Nested execution context chain.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub chain: Vec<String>,
    pub cwd: PathBuf,
    pub env: HashMap<String, String>,
    pub timeout: Duration,
}

impl ExecutionContext {
    pub fn host_root() -> Self {
        let mut env: HashMap<String, String> = std::env::vars().collect();
        Self {
            chain: vec!["host".into()],
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            env,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn child(&self, label: &str) -> Self {
        let mut chain = self.chain.clone();
        chain.push(label.to_string());
        Self {
            chain,
            cwd: self.cwd.clone(),
            env: self.env.clone(),
            timeout: self.timeout,
        }
    }

    pub fn with_env(mut self, key: &str, value: &str) -> Self {
        self.env.insert(key.to_string(), value.to_string());
        self
    }
}
