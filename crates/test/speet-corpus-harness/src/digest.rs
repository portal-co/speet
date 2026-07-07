//! SHA-256 hex digest for stdout comparison.

use sha2::{Digest, Sha256};

pub fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
