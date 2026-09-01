//! Sanity: corpus ELF `.text` matches inline EXIT_42 and both pipelines exit 42.

use binary_io::{BinArch, BinOs};
use speet_runtime::{default_host_api, load_text_from_object, Runtime};
use std::path::Path;
use std::sync::Arc;

const EXIT_42: &[u8] = &[
    0x13, 0x05, 0xA0, 0x02, 0x93, 0x08, 0xD0, 0x05, 0x73, 0x00, 0x00, 0x00,
];

#[test]
fn corpus_text_matches_inline_and_exits_42() {
    let mut rt = Runtime::new(Arc::new(default_host_api()));
    if !rt.llvm_available() {
        return;
    }
    let guest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/thin-runtime-corpus/rv64-linux/exit_42.elf");
    let (text, addr) = load_text_from_object(&guest).expect("load");
    assert_eq!(text.as_slice(), EXIT_42);
    assert_eq!(addr, 0x1000);
    let status = rt
        .recompile_rv64_and_run(EXIT_42, 0x1000, BinArch::AArch64, BinOs::MacOs)
        .expect("run inline");
    assert_eq!(status.code(), Some(42), "inline pipeline {:?}", status);
    let status = rt
        .recompile_rv64_and_run(&text, addr, BinArch::AArch64, BinOs::MacOs)
        .expect("run");
    assert_eq!(status.code(), Some(42), "corpus text pipeline {:?}", status);
}
