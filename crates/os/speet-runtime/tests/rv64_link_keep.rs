//! Link RV64 exit_42 to a fixed path for disassembly (manual debug).

use binary_io::{BinArch, BinOs};
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::recompile_rv64_to_wasm;
use speet_runtime::{default_host_api, Runtime};
use std::path::PathBuf;
use std::sync::Arc;

const EXIT_42: &[u8] = &[0x13, 0x05, 0xA0, 0x02, 0x93, 0x08, 0xD0, 0x05, 0x73, 0x00, 0x00, 0x00];

#[test]
#[ignore = "manual: writes /tmp/speet_rv64_new/guest for disassembly"]
fn link_rv64_exit_keep_guest() {
    let rt = Runtime::new(Arc::new(default_host_api()));
    if !rt.llvm_available() {
        return;
    }
    let wasm = recompile_rv64_to_wasm(EXIT_42, 0x1000);
    let obj = compile_wasm_to_object(&wasm, BinArch::AArch64, BinOs::MacOs).expect("obj");
    let dir = PathBuf::from("/tmp/speet_rv64_new");
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("guest");
    rt.link_guest_object(
        &obj,
        speet_recompile::drive::entry_param_count(&wasm),
        speet_riscv::RV64_SP_PARAM_INDEX,
        speet_recompile::frontend::halt_addr(0x1000, EXIT_42.len()),
        Some(speet_riscv::RV64_RA_PARAM_INDEX),
        BinArch::AArch64,
        BinOs::MacOs,
        &exe,
    )
    .expect("link");
    let status = std::process::Command::new(&exe).status().expect("run");
    eprintln!("exit={:?}", status.code());
    assert_eq!(status.code(), Some(42));
}
