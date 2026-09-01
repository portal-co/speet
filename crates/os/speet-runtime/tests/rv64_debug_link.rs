use binary_io::{BinArch, BinOs};
use speet_runtime::{default_host_api, Runtime};
use std::sync::Arc;

const EXIT_42: &[u8] = &[
    0x13, 0x05, 0xA0, 0x02, 0x93, 0x08, 0xD0, 0x05, 0x73, 0x00, 0x00, 0x00,
];

#[test]
fn debug_keep_exe() {
    let mut rt = Runtime::new(Arc::new(default_host_api()));
    if !rt.llvm_available() {
        return;
    }
    let wasm = rt.recompile_rv64_text(EXIT_42, 0x1000);
    let obj = rt
        .compile_to_object(&wasm, BinArch::AArch64, BinOs::MacOs)
        .unwrap();
    let dir = std::path::PathBuf::from("/tmp/speet_rv64_keep");
    let _ = std::fs::remove_dir_all(&dir);
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
    .unwrap();
    eprintln!("exe at {}", exe.display());
    let out = std::process::Command::new(&exe).output().unwrap();
    eprintln!("status={:?} code={:?}", out.status, out.status.code());
    eprintln!("stderr={}", String::from_utf8_lossy(&out.stderr));
}
