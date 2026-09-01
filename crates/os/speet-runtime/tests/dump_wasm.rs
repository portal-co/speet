use speet_recompile::frontend::recompile_rv64_to_wasm;
const EXIT_42: &[u8] = &[
    0x13, 0x05, 0xA0, 0x02, 0x93, 0x08, 0xD0, 0x05, 0x73, 0x00, 0x00, 0x00,
];
#[test]
fn dump() {
    let w = recompile_rv64_to_wasm(EXIT_42, 0x1000);
    std::fs::write("/tmp/exit_rv64.wasm", &w).unwrap();
    eprintln!("wasm {} bytes, funcs export _start", w.len());
}
