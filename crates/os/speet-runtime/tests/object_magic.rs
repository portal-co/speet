//! Object format sanity check.

use binary_io::{BinArch, BinOs};
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::recompile_rv64_to_wasm;

const EXIT_42: &[u8] = &[
    0x13, 0x05, 0xA0, 0x02, 0x93, 0x08, 0xD0, 0x05, 0x73, 0x00, 0x00, 0x00,
];

#[test]
fn compiled_objects_have_known_magic() {
    let wasm = recompile_rv64_to_wasm(EXIT_42, 0x1000);
    let elf = compile_wasm_to_object(&wasm, BinArch::X86_64, BinOs::Linux).expect("elf");
    assert_eq!(&elf[..4], b"\x7fELF");
    let macho = compile_wasm_to_object(&wasm, BinArch::AArch64, BinOs::MacOs).expect("macho");
    // Mach-O 64-bit little-endian
    assert_eq!(
        u32::from_le_bytes(macho[..4].try_into().unwrap()),
        0xfeedfacf
    );
}
