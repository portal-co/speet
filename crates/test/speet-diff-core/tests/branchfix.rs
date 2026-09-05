//! fixup_branches: (1) no-op on freshly generated code — every target the
//! generators patch is already in range; (2) after removing a word, all
//! branch targets are back in range and the module validates.

use speet_diff_core::case::Arch;
use speet_diff_core::generator::generate_case;
use speet_diff_core::build_case_module;

fn words(code: &[u8], big: bool) -> Vec<u32> {
    (0..code.len() / 4)
        .map(|i| {
            let b: [u8; 4] = code[i * 4..i * 4 + 4].try_into().unwrap();
            if big { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) }
        })
        .collect()
}

#[test]
fn fixup_is_noop_on_generated_code() {
    for arch in [Arch::AArch64, Arch::Arm, Arch::RiscV64, Arch::RiscV32, Arch::Mips] {
        for s in 1u64..30 {
            let case = generate_case(arch, s);
            let mut code = case.code.clone();
            let before = words(&code, arch == Arch::Mips);
            speet_diff_core::branchfix::fixup_branches(arch, &mut code);
            let after = words(&code, arch == Arch::Mips);
            assert_eq!(before, after, "{arch:?} seed {s}: fixup changed fresh code");
        }
    }
}

#[test]
fn fixup_repairs_after_word_removal_and_module_validates() {
    for arch in [Arch::AArch64, Arch::Arm, Arch::RiscV64, Arch::RiscV32, Arch::Mips] {
        for s in 1u64..12 {
            let case = generate_case(arch, s);
            if case.code.len() <= 8 {
                continue;
            }
            // Remove one middle word, then fixup.
            let mid = case.code.len() / 2 & !3;
            let mut code: Vec<u8> = Vec::new();
            code.extend_from_slice(&case.code[..mid]);
            code.extend_from_slice(&case.code[mid + 4..]);
            speet_diff_core::branchfix::fixup_branches(arch, &mut code);
            let trial = speet_diff_core::minimize::with_code_public(&case, code);
            // The translated module must compile+validate (a validate
            // failure here = a stale out-of-range branch survived fixup).
            // Execution is NOT asserted: retargeted branches may
            // legitimately run into RO stores / budget exhaustion.
            if let Err(e) = build_case_module(&trial) {
                panic!("{arch:?} seed {s}: module should validate after fixup: {e:?}");
            }
        }
    }
}
