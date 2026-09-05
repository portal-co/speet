//! Host LLVM decode verification (plan §4.1 "verified decode" —
//! independent-decoder cross-check): every word the generators emit must
//! decode under LLVM's disassembler (`llvm-mc --disassemble`) with no
//! `<invalid>` and mnemonics drawn only from the arch's intended
//! vocabulary. `llvm-mc` is located via `$LLVM_MC`, `xcrun --find`, or the
//! Homebrew prefix; a missing binary fails the test (fail-closed — never
//! silently skip: an unverifiable encoding pipeline is a bug, not a pass).

use std::collections::{HashMap, HashSet};
use std::process::Command;

use speet_diff_core::case::Arch;
use speet_diff_core::generator::generate_case;

fn find_llvm_mc() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("LLVM_MC") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(out) = Command::new("xcrun").args(["--find", "llvm-mc"]).output() {
        if out.status.success() {
            let p = std::path::PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
            if p.exists() {
                return Some(p);
            }
        }
    }
    for p in ["/opt/homebrew/opt/llvm/bin/llvm-mc", "/usr/local/opt/llvm/bin/llvm-mc"] {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(out) = Command::new("which").arg("llvm-mc").output() {
        if out.status.success() {
            let p = std::path::PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

/// Disassemble `words` (8-hex-digit strings) under `triple`. Returns one
/// decoded instruction text per word (`None` = decode failure / invalid).
fn disassemble(mc: &std::path::Path, triple: &str, words: &[String]) -> Vec<Option<String>> {
    use std::io::Write;
    let input: String = words
        .iter()
        .map(|w| format!("0x{} 0x{} 0x{} 0x{}", &w[6..8], &w[4..6], &w[2..4], &w[0..2]))
        .collect::<Vec<_>>()
        .join("\n");
    let mut child = Command::new(mc)
        .args([format!("--triple={triple}"), "--disassemble".into()])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn llvm-mc");
    child.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
    let out = child.wait_with_output().expect("llvm-mc output");
    assert!(
        out.status.success(),
        "llvm-mc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| {
            let l = l.trim();
            if l.is_empty() || l.contains("invalid") {
                None
            } else {
                // "\tmov\twd, ws" — after trim, the mnemonic is the
                // first tab-separated field (operands follow).
                Some(l.split('\t').next().unwrap_or("").to_string())
            }
        })
        .collect()
}

/// Intended mnemonic vocabulary per arch (the aliases LLVM prints count —
/// e.g. `mv` for `addi rd, rs, 0`, `sext.w` for `addiw rd, rs, 0`).
fn vocab(arch: Arch) -> HashSet<&'static str> {
    let v: &[&str] = match arch {
        Arch::X86_64 | Arch::X86_32 => &[
            "mov", "movabs", "add", "sub", "and", "or", "xor", "test", "cmp", "push", "pop",
            "xchg", "not", "lea", "movsx", "movsxd", "movzx", "imul", "shl", "shr", "sar",
            "ret", "nop", "jmp", "je", "jne", "jl", "jle", "jg", "jge", "jb", "jbe", "ja",
            "jae", "js", "jns", "jo", "jno", "jp", "jnp", "call",
        ],
        Arch::AArch64 => &[
            "mov", "movk", "movz", "movn", "add", "adds", "sub", "subs", "and", "ands",
            "orr", "eor", "lsl", "lsr", "asr", "ldr", "str", "ldrb", "strb", "ldrh", "strh",
            "ldur", "stur", "cbz", "cbnz", "b", "b.eq", "b.ne", "b.lt", "b.le", "b.gt",
            "b.ge", "b.lo", "b.ls", "b.hi", "b.hs", "csel", "ret", "nop",
        ],
        Arch::Arm => &[
            "mov", "mvn", "add", "adds", "sub", "subs", "and", "ands", "orr", "orrs", "eor",
            "eors", "bic", "ldr", "str", "b", "bl", "bx", "blx", "nop", "cmp",
        ],
        Arch::RiscV64 | Arch::RiscV32 => &[
            "add", "addi", "sub", "and", "andi", "or", "ori", "xor", "xori", "sll", "slli",
            "srl", "srli", "sra", "srai", "slliw", "srliw", "sraiw", "addiw", "addw",
            "subw", "sllw", "srlw", "sraw", "lui", "auipc", "ld", "lw", "lb", "lbu", "lh",
            "lhu", "sd", "sw", "sb", "sh", "beq", "bne", "blt", "bge", "bltu", "bgeu",
            "jal", "jalr", "ret", "mv", "li", "seqz", "snez", "not", "neg", "sext.w",
            "beqz", "bnez", "bgez", "bltz", "blez", "bgtz", "zext.w",
        ],
        Arch::Mips => &[
            "addiu", "addi", "addu", "add", "subu", "sub", "and", "andi", "or", "ori",
            "xor", "xori", "nor", "sll", "srl", "sra", "lui", "lw", "sw", "lb", "sb", "lh",
            "sh", "beq", "bne", "jr", "nop",
        ],
    };
    v.iter().copied().collect()
}

#[test]
fn generated_encodings_decode_under_llvm() {
    let Some(mc) = find_llvm_mc() else {
        panic!(
            "llvm-mc not found (set $LLVM_MC, or install llvm via Homebrew) — \
             the generators' encodings cannot be independently verified"
        );
    };
    let triple = |arch: Arch| match arch {
        Arch::X86_64 => "x86_64",
        Arch::X86_32 => "i386",
        Arch::AArch64 => "aarch64",
        Arch::Arm => "armv7",
        Arch::RiscV64 => "riscv64",
        Arch::RiscV32 => "riscv32",
        Arch::Mips => "mips",
    };

    for arch in [
        Arch::X86_64,
        Arch::X86_32,
        Arch::AArch64,
        Arch::Arm,
        Arch::RiscV64,
        Arch::RiscV32,
        Arch::Mips,
    ] {
        if arch.code_align() == 4 {
            // Word-aligned archs: collect unique words, batch-disassemble.
            let mut words: Vec<String> = Vec::new();
            let mut seen = HashSet::new();
            'collect: for s in 1u64..400 {
                let case = generate_case(arch, s);
                for i in (0..case.code.len()).step_by(4) {
                    let w = u32::from_le_bytes(case.code[i..i + 4].try_into().unwrap());
                    let hex = format!("{w:08x}");
                    if seen.insert(hex.clone()) {
                        words.push(hex);
                    }
                    if words.len() >= 400 {
                        break 'collect;
                    }
                }
            }
            let decoded = disassemble(&mc, triple(arch), &words);
            assert_eq!(decoded.len(), words.len(), "{arch:?}: llvm-mc row count");

            let mut failures = Vec::new();
            let mut off_vocab: Vec<String> = Vec::new();
            for (w, d) in words.iter().zip(&decoded) {
                match d {
                    None => failures.push(w.clone()),
                    Some(text) => {
                        let m = text.split_whitespace().next().unwrap_or("");
                        if !m.is_empty() && !vocab(arch).contains(m) {
                            off_vocab.push(format!("{m} (0x{w} → `{text}`)"));
                        }
                    }
                }
            }
            assert!(
                failures.is_empty(),
                "{arch:?}: {} words failed to decode under LLVM ({}): {failures:?}",
                failures.len(),
                triple(arch)
            );
            assert!(
                off_vocab.is_empty(),
                "{arch:?}: mnemonics outside the intended vocabulary: {off_vocab:?}"
            );
        } else {
            // Byte-granular (x86): disassemble whole case blobs; every
            // byte must be part of a decoded instruction.
            for s in 1u64..=15 {
                let case = generate_case(arch, s);
                let input = case
                    .code
                    .iter()
                    .map(|b| format!("0x{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                let out = Command::new(&mc)
                    .args([format!("--triple={}", triple(arch)), "--disassemble".into()])
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .and_then(|mut c| {
                        use std::io::Write;
                        c.stdin.as_mut().unwrap().write_all(input.as_bytes())?;
                        c.wait_with_output()
                    })
                    .expect("spawn llvm-mc");
                let text = String::from_utf8_lossy(&out.stdout);
                assert!(
                    out.status.success() && !text.contains("invalid"),
                    "{arch:?} seed {s}: LLVM decode error: {}{}",
                    text.trim(),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
    }
}
