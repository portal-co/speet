use speet_diff_core::case::{Arch, FuzzCase, RegState};
use speet_diff_core::{run_oracle, run_recompiled};

fn one_case(mut code: Vec<u8>) -> FuzzCase {
    // Terminate with `ret` so execution reaches the halt sentinel (the
    // trailing slot seals with bare `unreachable` otherwise).
    code.push(0xC3);
    let mut regs = RegState::default();
    regs.gprs[3] = speet_diff_core::generator::DATA_BASE; // RBX
    let sp = speet_diff_core::generator::STACK_BASE
        + ((speet_diff_core::generator::STACK_SIZE as u64) & !0xF)
        - 8;
    regs.gprs[4] = sp;
    regs.gprs[7] = 0x1234; // RDI
    regs.gprs[15] = 0xAAAA; // R15
    regs.rip = speet_diff_core::generator::CODE_BASE;
    let data = vec![0u8; speet_diff_core::generator::DATA_SIZE];
    let mut stack = vec![0u8; speet_diff_core::generator::STACK_SIZE];
    let off = (sp - speet_diff_core::generator::STACK_BASE) as usize;
    stack[off..off + 8]
        .copy_from_slice(&((speet_diff_core::generator::CODE_BASE + code.len() as u64).to_le_bytes()));
    FuzzCase {
        arch: Arch::X86_64,
        code,
        entry_pc: speet_diff_core::generator::CODE_BASE,
        regs,
        data,
        data_base: speet_diff_core::generator::DATA_BASE,
        stack,
        stack_base: speet_diff_core::generator::STACK_BASE,
        read_only: vec![(
            speet_diff_core::generator::RO_DATA_OFFSET,
            speet_diff_core::generator::RO_DATA_SIZE,
        )],
        step_budget: 4000,
        seed: 0,
    }
}

#[test]
fn mov_r32_r32_runs() {
    // mov r15d, edi  = 44 89 FF
    let case = one_case(vec![0x44, 0x89, 0xFF]);
    let o = run_oracle(&case);
    let r = run_recompiled(&case);
    println!("oracle: {o:?}");
    println!("recompiled: {r:?}");
    let o = o.expect("oracle ok");
    let r = r.expect("recompiled ok");
    // 44 89 FF is `mov edi, r15d` (opcode 89 /r: dst = r/m, src = reg).
    assert_eq!(r.regs.gprs[7], 0xAAAA, "edi = r15d");
    assert_eq!(o.regs.gprs[7], r.regs.gprs[7]);
    assert_eq!(o.regs.gprs[15], r.regs.gprs[15]);
}

#[test]
fn mov_r64_r64_runs() {
    // mov rax, rcx = 48 89 C8
    let case = one_case(vec![0x48, 0x89, 0xC8]);
    let o = run_oracle(&case).expect("oracle ok");
    let r = run_recompiled(&case).expect("recompiled ok");
    assert_eq!(r.regs.gprs[0], o.regs.gprs[0]);
}
