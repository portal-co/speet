//! AArch64 native helpers for ABI redirect stubs.

use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_aarch64::{
    out::{
        arg::{ArgKind, MemArgKind},
        WriterCore,
    },
    AArch64Arch,
};
use portal_solutions_blitz_common::asm::Reg;

fn reg(n: u8) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg {
        reg: Reg(n),
        size: MemorySize::_64,
    })
}

/// Translate a guest pointer held in `arg_reg` by adding `mem_base_reg`.
pub fn add_guest_mem_base<W, Ctx, E>(
    w: &mut W,
    ctx: &mut Ctx,
    arch: AArch64Arch,
    arg_reg: u8,
    mem_base_reg: u8,
) -> Result<(), E>
where
    W: WriterCore<Ctx, Error = E>,
{
    w.add(ctx, arch, &reg(arg_reg), &reg(arg_reg), &reg(mem_base_reg))
}
