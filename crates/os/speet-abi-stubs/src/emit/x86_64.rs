//! x86-64 native helpers for ABI redirect stubs.

use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_x86_64::{
    out::{
        arg::{ArgKind, MemArgKind},
        WriterCore,
    },
    RegisterClass, X64Arch,
};
use portal_solutions_blitz_common::asm::Reg;

/// Translate a guest pointer held in `arg_reg` by adding `mem_base_reg`.
pub fn add_guest_mem_base<W, Ctx, E>(
    w: &mut W,
    ctx: &mut Ctx,
    arch: X64Arch,
    arg_reg: u8,
    mem_base_reg: u8,
) -> Result<(), E>
where
    W: WriterCore<Ctx, Error = E>,
{
    let arg = ArgKind::Reg {
        reg: Reg(arg_reg),
        size: MemorySize::_64,
    };
    w.add(
        ctx,
        arch,
        &MemArgKind::NoMem(arg),
        &MemArgKind::NoMem(arg),
        &MemArgKind::NoMem(ArgKind::Reg {
            reg: Reg(mem_base_reg),
            size: MemorySize::_64,
        }),
    )
}
