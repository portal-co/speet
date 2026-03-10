#![no_std]

use core::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64};
pub trait Ctx {
    fn read(&self, addr: NonZeroU64, cell: &mut (dyn ArgCell + '_));
    fn write(&mut self, addr: NonZeroU64, cell: &mut (dyn Arg + '_));
    fn reg(&self, idx: u8, cell: &mut (dyn ArgCell + '_));
    fn set_reg(&mut self, idx: u8, cell: &mut (dyn Arg + '_));
    fn jalr(&mut self, addr: NonZeroU64, cell: &mut (dyn Arg + '_)) -> SysResult;
}
pub enum SysResult {
    Return,
    Jump(NonZeroU64),
}
pub trait OS {
    fn syscall(&mut self, ctx: &mut (dyn Ctx + '_)) -> SysResult;
    fn osfuncall(&mut self, addr: NonZeroU64, ctx: &mut (dyn Ctx + '_)) -> SysResult;
}
pub trait ArgCell {
    fn fill(&mut self, vals: &mut (dyn Iterator<Item = u8> + '_));
    fn done(self) -> impl Arg
    where
        Self: Sized;
}
pub trait Arg {
    fn as_addr(&self) -> Option<NonZeroU64>;
    fn len(&self) -> u64;
}
macro_rules! int_arg{
    ($($t:ty),*) => {
        $(impl Arg for $t {
            fn as_addr(&self) -> Option<NonZeroU64> {
                NonZeroU64::new(*self as u64)
            }
            fn len(&self) -> u64 {
                core::mem::size_of::<$t>() as u64
            }
        }
        impl ArgCell for $t {
            fn fill(&mut self, vals: &mut (dyn Iterator<Item = u8> + '_)) {
                let mut bytes = [0u8; core::mem::size_of::<$t>()];
                for b in bytes.iter_mut() {
                    *b = vals.next().unwrap_or(0);
                }
                *self = <$t>::from_le_bytes(bytes);
            }
            fn done(self) -> impl Arg {
                self
            }
        }
    )*

    };
}
int_arg!(u8, u16, u32, u64);
macro_rules! option_nonzero_arg {
    ($($t:ty),*) => {
        $(impl Arg for Option<$t> {
            fn as_addr(&self) -> Option<NonZeroU64> {
                NonZeroU64::new(self.as_ref().cloned().map(|a| a.get() as u64)?)
            }
            fn len(&self) -> u64 {
                core::mem::size_of::<$t>() as u64
            }
        })*
    };
}
option_nonzero_arg!(NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64);
