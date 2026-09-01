//! Hand-encodes the subset of DEX instructions used by the test fixtures,
//! matching `dex_bytecode::decode`'s bit layout exactly (verified against
//! `crates/managed/dex-bytecode/src/lib.rs`).

#![allow(dead_code)]

pub fn nop() -> Vec<u16> {
    vec![0x0000]
}

pub fn return_void() -> Vec<u16> {
    vec![0x000e]
}

/// `return vAA`
pub fn return_reg(src: u8) -> Vec<u16> {
    vec![0x000f | ((src as u16) << 8)]
}

/// `return-object vAA`
pub fn return_object(src: u8) -> Vec<u16> {
    vec![0x0011 | ((src as u16) << 8)]
}

/// `const/4 vA, #+B` — B is a signed 4-bit immediate (-8..=7).
pub fn const4(dst: u8, value: i8) -> Vec<u16> {
    assert!((-8..=7).contains(&value));
    let nibble = (value as u16) & 0xf;
    vec![0x0012 | ((dst as u16) << 8) | (nibble << 12)]
}

/// `const/16 vAA, #+BBBB`
pub fn const16(dst: u8, value: i16) -> Vec<u16> {
    vec![0x0013 | ((dst as u16) << 8), value as u16]
}

/// `move-object vA, vB`
pub fn move_object(dst: u8, src: u8) -> Vec<u16> {
    vec![0x0007 | ((dst as u16) << 8) | ((src as u16) << 12)]
}

/// `if-eqz vAA, +BBBB` (code-unit offset, relative to this instruction)
pub fn if_eqz(reg: u8, offset: i16) -> Vec<u16> {
    vec![0x0038 | ((reg as u16) << 8), offset as u16]
}

/// `if-nez vAA, +BBBB`
pub fn if_nez(reg: u8, offset: i16) -> Vec<u16> {
    vec![0x0039 | ((reg as u16) << 8), offset as u16]
}

/// `goto +AA` (8-bit signed code-unit offset)
pub fn goto(offset: i8) -> Vec<u16> {
    vec![0x0028 | (((offset as u8) as u16) << 8)]
}

/// `add-int/2addr vA, vB`
pub fn add_int_2addr(dst: u8, src: u8) -> Vec<u16> {
    vec![0x00b0 | ((dst as u16) << 8) | ((src as u16) << 12)]
}

/// `new-instance vAA, type@BBBB`
pub fn new_instance(dst: u8, type_idx: u16) -> Vec<u16> {
    vec![0x0022 | ((dst as u16) << 8), type_idx]
}

/// `new-array vA, vB, type@CCCC`
pub fn new_array(dst: u8, size_reg: u8, type_idx: u16) -> Vec<u16> {
    vec![0x0023 | ((dst as u16) << 8) | ((size_reg as u16) << 12), type_idx]
}

/// `iget vA, vB, field@CCCC`
pub fn iget(dst: u8, obj: u8, field_idx: u16) -> Vec<u16> {
    vec![0x0052 | ((dst as u16) << 8) | ((obj as u16) << 12), field_idx]
}

/// `iput vA, vB, field@CCCC`
pub fn iput(src: u8, obj: u8, field_idx: u16) -> Vec<u16> {
    vec![0x0059 | ((src as u16) << 8) | ((obj as u16) << 12), field_idx]
}

/// `array-length vA, vB`
pub fn array_length(dst: u8, array: u8) -> Vec<u16> {
    vec![0x0021 | ((dst as u16) << 8) | ((array as u16) << 12)]
}

/// `aget vAA, vBB, vCC`
pub fn aget(dst: u8, array: u8, index: u8) -> Vec<u16> {
    vec![0x0044 | ((dst as u16) << 8), (array as u16) | ((index as u16) << 8)]
}

/// `aput vAA, vBB, vCC`
pub fn aput(src: u8, array: u8, index: u8) -> Vec<u16> {
    vec![0x004b | ((src as u16) << 8), (array as u16) | ((index as u16) << 8)]
}

/// `instance-of vA, vB, type@CCCC`
pub fn instance_of(dst: u8, obj: u8, type_idx: u16) -> Vec<u16> {
    vec![0x0020 | ((dst as u16) << 8) | ((obj as u16) << 12), type_idx]
}

/// `check-cast vAA, type@BBBB`
pub fn check_cast(reg: u8, type_idx: u16) -> Vec<u16> {
    vec![0x001f | ((reg as u16) << 8), type_idx]
}

/// Concatenate several encoded instructions into one flat code-unit stream.
pub fn assemble(insns: &[Vec<u16>]) -> Vec<u16> {
    insns.iter().flat_map(|i| i.iter().copied()).collect()
}
