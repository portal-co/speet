//! # WebAssembly Helper Instructions
//!
//! This crate provides pure helper functions for generating WebAssembly instruction
//! sequences for common operations. These helpers have no dependency on yecta or any
//! reactor system - they simply return vectors of instructions.
//!
//! ## Features
//!
//! - **mulh operations**: High-bits computation for 64x64->128-bit multiplication
//!   - Signed × Signed
//!   - Unsigned × Unsigned  
//!   - Signed × Unsigned
#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use wasm_encoder::Instruction;

/// Temporary variable indices used by mulh operations
///
/// These correspond to WebAssembly local variable indices that will be used
/// to store intermediate results during multiplication.
#[derive(Debug, Clone, Copy)]
pub struct MulhTemps {
    /// Temporary for storing first operand (a)
    pub temp_a: u32,
    /// Temporary for storing second operand (b)
    pub temp_b: u32,
    /// Temporary for accumulating middle terms
    pub temp_mid: u32,
}

impl MulhTemps {
    /// Create a new MulhTemps with consecutive indices starting from `base`
    pub fn new(base: u32) -> Self {
        Self {
            temp_a: base,
            temp_b: base + 1,
            temp_mid: base + 2,
        }
    }
}

/// Generate instructions for computing high 64 bits of signed 64x64 -> 128-bit multiplication
///
/// This implements the algorithm for computing the upper 64 bits of a signed multiplication
/// by breaking the operands into high and low 32-bit halves and computing partial products.
///
/// # Arguments
/// * `src1` - Local index containing first signed 64-bit operand
/// * `src2` - Local index containing second signed 64-bit operand
/// * `temps` - Temporary variable indices for intermediate results
///
/// # Stack Effect
/// - Input: (empty)
/// - Output: i64 (high 64 bits of src1 * src2)
///
/// # Algorithm
/// For signed multiplication, we compute:
/// ```text
/// result = a_hi * b_hi * 2^64 + (a_hi * b_lo + a_lo * b_hi) * 2^32
/// ```
/// Where all operations respect sign extension for signed values.
pub fn mulh_signed(src1: u32, src2: u32, temps: MulhTemps) -> Vec<Instruction<'static>> {
    let mut instrs = Vec::new();

    // Load src1 and src2 to locals for reuse
    instrs.push(Instruction::LocalGet(src1));
    instrs.push(Instruction::LocalSet(temps.temp_a));
    instrs.push(Instruction::LocalGet(src2));
    instrs.push(Instruction::LocalSet(temps.temp_b));

    // Start with a_hi * b_hi
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS); // a_hi (sign-extended)
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS); // b_hi (sign-extended)
    instrs.push(Instruction::I64Mul); // a_hi * b_hi

    // Compute middle term: a_hi * b_lo (full 64-bit result)
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS); // a_hi
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(0xFFFFFFFF));
    instrs.push(Instruction::I64And); // b_lo
    instrs.push(Instruction::I64Mul); // a_hi * b_lo (64-bit result)
    instrs.push(Instruction::LocalSet(temps.temp_mid)); // save for carry computation

    // Add high 32 bits of (a_hi * b_lo) to result
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS); // arithmetic shift for signed
    instrs.push(Instruction::I64Add);

    // Compute other middle term: a_lo * b_hi (full 64-bit result)
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(0xFFFFFFFF));
    instrs.push(Instruction::I64And); // a_lo
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS); // b_hi
    instrs.push(Instruction::I64Mul); // a_lo * b_hi (64-bit result)

    // Add it to the middle term accumulator for carry calculation
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Add); // sum of middle terms (low parts)
    instrs.push(Instruction::LocalSet(temps.temp_mid));

    // Add high 32 bits of the summed middle terms
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS); // arithmetic shift
    instrs.push(Instruction::I64Add);

    instrs
}

/// Generate instructions for computing high 64 bits of unsigned 64x64 -> 128-bit multiplication
///
/// This implements the algorithm for computing the upper 64 bits of an unsigned multiplication
/// by breaking the operands into high and low 32-bit halves and computing partial products.
///
/// # Arguments
/// * `src1` - Local index containing first unsigned 64-bit operand
/// * `src2` - Local index containing second unsigned 64-bit operand
/// * `temps` - Temporary variable indices for intermediate results
///
/// # Stack Effect
/// - Input: (empty)
/// - Output: i64 (high 64 bits of src1 * src2)
///
/// # Algorithm
/// For unsigned multiplication, we compute:
/// ```text
/// result = a_hi * b_hi * 2^64 + (a_hi * b_lo + a_lo * b_hi) * 2^32
/// ```
/// Where all operations use unsigned shifts and extensions.
pub fn mulh_unsigned(src1: u32, src2: u32, temps: MulhTemps) -> Vec<Instruction<'static>> {
    let mut instrs = Vec::new();

    // Load operands to temps for reuse
    instrs.push(Instruction::LocalGet(src1));
    instrs.push(Instruction::LocalSet(temps.temp_a));
    instrs.push(Instruction::LocalGet(src2));
    instrs.push(Instruction::LocalSet(temps.temp_b));

    // Start with a_hi * b_hi (all unsigned)
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrU); // a_hi (unsigned)
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrU); // b_hi (unsigned)
    instrs.push(Instruction::I64Mul);

    // Compute middle term: a_hi * b_lo
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrU); // a_hi
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(0xFFFFFFFF));
    instrs.push(Instruction::I64And); // b_lo
    instrs.push(Instruction::I64Mul);
    instrs.push(Instruction::LocalSet(temps.temp_mid));

    // Add high 32 bits of (a_hi * b_lo)
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrU);
    instrs.push(Instruction::I64Add);

    // Compute other middle term: a_lo * b_hi
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(0xFFFFFFFF));
    instrs.push(Instruction::I64And); // a_lo
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrU); // b_hi
    instrs.push(Instruction::I64Mul);

    // Add to middle term for carry calculation
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Add);
    instrs.push(Instruction::LocalSet(temps.temp_mid));

    // Add high 32 bits of summed middle terms
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrU);
    instrs.push(Instruction::I64Add);

    instrs
}

/// Generate instructions for computing high 64 bits of signed-unsigned 64x64 -> 128-bit multiplication
///
/// This implements the algorithm for computing the upper 64 bits of a mixed-sign multiplication
/// where the first operand is signed and the second is unsigned.
///
/// # Arguments
/// * `src1` - Local index containing first **signed** 64-bit operand
/// * `src2` - Local index containing second **unsigned** 64-bit operand
/// * `temps` - Temporary variable indices for intermediate results
///
/// # Stack Effect
/// - Input: (empty)
/// - Output: i64 (high 64 bits of src1 * src2)
///
/// # Algorithm
/// For mixed-sign multiplication, we compute:
/// ```text
/// result = a_hi * b_hi * 2^64 + (a_hi * b_lo + a_lo * b_hi) * 2^32
/// ```
/// Where src1 parts use signed operations and src2 parts use unsigned operations.
pub fn mulh_signed_unsigned(src1: u32, src2: u32, temps: MulhTemps) -> Vec<Instruction<'static>> {
    let mut instrs = Vec::new();

    // Load operands to temps
    instrs.push(Instruction::LocalGet(src1));
    instrs.push(Instruction::LocalSet(temps.temp_a));
    instrs.push(Instruction::LocalGet(src2));
    instrs.push(Instruction::LocalSet(temps.temp_b));

    // src1 is signed, src2 is unsigned

    // Start with a_hi * b_hi (a_hi signed, b_hi unsigned)
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS); // a_hi (signed)
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrU); // b_hi (unsigned)
    instrs.push(Instruction::I64Mul);

    // Compute middle term: a_hi * b_lo (a_hi signed, b_lo unsigned)
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS); // a_hi (signed)
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(0xFFFFFFFF));
    instrs.push(Instruction::I64And); // b_lo
    instrs.push(Instruction::I64Mul);
    instrs.push(Instruction::LocalSet(temps.temp_mid));

    // Add high 32 bits of (a_hi * b_lo) - use signed shift
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS);
    instrs.push(Instruction::I64Add);

    // Compute other middle term: a_lo * b_hi (a_lo unsigned, b_hi unsigned)
    instrs.push(Instruction::LocalGet(temps.temp_a));
    instrs.push(Instruction::I64Const(0xFFFFFFFF));
    instrs.push(Instruction::I64And); // a_lo
    instrs.push(Instruction::LocalGet(temps.temp_b));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrU); // b_hi (unsigned)
    instrs.push(Instruction::I64Mul);

    // Add to middle term for carry calculation
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Add);
    instrs.push(Instruction::LocalSet(temps.temp_mid));

    // Add high 32 bits of summed middle terms - use signed shift
    instrs.push(Instruction::LocalGet(temps.temp_mid));
    instrs.push(Instruction::I64Const(32));
    instrs.push(Instruction::I64ShrS);
    instrs.push(Instruction::I64Add);

    instrs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mulh_temps_new() {
        let temps = MulhTemps::new(65);
        assert_eq!(temps.temp_a, 65);
        assert_eq!(temps.temp_b, 66);
        assert_eq!(temps.temp_mid, 67);
    }

    #[test]
    fn test_mulh_signed_generates_instructions() {
        let temps = MulhTemps::new(65);
        let instrs = mulh_signed(10, 11, temps);
        assert!(!instrs.is_empty());
        // Should start by loading operands
        assert!(matches!(instrs[0], Instruction::LocalGet(10)));
    }

    #[test]
    fn test_mulh_unsigned_generates_instructions() {
        let temps = MulhTemps::new(65);
        let instrs = mulh_unsigned(10, 11, temps);
        assert!(!instrs.is_empty());
        assert!(matches!(instrs[0], Instruction::LocalGet(10)));
    }

    #[test]
    fn test_mulh_signed_unsigned_generates_instructions() {
        let temps = MulhTemps::new(65);
        let instrs = mulh_signed_unsigned(10, 11, temps);
        assert!(!instrs.is_empty());
        assert!(matches!(instrs[0], Instruction::LocalGet(10)));
    }
}
