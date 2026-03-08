//! SHA3-256 type-hash for runtime object identification.

use tiny_keccak::{Hasher, Sha3};

/// A 32-byte SHA3-256 type hash that identifies a class at runtime.
///
/// Every heap object carries a `TypeHash` in its header (at offset 0).
///
/// ## Hash values
///
/// - **Reference types**: SHA3-256 of the fully-qualified class name *without*
///   array-dimension brackets (e.g. `"java/lang/String"`, `"com/example/Foo"`).
/// - **Primitive array element types**: all bytes are zero except the *last*
///   byte, which holds the [`PrimitiveType`] discriminant (1–8).
///
/// This encoding means that a `String[]` and `int[]` are distinguishable:
/// the former hashes `"java/lang/String"` and carries `array_dim = 1`, while
/// the latter uses the [`PrimitiveType::Int`] sentinel with `array_dim = 1`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TypeHash(pub [u8; 32]);

impl TypeHash {
    /// Compute the SHA3-256 hash of `name`.
    ///
    /// `name` should be the fully-qualified class name without array brackets
    /// and without the `L`/`;` descriptors used in JVM/DEX type signatures.
    /// For example, pass `"java/lang/String"` rather than `"Ljava/lang/String;"`.
    pub fn of_class(name: &str) -> TypeHash {
        let mut sha3 = Sha3::v256();
        sha3.update(name.as_bytes());
        let mut out = [0u8; 32];
        sha3.finalize(&mut out);
        TypeHash(out)
    }

    /// Return the sentinel hash for a primitive array element type.
    ///
    /// The result is all-zero bytes except the last byte, which is set to the
    /// [`PrimitiveType`] discriminant value (1–8).
    pub const fn primitive(p: PrimitiveType) -> TypeHash {
        let mut bytes = [0u8; 32];
        bytes[31] = p as u8;
        TypeHash(bytes)
    }

    /// Split the 32-byte hash into four little-endian `i64` chunks.
    ///
    /// Used when passing the hash to a wasm allocator function as four
    /// separate `i64` arguments (avoids storing the hash in linear memory
    /// just for an allocation call).
    pub fn as_i64_chunks(&self) -> [i64; 4] {
        let b = &self.0;
        [
            i64::from_le_bytes([b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]]),
            i64::from_le_bytes([b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]]),
            i64::from_le_bytes([b[16],b[17],b[18],b[19],b[20],b[21],b[22],b[23]]),
            i64::from_le_bytes([b[24],b[25],b[26],b[27],b[28],b[29],b[30],b[31]]),
        ]
    }

    /// Split the 32-byte hash into eight little-endian `i32` chunks.
    pub fn as_i32_chunks(&self) -> [i32; 8] {
        let b = &self.0;
        [
            i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            i32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            i32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            i32::from_le_bytes([b[12], b[13], b[14], b[15]]),
            i32::from_le_bytes([b[16], b[17], b[18], b[19]]),
            i32::from_le_bytes([b[20], b[21], b[22], b[23]]),
            i32::from_le_bytes([b[24], b[25], b[26], b[27]]),
            i32::from_le_bytes([b[28], b[29], b[30], b[31]]),
        ]
    }
}

/// Discriminant for primitive element types used in typed arrays.
///
/// Stored in the last byte of a [`TypeHash`] sentinel (all other bytes are
/// zero), allowing the runtime to distinguish e.g. `int[]` from `float[]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PrimitiveType {
    Boolean = 1,
    Byte    = 2,
    Char    = 3,
    Short   = 4,
    Int     = 5,
    Long    = 6,
    Float   = 7,
    Double  = 8,
}
