//! Width, load-kind, and store-kind enumerations shared by all architecture
//! recompilers.
//!
//! The `MemoryEmitter` struct that used to live here has been removed; all
//! load/store code generation is now handled by [`crate::mapper::DirectMemory`]
//! (backed by a [`crate::mapper::MemoryAccess`] implementation) together with
//! [`speet_ordering::MemorySink`].

// ── Width helpers ──────────────────────────────────────────────────────────────

/// Whether the guest address space is 32-bit or 64-bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressWidth {
    /// 32-bit guest registers; addresses are `i32` on the wasm stack.
    W32,
    /// 64-bit guest registers; guest addresses are `i64` and may need wrapping
    /// to `i32` when `memory64` is *not* enabled.
    W64 {
        /// When `true` the Wasm memory uses 64-bit addressing (memory64
        /// proposal); no `I32WrapI64` is emitted.  When `false` the address
        /// must be wrapped before use.
        memory64: bool,
    },
}

impl AddressWidth {
    /// `true` when the native register width is 64 bits.
    #[inline]
    pub fn is_64(&self) -> bool {
        matches!(self, Self::W64 { .. })
    }

    /// `true` when wasm memory instructions use 64-bit (`i64`) addresses.
    #[inline]
    pub fn memory64(&self) -> bool {
        matches!(self, Self::W64 { memory64: true })
    }
}

/// Width of the integer register file — used when choosing sign-/zero-extend
/// sequences after narrow loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntWidth {
    I32,
    I64,
}

// ── Load/Store descriptors ─────────────────────────────────────────────────────

/// What memory load to perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadKind {
    /// Load a signed 8-bit value, sign-extended.
    I8S,
    /// Load an unsigned 8-bit value, zero-extended.
    I8U,
    /// Load a signed 16-bit value, sign-extended.
    I16S,
    /// Load an unsigned 16-bit value, zero-extended.
    I16U,
    /// Load a 32-bit value (sign-extended when target register is 64-bit).
    I32S,
    /// Load a 32-bit value, zero-extended (RV64 `LWU`-style).
    I32U,
    /// Load a 64-bit value.
    I64,
    /// Load a 32-bit float, promoting to f64.
    F32,
    /// Load a 64-bit float.
    F64,
}

/// What memory store to perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreKind {
    I8,
    I16,
    I32,
    I64,
    /// Store f64 demoted to f32.
    F32,
    F64,
}


