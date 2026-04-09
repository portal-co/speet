//! [`SlotAssigner`] — controls which guest PCs receive WASM function slots.
//!
//! This trait is the fifth type parameter of [`Reactor`](crate::Reactor), giving
//! callers full control over slot allocation without touching the recompiler core.
//!
//! ## Standard implementations
//!
//! | Type | Omission | PC→slot | Use when |
//! |------|----------|---------|----------|
//! | [`PassthroughSlots`] | None | formula-based (legacy) | migrating existing code |
//! | [`FunctionCount`]    | N/A  | panics (no PCs)         | managed frontends (WASM→WASM, DEX) |
//!
//! Native-ISA recompilers should prefer `PcSlotMap` from `speet-reach`, which
//! fixes the latent PC→slot-index bug present in the formula-based approach.

/// Controls which guest instruction PCs receive WASM function slots,
/// and maps each included PC to its 0-based local slot index.
///
/// The local slot index is the offset within this binary's function allocation
/// (i.e. relative to `base_func_offset`).  The Reactor assigns these indices
/// sequentially via `next_with` in the order instructions are translated;
/// `slot_for_pc` must reflect that same order and spacing.
pub trait SlotAssigner: Send + Sync {
    /// Total WASM function slots this binary will produce.
    ///
    /// This value is passed to `FuncSchedule::push` before translation begins.
    /// It must equal the number of `next_with` calls made during translation.
    fn total_slots(&self) -> u32;

    /// Map a guest PC to its 0-based local slot index, or `None` if the
    /// instruction at this PC is omitted and must not receive a function slot.
    ///
    /// **Contract:** all PCs that will have `next_with` called on their behalf
    /// must return `Some`, in strictly increasing slot-index order matching
    /// the order they appear in the binary.
    ///
    /// For managed frontends where functions are not addressed by PC,
    /// this method may be left unimplemented (default: panic).
    fn slot_for_pc(&self, pc: u64) -> Option<u32> {
        let _ = pc;
        panic!("slot_for_pc is not applicable for this SlotAssigner");
    }
}

// ── PassthroughSlots ──────────────────────────────────────────────────────────

/// Legacy formula-based slot assigner — backward-compatible default.
///
/// Encodes the existing per-arch formula `slot = (pc - base_pc) / stride`.
/// No lookup table; no omission.  Retained only for **migration**; new
/// native-ISA code should use `PcSlotMap` from `speet-reach` instead.
///
/// # Warning
///
/// The formula is incorrect for variable-length ISAs (x86-64) and for any
/// ISA where the stride does not equal the instruction byte length.  It is
/// provided solely so that existing callers compile unchanged.
pub struct PassthroughSlots {
    /// Base address of the first instruction.
    pub base_pc: u64,
    /// Byte stride between consecutive instruction slots.
    /// - RISC-V: 2 (16-bit compressed) or 4 (standard) — *existing code uses 2*
    /// - MIPS: 4
    /// - x86-64: 1 (every byte is a potential slot — almost certainly wrong)
    pub stride: u32,
    /// Pre-counted instruction total (from `count_fns` pre-pass or byte range).
    pub count: u32,
}

impl PassthroughSlots {
    /// Construct a new `PassthroughSlots` with the given parameters.
    pub fn new(base_pc: u64, stride: u32, count: u32) -> Self {
        Self { base_pc, stride, count }
    }
}

impl Default for PassthroughSlots {
    /// Returns a sentinel `PassthroughSlots` that panics on use.
    ///
    /// This default exists solely so that `Reactor<…, PassthroughSlots>` can
    /// implement `Default`.  Calling `total_slots()` or `slot_for_pc()` on the
    /// sentinel will panic — callers must replace it before use.
    fn default() -> Self {
        Self { base_pc: 0, stride: 0, count: 0 }
    }
}

impl SlotAssigner for PassthroughSlots {
    fn total_slots(&self) -> u32 {
        self.count
    }

    fn slot_for_pc(&self, pc: u64) -> Option<u32> {
        if self.stride == 0 {
            panic!("PassthroughSlots: not initialized (stride == 0)");
        }
        let offset = pc.wrapping_sub(self.base_pc);
        Some((offset / self.stride as u64) as u32)
    }
}

// ── FunctionCount ─────────────────────────────────────────────────────────────

/// Slot assigner for managed frontends (WASM→WASM, DEX, etc.).
///
/// Functions are not addressed by guest PC in managed frontends; only a total
/// count is needed.  `slot_for_pc` panics — it should never be called.
pub struct FunctionCount(pub u32);

impl SlotAssigner for FunctionCount {
    fn total_slots(&self) -> u32 {
        self.0
    }
    // slot_for_pc: uses default impl (panics) — correct for managed frontends.
}
