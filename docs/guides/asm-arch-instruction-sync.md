# asm-arch ↔ speet Instruction Sync Guide

**Crates:** `crates/native/speet-x86_64`, `crates/native/speet-aarch64`, `crates/native/speet-riscv` (RV32), `crates/native/speet-arm`, `crates/native/speet-x86`
**External dependency:** `asm-arch` (sibling workspace at `/Users/g/Code-local/portal-hot/asm-arch`)
**Design doc:** [recompiler-guide.md](../recompiler-guide.md)

---

## 1. Why this sync exists

`asm-arch` is the machine-code **emitter** that wasm-blitz lowers WASM → native through (each
arch's `WriterCore` trait, e.g. `asm-x86-64/src/out.rs`, `asm-aarch64/src/out.rs`). speet's
per-arch crates here are the **frontend**: they decode native machine code and lower it to WASM.

These two halves should round-trip: any instruction family asm-arch's backend can *emit* into a
native binary, speet's frontend must be able to *ingest* when decoding that binary back into
WASM. If asm-arch grows a new `WriterCore` method and speet's frontend doesn't grow a matching
decode arm, the pipeline develops a one-way hole — a native binary containing that instruction
(itself possibly produced by a previous wasm-blitz pass) becomes silently unrecompilable.

**Scope note:** this is sync to asm-arch's emission surface, not general x86/ARM ISA
completeness. speet's frontends already decode plenty of guest instructions with no asm-arch
counterpart (e.g. x86 `Andnpd`/`Pand`/`Por`/`Test`) — that's fine and expected; it's extra
real-world guest coverage, not part of this invariant. The invariant only runs one direction:
**every asm-arch `WriterCore` method must have a corresponding speet decode arm.** The reverse is
not required.

## 2. How to detect drift

1. Enumerate the current `WriterCore` method list for each arch:
   ```
   grep -n "    fn [a-z_0-9]*(" asm-arch/crates/asm-x86-64/src/out.rs   # x86_64 (or out/out.rs, check layout)
   grep -n "    fn [a-z_0-9]*(" asm-arch/crates/asm-aarch64/src/out.rs  # aarch64
   ```
   Skip the backend-internal ones: `db`, `current_offset`, `align_to`, `set_label`,
   `*_label` (label/relocation emission, not a decodable instruction), `get_ip`.
2. For each method, find the real instruction(s) it encodes — read the binary writer impl
   (`asm-x86-64/src/out/iced.rs` / `asm-aarch64/src/out/bin.rs`) to see exactly which
   `iced_x86::Code` / ARM encoding it emits. **Don't trust the method name alone** — e.g. x86's
   `mul` method does *not* emit the literal one-operand `MUL` instruction; it emits 2-operand
   `IMUL r, r/m` (truncating multiply, matching WASM `i32.mul`/`i64.mul` semantics). Always
   verify against the actual emitted encoding before concluding something is a gap.
3. Check the corresponding speet frontend dispatch recognizes that instruction:
   - x86_64: grep `Mnemonic::` arms in `speet-x86_64/src/direct.rs`.
   - aarch64: grep the per-`Operation`-class match arms across `speet-aarch64/src/direct.rs`
     and `direct/{alu,fp,mem}.rs` (dispatch is two-level: `Operation::CLASS` routes to a
     `translate_*` function, which then matches the disarm64 enum variants within that class).
   - Confirm with a decode-coverage unit test: decode real assembled bytes for the instruction,
     translate, and assert `unsupported_insns()` is empty (see §4).
4. Update the matrix in §3 with any new/changed rows.

**disarm64 gotcha (aarch64 only):** disarm64 sometimes merges several real-instruction width
combinations into a *single* enum variant, distinguishable only by raw instruction bits — not by
separate variants. This bit speet once already (the `SCVTF`/`UCVTF`/`FCVTZS`/`FCVTZU` bug fixed
alongside this guide: all four `{s,d} × {w,x}` combos shared one enum variant each, and the
handler had hardcoded the narrowest combo). When adding a new aarch64 decode arm, check the
variant's qualifier list in `disarm64`'s generated `decoder_full.rs` (search by mask/base) before
assuming "one enum variant = one instruction form."

## 3. Per-arch coverage matrix

Status legend: **✅** decoded and translated · **N/A** backend-internal, no guest instruction ·
**⚠** decoded but with a known semantic limitation (see footnote).

### x86_64 (`speet-x86_64/src/direct.rs`)

| `WriterCore` method | Guest mnemonic(s) | Status |
|---|---|---|
| `hlt` | `Hlt` | ✅ |
| `xchg` | `Xchg` | ⚠ [^xchg] |
| `mov` / `mov64` | `Mov` (incl. `movabs` imm64 form) | ✅ |
| `sub` | `Sub` | ✅ |
| `add` | `Add` | ✅ |
| `movsx` | `Movsx` | ✅ |
| `movzx` | `Movzx` | ✅ |
| `push` | `Push` | ✅ |
| `pop` | `Pop` | ✅ |
| `pushf` | `Pushf`/`Pushfd`/`Pushfq` | ✅ |
| `popf` | `Popf`/`Popfd`/`Popfq` | ✅ |
| `call` | `Call` | ✅ |
| `jmp` | `Jmp` | ✅ |
| `cmp` / `cmp0` | `Cmp` (+ `Test`) | ✅ |
| `cmovcc` | `Cmovcc` (all condition codes) | ✅ |
| `not` | `Not` | ✅ |
| `lea` | `Lea` | ✅ |
| `get_ip` | — | N/A (backend RIP-relative literal addressing, no guest form) |
| `ret` | `Ret` | ✅ |
| `mul` | *(none — see §2 step 2)* | N/A — emits 2-operand truncating `IMUL`, already covered by the `Imul` arm below |
| `div` | `Div` | ✅ |
| `idiv` | `Idiv` | ✅ |
| `and` | `And` | ✅ |
| `or` | `Or` | ✅ |
| `eor` | `Xor` | ✅ |
| `shl` | `Shl` | ✅ |
| `shr` | `Shr` | ✅ |
| `sar` | `Sar` | ✅ |
| `fadd`/`fsub`/`fmul`/`fdiv` | `Addsd`/`Subsd`/`Mulsd`/`Divsd` | ✅ |
| `fadd_s`/`fsub_s`/`fmul_s`/`fdiv_s` | `Addss`/`Subss`/`Mulss`/`Divss` | ✅ |
| `fmov` | `Movsd`/`Movapd`/`Movq` | ✅ |
| `fmov_s` | `Movss`/`Movd` | ✅ |
| `fmin`/`fmax` | `Minsd`/`Maxsd` | ✅ |
| `fmin_s`/`fmax_s` | `Minss`/`Maxss` | ✅ |
| `fsqrt`/`fsqrt_s` | `Sqrtsd`/`Sqrtss` | ✅ |
| `fcvt_s_d` | `Cvtss2sd` | ✅ |
| `fcvt_d_s` | `Cvtsd2ss` | ✅ |
| `fmov_gp_to_d`/`fmov_d_to_gp` | `Movq` (xmm↔gpr, 64-bit) | ✅ |
| `fmov_gp_to_s`/`fmov_s_to_gp` | `Movd` (xmm↔gpr, 32-bit) | ✅ |
| `fcmp` | `Ucomisd`/`Comisd` | ✅ |
| `fcmp_s` | `Ucomiss`/`Comiss` | ✅ |
| `scvtf_d_x`/`scvtf_d_w` | `Cvtsi2sd` (64-/32-bit GPR src) | ✅ |
| `scvtf_s_x`/`scvtf_s_w` | `Cvtsi2ss` (64-/32-bit GPR src) | ✅ |
| `fcvtzs_x_d`/`fcvtzs_w_d` | `Cvttsd2si` (64-/32-bit dst) | ✅ |
| `fcvtzs_x_s`/`fcvtzs_w_s` | `Cvttss2si` (64-/32-bit dst) | ✅ |
| `db`/`current_offset`/`align_to`/`*_label` | — | N/A (backend-internal) |

x86 has **no** `ucvtf`/`fcvtzu_*` `WriterCore` methods (SSE2 has no native unsigned scalar
convert) — there is intentionally no matching speet decode requirement for unsigned int↔FP on
x86. Don't add one looking for "parity" with aarch64; there's nothing on the asm-arch side to
sync to.

Register model: XMM0–15 are stored as raw-bits `i64` locals (`xmm_slot`, `BASE_PARAMS = 42`, see
`speet-x86_64/src/lib.rs`). FP handlers reinterpret i64→f64/f32 around native WASM FP ops and
back, preserving exact bits (NaN payloads, f32 low half). This deliberately differs from
aarch64's model (next section).

### aarch64 (`speet-aarch64/src/direct.rs` + `direct/{alu,fp,mem}.rs`)

| `WriterCore` method | Guest mnemonic(s) / disarm64 class | Status |
|---|---|---|
| `mov`/`mov_imm` | `MOVZ`/`MOVN`/`MOVK` (`MOVEWIDE`), `ORR` alias forms | ✅ |
| `sub`/`add`/`add_uxtw` | `ADDSUB_IMM`/`ADDSUB_SHIFT`/`ADDSUB_EXT` | ✅ |
| `sxt`/`uxt` | `BITFIELD` (SBFM/UBFM aliases) | ✅ |
| `str`/`ldr` | `LDST_POS`/`LDST_IMM9`/`LDST_REGOFF` | ✅ |
| `stp`/`ldp` | `LDSTPAIR_OFF`/`LDSTPAIR_INDEXED` | ✅ |
| `bl` | `BRANCH_IMM::BL_ADDR_PCREL26` | ✅ |
| `br` | `BRANCH_REG::BR_Rn` (+ `BLR_Rn`/`RET_Rn` handled alongside) | ✅ |
| `b` | `BRANCH_IMM::B_ADDR_PCREL26` | ✅ |
| `cmp` | `ADDSUB_*` (CMP = SUBS alias, flags-only) | ✅ |
| `csel` | `CONDSEL::CSEL/CSINC/CSINV/CSNEG` | ✅ |
| `bcond` | `CONDBRANCH`, `COMPBRANCH` (CBZ/CBNZ) | ✅ |
| `and`/`orr`/`eor` | `LOG_SHIFT`/`LOG_IMM` | ✅ |
| `lsl`/`lsr`/`asr` | `DP_2SRC::LSLV/LSRV/ASRV` (+ `BITFIELD` immediate forms) | ✅ |
| `mvn` | `LOG_SHIFT` (ORN with XZR) | ✅ |
| `adr` | `PCRELADDR` | ✅ |
| `ret` | `BRANCH_REG::RET_Rn` | ✅ |
| `mrs_nzcv`/`msr_nzcv` | `IC_SYSTEM::MRS_Rt_SYSREG`/`MSR_SYSREG_Rt` | ✅ |
| `mul` | `DP_3SRC::MADD_Rd_Rn_Rm_Ra` (MUL = alias, Ra=XZR) | ✅ |
| `udiv`/`sdiv` | `DP_2SRC::UDIV/SDIV` | ✅ |
| `fadd`/`fsub`/`fmul`/`fdiv` (+ `_s`) | `FLOATDP2::FADD/FSUB/FMUL/FDIV` (D and S variants) | ⚠ [^floatdp] |
| `fmov`/`fmov_s` | `FLOATDP1::FMOV_Fd_Fn` (D/S) | ✅ |
| `fmax`/`fmin`/`fmax_s`/`fmin_s` | `FLOATDP2::FMAX/FMIN/FMAXNM/FMINNM` | ⚠ [^floatdp] |
| `fmov_gp_to_d`/`fmov_d_to_gp` | `FLOAT2INT` `FMOV` Xn↔Dn forms | ✅ |
| `fmov_gp_to_s`/`fmov_s_to_gp` | `FLOAT2INT` `FMOV` Wn↔Sn forms | ✅ |
| `fsqrt`/`fabs`/`fneg` (+ `_s`) | `FLOATDP1::FSQRT/FABS/FNEG` (D/S) | ⚠ [^floatdp] (fsqrt only; fabs/fneg are bit ops, exact) |
| `scvtf_d_x`/`scvtf_d_w`/`scvtf_s_x`/`scvtf_s_w` | `FLOAT2INT::SCVTF_Fd_S_D_Rn_W` (merged variant, branch on `sf`/`ftype`) | ✅ |
| `ucvtf_d_x`/`ucvtf_d_w`/`ucvtf_s_x`/`ucvtf_s_w` | `FLOAT2INT::UCVTF_Fd_S_D_Rn_W` (merged variant) | ✅ |
| `fcvtzs_x_d`/`fcvtzs_w_d`/`fcvtzs_x_s`/`fcvtzs_w_s` | `FLOAT2INT::FCVTZS_Rd_W_Fn_S_D` (merged variant) | ✅ |
| `fcvtzu_x_d`/`fcvtzu_w_d`/`fcvtzu_x_s`/`fcvtzu_w_s` | `FLOAT2INT::FCVTZU_Rd_W_Fn_S_D` (merged variant) | ✅ |
| `fcvt_s_d`/`fcvt_d_s` | `FLOATDP1::FCVT_Fd_Fn` | ✅ |
| `fcmp`/`fcmp_s` | `FLOATCMP::FCMP*` (incl. `FPIMM0` zero-compare forms) | ✅ |
| `brk` | `EXCEPTION::BRK_EXCEPTION` | ✅ |
| `current_offset`/`align_to`/`set_label`/`adr_label`/`adrp_label`/`add_lo12_label`/`b_label`/`bcond_label`/`bl_label` | — | N/A (backend-internal) |

Register model: V0–V31 are stored as `f64` locals (`fp_slot`), always — even for
single-precision values, which are kept exactly-promoted to f64. This deliberately differs from
x86's raw-bits model (previous section); see [^floatdp] for the consequence.

[^floatdp]: **Known limitation, not fixed by this guide's audit.** Because V-registers are
    stored as f64 always, single-precision `FLOATDP1`/`FLOATDP2`/`FLOATDP3` arithmetic
    (`FADD`/`FSUB`/`FMUL`/`FDIV`/`FNMUL`/`FMIN`/`FMAX`/`FSQRT`/`FMADD`/etc. in their S-register
    forms) runs the *same* full-f64 WASM op for both S and D forms — it does **not**
    demote/promote through f32 for the single-precision case. Real hardware would round once at
    f32 precision; this lowering effectively computes at f64 precision and only narrows (via the
    eventual single-precision read) at the very end, which can disagree with hardware in the low
    mantissa bits/rounding for single-precision arithmetic chains. By contrast,
    `FLOAT2INT` (the GPR/FP boundary — `SCVTF`/`UCVTF`/`FCVTZS`/`FCVTZU`/`FMOV` gp↔fp) *does*
    correctly demote/promote through f32 where genuinely single-precision, since that's the one
    place bit-exactness is cheap to get right with this storage model. Closing the
    `FLOATDP*` gap properly would need each single-precision V-reg op to demote to f32, operate,
    and promote back — a larger rearchitecture than a drive-by fix; out of scope for this sync
    pass. Tracked as a known approximation, not a regression.

[^xchg]: `Xchg` is decoded and translated, but the existing implementation clobbers the
    `ZF_LOCAL` flag local as scratch space during the swap — a pre-existing bug independent of
    this sync (not introduced or fixed by it). Flagged here so the next person auditing flag
    correctness doesn't have to rediscover it.

### ILP32 pairs (Phase 1–4 thin surface)

These pairs are first-class for the sync invariant. Emitter crates are thinner than their
64-bit twins; the bar is “every `WriterCore` method the ILP32 asm crate exposes has a speet
decode arm,” not parity with aarch64/x86_64 matrices above.

#### riscv32 (`asm-riscv32` ↔ `speet-riscv` with `Xlen::Rv32`)

| `WriterCore` method | Guest / notes | Status |
|---|---|---|
| `mv` / `add` / `addi` / `li` | RV32I integer | ✅ (via `speet-riscv`) |
| `lw` / `sw` / `lb`/`sb`/`lh`/`sh` | loads/stores | ✅ |
| `ld` / `sd` | soft 8-byte WASM slots (two words / low-word) | ⚠ emitter soft-expands; guest RV32 has no `ld`/`sd` |
| `jal` / `jalr` / `call` / `ret` | control + Flag on jal/ret | ✅ |
| branches (`beq`/…) | via `Writer` label helpers | ✅ |

#### arm32 (`asm-arm` ↔ `speet-arm`)

| `WriterCore` method | Guest / notes | Status |
|---|---|---|
| `mov` / `mov_imm` / `add` / `sub` | A32 DP | ✅ (thin) |
| `and` / `orr` / `eor` / `cmp` | A32 DP | ✅ / expand |
| `ldr` / `str` | imm offset | ✅ (thin) |
| `b` / `bl` / `bx` / `ret` | Flag on BL / BX lr | ✅ |
| Thumb-2 emit | out of scope for asm-arm (A32 emit only) | N/A emitter; speet Thumb decode stubbed |

#### x86-32 (`asm-x86` ↔ `speet-x86`)

| `WriterCore` method | Guest / notes | Status |
|---|---|---|
| `mov` / `add` / `sub` / `and` / `or` / `xor` | i686 integer | ✅ (thin) |
| `push` / `pop` / `leave` | stack | ✅ |
| `lea` / `cmp` | address / flags | ✅ / Jcc expand |
| `call` / `jmp` / `ret` | Flag on call/ret | ✅ |

Drift detection for ILP32: grep `WriterCore` methods in
`asm-arch/crates/asm-{riscv32,arm,x86}/src/out.rs` and match against
`speet-{riscv,arm,x86}` decode paths (RV32 shares `speet-riscv`).

## 4. Verification bar for sync work

This subgoal's precedent (and the bar future sync work should match, absent new infra):

- **Decode-coverage unit tests** — decode real assembled instruction bytes (via `clang` on the
  matching host arch, or hand-encoded), translate through the recompiler, and assert
  `unsupported_insns()` is empty (no silent fallback to `Unreachable`). See
  `speet-x86_64/tests/int_gap_tests.rs`, `speet-aarch64/tests/fp_cvt_width_tests.rs` for the
  pattern (`assert_all_supported`).
- **Manual encoding analysis** for bit-level correctness questions (e.g. the disarm64
  merged-variant cases) — read the generated decoder's mask/qualifier metadata directly rather
  than guessing from the variant name.
- **Execution-correctness (wasmi numeric results) is not yet wired for x86/aarch64 corpus
  tests.** `speet-e2e`'s `run_module()` harness discards the WASM `Store` after execution and has
  no register-readback mechanism for these two arches (the `__speet_hint`/`REG_SAVE_BASE`
  snapshot mechanism is RISC-V-specific). If you need to verify *numeric* correctness (not just
  "decodes without falling through"), you currently have to build that infra first — it doesn't
  exist as a drop-in. Decode-coverage + manual analysis is the accepted bar until someone invests
  in that.

## 5. Checklist — adding a new instruction

1. Find which `WriterCore` method(s) it corresponds to and read the **binary** writer impl
   (not just the trait signature) to pin down the exact encoding/semantics (§2 step 2).
2. x86: add a `Mnemonic::X => { ... }` arm in `speet-x86_64/src/direct.rs`. If it needs a new
   register class (like the X1 phase's XMM file), extend the register model in
   `speet-x86_64/src/lib.rs` (`BASE_PARAMS`, the slot layout, `resolve_*`) first.
   aarch64: find or add the `Operation::CLASS` arm in `direct.rs`, then add the disarm64 variant
   match in the relevant `direct/{alu,fp,mem}.rs` `translate_*` function. **Check disarm64's
   `decoder_full.rs` for merged variants before assuming one variant = one width/form (§2).**
3. Add a decode-coverage unit test (`assert_all_supported` pattern, §4) covering every distinct
   encoding form/width the new arm handles.
4. If it's part of a larger corpus-driven family, add/extend a `test-data/*-corpus/*.s` file and
   regenerate via `generate_tests.py`.
5. Update the matrix in §3.
6. `cargo build`/`cargo test` the affected crate(s); run the relevant `speet-e2e` cross-matrix
   tests (`cargo test -p speet-e2e --test e2e -- <arch>`) to confirm no regression.
