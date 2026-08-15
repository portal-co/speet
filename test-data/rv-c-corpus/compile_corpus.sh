#!/usr/bin/env bash
# Rebuild RV C corpus (mirrors c-corpus lib + programs layout).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CCORPUS="$(cd "$ROOT/../c-corpus" && pwd)"
cd "$ROOT"

LIB_INC="-I$CCORPUS/lib/common"
LIB_COMMON=(checksum.c buffer.c)
CFLAGS_FREESTANDING="-O1 -fno-vectorize -fno-slp-vectorize -fno-stack-protector -fno-ident -fno-asynchronous-unwind-tables -ffreestanding -nostdlib"
CFLAGS_LINKED="-O1 -fno-vectorize -fno-slp-vectorize -fno-stack-protector"

find_tool() {
  local _name="$1"; shift
  for c in "$@"; do [ -n "$c" ] && [ -x "$c" ] && echo "$c" && return; done
  echo ""
}

CLANG="$(find_tool "" \
  /opt/homebrew/opt/llvm/bin/clang \
  /opt/homebrew/opt/llvm@20/bin/clang \
  /usr/local/opt/llvm/bin/clang)"
[ -z "$CLANG" ] && CLANG="$(command -v clang 2>/dev/null || true)"
OBJCOPY="$(find_tool llvm-objcopy \
  /opt/homebrew/opt/llvm/bin/llvm-objcopy \
  /opt/homebrew/opt/llvm@20/bin/llvm-objcopy \
  /usr/local/opt/llvm/bin/llvm-objcopy)"
NM="$(find_tool llvm-nm \
  /opt/homebrew/opt/llvm/bin/llvm-nm \
  /opt/homebrew/opt/llvm@20/bin/llvm-nm \
  /usr/local/opt/llvm/bin/llvm-nm)"
LLVM_LINK="$(find_tool llvm-link \
  /opt/homebrew/opt/llvm/bin/llvm-link \
  /opt/homebrew/opt/llvm@20/bin/llvm-link \
  /usr/local/opt/llvm/bin/llvm-link)"

# A cross-target ELF linker (rustup ships one for its own use -- lld isn't
# per-target, so the host-triple copy links riscv32/riscv64 ELF just fine).
# Needed by build_text below to resolve PC-relative call relocations
# (R_RISCV_CALL_PLT etc) before extracting .text -- see that function's
# comment for why unlinked .o files produce corrupt call targets.
LLD_DIR=""
if command -v rustc >/dev/null 2>&1; then
  _sysroot="$(rustc --print sysroot 2>/dev/null || true)"
  if [ -n "$_sysroot" ]; then
    _host_triple="$(rustc -vV 2>/dev/null | awk '/^host:/ {print $2}')"
    _cand="$_sysroot/lib/rustlib/$_host_triple/bin/gcc-ld"
    [ -x "$_cand/ld.lld" ] && LLD_DIR="$_cand"
  fi
fi

if [ -z "$CLANG" ] || [ -z "$OBJCOPY" ] || [ -z "$NM" ] || [ -z "$LLVM_LINK" ]; then
  echo "error: need clang, llvm-objcopy, llvm-nm, llvm-link" >&2
  exit 1
fi
if [ -z "$LLD_DIR" ]; then
  echo "error: need a cross-target ld.lld (checked rustup's \$(rustc --print sysroot)/lib/rustlib/<host>/bin/gcc-ld/ld.lld)" >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
MANIFEST="$ROOT/manifest.toml"
: > "$MANIFEST"

write_entry() {
  local obj="$1" out="$2"
  local off
  off=$("$NM" "$obj" 2>/dev/null | awk '$3=="main" {print $1; exit}')
  [ -n "$off" ] || { echo "error: no main in $obj" >&2; return 1; }
  printf '0x%s\n' "$off" > "${out}.entry"
}

append_manifest() {
  local program="$1" triple="$2" arch="$3" linked="$4" text="$5" entry="$6"
  cat >> "$MANIFEST" <<EOF
[[artifact]]
program = "$program"
triple = "$triple"
linked = "$linked"
text = "$text"
entry = "$entry"
arch = "$arch"
os = "linux"

EOF
}

build_text() {
  local dir="$1" triple="$2" program="$3" main="$4"
  local base="$dir/$program"
  echo "  text $base.text.elf"
  local -a bcs=()
  for src in "${LIB_COMMON[@]}"; do
    local bc="$tmp/$(basename "${src%.c}").bc"
    "$CLANG" --target="$triple" -emit-llvm -c $CFLAGS_FREESTANDING $LIB_INC "$CCORPUS/lib/common/$src" -o "$bc"
    bcs+=("$bc")
  done
  local mbc="$tmp/${program}-main.bc"
  "$CLANG" --target="$triple" -emit-llvm -c $CFLAGS_FREESTANDING $LIB_INC "$CCORPUS/programs/$main/main.c" -o "$mbc"
  bcs+=("$mbc")
  local merged="$tmp/${program}-merged.bc"
  "$LLVM_LINK" -o "$merged" "${bcs[@]}"
  local reloc="$tmp/${program}.o"
  "$CLANG" --target="$triple" -c $CFLAGS_FREESTANDING "$merged" -o "$reloc"
  # `-c`-only compilation leaves PC-relative call/branch sites (e.g. a local
  # function call's `auipc ra,0 / jalr ra,0(ra)`) as unresolved relocations
  # (R_RISCV_CALL_PLT etc) with zeroed placeholder immediates -- the
  # assembler can't know the callee's final address without linking. The
  # old code fed this straight to `objcopy --strip-all`, which discards the
  # relocation section along with everything else, permanently losing the
  # information needed to compute the real call target: the emitted bytes
  # decode as a jump to the auipc's own PC instead of the callee. Link at a
  # fixed, zero base (matching this whole corpus's "`.text` starts at guest
  # VA 0" convention) so the linker resolves every relocation -- and often
  # relaxes auipc+jalr pairs back down to a single `jal` -- before `.text`
  # is extracted.
  # `-static` avoids a PT_DYNAMIC segment (.dynsym/.dynamic/etc, which the
  # default freestanding link otherwise emits despite nothing actually
  # being dynamically linked); `-Wl,-n` (nmagic) disables page-alignment
  # padding between segments, which otherwise inflates `.text`'s declared
  # size out to the next 4 KiB boundary and pulls the following sections'
  # bytes into the `--only-section=.text` extraction below. Together they
  # keep `.text.elf` exactly as tight as the code actually is.
  local linked="$tmp/${program}.linked.o"
  "$CLANG" --target="$triple" $CFLAGS_FREESTANDING -static \
      -B"$LLD_DIR" -fuse-ld=lld \
      -Wl,-Ttext=0x0 -Wl,--image-base=0 -Wl,--no-dynamic-linker -Wl,-e,main -Wl,-n \
      "$reloc" -o "$linked"
  write_entry "$linked" "$base"
  "$OBJCOPY" --strip-all --only-section=.text "$linked" "${base}.text.elf"
}

build_linked() {
  local dir="$1" triple="$2" program="$3"
  local linked="$dir/${program}.linked.elf"
  echo "  link $linked"
  if ! "$CLANG" --target="$triple" $CFLAGS_LINKED $LIB_INC \
      $(printf "$CCORPUS/lib/common/%s " "${LIB_COMMON[@]}") \
      "$CCORPUS/programs/${program}/main.c" -o "$linked" 2>/dev/null; then
    echo "  (skip $linked)"
    return 0
  fi
  write_entry "$linked" "$dir/$program" || return 0
  "$OBJCOPY" --strip-all --only-section=.text "$linked" "$dir/${program}.text.elf"
}

mkdir -p rv32 rv64
build_text rv32 riscv32-unknown-linux-gnu arith arith
append_manifest arith riscv32-unknown-linux-gnu rv32 rv32/arith.linked.elf rv32/arith.text.elf rv32/arith.entry
build_text rv32 riscv32-unknown-linux-gnu frame frame
append_manifest frame riscv32-unknown-linux-gnu rv32 rv32/frame.linked.elf rv32/frame.text.elf rv32/frame.entry

build_text rv64 riscv64-unknown-linux-gnu arith arith
append_manifest arith riscv64-unknown-linux-gnu rv64 rv64/arith.linked.elf rv64/arith.text.elf rv64/arith.entry
build_text rv64 riscv64-unknown-linux-gnu frame frame
append_manifest frame riscv64-unknown-linux-gnu rv64 rv64/frame.linked.elf rv64/frame.text.elf rv64/frame.entry
build_linked rv64 riscv64-unknown-linux-gnu exit42
[ -f rv64/exit42.linked.elf ] && append_manifest exit42 riscv64-unknown-linux-gnu rv64 rv64/exit42.linked.elf rv64/exit42.text.elf rv64/exit42.entry

echo "Done. manifest: $MANIFEST"
