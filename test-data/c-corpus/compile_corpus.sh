#!/usr/bin/env bash
# Rebuild C corpus: multi-TU programs → linked guests + .text.elf + manifest.toml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

LIB_INC="-I$ROOT/lib/common"
LIB_COMMON=(lib/common/checksum.c lib/common/buffer.c)

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
LD="$(find_tool ld.lld \
  /opt/homebrew/opt/llvm/bin/ld.lld \
  /opt/homebrew/opt/llvm@20/bin/ld.lld \
  /usr/local/opt/llvm/bin/ld.lld)"
if [ -z "$LD" ] && [ -n "$CLANG" ]; then
  _lld="$(dirname "$CLANG")/ld.lld"
  [ -x "$_lld" ] && LD="$_lld"
fi

LLVM_LINK="$(find_tool llvm-link \
  /opt/homebrew/opt/llvm/bin/llvm-link \
  /opt/homebrew/opt/llvm@20/bin/llvm-link \
  /usr/local/opt/llvm/bin/llvm-link)"
if [ -z "$LLVM_LINK" ] && [ -n "$CLANG" ]; then
  _ll="$(dirname "$CLANG")/llvm-link"
  [ -x "$_ll" ] && LLVM_LINK="$_ll"
fi

if [ -z "$CLANG" ] || [ -z "$OBJCOPY" ] || [ -z "$NM" ]; then
  echo "error: need clang, llvm-objcopy, llvm-nm" >&2
  exit 1
fi
if [ -z "$LLVM_LINK" ]; then
  echo "error: need llvm-link for freestanding multi-TU merge" >&2
  exit 1
fi

SDK=""
if [ "$(uname -s)" = "Darwin" ]; then
  SDK="$(xcrun --show-sdk-path 2>/dev/null || true)"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
MANIFEST="$ROOT/manifest.toml"
: > "$MANIFEST"

# main symbol offset within .text (hex, no 0x prefix) → ${out}.entry
write_entry() {
  local obj="$1" out="$2"
  local off
  off=$("$NM" "$obj" 2>/dev/null | awk '$3=="main" || $3=="_main" {print $1; exit}')
  if [ -z "$off" ]; then
    echo "error: no main symbol in $obj" >&2
    return 1
  fi
  printf '0x%s\n' "$off" > "${out}.entry"
}

append_manifest() {
  local program="$1" triple="$2" arch="$3" os="$4" linked="$5" text="$6" entry="$7"
  cat >> "$MANIFEST" <<EOF
[[artifact]]
program = "$program"
triple = "$triple"
linked = "$linked"
text = "$text"
entry = "$entry"
arch = "$arch"
os = "$os"

EOF
}

# Freestanding multi-TU → .text.elf (recompiler input)
build_freestanding_text() {
  local out_dir="$1" triple="$2" program="$3"
  shift 3
  local -a extra_srcs=("$@")
  local base="$out_dir/$program"
  echo "  text $base.text.elf"
  local -a objs=()
  local -a bcs=()
  for src in "${LIB_COMMON[@]}" "${extra_srcs[@]}"; do
    local bc="$tmp/$(basename "${src%.c}").bc"
    "$CLANG" --target="$triple" -emit-llvm -c $CFLAGS_FREESTANDING $LIB_INC "$ROOT/$src" -o "$bc"
    bcs+=("$bc")
  done
  local merged_bc="$tmp/${program}-merged.bc"
  "$LLVM_LINK" -o "$merged_bc" "${bcs[@]}"
  local reloc="$tmp/${program}-reloc.o"
  "$CLANG" --target="$triple" -c $CFLAGS_FREESTANDING "$merged_bc" -o "$reloc"
  write_entry "$reloc" "$base"
  "$OBJCOPY" --strip-all --only-section=.text "$reloc" "${base}.text.elf"
}

# Libc-linked guest → .linked.elf/.macho + .text.elf + .entry
build_linked_program() {
  local out_dir="$1" triple="$2" program="$3" out_name="$4"
  shift 4
  local -a extra_srcs=("$@")
  local linked="$out_dir/${out_name}.linked.elf"
  local text="$out_dir/${out_name}.text.elf"
  echo "  link $linked"
  local -a sdk_args=()
  [ -n "$SDK" ] && sdk_args=(-isysroot "$SDK")
  if ! "$CLANG" --target="$triple" "${sdk_args[@]}" $CFLAGS_LINKED $LIB_INC \
      "${LIB_COMMON[@]/#/$ROOT/}" "$ROOT/programs/$program/main.c" \
      $(printf ' %s' "${extra_srcs[@]/#/$ROOT/}") \
      -o "$linked" 2>/dev/null; then
    echo "  (skip $linked: link failed — need target sysroot)"
    return 0
  fi
  write_entry "$linked" "$out_dir/$out_name" || return 0
  "$OBJCOPY" --strip-all --only-section=.text "$linked" "$text"
}

build_linked_macho() {
  local out_dir="$1" triple="$2" arch_flag="$3" program="$4" out_name="$5"
  local linked="$out_dir/${out_name}.linked.macho"
  local text="$out_dir/${out_name}.text.elf"
  echo "  macho $linked"
  local -a sdk_args=()
  [ -n "$SDK" ] && sdk_args=(-isysroot "$SDK")
  if ! "$CLANG" --target="$triple" -arch "$arch_flag" -mmacosx-version-min=11.0 \
      "${sdk_args[@]}" $CFLAGS_LINKED $LIB_INC \
      "${LIB_COMMON[@]/#/$ROOT/}" "$ROOT/programs/$program/main.c" \
      -o "$linked" 2>/dev/null; then
    echo "  (skip $linked: macho link failed)"
    return 0
  fi
  write_entry "$linked" "$out_dir/$out_name" || return 0
  "$OBJCOPY" --strip-all --only-section=.text "$linked" "$text"
  # Legacy alias for thin-runtime tests
  cp "$linked" "$out_dir/${out_name}.macho"
}

echo "clang:    $CLANG"
echo "objcopy:  $OBJCOPY"
echo ""

# --- x86_64-linux ---
build_freestanding_text x86_64-linux x86_64-linux-gnu arith programs/arith/main.c
append_manifest arith x86_64-linux-gnu x86_64 linux x86_64-linux/arith.linked.elf x86_64-linux/arith.text.elf x86_64-linux/arith.entry

build_freestanding_text x86_64-linux x86_64-linux-gnu frame programs/frame/main.c
append_manifest frame x86_64-linux-gnu x86_64 linux x86_64-linux/frame.linked.elf x86_64-linux/frame.text.elf x86_64-linux/frame.entry

build_linked_program x86_64-linux x86_64-linux-gnu exit42 exit42 lib/common/io.c
[ -f x86_64-linux/exit42.linked.elf ] && append_manifest exit42 x86_64-linux-gnu x86_64 linux x86_64-linux/exit42.linked.elf x86_64-linux/exit42.text.elf x86_64-linux/exit42.entry
# Legacy alias
[ -f x86_64-linux/exit42.linked.elf ] && cp x86_64-linux/exit42.linked.elf x86_64-linux/exit.elf

build_linked_program x86_64-linux x86_64-linux-gnu hello hello lib/common/io.c
[ -f x86_64-linux/hello.linked.elf ] && append_manifest hello x86_64-linux-gnu x86_64 linux x86_64-linux/hello.linked.elf x86_64-linux/hello.text.elf x86_64-linux/hello.entry

build_freestanding_text x86_64-linux x86_64-linux-gnu checksum programs/checksum/main.c
append_manifest checksum x86_64-linux-gnu x86_64 linux x86_64-linux/checksum.linked.elf x86_64-linux/checksum.text.elf x86_64-linux/checksum.entry

# --- aarch64-linux ---
build_freestanding_text aarch64-linux aarch64-linux-gnu arith programs/arith/main.c
append_manifest arith aarch64-linux-gnu aarch64 linux aarch64-linux/arith.linked.elf aarch64-linux/arith.text.elf aarch64-linux/arith.entry

build_freestanding_text aarch64-linux aarch64-linux-gnu frame programs/frame/main.c
append_manifest frame aarch64-linux-gnu aarch64 linux aarch64-linux/frame.linked.elf aarch64-linux/frame.text.elf aarch64-linux/frame.entry

build_freestanding_text aarch64-linux aarch64-linux-gnu pairs programs/pairs/main.c
append_manifest pairs aarch64-linux-gnu aarch64 linux aarch64-linux/pairs.linked.elf aarch64-linux/pairs.text.elf aarch64-linux/pairs.entry

build_linked_program aarch64-linux aarch64-linux-gnu exit42 exit42 lib/common/io.c
[ -f aarch64-linux/exit42.linked.elf ] && append_manifest exit42 aarch64-linux-gnu aarch64 linux aarch64-linux/exit42.linked.elf aarch64-linux/exit42.text.elf aarch64-linux/exit42.entry
[ -f aarch64-linux/exit42.linked.elf ] && cp aarch64-linux/exit42.linked.elf aarch64-linux/exit.elf

build_linked_program aarch64-linux aarch64-linux-gnu hello hello lib/common/io.c
[ -f aarch64-linux/hello.linked.elf ] && append_manifest hello aarch64-linux-gnu aarch64 linux aarch64-linux/hello.linked.elf aarch64-linux/hello.text.elf aarch64-linux/hello.entry

build_freestanding_text aarch64-linux aarch64-linux-gnu checksum programs/checksum/main.c
append_manifest checksum aarch64-linux-gnu aarch64 linux aarch64-linux/checksum.linked.elf aarch64-linux/checksum.text.elf aarch64-linux/checksum.entry

# --- macOS linked guests ---
if [ -n "$SDK" ]; then
  build_linked_macho x86_64-macos x86_64-apple-macosx x86_64 exit42 exit42
  [ -f x86_64-macos/exit42.linked.macho ] && append_manifest exit42 x86_64-apple-macosx x86_64 macos x86_64-macos/exit42.linked.macho x86_64-macos/exit42.text.elf x86_64-macos/exit42.entry
  [ -f x86_64-macos/exit42.linked.macho ] && cp x86_64-macos/exit42.linked.macho x86_64-macos/exit.macho

  build_linked_macho aarch64-macos aarch64-apple-darwin arm64 exit42 exit42
  [ -f aarch64-macos/exit42.linked.macho ] && append_manifest exit42 aarch64-apple-darwin aarch64 macos aarch64-macos/exit42.linked.macho aarch64-macos/exit42.text.elf aarch64-macos/exit42.entry
  [ -f aarch64-macos/exit42.linked.macho ] && cp aarch64-macos/exit42.linked.macho aarch64-macos/exit.macho
else
  echo "  (skip macOS linked guests: no SDK)"
fi

echo "Done. manifest: $MANIFEST"
