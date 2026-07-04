#!/usr/bin/env bash
# Rebuild C corpus objects: .text.elf (recompiler input) + linked exit guests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CFLAGS="-O1 -fno-vectorize -fno-slp-vectorize -fno-stack-protector -fno-ident -fno-asynchronous-unwind-tables -ffreestanding -nostdlib"

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

if [ -z "$CLANG" ] || [ -z "$OBJCOPY" ] || [ -z "$NM" ]; then
  echo "error: need clang, llvm-objcopy, llvm-nm" >&2
  exit 1
fi

SDK=""
if [ "$(uname -s)" = "Darwin" ]; then
  SDK="$(xcrun --show-sdk-path 2>/dev/null || true)"
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

# main symbol offset within .text (hex, no 0x prefix) → ${out}.entry
write_entry() {
  local obj="$1" out="$2"
  local off
  off=$("$NM" "$obj" 2>/dev/null | awk '$3=="main" {print $1; exit}')
  if [ -z "$off" ]; then
    echo "error: no main symbol in $obj" >&2
    return 1
  fi
  printf '0x%s\n' "$off" > "${out}.entry"
}

build_text_elf() {
  local dir="$1" triple="$2" src="$3"
  local base="${src%.c}"
  echo "  text $dir/$src"
  "$CLANG" --target="$triple" -c $CFLAGS "$dir/$src" -o "$tmp"
  write_entry "$tmp" "$dir/$base"
  "$OBJCOPY" --strip-all --only-section=.text "$tmp" "$dir/${base}.text.elf"
}

build_linked_elf() {
  local dir="$1" triple="$2" src="$3" out="$4"
  echo "  link $dir/$out"
  if ! "$CLANG" --target="$triple" -O1 -fno-vectorize -fno-slp-vectorize \
      -fno-stack-protector "$dir/$src" -o "$dir/$out" 2>/dev/null; then
    echo "  (skip $dir/$out: link failed — need target sysroot)"
  fi
}

build_linked_macho() {
  local dir="$1" triple="$2" arch_flag="$3" src="$4" out="$5"
  echo "  macho $dir/$out"
  local -a sdk_args=()
  [ -n "$SDK" ] && sdk_args=(-isysroot "$SDK")
  if ! "$CLANG" --target="$triple" -arch "$arch_flag" -mmacosx-version-min=11.0 \
      "${sdk_args[@]}" -O1 -fno-vectorize -fno-slp-vectorize -fno-stack-protector \
      "$dir/$src" -o "$dir/$out" 2>/dev/null; then
    echo "  (skip $dir/$out: macho link failed)"
  fi
}

echo "clang:    $CLANG"
echo "objcopy:  $OBJCOPY"
echo ""

build_text_elf x86_64-linux x86_64-linux-gnu arith.c
build_text_elf x86_64-linux x86_64-linux-gnu frame.c
build_linked_elf x86_64-linux x86_64-linux-gnu exit.c exit.elf

build_text_elf aarch64-linux aarch64-linux-gnu arith.c
build_text_elf aarch64-linux aarch64-linux-gnu frame.c
build_text_elf aarch64-linux aarch64-linux-gnu pairs.c
build_linked_elf aarch64-linux aarch64-linux-gnu exit.c exit.elf

if [ -n "$SDK" ]; then
  build_linked_macho x86_64-macos x86_64-apple-macosx x86_64 exit.c exit.macho
  build_linked_macho aarch64-macos aarch64-apple-darwin arm64 exit.c exit.macho
else
  echo "  (skip macOS linked guests: no SDK)"
fi

echo "Done."
