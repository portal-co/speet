#!/usr/bin/env bash
# Rebuild RV C corpus .text.elf blobs and linked exit guest.
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

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

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
  local dir="$1" triple="$2" march="$3" mabi="$4" src="$5"
  local base="${src%.c}"
  echo "  text $dir/$src"
  "$CLANG" --target="$triple" -march="$march" -mabi="$mabi" -c $CFLAGS \
    "$dir/$src" -o "$tmp"
  write_entry "$tmp" "$dir/$base"
  "$OBJCOPY" --strip-all --only-section=.text "$tmp" "$dir/${base}.text.elf"
}

build_linked_elf() {
  local dir="$1" triple="$2" march="$3" mabi="$4" src="$5" out="$6"
  echo "  link $dir/$out"
  "$CLANG" --target="$triple" -march="$march" -mabi="$mabi" $CFLAGS \
    -nostdlib -static "$dir/$src" -o "$dir/$out" 2>/dev/null || \
  "$CLANG" --target="$triple" -march="$march" -mabi="$mabi" $CFLAGS \
    "$dir/$src" -o "$dir/$out"
}

build_text_elf rv32 riscv32-unknown-elf rv32im ilp32 arith.c
build_text_elf rv32 riscv32-unknown-elf rv32im ilp32 frame.c
build_text_elf rv64 riscv64-unknown-elf rv64im lp64 arith.c
build_text_elf rv64 riscv64-unknown-elf rv64im lp64 frame.c
build_linked_elf rv64 riscv64-unknown-elf rv64im lp64 exit.c exit.elf

echo "Done."
