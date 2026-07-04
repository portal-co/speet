#!/usr/bin/env bash
# Rebuild thin-runtime-corpus guest objects with LLVM only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

LLVM_MC="${LLVM_MC:-llvm-mc}"
LLVM_OBJCOPY="${LLVM_OBJCOPY:-llvm-objcopy}"

find_tool() {
  local name="$1"; shift
  if command -v "$name" &>/dev/null; then echo "$name"; return; fi
  for c in "$@"; do [ -x "$c" ] && echo "$c" && return; done
  echo ""
}

MC="$(find_tool "$LLVM_MC" \
  /opt/homebrew/opt/llvm/bin/llvm-mc \
  /usr/local/opt/llvm/bin/llvm-mc)"
OC="$(find_tool "$LLVM_OBJCOPY" \
  /opt/homebrew/opt/llvm/bin/llvm-objcopy \
  /usr/local/opt/llvm/bin/llvm-objcopy)"

if [ -z "$MC" ] || [ -z "$OC" ]; then
  echo "error: need llvm-mc and llvm-objcopy" >&2
  exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

build_elf() {
  local dir="$1" triple="$2" src="$3" out="$4"
  echo "  ELF $dir/$src"
  "$MC" --triple="$triple" --filetype=obj -o "$tmp" "$dir/$src"
  "$OC" --strip-all --only-section=.text "$tmp" "$dir/$out"
}

build_macho() {
  local dir="$1" triple="$2" src="$3" out="$4"
  echo "  Mach-O $dir/$src"
  "$MC" --triple="$triple" --filetype=obj -o "$dir/$out" "$dir/$src"
}

build_elf rv64-linux riscv64-unknown-elf exit_42.s exit_42.elf
build_elf x86_64-linux x86_64-unknown-elf exit_42.s exit_42.elf
build_elf aarch64-linux aarch64-unknown-elf exit_42.s exit_42.elf
build_macho x86_64-macos x86_64-apple-macosx exit_42.s exit_42.macho
build_macho aarch64-macos aarch64-apple-darwin exit_42.s exit_42.macho

echo "Done."
