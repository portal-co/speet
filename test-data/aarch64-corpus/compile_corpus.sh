#!/usr/bin/env bash
# Rebuild the pre-assembled AArch64 corpus ELF objects from their .s sources.
#
# Each .s file is assembled with llvm-mc and then stripped to just the
# .text section with llvm-objcopy. The resulting .elf files are committed
# to the repository so tests never need to invoke an assembler.
#
# Usage:
#   ./compile_corpus.sh
#   LLVM_MC=... LLVM_OBJCOPY=... ./compile_corpus.sh
#
# Requirements: llvm-mc and llvm-objcopy (LLVM 17+)

set -euo pipefail
cd "$(dirname "$0")"

LLVM_MC="${LLVM_MC:-llvm-mc}"
LLVM_OBJCOPY="${LLVM_OBJCOPY:-llvm-objcopy}"

find_tool() {
    local name="$1"; shift
    if command -v "$name" &>/dev/null; then echo "$name"; return; fi
    for candidate in "$@"; do
        if command -v "$candidate" &>/dev/null; then echo "$candidate"; return; fi
    done
    for ver in 22 21 20 19 18 17; do
        if command -v "${name}-${ver}" &>/dev/null; then echo "${name}-${ver}"; return; fi
    done
    echo >&2 "error: $name not found on PATH"; exit 1
}

MC="$(find_tool "$LLVM_MC" /opt/homebrew/opt/llvm/bin/llvm-mc)"
OC="$(find_tool "$LLVM_OBJCOPY" /opt/homebrew/opt/llvm/bin/llvm-objcopy)"

for s in *.s; do
    base="${s%.s}"
    echo "  Assembling ${s}..."
    "$MC" --triple=aarch64-unknown-elf --filetype=obj -o "/tmp/${base}.o" "$s"
    "$OC" --strip-all --only-section=.text "/tmp/${base}.o" "${base}.elf"
    echo "    → ${base}.elf ($(wc -c < "${base}.elf") bytes)"
done
echo "Done."
