#!/usr/bin/env bash
# Rebuild the pre-assembled MIPS corpus ELF objects from their `.s` sources.
#
# Each `.s` file is assembled with `llvm-mc --triple=mips-unknown-elf` into a
# relocatable ELF object, then stripped down to just the `.text` section with
# `llvm-objcopy --strip-all --only-section=.text`.  The resulting `.elf` files
# contain no debug information, no symbol table, and no source-file metadata —
# only the raw instruction bytes inside the `.text` section.
#
# The pre-built `.elf` files are committed to the repository so that the corpus
# tests never need to invoke an assembler at test-run time.
#
# Usage:
#   ./compile_corpus.sh                   # use whatever llvm-mc/llvm-objcopy
#                                         # are on PATH
#   LLVM_MC=...  LLVM_OBJCOPY=...  \
#     ./compile_corpus.sh                 # override tool paths
#
# Requirements: llvm-mc and llvm-objcopy (LLVM 17+)

set -euo pipefail
cd "$(dirname "$0")"

LLVM_MC="${LLVM_MC:-llvm-mc}"
LLVM_OBJCOPY="${LLVM_OBJCOPY:-llvm-objcopy}"

# Resolve tools through the standard Homebrew / system candidate list when the
# bare name is not on PATH.
find_tool() {
    local name="$1"; shift
    if command -v "$name" &>/dev/null; then echo "$name"; return; fi
    for candidate in "$@"; do
        if [ -x "$candidate" ]; then echo "$candidate"; return; fi
    done
    echo ""
}

MC_CANDIDATES=(
    /opt/homebrew/Cellar/llvm@20/20.1.8/bin/llvm-mc
    /opt/homebrew/Cellar/llvm@19/19.1.7/bin/llvm-mc
    /opt/homebrew/Cellar/llvm@18/18.1.8/bin/llvm-mc
    /opt/homebrew/opt/llvm/bin/llvm-mc
    /opt/homebrew/opt/llvm@20/bin/llvm-mc
    /opt/homebrew/opt/llvm@19/bin/llvm-mc
    /opt/homebrew/opt/llvm@18/bin/llvm-mc
    /usr/bin/llvm-mc-20 /usr/bin/llvm-mc-19 /usr/bin/llvm-mc-18 /usr/bin/llvm-mc-17
    /usr/local/bin/llvm-mc
)
OC_CANDIDATES=(
    /opt/homebrew/Cellar/llvm@20/20.1.8/bin/llvm-objcopy
    /opt/homebrew/Cellar/llvm@19/19.1.7/bin/llvm-objcopy
    /opt/homebrew/Cellar/llvm@18/18.1.8/bin/llvm-objcopy
    /opt/homebrew/opt/llvm/bin/llvm-objcopy
    /opt/homebrew/opt/llvm@20/bin/llvm-objcopy
    /opt/homebrew/opt/llvm@19/bin/llvm-objcopy
    /opt/homebrew/opt/llvm@18/bin/llvm-objcopy
    /usr/bin/llvm-objcopy-20 /usr/bin/llvm-objcopy-19 /usr/bin/llvm-objcopy-18
    /usr/local/bin/llvm-objcopy
)

LLVM_MC="$(find_tool "$LLVM_MC" "${MC_CANDIDATES[@]}")"
LLVM_OBJCOPY="$(find_tool "$LLVM_OBJCOPY" "${OC_CANDIDATES[@]}")"

if [ -z "$LLVM_MC" ]; then
    echo "error: llvm-mc not found. Install LLVM or set LLVM_MC=/path/to/llvm-mc" >&2
    exit 1
fi
if [ -z "$LLVM_OBJCOPY" ]; then
    echo "error: llvm-objcopy not found. Install LLVM or set LLVM_OBJCOPY=/path/to/llvm-objcopy" >&2
    exit 1
fi

echo "llvm-mc:      $LLVM_MC"
echo "llvm-objcopy: $LLVM_OBJCOPY"
echo ""

ok=0; fail=0
tmp=$(mktemp /tmp/speet_mips_XXXXXX.o)
trap 'rm -f "$tmp"' EXIT

for src in *.s; do
    out="${src%.s}.elf"
    printf "  %-45s" "$src"
    if "$LLVM_MC" --triple=mips-unknown-elf --filetype=obj -o "$tmp" "$src" \
       && "$LLVM_OBJCOPY" --strip-all --only-section=.text "$tmp" "$out"; then
        echo "✓  $out"
        ok=$((ok + 1))
    else
        echo "✗  FAILED"
        fail=$((fail + 1))
    fi
done

echo ""
echo "Done: $ok succeeded, $fail failed."
[ "$fail" -eq 0 ]
