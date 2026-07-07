#!/usr/bin/env bash
# Idempotent emulator provisioning for C corpus guest-runner paths.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${SPEET_EMULATOR_CACHE:-${HOME:-/tmp}/.cache/speet/emulators}"
mkdir -p "$CACHE"

install_blink() {
  local dest="$CACHE/blink"
  if [ -x "$dest" ]; then
    echo "blink: $dest (cached)"
    return 0
  fi
  if command -v blink >/dev/null 2>&1; then
    ln -sf "$(command -v blink)" "$dest" 2>/dev/null || cp "$(command -v blink)" "$dest"
    chmod +x "$dest"
    echo "blink: linked from PATH → $dest"
    return 0
  fi
  echo "blink: not found — set BLINK or install jart/blink manually" >&2
  return 1
}

install_qemu() {
  local name="$1"
  local dest="$CACHE/$name"
  if [ -x "$dest" ]; then
    echo "$name: $dest (cached)"
    return 0
  fi
  if command -v "$name" >/dev/null 2>&1; then
    ln -sf "$(command -v "$name")" "$dest" 2>/dev/null || cp "$(command -v "$name")" "$dest"
    chmod +x "$dest"
    echo "$name: linked from PATH → $dest"
    return 0
  fi
  if [ "$(uname -s)" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
    echo "$name: try apt-get install qemu-user-static" >&2
  fi
  if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    echo "$name: try brew install qemu" >&2
  fi
  return 1
}

case "${1:-all}" in
  blink) install_blink ;;
  qemu-riscv64) install_qemu qemu-riscv64 ;;
  qemu-riscv32) install_qemu qemu-riscv32 ;;
  qemu-aarch64) install_qemu qemu-aarch64 ;;
  qemu-x86_64) install_qemu qemu-x86_64 ;;
  all)
    install_blink || true
    install_qemu qemu-riscv64 || true
    install_qemu qemu-aarch64 || true
    install_qemu qemu-x86_64 || true
    ;;
  *)
    echo "usage: $0 [blink|qemu-riscv64|qemu-riscv32|qemu-aarch64|qemu-x86_64|all]" >&2
    exit 1
    ;;
esac
