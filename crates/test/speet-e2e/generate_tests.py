#!/usr/bin/env python3
"""Generate the test-invocation section of tests/riscv_e2e.rs.

Run from the repo root or from crates/test/speet-e2e/:
    python3 crates/test/speet-e2e/generate_tests.py

The script rewrites everything between the sentinel comments
    // @generated-tests-begin
and
    // @generated-tests-end
inside tests/riscv_e2e.rs, leaving the rest of the file untouched.
"""

import os
import re
import sys
from itertools import combinations
from pathlib import Path

# ── Locate the repo root ──────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
# Walk up until we find test-data/rv-corpus
repo_root = SCRIPT_DIR
for _ in range(6):
    if (repo_root / "test-data" / "rv-corpus").exists():
        break
    repo_root = repo_root.parent
else:
    sys.exit("Could not locate repo root (looking for test-data/rv-corpus)")

CORPUS_DIR = repo_root / "test-data" / "rv-corpus"
TEST_FILE  = SCRIPT_DIR / "tests" / "riscv_e2e.rs"

# ── Discover corpus binaries ──────────────────────────────────────────────────

def arch_of(dir_name: str) -> str:
    """Map corpus sub-directory name → Arch enum variant."""
    if dir_name.startswith("rv64"):
        return "Arch::Rv64"
    if dir_name.startswith("rv32"):
        return "Arch::Rv32"
    raise ValueError(f"Unrecognised corpus directory: {dir_name!r}")

def ident_of(rel: str) -> str:
    """Turn a corpus-relative path like 'rv32i/01_foo_bar' into 'rv32i_01'."""
    parts = rel.split("/")
    # Use just the numeric prefix from the filename (e.g. "01" from "01_foo")
    num = parts[-1].split("_")[0]
    # Sanitise directory part (replace non-ident chars)
    dir_part = re.sub(r"[^0-9a-zA-Z]", "_", parts[0])
    return f"{dir_part}_{num}"

# Collect all ELF binaries: files with no extension inside corpus sub-dirs.
corpus: list[tuple[str, str, str]] = []  # (ident, rel_path, arch)
for sub in sorted(CORPUS_DIR.iterdir()):
    if not sub.is_dir() or sub.name.startswith("."):
        continue
    try:
        arch = arch_of(sub.name)
    except ValueError:
        continue
    for f in sorted(sub.iterdir()):
        if f.is_file() and f.suffix == "":
            rel = f"{sub.name}/{f.name}"
            corpus.append((ident_of(rel), rel, arch))

# ── C objects ─────────────────────────────────────────────────────────────────

c_objects: list[tuple[str, str, str]] = [
    # (ident,         env_var,          arch)
    ("rv32c_arith",  "E2E_RV32_ARITH", "Arch::Rv32"),
    ("rv64c_arith",  "E2E_RV64_ARITH", "Arch::Rv64"),
    ("x86c_arith",   "E2E_X86_ARITH",  "Arch::X86_64"),
]

# ── Code builders ─────────────────────────────────────────────────────────────

EH_VARIANTS = [("no_eh", "Eh::None"), ("eh", "Eh::With")]

def both_eh(lines_fn):
    """Call lines_fn(eh_suffix, eh_expr) for each EH variant; return joined."""
    out = []
    for suf, expr in EH_VARIANTS:
        out.extend(lines_fn(suf, expr))
    return out

def smoke_corpus(ident, rel, arch):
    def f(suf, eh):
        return [f'smoke!(smoke_{ident}_{suf}, "{rel}", arch={arch}, {eh});']
    return both_eh(f)

def run_corpus(ident, rel, arch):
    def f(suf, eh):
        return [f'run!(run_{ident}_{suf}, "{rel}", arch={arch}, {eh});']
    return both_eh(f)

def smoke_c(ident, env, arch):
    def f(suf, eh):
        return [f'smoke_c!(smoke_{ident}_{suf}, env="{env}", arch={arch}, {eh});']
    return both_eh(f)

def run_c(ident, env, arch):
    def f(suf, eh):
        return [f'run_c!(run_{ident}_{suf}, env="{env}", arch={arch}, {eh});']
    return both_eh(f)

def link_corpus_pair(a, b):
    """Generate link! tests for two corpus binaries."""
    (id_a, rel_a, arch_a), (id_b, rel_b, arch_b) = a, b
    name_base = f"link_{id_a}_x_{id_b}"
    def f(suf, eh):
        return [
            f'link!({name_base}_{suf},',
            f'    [("{rel_a}", arch={arch_a}, entry="entry_0"),',
            f'     ("{rel_b}", arch={arch_b}, entry="entry_1")],',
            f'    {eh});',
        ]
    return both_eh(f)

def link_corpus_c(corpus_entry, c_entry):
    """Generate link_c! tests for one corpus + one C binary."""
    (id_a, rel_a, arch_a) = corpus_entry
    (id_b, env_b, arch_b) = c_entry
    name_base = f"link_{id_a}_x_{id_b}"
    def f(suf, eh):
        return [
            f'link_c!({name_base}_{suf},',
            f'    [("{rel_a}", is_corpus=true,  arch={arch_a}, entry="entry_0"),',
            f'     ("{env_b}", is_corpus=false, arch={arch_b}, entry="entry_1")],',
            f'    {eh});',
        ]
    return both_eh(f)

def link_c_pair(a, b):
    """Generate link_c! tests for two C objects."""
    (id_a, env_a, arch_a), (id_b, env_b, arch_b) = a, b
    name_base = f"link_{id_a}_x_{id_b}"
    def f(suf, eh):
        return [
            f'link_c!({name_base}_{suf},',
            f'    [("{env_a}", is_corpus=false, arch={arch_a}, entry="entry_0"),',
            f'     ("{env_b}", is_corpus=false, arch={arch_b}, entry="entry_1")],',
            f'    {eh});',
        ]
    return both_eh(f)

# ── Assemble all generated lines ──────────────────────────────────────────────

def section(title, lines):
    return [f"// ── {title} {'─' * max(1, 76 - len(title))}", ""] + lines + [""]

out: list[str] = []

# Smoke – corpus
smoke_lines = []
for entry in corpus:
    smoke_lines.extend(smoke_corpus(*entry))
out += section("Corpus smoke tests", smoke_lines)

# Smoke – C
sc_lines = []
for entry in c_objects:
    sc_lines.extend(smoke_c(*entry))
out += section("C smoke tests", sc_lines)

# Run – corpus
run_lines = []
for entry in corpus:
    run_lines.extend(run_corpus(*entry))
out += section("Corpus run tests", run_lines)

# Run – C
rc_lines = []
for entry in c_objects:
    rc_lines.extend(run_c(*entry))
out += section("C run tests", rc_lines)

# Link – corpus × corpus (all unique pairs)
lcc_lines = []
for a, b in combinations(corpus, 2):
    lcc_lines.extend(link_corpus_pair(a, b))
out += section("Corpus-corpus link tests", lcc_lines)

# Link – corpus × C (all combinations)
lcC_lines = []
for corp in corpus:
    for c in c_objects:
        lcC_lines.extend(link_corpus_c(corp, c))
out += section("Corpus-C link tests", lcC_lines)

# Link – C × C (all unique pairs)
lCC_lines = []
for a, b in combinations(c_objects, 2):
    lCC_lines.extend(link_c_pair(a, b))
out += section("C-C link tests", lCC_lines)

generated = "\n".join(out).rstrip() + "\n"

# ── Splice into the test file ─────────────────────────────────────────────────

BEGIN = "// @generated-tests-begin"
END   = "// @generated-tests-end"

src = TEST_FILE.read_text()

if BEGIN not in src or END not in src:
    sys.exit(
        f"Sentinel comments not found in {TEST_FILE}.\n"
        f"Add these two lines to the file to mark where generated tests go:\n"
        f"  {BEGIN}\n"
        f"  {END}"
    )

before = src[:src.index(BEGIN) + len(BEGIN)]
after  = src[src.index(END):]
new_src = before + "\n\n" + generated + "\n" + after

TEST_FILE.write_text(new_src)
print(f"Wrote {len(generated.splitlines())} generated lines to {TEST_FILE}")

# Print a quick summary
n_smoke  = sum(1 for l in generated.splitlines() if l.startswith("smoke!(") or l.startswith("smoke_c!("))
n_run    = sum(1 for l in generated.splitlines() if l.startswith("run!(") or l.startswith("run_c!("))
n_link   = sum(1 for l in generated.splitlines() if l.startswith("link!(") or l.startswith("link_c!("))
print(f"  {len(corpus)} corpus binaries, {len(c_objects)} C objects")
print(f"  smoke: {n_smoke}  run: {n_run}  link: {n_link}")
