#!/usr/bin/env python3
"""Generate the test-invocation section of tests/e2e.rs.

Run from the repo root or from crates/test/speet-e2e/:
    python3 crates/test/speet-e2e/generate_tests.py

The script rewrites everything between the sentinel comments
    // @generated-tests-begin
and
    // @generated-tests-end
inside tests/e2e.rs, leaving the rest of the file untouched.
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
TEST_FILE  = SCRIPT_DIR / "tests" / "e2e.rs"

# ── Discover corpus binaries ──────────────────────────────────────────────────

def arch_of(dir_name: str) -> str:
    """Map corpus sub-directory name → Arch enum variant."""
    if dir_name.startswith("rv64"):
        return "Arch::Rv64"
    if dir_name.startswith("rv32"):
        return "Arch::Rv32"
    raise ValueError(f"Unrecognised corpus directory: {dir_name!r}")

def ident_of(rel: str) -> str:
    """Turn a corpus-relative path like 'rv32i/01_foo_bar' into 'rv32i_01'.
    The .elf suffix is stripped before splitting."""
    parts = rel.split("/")
    stem = parts[-1].removesuffix(".elf")
    # Use just the numeric prefix from the filename (e.g. "01" from "01_foo")
    num = stem.split("_")[0]
    # Sanitise directory part (replace non-ident chars)
    dir_part = re.sub(r"[^0-9a-zA-Z]", "_", parts[0])
    return f"{dir_part}_{num}"

# Fixtures excluded from the generic corpus: these require a syscall/WASI
# environment (ecall) that the generic e2e runner does not provide, and are
# exercised by dedicated harnesses (e.g. linux_wasi_tests.rs).  They are kept
# as .s-only references and are not assembled by compile_corpus.sh, but guard
# against an accidentally-present .elf here too.
CORPUS_EXCLUDE: set[str] = {
    "rv64i/02_write_and_exit",
}

# Collect all ELF binaries: .elf files inside corpus sub-dirs.
# The corpus() helper appends ".elf", so we pass the stem (without extension).
corpus: list[tuple[str, str, str]] = []  # (ident, rel_path_stem, arch)
for sub in sorted(CORPUS_DIR.iterdir()):
    if not sub.is_dir() or sub.name.startswith("."):
        continue
    try:
        arch = arch_of(sub.name)
    except ValueError:
        continue
    for f in sorted(sub.iterdir()):
        if f.is_file() and f.suffix == ".elf":
            stem = f.stem  # filename without .elf
            rel = f"{sub.name}/{stem}"
            if rel in CORPUS_EXCLUDE:
                continue
            corpus.append((ident_of(rel), rel, arch))

# Fallback: if the rv-corpus submodule isn't checked out, the filesystem scan
# above finds nothing. Rather than silently *drop* the corpus tests on
# regeneration, recover the corpus list (ident, rel, arch) from the existing
# `smoke!(...)` lines in the current test file. This keeps regeneration safe in
# environments without the submodule.
if not corpus and TEST_FILE.exists():
    existing = TEST_FILE.read_text()
    seen: set[str] = set()
    for m in re.finditer(
        r'smoke!\(smoke_(\w+?)_(?:no_eh|eh),\s*"([^"]+)",\s*arch=(Arch::\w+),', existing
    ):
        ident, rel, arch = m.group(1), m.group(2), m.group(3)
        if ident not in seen:
            seen.add(ident)
            corpus.append((ident, rel, arch))
    if corpus:
        print(f"  (rv-corpus submodule absent — recovered {len(corpus)} "
              f"corpus entries from existing tests)")

# ── C objects ─────────────────────────────────────────────────────────────────

c_objects: list[tuple[str, str, str]] = [
    # (ident,         env_var,          arch)
    ("rv32c_arith",  "E2E_RV32_ARITH", "Arch::Rv32"),
    ("rv64c_arith",  "E2E_RV64_ARITH", "Arch::Rv64"),
    ("x86c_arith",   "E2E_X86_ARITH",  "Arch::X86_64"),
]

# ── WASM fixtures ─────────────────────────────────────────────────────────────
#
# Each entry: (ident, builder_expr, entry_name, has_branches, has_memory)
#
#   has_branches — fixture contains if/br_if; condition-trap tests are generated
#   has_memory   — fixture uses linear memory; mapper+run tests are skipped
#                  (page table not initialised at runtime)

WASM_FIXTURES: list[tuple[str, str, str, bool, bool]] = [
    ("arith",     "wasm_arith()",     "compute",   False, False),
    ("branches",  "wasm_branches()",  "test",      True,  False),
    ("memory_rw", "wasm_memory_rw()", "roundtrip", False, True),
]

MAPPER_VARIANTS: list[tuple[str, str]] = [
    ("no_mapper",    "None"),
    ("with_mapper",  "Some(make_test_mapper())"),
]

COND_TRAP_VARIANTS: list[tuple[str, str]] = [
    ("no_cond_trap",   "None"),
    ("with_flip_trap", "Some(Box::new(FlipConditionTrap))"),
]

# ── Code builders ─────────────────────────────────────────────────────────────

# Third element enables `set_speculative_calls(true)` on the recompiler
# (x86-64/RISC-V only — ignored for AArch64). Only paired with `Eh::With`:
# speculative calls require an escape tag, which `Eh::None` never sets.
EH_VARIANTS = [
    ("no_eh", "Eh::None", "false"),
    ("eh", "Eh::With", "false"),
    ("eh_spec", "Eh::With", "true"),
]

def both_eh(lines_fn):
    """Call lines_fn(eh_suffix, eh_expr, spec_expr) for each EH variant; return joined."""
    out = []
    for suf, expr, spec in EH_VARIANTS:
        out.extend(lines_fn(suf, expr, spec))
    return out

def smoke_corpus(ident, rel, arch):
    def f(suf, eh, spec):
        return [f'smoke!(smoke_{ident}_{suf}, "{rel}", arch={arch}, {eh}, speculative={spec});']
    return both_eh(f)

def run_corpus(ident, rel, arch):
    def f(suf, eh, spec):
        return [f'run!(run_{ident}_{suf}, "{rel}", arch={arch}, {eh}, speculative={spec});']
    return both_eh(f)

def run_trap_corpus(ident, rel, arch):
    def f(suf, eh, spec):
        return [f'run_trap!(run_trap_{ident}_{suf}, "{rel}", arch={arch}, {eh}, speculative={spec});']
    return both_eh(f)

def smoke_c(ident, env, arch):
    def f(suf, eh, spec):
        return [f'smoke_c!(smoke_{ident}_{suf}, env="{env}", arch={arch}, {eh}, speculative={spec});']
    return both_eh(f)

def run_c(ident, env, arch):
    def f(suf, eh, spec):
        return [f'run_c!(run_{ident}_{suf}, env="{env}", arch={arch}, {eh}, speculative={spec});']
    return both_eh(f)

def link_corpus_pair(a, b):
    """Generate link! tests for two corpus binaries."""
    (id_a, rel_a, arch_a), (id_b, rel_b, arch_b) = a, b
    name_base = f"link_{id_a}_x_{id_b}"
    def f(suf, eh, spec):
        return [
            f'link!({name_base}_{suf},',
            f'    [("{rel_a}", arch={arch_a}, entry="entry_0"),',
            f'     ("{rel_b}", arch={arch_b}, entry="entry_1")],',
            f'    {eh}, speculative={spec});',
        ]
    return both_eh(f)

def link_corpus_c(corpus_entry, c_entry):
    """Generate link_c! tests for one corpus + one C binary."""
    (id_a, rel_a, arch_a) = corpus_entry
    (id_b, env_b, arch_b) = c_entry
    name_base = f"link_{id_a}_x_{id_b}"
    def f(suf, eh, spec):
        return [
            f'link_c!({name_base}_{suf},',
            f'    [("{rel_a}", is_corpus=true,  arch={arch_a}, entry="entry_0"),',
            f'     ("{env_b}", is_corpus=false, arch={arch_b}, entry="entry_1")],',
            f'    {eh}, speculative={spec});',
        ]
    return both_eh(f)

def link_c_pair(a, b):
    """Generate link_c! tests for two C objects."""
    (id_a, env_a, arch_a), (id_b, env_b, arch_b) = a, b
    name_base = f"link_{id_a}_x_{id_b}"
    def f(suf, eh, spec):
        return [
            f'link_c!({name_base}_{suf},',
            f'    [("{env_a}", is_corpus=false, arch={arch_a}, entry="entry_0"),',
            f'     ("{env_b}", is_corpus=false, arch={arch_b}, entry="entry_1")],',
            f'    {eh}, speculative={spec});',
        ]
    return both_eh(f)

# ── WASM test generators ──────────────────────────────────────────────────────

def wasm_smoke_fixture(ident, builder, mapper_suf, mapper_expr, trap_suf, trap_expr):
    name = f"smoke_wasm_{ident}_{mapper_suf}_{trap_suf}"
    return [f'wasm_smoke!({name}, {builder}, mapper = {mapper_expr}, cond_trap = {trap_expr});']

def wasm_run_fixture(ident, builder, entry, mapper_suf, mapper_expr, trap_suf, trap_expr):
    name = f"run_wasm_{ident}_{mapper_suf}_{trap_suf}"
    return [
        f'wasm_run!({name}, {builder}, entry = "{entry}",',
        f'    mapper = {mapper_expr}, cond_trap = {trap_expr});',
    ]

def wasm_cond_trap_tests(ident, builder, entry):
    """Generate hook-cond-trap run tests (only for fixtures with branches)."""
    lines = []
    for decide_name, decide_fn, inp, expected in [
        ("passthrough", "|v| v", 1, 1),
        ("passthrough_zero", "|v| v", 0, 0),
        ("override_false", "|_| 0", 1, 0),
        ("override_true",  "|_| 1", 0, 1),
    ]:
        name = f"run_wasm_{ident}_hook_{decide_name}"
        lines += [
            f'wasm_run_cond_trap!({name}, {builder}, entry = "{entry}",',
            f'    input = {inp}, decide_fn = {decide_fn}, expected = {expected});',
        ]
    return lines

# ── Native-backend cross generators ───────────────────────────────────────────
#
# For each frontend (corpus / C / WASM fixture) we also lower the recompiled
# module through the *native* backend (wasm-blitz) for both architectures and
# object formats. This crosses every frontend with {wasm, native} backends.

def native_corpus(ident, rel, arch):
    def f(suf, eh, spec):
        return [f'native!(native_{ident}_{suf}, "{rel}", arch={arch}, {eh}, speculative={spec});']
    return both_eh(f)

def native_c(ident, env, arch):
    def f(suf, eh, spec):
        return [f'native_c!(native_{ident}_{suf}, env="{env}", arch={arch}, {eh}, speculative={spec});']
    return both_eh(f)

def native_wasm_fixture(ident, builder, mapper_suf, mapper_expr, trap_suf, trap_expr):
    name = f"native_wasm_{ident}_{mapper_suf}_{trap_suf}"
    return [f'native_wasm!({name}, {builder}, mapper = {mapper_expr}, cond_trap = {trap_expr});']

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

# Run with trap – corpus (all binaries; validates cell-local indices)
rt_lines = []
for entry in corpus:
    rt_lines.extend(run_trap_corpus(*entry))
out += section("Corpus run-with-trap tests", rt_lines)

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

# WASM smoke – all fixtures × all mapper variants × all cond_trap variants
wsmoke_lines = []
for (ident, builder, entry, has_branches, has_memory) in WASM_FIXTURES:
    for (mapper_suf, mapper_expr) in MAPPER_VARIANTS:
        for (trap_suf, trap_expr) in COND_TRAP_VARIANTS:
            wsmoke_lines.extend(wasm_smoke_fixture(
                ident, builder, mapper_suf, mapper_expr, trap_suf, trap_expr))
out += section("WASM smoke tests", wsmoke_lines)

# WASM run – all fixtures × all mapper variants × all cond_trap variants
# Skip mapper+run for fixtures with memory (page table not initialised at runtime).
wrun_lines = []
for (ident, builder, entry, has_branches, has_memory) in WASM_FIXTURES:
    for (mapper_suf, mapper_expr) in MAPPER_VARIANTS:
        if has_memory and mapper_suf != "no_mapper":
            continue  # mapper runtime requires initialised page table
        for (trap_suf, trap_expr) in COND_TRAP_VARIANTS:
            wrun_lines.extend(wasm_run_fixture(
                ident, builder, entry, mapper_suf, mapper_expr, trap_suf, trap_expr))
out += section("WASM run tests", wrun_lines)

# WASM cond-trap hook tests (only for fixtures with branches)
wcond_lines = []
for (ident, builder, entry, has_branches, has_memory) in WASM_FIXTURES:
    if has_branches:
        wcond_lines.extend(wasm_cond_trap_tests(ident, builder, entry))
out += section("WASM condition-trap hook tests", wcond_lines)

# Native backend cross — corpus
nat_lines = []
for entry in corpus:
    nat_lines.extend(native_corpus(*entry))
out += section("Native-backend corpus tests", nat_lines)

# Native backend cross — C
natc_lines = []
for entry in c_objects:
    natc_lines.extend(native_c(*entry))
out += section("Native-backend C tests", natc_lines)

# Native backend cross — WASM fixtures (× mapper × cond_trap)
natw_lines = []
for (ident, builder, entry, has_branches, has_memory) in WASM_FIXTURES:
    for (mapper_suf, mapper_expr) in MAPPER_VARIANTS:
        for (trap_suf, trap_expr) in COND_TRAP_VARIANTS:
            natw_lines.extend(native_wasm_fixture(
                ident, builder, mapper_suf, mapper_expr, trap_suf, trap_expr))
out += section("Native-backend WASM tests", natw_lines)

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
n_smoke      = sum(1 for l in generated.splitlines() if l.startswith("smoke!(") or l.startswith("smoke_c!("))
n_run        = sum(1 for l in generated.splitlines() if l.startswith("run!(") or l.startswith("run_c!("))
n_run_trap   = sum(1 for l in generated.splitlines() if l.startswith("run_trap!("))
n_link       = sum(1 for l in generated.splitlines() if l.startswith("link!(") or l.startswith("link_c!("))
n_wasm_smoke = sum(1 for l in generated.splitlines() if l.startswith("wasm_smoke!("))
n_wasm_run   = sum(1 for l in generated.splitlines() if l.startswith("wasm_run!(") and "cond_trap" not in l.split("wasm_run!(")[1].split(",")[0])
n_wasm_cond  = sum(1 for l in generated.splitlines() if l.startswith("wasm_run_cond_trap!("))
n_native = sum(1 for l in generated.splitlines() if l.startswith("native!(") or l.startswith("native_c!(") or l.startswith("native_wasm!("))
print(f"  {len(corpus)} corpus binaries, {len(c_objects)} C objects, {len(WASM_FIXTURES)} WASM fixtures")
print(f"  wasmi-backend — smoke: {n_smoke}  run: {n_run}  run_trap: {n_run_trap}  link: {n_link}")
print(f"  wasm-fixtures — smoke: {n_wasm_smoke}  run: {n_wasm_run}  cond_trap: {n_wasm_cond}")
print(f"  native-backend cross: {n_native}")
