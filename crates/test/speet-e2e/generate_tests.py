#!/usr/bin/env python3
"""Generate the test-invocation section of tests/e2e.rs.

Run from the repo root or from crates/test/speet-e2e/:
    python3 crates/test/speet-e2e/generate_tests.py

The script rewrites everything between the sentinel comments
    // @generated-tests-begin
and
    // @generated-tests-end
inside tests/e2e.rs, leaving the rest of the file untouched.

Canonical matrix: binaries × dual-lane paths × escape configs, filtered to
supported cells (see harness/capabilities.rs).
"""

import os
import re
import sys
from itertools import combinations
from pathlib import Path

# ── Locate the repo root ──────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
repo_root = SCRIPT_DIR
for _ in range(6):
    if (repo_root / "test-data" / "rv-corpus").exists():
        break
    repo_root = repo_root.parent
else:
    sys.exit("Could not locate repo root (looking for test-data/rv-corpus)")

CORPUS_DIR = repo_root / "test-data" / "rv-corpus"
AARCH64_DIR = repo_root / "test-data" / "aarch64-corpus"
X86_DIR = repo_root / "test-data" / "x86_64-corpus"
MIPS_DIR = repo_root / "test-data" / "mips-corpus"
TEST_FILE = SCRIPT_DIR / "tests" / "e2e.rs"

# ── Escape configs + capability mirror (Python-side of capabilities.rs) ───────

ESCAPE_CONFIGS = [
    ("no_eh", "EscapeConfig::None"),
    ("eh", "EscapeConfig::Exception"),
    ("eh_spec", "EscapeConfig::ExceptionSpec"),
    ("flag_spec", "EscapeConfig::FlagSpec"),
]

SPECULATIVE_SUFS = {"eh_spec", "flag_spec"}
ARCH_SUPPORTS_SPEC = {"Arch::Rv32", "Arch::Rv64", "Arch::X86_64", "Arch::AArch64"}


def configs_for_arch(arch: str):
    """Yield (suffix, EscapeConfig expr) legal for this guest arch."""
    for suf, expr in ESCAPE_CONFIGS:
        if suf in SPECULATIVE_SUFS and arch not in ARCH_SUPPORTS_SPEC:
            continue
        yield suf, expr


def both_escape(lines_fn, arch: str):
    out = []
    for suf, expr in configs_for_arch(arch):
        out.extend(lines_fn(suf, expr))
    return out


# ── Discover corpus binaries ──────────────────────────────────────────────────

def arch_of(dir_name: str) -> str:
    if dir_name.startswith("rv64"):
        return "Arch::Rv64"
    if dir_name.startswith("rv32"):
        return "Arch::Rv32"
    raise ValueError(f"Unrecognised corpus directory: {dir_name!r}")


def ident_of(rel: str) -> str:
    parts = rel.split("/")
    stem = parts[-1].removesuffix(".elf")
    num = stem.split("_")[0]
    dir_part = re.sub(r"[^0-9a-zA-Z]", "_", parts[0])
    return f"{dir_part}_{num}"


CORPUS_EXCLUDE: set[str] = {
    "rv64i/02_write_and_exit",
}

def is_rv_corpus_binary(path: Path) -> bool:
    """True for a compiled corpus binary (.elf or extensionless ELF next to .s)."""
    if not path.is_file() or path.name.startswith("."):
        return False
    if path.suffix == ".elf":
        return True
    if path.suffix == ".s":
        return False
    # Extensionless checked-in ELF sitting beside `<name>.s`.
    return path.suffix == "" and (path.parent / f"{path.name}.s").exists()


corpus: list[tuple[str, str, str]] = []  # (ident, rel_path_stem, arch)
for sub in sorted(CORPUS_DIR.iterdir()) if CORPUS_DIR.is_dir() else []:
    if not sub.is_dir() or sub.name.startswith("."):
        continue
    try:
        arch = arch_of(sub.name)
    except ValueError:
        continue
    for f in sorted(sub.iterdir()):
        if not is_rv_corpus_binary(f):
            continue
        stem = f.stem if f.suffix == ".elf" else f.name
        rel = f"{sub.name}/{stem}"
        if rel in CORPUS_EXCLUDE:
            continue
        corpus.append((ident_of(rel), rel, arch))

if not corpus and TEST_FILE.exists():
    existing = TEST_FILE.read_text()
    seen: set[str] = set()
    for m in re.finditer(
        r'smoke!\(smoke_(\w+?)_(?:no_eh|eh|eh_spec|flag_spec),\s*"([^"]+)",\s*arch=(Arch::\w+),',
        existing,
    ):
        ident, rel, arch = m.group(1), m.group(2), m.group(3)
        if ident not in seen:
            seen.add(ident)
            corpus.append((ident, rel, arch))
    if corpus:
        print(f"  (rv-corpus binaries absent — recovered {len(corpus)} "
              f"corpus entries from existing tests)")


def discover_flat_corpus(dir_path: Path, arch: str, prefix: str) -> list[tuple[str, str, str]]:
    """Flat corpus: test-data/<name>/<stem>.elf → (ident, stem, arch)."""
    out = []
    if not dir_path.is_dir():
        return out
    for f in sorted(dir_path.iterdir()):
        if f.is_file() and f.suffix == ".elf":
            stem = f.stem
            num = stem.split("_")[0]
            out.append((f"{prefix}_{num}", stem, arch))
    return out


aarch64_corpus = discover_flat_corpus(AARCH64_DIR, "Arch::AArch64", "aarch64")
x86_corpus = discover_flat_corpus(X86_DIR, "Arch::X86_64", "x86_64")
mips_corpus = discover_flat_corpus(MIPS_DIR, "Arch::Mips", "mips")

# Fallback lists matching the previously hand-written smoke blocks when ELFs
# are present as .s-only or submodule not built.
if not aarch64_corpus:
    aarch64_corpus = [
        ("aarch64_01", "01_integer_computational", "Arch::AArch64"),
        ("aarch64_02", "02_control_transfer", "Arch::AArch64"),
        ("aarch64_03", "03_load_store", "Arch::AArch64"),
        ("aarch64_04", "04_integer_ext", "Arch::AArch64"),
        ("aarch64_05", "05_load_store_ext", "Arch::AArch64"),
        ("aarch64_06", "06_floating_point", "Arch::AArch64"),
    ]
if not x86_corpus:
    x86_corpus = [
        ("x86_64_01", "01_integer_computational", "Arch::X86_64"),
        ("x86_64_02", "02_control_transfer", "Arch::X86_64"),
        ("x86_64_03", "03_load_store", "Arch::X86_64"),
        ("x86_64_04", "04_flags_and_setcc", "Arch::X86_64"),
        ("x86_64_05", "05_edge_cases", "Arch::X86_64"),
    ]
if not mips_corpus:
    mips_corpus = [
        ("mips_01", "01_integer_computational", "Arch::Mips"),
        ("mips_02", "02_control_transfer", "Arch::Mips"),
        ("mips_03", "03_load_store", "Arch::Mips"),
        ("mips_04", "04_multiply_divide", "Arch::Mips"),
        ("mips_05", "05_edge_cases", "Arch::Mips"),
    ]

# ── C objects ─────────────────────────────────────────────────────────────────

c_objects: list[tuple[str, str, str]] = [
    ("rv32c_arith", "E2E_RV32_ARITH", "Arch::Rv32"),
    ("rv64c_arith", "E2E_RV64_ARITH", "Arch::Rv64"),
    ("x86c_arith", "E2E_X86_ARITH", "Arch::X86_64"),
]

# ── WASM fixtures ─────────────────────────────────────────────────────────────

WASM_FIXTURES: list[tuple[str, str, str, bool, bool]] = [
    ("arith", "wasm_arith()", "compute", False, False),
    ("branches", "wasm_branches()", "test", True, False),
    ("memory_rw", "wasm_memory_rw()", "roundtrip", False, True),
]

MAPPER_VARIANTS: list[tuple[str, str]] = [
    ("no_mapper", "None"),
    ("with_mapper", "Some(make_test_mapper())"),
]

COND_TRAP_VARIANTS: list[tuple[str, str]] = [
    ("no_cond_trap", "None"),
    ("with_flip_trap", "Some(Box::new(FlipConditionTrap))"),
]

# ── Code builders (wasmi / blitz) ─────────────────────────────────────────────

def smoke_corpus(ident, rel, arch):
    def f(suf, cfg):
        return [f'smoke!(smoke_{ident}_{suf}, "{rel}", arch={arch}, {cfg});']
    return both_escape(f, arch)


def run_corpus(ident, rel, arch):
    def f(suf, cfg):
        return [f'run!(run_{ident}_{suf}, "{rel}", arch={arch}, {cfg});']
    return both_escape(f, arch)


def run_trap_corpus(ident, rel, arch):
    def f(suf, cfg):
        return [f'run_trap!(run_trap_{ident}_{suf}, "{rel}", arch={arch}, {cfg});']
    return both_escape(f, arch)


def smoke_c(ident, env, arch):
    def f(suf, cfg):
        return [f'smoke_c!(smoke_{ident}_{suf}, env="{env}", arch={arch}, {cfg});']
    return both_escape(f, arch)


def run_c(ident, env, arch):
    def f(suf, cfg):
        return [f'run_c!(run_{ident}_{suf}, env="{env}", arch={arch}, {cfg});']
    return both_escape(f, arch)


def link_corpus_pair(a, b):
    (id_a, rel_a, arch_a), (id_b, rel_b, arch_b) = a, b
    # Link pairs only for configs both arches support (intersection).
    name_base = f"link_{id_a}_x_{id_b}"
    out = []
    for suf, cfg in configs_for_arch(arch_a):
        if suf in SPECULATIVE_SUFS and arch_b not in ARCH_SUPPORTS_SPEC:
            continue
        if suf in SPECULATIVE_SUFS and arch_a not in ARCH_SUPPORTS_SPEC:
            continue
        out += [
            f'link!({name_base}_{suf},',
            f'    [("{rel_a}", arch={arch_a}, entry="entry_0"),',
            f'     ("{rel_b}", arch={arch_b}, entry="entry_1")],',
            f'    {cfg});',
        ]
    return out


def link_corpus_c(corpus_entry, c_entry):
    (id_a, rel_a, arch_a) = corpus_entry
    (id_b, env_b, arch_b) = c_entry
    name_base = f"link_{id_a}_x_{id_b}"
    out = []
    for suf, cfg in configs_for_arch(arch_a):
        if suf in SPECULATIVE_SUFS and arch_b not in ARCH_SUPPORTS_SPEC:
            continue
        out += [
            f'link_c!({name_base}_{suf},',
            f'    [("{rel_a}", is_corpus=true,  arch={arch_a}, entry="entry_0"),',
            f'     ("{env_b}", is_corpus=false, arch={arch_b}, entry="entry_1")],',
            f'    {cfg});',
        ]
    return out


def link_c_pair(a, b):
    (id_a, env_a, arch_a), (id_b, env_b, arch_b) = a, b
    name_base = f"link_{id_a}_x_{id_b}"
    out = []
    for suf, cfg in configs_for_arch(arch_a):
        if suf in SPECULATIVE_SUFS and arch_b not in ARCH_SUPPORTS_SPEC:
            continue
        out += [
            f'link_c!({name_base}_{suf},',
            f'    [("{env_a}", is_corpus=false, arch={arch_a}, entry="entry_0"),',
            f'     ("{env_b}", is_corpus=false, arch={arch_b}, entry="entry_1")],',
            f'    {cfg});',
        ]
    return out


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
    lines = []
    for decide_name, decide_fn, inp, expected in [
        ("passthrough", "|v| v", 1, 1),
        ("passthrough_zero", "|v| v", 0, 0),
        ("override_false", "|_| 0", 1, 0),
        ("override_true", "|_| 1", 0, 1),
    ]:
        name = f"run_wasm_{ident}_hook_{decide_name}"
        lines += [
            f'wasm_run_cond_trap!({name}, {builder}, entry = "{entry}",',
            f'    input = {inp}, decide_fn = {decide_fn}, expected = {expected});',
        ]
    return lines


def native_corpus(ident, rel, arch):
    def f(suf, cfg):
        return [f'native!(native_{ident}_{suf}, "{rel}", arch={arch}, {cfg});']
    return both_escape(f, arch)


def native_c(ident, env, arch):
    def f(suf, cfg):
        return [f'native_c!(native_{ident}_{suf}, env="{env}", arch={arch}, {cfg});']
    return both_escape(f, arch)


def native_wasm_fixture(ident, builder, mapper_suf, mapper_expr, trap_suf, trap_expr):
    name = f"native_wasm_{ident}_{mapper_suf}_{trap_suf}"
    return [f'native_wasm!({name}, {builder}, mapper = {mapper_expr}, cond_trap = {trap_expr});']


# ── Non-RV corpora ────────────────────────────────────────────────────────────

def smoke_run_native_flat(macro_smoke, macro_run, macro_native, entries):
    lines = []
    for ident, rel, arch in entries:
        for suf, cfg in configs_for_arch(arch):
            lines.append(f'{macro_smoke}!({ident}_{suf}, "{rel}", {cfg});')
            lines.append(f'{macro_run}!(run_{ident}_{suf}, "{rel}", {cfg});')
        # blitz compile-check: one Jump cell per binary (escape×blitz covered on RV).
        lines.append(f'{macro_native}!(native_{ident}, "{rel}");')
    return lines


# ── Dual-lane WASI / thin fixtures ────────────────────────────────────────────

def dual_lane_lines():
    """linux_wasi / thin_native / darwin_wasi cells for hand-byte + corpus fixtures."""
    lines = []
    # linux_wasi: exit_42 × {no_eh, flag_spec}
    for suf, cfg in [("no_eh", "EscapeConfig::None"), ("flag_spec", "EscapeConfig::FlagSpec")]:
        lines.append(
            f'linux_wasi!(linux_wasi_exit_42_{suf}, bytes = FIXTURE_EXIT_42, addr = 0x1000, '
            f'{cfg}, expect_exit = 42);'
        )
        lines.append(
            f'linux_wasi!(linux_wasi_write_exit_{suf}, bytes = FIXTURE_WRITE_EXIT, addr = 0x1000, '
            f'{cfg}, seed_addr = 520, seed = b"hello\\n", expect_stdout = b"hello\\n", expect_exit = 0);'
        )
        lines.append(
            f'thin_native!(thin_native_exit_42_{suf}, bytes = FIXTURE_EXIT_42, addr = 0x1000, '
            f'{cfg}, expect_exit = 42);'
        )
    # darwin_wasi: Jump + FlagSpec (ExceptionSpec needs TagSection)
    for suf, cfg in [("no_eh", "EscapeConfig::None"), ("flag_spec", "EscapeConfig::FlagSpec")]:
        lines.append(
            f'darwin_wasi!(darwin_wasi_write_exit_{suf}, bytes = FIXTURE_DARWIN_WRITE_EXIT, '
            f'addr = 0x1000, {cfg}, seed_addr = 520, seed = b"hello\\n", '
            f'expect_stdout = b"hello\\n", expect_exit = 0);'
        )
    return lines


# ── Assemble all generated lines ──────────────────────────────────────────────

def section(title, lines):
    return [f"// ── {title} {'─' * max(1, 76 - len(title))}", ""] + lines + [""]


out: list[str] = []

smoke_lines = []
for entry in corpus:
    smoke_lines.extend(smoke_corpus(*entry))
out += section("Corpus smoke tests (wasmi)", smoke_lines)

sc_lines = []
for entry in c_objects:
    sc_lines.extend(smoke_c(*entry))
out += section("C smoke tests (wasmi)", sc_lines)

run_lines = []
for entry in corpus:
    run_lines.extend(run_corpus(*entry))
out += section("Corpus run tests (wasmi)", run_lines)

rc_lines = []
for entry in c_objects:
    rc_lines.extend(run_c(*entry))
out += section("C run tests (wasmi)", rc_lines)

rt_lines = []
for entry in corpus:
    rt_lines.extend(run_trap_corpus(*entry))
out += section("Corpus run-with-trap tests (wasmi)", rt_lines)

lcc_lines = []
for a, b in combinations(corpus, 2):
    lcc_lines.extend(link_corpus_pair(a, b))
out += section("Corpus-corpus link tests (wasmi)", lcc_lines)

lcC_lines = []
for corp in corpus:
    for c in c_objects:
        lcC_lines.extend(link_corpus_c(corp, c))
out += section("Corpus-C link tests (wasmi)", lcC_lines)

lCC_lines = []
for a, b in combinations(c_objects, 2):
    lCC_lines.extend(link_c_pair(a, b))
out += section("C-C link tests (wasmi)", lCC_lines)

wsmoke_lines = []
for (ident, builder, entry, has_branches, has_memory) in WASM_FIXTURES:
    for (mapper_suf, mapper_expr) in MAPPER_VARIANTS:
        for (trap_suf, trap_expr) in COND_TRAP_VARIANTS:
            wsmoke_lines.extend(wasm_smoke_fixture(
                ident, builder, mapper_suf, mapper_expr, trap_suf, trap_expr))
out += section("WASM smoke tests", wsmoke_lines)

wrun_lines = []
for (ident, builder, entry, has_branches, has_memory) in WASM_FIXTURES:
    for (mapper_suf, mapper_expr) in MAPPER_VARIANTS:
        if has_memory and mapper_suf != "no_mapper":
            continue
        for (trap_suf, trap_expr) in COND_TRAP_VARIANTS:
            wrun_lines.extend(wasm_run_fixture(
                ident, builder, entry, mapper_suf, mapper_expr, trap_suf, trap_expr))
out += section("WASM run tests", wrun_lines)

wcond_lines = []
for (ident, builder, entry, has_branches, has_memory) in WASM_FIXTURES:
    if has_branches:
        wcond_lines.extend(wasm_cond_trap_tests(ident, builder, entry))
out += section("WASM condition-trap hook tests", wcond_lines)

nat_lines = []
for entry in corpus:
    nat_lines.extend(native_corpus(*entry))
out += section("Native-backend (blitz) corpus tests", nat_lines)

natc_lines = []
for entry in c_objects:
    natc_lines.extend(native_c(*entry))
out += section("Native-backend (blitz) C tests", natc_lines)

natw_lines = []
for (ident, builder, entry, has_branches, has_memory) in WASM_FIXTURES:
    for (mapper_suf, mapper_expr) in MAPPER_VARIANTS:
        for (trap_suf, trap_expr) in COND_TRAP_VARIANTS:
            natw_lines.extend(native_wasm_fixture(
                ident, builder, mapper_suf, mapper_expr, trap_suf, trap_expr))
out += section("Native-backend (blitz) WASM tests", natw_lines)

# Fold former hand-written aarch64 / x86 / mips blocks into the matrix.
out += section(
    "AArch64 corpus (wasmi + blitz)",
    smoke_run_native_flat("smoke_aarch64", "run_aarch64", "native_aarch64", aarch64_corpus),
)
out += section(
    "x86-64 corpus (wasmi + blitz)",
    smoke_run_native_flat("smoke_x86_64", "run_x86_64", "native_x86_64", x86_corpus),
)
out += section(
    "MIPS corpus (wasmi + blitz)",
    smoke_run_native_flat("smoke_mips", "run_mips", "native_mips", mips_corpus),
)

out += section("Dual-lane linux-wasi / thin-native / darwin-wasi", dual_lane_lines())

generated = "\n".join(out).rstrip() + "\n"

# ── Splice into the test file ─────────────────────────────────────────────────

BEGIN = "// @generated-tests-begin"
END = "// @generated-tests-end"

src = TEST_FILE.read_text()

if BEGIN not in src or END not in src:
    sys.exit(
        f"Sentinel comments not found in {TEST_FILE}.\n"
        f"Add these two lines to the file to mark where generated tests go:\n"
        f"  {BEGIN}\n"
        f"  {END}"
    )

before = src[:src.index(BEGIN) + len(BEGIN)]
after = src[src.index(END):]
new_src = before + "\n\n" + generated + "\n" + after

TEST_FILE.write_text(new_src)
print(f"Wrote {len(generated.splitlines())} generated lines to {TEST_FILE}")

n_smoke = sum(1 for l in generated.splitlines() if l.startswith("smoke!(") or l.startswith("smoke_c!(") or l.startswith("smoke_aarch64!(") or l.startswith("smoke_x86_64!(") or l.startswith("smoke_mips!("))
n_run = sum(1 for l in generated.splitlines() if l.startswith("run!(") or l.startswith("run_c!(") or l.startswith("run_aarch64!(") or l.startswith("run_x86_64!(") or l.startswith("run_mips!("))
n_run_trap = sum(1 for l in generated.splitlines() if l.startswith("run_trap!("))
n_link = sum(1 for l in generated.splitlines() if l.startswith("link!(") or l.startswith("link_c!("))
n_wasm_smoke = sum(1 for l in generated.splitlines() if l.startswith("wasm_smoke!("))
n_wasm_run = sum(1 for l in generated.splitlines() if l.startswith("wasm_run!(") and "cond_trap" not in l.split("wasm_run!(")[1].split(",")[0])
n_wasm_cond = sum(1 for l in generated.splitlines() if l.startswith("wasm_run_cond_trap!("))
n_native = sum(1 for l in generated.splitlines() if l.startswith("native!(") or l.startswith("native_c!(") or l.startswith("native_wasm!(") or l.startswith("native_aarch64!(") or l.startswith("native_x86_64!(") or l.startswith("native_mips!("))
n_lane = sum(1 for l in generated.splitlines() if l.startswith("linux_wasi!(") or l.startswith("thin_native!(") or l.startswith("darwin_wasi!("))
print(f"  {len(corpus)} rv corpus, {len(aarch64_corpus)} aarch64, {len(x86_corpus)} x86, {len(mips_corpus)} mips, {len(c_objects)} C, {len(WASM_FIXTURES)} WASM")
print(f"  escape configs: {[s for s, _ in ESCAPE_CONFIGS]}")
print(f"  wasmi — smoke: {n_smoke}  run: {n_run}  run_trap: {n_run_trap}  link: {n_link}")
print(f"  wasm-fixtures — smoke: {n_wasm_smoke}  run: {n_wasm_run}  cond_trap: {n_wasm_cond}")
print(f"  blitz native-backend: {n_native}")
print(f"  dual-lane cells: {n_lane}")
