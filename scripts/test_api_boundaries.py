#!/usr/bin/env python3
"""Reject production coupling among the three consumer/library trees."""

from pathlib import Path
import hashlib
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TREES = {
    "DiffUser.cpp": ROOT / "source/DiffUser.cpp",
    "SDKIT3": ROOT / "source/sdkit3-port-source",
    "llama.cpp": ROOT / "source/llama.cpp",
}
RULES = {
    "DiffUser.cpp": re.compile(r"API\.test|\bggml\b|\bggml_|\bGGML_|sdkit3-port-source|stable-diffusion(?:\.cpp|\.h)|(?:^|[/\\])llama\.cpp", re.I),
    "SDKIT3": re.compile(r"(?:^|[/\\])API(?:\.cpp|\.bridge)(?:[/\\]|$)", re.I),
    "llama.cpp": re.compile(r"(?:^|[/\\])API(?:\.cpp|\.bridge)(?:[/\\]|$)|sdkit3-port-source|stable-diffusion(?:\.cpp|\.h)", re.I),
}
SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".cmake"}

violations = []
for name, tree in TREES.items():
    for path in tree.rglob("*"):
        if not path.is_file() or (path.suffix not in SUFFIXES and path.name != "CMakeLists.txt"):
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("//", "/*", "*")):
                continue
            if RULES[name].search(line):
                violations.append(f"{path.relative_to(ROOT)}:{number}: {stripped}")

# Exact engine-source copies are also coupling, even without an include path.
source_suffixes = SUFFIXES - {".cmake"}
api_hashes = {}
for path in TREES["DiffUser.cpp"].rglob("*"):
    if path.is_file() and path.suffix in source_suffixes and path.stat().st_size:
        api_hashes.setdefault(hashlib.sha256(path.read_bytes()).digest(), []).append(path)
for engine in ("SDKIT3", "llama.cpp"):
    for path in TREES[engine].rglob("*"):
        if not path.is_file() or path.suffix not in source_suffixes or not path.stat().st_size:
            continue
        digest = hashlib.sha256(path.read_bytes()).digest()
        for api_path in api_hashes.get(digest, ()):
            violations.append(
                f"exact source copy: {api_path.relative_to(ROOT)} == {path.relative_to(ROOT)}"
            )

if violations:
    print("Production library boundary violations:", *violations, sep="\n")
    sys.exit(1)
print("DiffUser.cpp, SDKIT3, and llama.cpp production boundaries are independent")
