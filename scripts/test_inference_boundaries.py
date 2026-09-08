#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "source" / "INFERENCE.cpp"
FORBIDDEN = re.compile(
    r"#\s*include[^\n]*(?:API\.cpp|sdkit3|stable-diffusion|ggml|llama|crow|asio|/definition/|/translation/|library/include/api)",
    re.IGNORECASE,
)

violations = []
for folder in (PRODUCTION / "include", PRODUCTION / "src", PRODUCTION / "features"):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix in {".h", ".hpp", ".c", ".cc", ".cpp", ".cxx"}:
            for line_number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if FORBIDDEN.search(line):
                    violations.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")

if violations:
    print("INFERENCE.cpp production dependency violations:", *violations, sep="\n")
    sys.exit(1)
print("INFERENCE.cpp production sources use no engines, transports, UI, or non-Common DiffUser surfaces")
