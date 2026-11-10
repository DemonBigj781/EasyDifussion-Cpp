#!/usr/bin/env python3
"""Compile/run the isolated Theory CPU contract. No model or GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/DiffUser.cpp"
MANIFEST = ROOT / "docs/integration/theory-lifecycle-source.json"
CXX = os.environ.get("CXX") or shutil.which("c++")

class TheoryCpuLifecycleTests(unittest.TestCase):
    def test_imports_match_reviewed_theory_source(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["source_commit"], "4755b7abe48b49207e33bbff786dc3e2a7032491")
        self.assertEqual(len(manifest["files"]), 11)
        for path, digest in manifest["files"].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    @unittest.skipUnless(CXX, "host C++ compiler is required")
    def test_cpu_contract_compiles_and_runs(self):
        manifest = json.loads(MANIFEST.read_text())
        sources = [str(ROOT / path) for path in manifest["files"] if path.endswith(".cpp")]
        with tempfile.TemporaryDirectory(prefix="theory-cpu-contract-") as temporary:
            binary = Path(temporary) / "cpu-lifecycle"
            subprocess.run([CXX, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(SOURCE), *sources,
                            str(SOURCE / "tests/cpu_lifecycle.cpp"), "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=120)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=15)
            self.assertIn("32 cycles and unsupported-route checks passed", result.stdout)

if __name__ == "__main__":
    unittest.main()
