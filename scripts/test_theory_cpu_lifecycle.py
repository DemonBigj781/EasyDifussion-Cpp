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
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                                 manifest.get("integrated_sha256", {}).get(path, digest))

    @unittest.skipUnless(CXX, "host C++ compiler is required")
    def test_cpu_contract_compiles_and_runs(self):
        self.compile_and_run("cpu_lifecycle.cpp", "32 cycles and unsupported-route checks passed")

    @unittest.skipUnless(CXX, "host C++ compiler is required")
    def test_exclusive_ownership_and_concurrent_registry(self):
        self.compile_and_run("cpu_ownership.cpp", "1024 concurrent cycles passed")

    def compile_and_run(self, fixture, expected):
        manifest = json.loads(MANIFEST.read_text())
        sources = [str(ROOT / path) for path in manifest["files"] if path.endswith(".cpp")]
        sources.append(str(SOURCE / "library/src/cpu_byte_buffer.cpp"))
        flags = ["-pthread"]
        sanitizers = os.environ.get("THEORY_TEST_SANITIZERS")
        if sanitizers:
            flags += ["-fsanitize=" + sanitizers, "-fno-omit-frame-pointer"]
        with tempfile.TemporaryDirectory(prefix="theory-cpu-contract-") as temporary:
            binary = Path(temporary) / "cpu-lifecycle"
            compiled = subprocess.run([CXX, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(SOURCE), "-I", str(SOURCE / "library/include"), *flags, *sources,
                            str(SOURCE / "tests" / fixture), "-o", str(binary)],
                           check=False, capture_output=True, text=True, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(binary)], check=False, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)

if __name__ == "__main__":
    unittest.main()
