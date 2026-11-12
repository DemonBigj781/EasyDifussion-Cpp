#!/usr/bin/env python3
"""Compile real config/OptionsManager and companion routing without loading models."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

from test_application_patches import prepare_source_fixture, materialize_fixture_headers

ROOT = Path(__file__).resolve().parents[2]
COSMO = ROOT / "cosmopolitan"
SDKIT = ROOT / "source/sdkit3-port-source"


class SplitModelSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="cosmo-split-settings-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        prepare_source_fixture(cls.directory)
        materialize_fixture_headers(cls.directory)
        staged = cls.directory / "sdkit3-port-source"
        for name in ("stb_image.h", "stb_image_write.h"):
            relative = Path("stable-diffusion.cpp/thirdparty") / name
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(SDKIT / relative, target)
        cls.compiler = shlex.split(os.environ.get("CXX", "c++"))
        cls.includes = ["-I" + str(path) for path in (
            staged / "include", staged / "third_party", COSMO / "src",
            SDKIT / "stable-diffusion.cpp/thirdparty")]
        cls.options = cls.directory / "split-options"
        cls.build(cls.options, [COSMO / "src/config.cpp", staged / "src/options_manager.cpp",
                              COSMO / "tests/split_options_fixture.cpp"])
        cls.config = cls.directory / "config-probe"
        cls.build(cls.config, [COSMO / "src/config.cpp", COSMO / "tests/config_probe.cpp"])
        # Exercise the actual production routing block with a file-type inspector
        # stub. The test does not load weights or assert model compatibility.
        source = (SDKIT / "src/image_generator.cpp").read_text()
        begin = source.index("    // Collect additional modules for comparison")
        end = source.index('    if (options.has("uni_controlnet_clip_vision_path")', begin)
        fixture = (COSMO / "tests/split_routing_fixture.cpp").read_text()
        routing_source = cls.directory / "split-routing.cpp"
        routing_source.write_text(fixture.replace("/* PRODUCTION_COMPANION_ROUTING */", source[begin:end]))
        cls.routing = cls.directory / "split-routing"
        cls.build(cls.routing, [routing_source])

    @classmethod
    def build(cls, target, sources):
        result = subprocess.run(cls.compiler + ["-std=c++20", "-pthread", "-Wall", "-Wextra", "-Werror",
                                *cls.includes, *map(str, sources), "-o", str(target)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_checked(self, command):
        result = subprocess.run(list(map(str, command)), cwd=self.directory,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_saved_companions_roundtrip_and_invalid_updates_are_atomic(self):
        self.run_checked(["python3", COSMO / "tests/verify_config.py", "--probe", self.config])

    def test_overrides_never_mutate_saved_options_and_recover_after_rejection(self):
        self.run_checked([self.options, "shell", "--config", self.directory / "overlay.json"])

    def test_companion_roles_reach_slots_and_bad_inputs_fail_explicitly(self):
        self.run_checked([self.routing, self.directory])

    def test_image_generator_compiles_with_patched_application_headers(self):
        vendor = SDKIT / "stable-diffusion.cpp"
        includes = ["-I" + str(path) for path in (COSMO / "backend/include", vendor / "include",
                                                  vendor / "src", vendor / "ggml/include")]
        self.run_checked(self.compiler + ["-std=c++20", "-DGGML_MAX_NAME=160", "-fsyntax-only",
                         *self.includes, *includes,
                         self.directory / "sdkit3-port-source/src/image_generator.cpp"])


if __name__ == "__main__":
    unittest.main()
