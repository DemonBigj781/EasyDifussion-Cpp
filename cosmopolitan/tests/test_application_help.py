#!/usr/bin/env python3
"""Run real CLI/config/help code with inert startup stubs, without any drivers."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
COSMO = ROOT / "cosmopolitan"


class ApplicationHelpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="cosmo-application-help-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        includes = [COSMO / "src", COSMO / "backend/include",
                    ROOT / "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty"]
        flags = ["-DCOSMO_WEBGPU_BACKEND=1", "-Wall", "-Wextra", "-Werror"]
        flags += ["-I" + str(path) for path in includes]
        main = cls.directory / "main.o"
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                       ["-std=c11", *flags, "-c", str(COSMO / "src/main.c"), "-o", str(main)],
                       check=True, capture_output=True, text=True)
        cls.executable = cls.directory / "application-help"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++20", "-pthread", *flags, str(main),
            str(COSMO / "src/config.cpp"), str(COSMO / "src/application_commands.cpp"),
            str(COSMO / "tests/application_help_fixture.cpp"), "-o", str(cls.executable),
        ], check=True, capture_output=True, text=True)

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(dir=self.directory)
        self.addCleanup(self.work.cleanup)
        self.path = Path(self.work.name)

    def run_cli(self, *args):
        return subprocess.run([str(self.executable), *args], cwd=self.path,
                              capture_output=True, text=True, timeout=10)

    def assert_help_only(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("infer image --output FILE.png", result.stdout)
        self.assertIn("infer text", result.stdout)
        self.assertNotIn("FIXTURE_", result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")

    def assert_normal_startup(self, result):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "FIXTURE_BACKEND_CONFIGURE", "FIXTURE_BACKEND_INITIALIZE", "FIXTURE_APPLICATION_CREATE",
        ])
        self.assertIn("fixture stops before application services", result.stderr)

    def test_inference_help_never_starts_backends_or_services(self):
        for kind in ("image", "text"):
            for help_flag in ("--help", "-h"):
                for provider in (None, "auto", "native", "embedded"):
                    with self.subTest(kind=kind, help=help_flag, provider=provider):
                        args = ["infer", kind]
                        if provider:
                            args += ["--backend", "webgpu", "--provider", provider,
                                     "--device", "unavailable-device"]
                        self.assert_help_only(self.run_cli(*args, help_flag))
                        self.assertEqual(list(self.path.iterdir()), [])

    def test_help_with_global_options_before_command(self):
        self.assert_help_only(self.run_cli("--backend", "cpu", "--provider", "native",
                                           "infer", "image", "--help"))
        self.assertEqual(list(self.path.iterdir()), [])

    def test_help_does_not_read_configuration_or_request_file(self):
        config = self.path / "invalid.json"
        config.write_text("not valid JSON")
        self.assert_help_only(self.run_cli("infer", "image", "--config", str(config),
                                           "--request-file", "missing-request.json", "--help"))
        self.assertEqual(config.read_text(), "not valid JSON")
        self.assertEqual(list(self.path.iterdir()), [config])

    def test_help_after_ordinary_options_does_not_create_output(self):
        self.assert_help_only(self.run_cli("infer", "image", "--prompt", "a scene",
                                           "--output", "image.png", "--steps", "1", "--help"))
        self.assertEqual(list(self.path.iterdir()), [])

    def test_help_shaped_values_are_not_help_flags(self):
        for kind in ("image", "text"):
            for option in ("--prompt", "--request-file"):
                for value in ("--help", "-h"):
                    with self.subTest(kind=kind, option=option, value=value):
                        self.assert_normal_startup(self.run_cli("infer", kind, option, value))

    def test_real_help_after_help_shaped_value(self):
        self.assert_help_only(self.run_cli("infer", "text", "--prompt", "--help", "-h"))
        self.assertEqual(list(self.path.iterdir()), [])

    def test_non_help_inference_keeps_startup(self):
        self.assert_normal_startup(self.run_cli("infer", "image", "--prompt", "a scene"))

    def test_shell_script_is_not_top_level_inference_help(self):
        self.assert_normal_startup(self.run_cli("shell", "-c", "infer image --help"))

    def test_global_help_still_bypasses_startup(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Usage:", result.stdout)
        self.assertNotIn("FIXTURE_", result.stdout + result.stderr)
        self.assertEqual(list(self.path.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
