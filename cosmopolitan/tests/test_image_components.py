#!/usr/bin/env python3
"""Compile the real image frontend with model-free engine stubs; never infer."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ImageComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.executable = cls.directory / "image-cli"
        compile_command = [
            os.environ.get("CC", "cc"), "-std=gnu11", "-Werror=implicit-function-declaration",
            "-I" + str(ROOT / "cosmopolitan/src"),
            "-I" + str(ROOT / "source/llama.cpp/ggml/include"),
            "-I" + str(ROOT / "source/sdkit3-port-source/stable-diffusion.cpp/include"),
            str(ROOT / "cosmopolitan/src/image_app.c"),
            str(ROOT / "cosmopolitan/tests/image_cli_fixture.c"), "-o", str(cls.executable)
        ]
        subprocess.run(compile_command, check=True, capture_output=True, text=True)
        cls.webgpu_executable = cls.directory / "image-cli-webgpu"
        compile_command[-1] = str(cls.webgpu_executable)
        compile_command[1:1] = ["-DCOSMO_WEBGPU_BACKEND", "-I" + str(ROOT / "cosmopolitan/backend/include")]
        subprocess.run(compile_command, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.files = {}
        for name in ("model", "diffusion-model", "clip-l", "clip-g", "t5xxl", "vae"):
            p = self.directory / (name + ".weights")
            p.write_bytes(b"not a model: fixture never loads weights")
            self.files[name] = p
        self.output = self.directory / "out.png"
        self.output.unlink(missing_ok=True)

    def run_cli(self, *args):
        return subprocess.run([str(self.executable), "--prompt", "test", "--output", str(self.output), *map(str, args)],
                              capture_output=True, text=True)

    def test_bundled_checkpoint_preserved(self):
        p = self.run_cli("--model", self.files["model"])
        self.assertEqual(p.returncode, 1)  # Fixture intentionally stops at context creation.
        self.assertIn("FIXTURE_MODEL=" + str(self.files["model"]), p.stdout)
        self.assertIn("FIXTURE_DIFFUSION=\n", p.stdout)

    def test_separate_components_reach_engine(self):
        args = []
        for name in ("diffusion-model", "clip-l", "clip-g", "t5xxl", "vae"):
            args.extend(["--" + name, self.files[name]])
        p = self.run_cli(*args)
        self.assertEqual(p.returncode, 1)
        for label, name in (("DIFFUSION", "diffusion-model"), ("CLIP_L", "clip-l"),
                            ("CLIP_G", "clip-g"), ("T5", "t5xxl"), ("VAE", "vae")):
            self.assertIn("FIXTURE_" + label + "=" + str(self.files[name]), p.stdout)
        self.assertIn("FIXTURE_MODEL=\n", p.stdout)
        self.assertFalse(self.output.exists())

    def test_requires_exactly_one_base(self):
        for args in ([], ["--model", self.files["model"], "--diffusion-model", self.files["diffusion-model"]]):
            with self.subTest(args=args):
                p = self.run_cli(*args)
                self.assertEqual(p.returncode, 2)
                self.assertNotIn("FIXTURE_", p.stdout)

    def test_each_empty_component_rejected(self):
        for name in self.files:
            with self.subTest(name=name):
                args = [] if name in ("model", "diffusion-model") else ["--model", self.files["model"]]
                p = self.run_cli(*args, "--" + name, "")
                self.assertEqual(p.returncode, 2)
                self.assertNotIn("FIXTURE_", p.stdout)

    def test_each_missing_component_rejected(self):
        for name in self.files:
            with self.subTest(name=name):
                args = [] if name in ("model", "diffusion-model") else ["--model", self.files["model"]]
                p = self.run_cli(*args, "--" + name, self.directory / "absent")
                self.assertEqual(p.returncode, 1)
                self.assertNotIn("FIXTURE_", p.stdout)

    def test_output_cannot_alias_any_component(self):
        for name in self.files:
            for link in (os.link, os.symlink):
                with self.subTest(name=name, link=link.__name__):
                    self.output.unlink(missing_ok=True)
                    link(self.files[name], self.output)
                    args = [] if name in ("model", "diffusion-model") else ["--model", self.files["model"]]
                    p = self.run_cli(*args, "--" + name, self.files[name])
                    self.assertEqual(p.returncode, 2)
                    self.assertNotIn("FIXTURE_", p.stdout)
                    self.assertEqual(self.files[name].read_bytes(), b"not a model: fixture never loads weights")

    def test_duplicate_component_rejected(self):
        p = self.run_cli("--model", self.files["model"], "--clip-l", self.files["clip-l"], "--clip-l", self.files["clip-l"])
        self.assertEqual(p.returncode, 2)

    def test_cpu_does_not_initialize_unavailable_webgpu(self):
        p = subprocess.run([str(self.webgpu_executable), "--backend", "cpu", "--model", str(self.files["model"]),
                            "--prompt", "test", "--output", str(self.output)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("FIXTURE_MODEL=", p.stdout)
        self.assertNotIn("FIXTURE_WEBGPU_INITIALIZED", p.stdout)

    def test_webgpu_failure_does_not_silently_use_cpu(self):
        p = subprocess.run([str(self.webgpu_executable), "--backend", "webgpu", "--model", str(self.files["model"]),
                            "--prompt", "test", "--output", str(self.output)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("FIXTURE_WEBGPU_INITIALIZED", p.stdout)
        self.assertNotIn("FIXTURE_MODEL=", p.stdout)

    def test_help_lists_split_inputs(self):
        p = subprocess.run([str(self.executable), "--help"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        for name in self.files:
            self.assertIn("--" + name, p.stdout)


if __name__ == "__main__":
    unittest.main()
