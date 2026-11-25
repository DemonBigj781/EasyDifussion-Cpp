import importlib.util
import pathlib
import unittest
from unittest import mock


def _load_backend_args():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    module_path = repo_root / "ui" / "easydiffusion" / "backend_args.py"
    spec = importlib.util.spec_from_file_location("backend_args", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBackendCommandlineArgs(unittest.TestCase):
    def setUp(self):
        self.backend_args = _load_backend_args()

    def test_shell_quoting_becomes_an_argv_list(self):
        self.assertEqual(
            self.backend_args.parse_backend_commandline_args('--vae-tiles 32 --name "two words"'),
            ["--vae-tiles", "32", "--name", "two words"],
        )

    def test_list_is_normalized_to_strings(self):
        self.assertEqual(
            self.backend_args.parse_backend_commandline_args(["--vae-tiles", 32]),
            ["--vae-tiles", "32"],
        )

    def test_managed_options_are_rejected_in_both_forms(self):
        directory_options = (
            *self.backend_args.MODEL_DIRECTORY_FLAGS.values(),
            *self.backend_args.LEGACY_MODEL_DIRECTORY_FLAGS.values(),
        )
        for value in ("--port 9999", *(f"{flag}=/tmp/models" for flag in directory_options)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.backend_args.parse_backend_commandline_args(value)

    def test_only_model_directories_are_managed(self):
        args = ["--control-net-sd1-path", "/tmp/model.safetensors", "--convert-model", "/tmp/source"]
        self.assertEqual(self.backend_args.parse_backend_commandline_args(args), args)

    def test_native_bundle_directory_dialect_is_probed(self):
        module = self.backend_args
        def option_lines(flags):
            return "\n".join(f"  {flag} <dir>                 Model directory" for flag in flags)

        old = option_lines(module.LEGACY_MODEL_DIRECTORY_FLAGS.values())
        new = option_lines(module.MODEL_DIRECTORY_FLAGS.values())
        cases = (
            ("", old, module.LEGACY_MODEL_DIRECTORY_FLAGS),
            (new, "", module.MODEL_DIRECTORY_FLAGS),
            (new, old, module.MODEL_DIRECTORY_FLAGS),
            ("  --something            See --dir-lora <dir>", old, module.LEGACY_MODEL_DIRECTORY_FLAGS),
        )
        for stdout, stderr, expected in cases:
            env = {"EXAMPLE": "1"}
            with self.subTest(expected=expected), mock.patch.object(module.subprocess, "run") as run:
                run.return_value.stdout = stdout
                run.return_value.stderr = stderr
                self.assertEqual(module.probe_model_directory_flags("/bin/sdk", cwd="/tmp", env=env), expected)
                run.assert_called_once_with(["/bin/sdk", "--help"], cwd="/tmp", env=env,
                                            timeout=5, capture_output=True, text=True,
                                            errors="replace", check=True)

    def test_incomplete_or_unavailable_directory_dialect_is_detected(self):
        module = self.backend_args
        with mock.patch.object(module.subprocess, "run") as run:
            run.return_value.stdout = "  --dir-lora <dir>  LoRA models"
            run.return_value.stderr = ""
            with self.assertRaisesRegex(RuntimeError, "complete supported"):
                module.probe_model_directory_flags("/bin/sdk", cwd="/tmp", env={})
            run.side_effect = module.subprocess.TimeoutExpired(["/bin/sdk", "--help"], 5)
            with self.assertRaisesRegex(RuntimeError, "Could not read native sdkit"):
                module.probe_model_directory_flags("/bin/sdk", cwd="/tmp", env={})

    def test_help_is_rejected(self):
        with self.assertRaises(ValueError):
            self.backend_args.parse_backend_commandline_args("--help")

    def test_shell_operators_are_plain_arguments(self):
        self.assertEqual(
            self.backend_args.parse_backend_commandline_args("--name ok;touch /tmp/not-run"),
            ["--name", "ok;touch", "/tmp/not-run"],
        )


if __name__ == "__main__":
    unittest.main()
