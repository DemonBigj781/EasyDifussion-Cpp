import importlib.util
import pathlib
import unittest


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
        for value in ("--port 9999", "--ckpt-dir=/tmp/models"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.backend_args.parse_backend_commandline_args(value)

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
