import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnostics", ROOT / "ui/easydiffusion/diagnostics.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnosticsTests(unittest.TestCase):
    def test_log_tail_preserves_traceback_and_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            log = pathlib.Path(folder) / "log"
            log.write_text("old\nERROR failed\nTraceback:\n  model failed\n", encoding="utf8")
            result = diagnostics.read_log_tail(log, 3)
            self.assertEqual(result["lines"], ["ERROR failed", "Traceback:", "  model failed"])
            self.assertTrue(result["truncated"])
            log.write_text("x" * 600000 + "\nERROR recent\n", encoding="utf8")
            self.assertEqual(diagnostics.read_log_tail(log, 3)["lines"], ["ERROR recent"])
            with self.assertRaises(ValueError):
                diagnostics.read_log_tail(log, 1001)

    def test_navigation_metadata_symlinks_and_boundary(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as external:
            root = pathlib.Path(folder)
            (root / "folder").mkdir()
            (root / "file.txt").write_text("example")
            (pathlib.Path(external) / "target.txt").write_text("external")
            (root / "linked").symlink_to(external, target_is_directory=True)
            (root / "broken").symlink_to(root / "missing")
            result = diagnostics.inspect_path(root, ".")
            self.assertEqual(result["path"], ".")
            self.assertEqual(result["type"], "directory")
            self.assertEqual(len(result["entries"]), 4)
            self.assertEqual(diagnostics.inspect_path(root, "file.txt")["size"], 7)
            self.assertEqual(diagnostics.inspect_path(root, "linked/target.txt")["size"], 8)
            self.assertTrue(diagnostics.inspect_path(root, "linked")["symlink"])
            self.assertEqual(diagnostics.inspect_path(root, "broken")["type"], "broken symlink")
            for path in ("..", "../outside", "/etc", "folder/../../outside"):
                with self.assertRaises(ValueError):
                    diagnostics.inspect_path(root, path)
            with self.assertRaises(FileNotFoundError):
                diagnostics.inspect_path(root, "missing")


if __name__ == "__main__":
    unittest.main()
