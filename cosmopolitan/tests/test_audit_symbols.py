"""Compile database selection must match the packaged foundation identity."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_symbols import compile_commands_path


class CompileCommandsSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.legacy = self.out / "build/compile_commands.json"
        self.legacy.parent.mkdir()
        self.legacy.write_text('[{"stale": true}]')
        self.identity = "ab" * 32

    def manifest(self, metadata):
        (self.out / "LINK.json").write_text(json.dumps(metadata))

    def test_legacy_without_manifest(self):
        self.assertEqual(compile_commands_path(self.out), self.legacy)

    def test_legacy_manifest(self):
        self.manifest({"cosmocc": {"version": "3.9.7"}})
        self.assertEqual(compile_commands_path(self.out), self.legacy)

    def test_selected_identity_not_stale_or_other_build(self):
        selected = self.out / ("build-source-" + self.identity) / "compile_commands.json"
        selected.parent.mkdir()
        selected.write_text('[{"selected": true}]')
        other = self.out / ("build-source-" + "c" * 64)
        other.mkdir()
        (other / "compile_commands.json").write_text('[]')
        self.manifest({"foundation_handoff": {"selection_identity_sha256": self.identity}})
        self.assertEqual(compile_commands_path(self.out), selected)
        self.assertEqual(json.loads(compile_commands_path(self.out).read_text()), [{"selected": True}])

    def test_missing_selected_database_does_not_fall_back(self):
        self.manifest({"foundation_handoff": {"selection_identity_sha256": self.identity}})
        with self.assertRaises(FileNotFoundError):
            compile_commands_path(self.out).read_text()

    def test_invalid_identities_rejected(self):
        for identity in (None, 123, "", "a" * 63, "a" * 65, "A" * 64,
                         "g" * 64, "../build", "a" * 64 + "\n"):
            with self.subTest(identity=identity):
                self.manifest({"foundation_handoff": {"selection_identity_sha256": identity}})
                with self.assertRaisesRegex(RuntimeError, "selection_identity_sha256"):
                    compile_commands_path(self.out)

    def test_invalid_handoffs_rejected(self):
        for handoff in (None, [], "legacy", {}):
            with self.subTest(handoff=handoff):
                self.manifest({"foundation_handoff": handoff})
                with self.assertRaises(RuntimeError):
                    compile_commands_path(self.out)

    def test_source_runtime_missing_handoff_rejected(self):
        self.manifest({"cosmocc": {"version": "source-built"}})
        with self.assertRaisesRegex(RuntimeError, "missing foundation_handoff"):
            compile_commands_path(self.out)

    def test_malformed_manifest_not_ignored(self):
        (self.out / "LINK.json").write_text('{broken')
        with self.assertRaises(json.JSONDecodeError):
            compile_commands_path(self.out)
        self.manifest([])
        with self.assertRaises(RuntimeError):
            compile_commands_path(self.out)


if __name__ == "__main__":
    unittest.main()
