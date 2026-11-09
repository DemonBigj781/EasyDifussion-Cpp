#!/usr/bin/env python3
"""Contract tests. Do not run until the implementation-phase test gate opens."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "foundation_handoff.py"
SPEC = importlib.util.spec_from_file_location("foundation_handoff", SOURCE)
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)
REPOSITORY = "https://github.com/example/foundation.git"
REVISION = "0123456789abcdef" * 2 + "01234567"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def archive_member(name, payload):
    header = (name.ljust(16) + b"0".ljust(12) + b"0".ljust(6) +
              b"0".ljust(6) + b"100644".ljust(8) +
              str(len(payload)).encode("ascii").ljust(10) + b"`\n")
    return header + payload + (b"\n" if len(payload) % 2 else b"")


def archive_bytes(payload=b"fixture member, never executed"):
    return b"!<arch>\n" + archive_member(b"fixture.o/", payload)


class FoundationHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = {
            "format": handoff.FORMAT,
            "schema_version": 1,
            "foundation": {"repository": REPOSITORY, "revision": REVISION,
                           "source_tree_sha256": digest(b"source tree"), "clean": True},
            "target": {"architecture": "x86_64", "abi": "cosmopolitan-ape",
                       "c_triple": "x86_64-linux-cosmo", "rust_triple": "x86_64-unknown-linux-musl"},
            "runtime": {"kind": "source-built",
                        "provenance": {"repository": REPOSITORY, "revision": REVISION,
                                       "patches_sha256": digest(b"patch inventory")},
                        "provenance_record": "provenance/runtime.json",
                        "headers": {"path": "include", "tree_sha256": ""},
                        "crt_objects": ["lib/crt.o"], "archives": ["lib/runtime.a"],
                        "cxx_abi_archives": ["lib/cxx.a"]},
            "toolchain": {name: "bin/" + name for name in handoff.TOOLS},
            "runtime_identity_sha256": "",
            "components": {},
            "artifacts": {},
        }
        self.data["toolchain"]["support_files"] = ["lib/compiler-support.a"]
        (self.root / "include").mkdir()
        (self.root / "include/stdio.h").write_text("/* fixture header */\n")
        self.data["runtime"]["headers"]["tree_sha256"] = handoff.header_tree_sha256(self.root, "include")
        for name in handoff.TOOLS:
            self.artifact("bin/" + name, "executable", b"#!/bin/sh\nexit 1\n")
        self.artifact("lib/crt.o", "object", b"fixture object, never executed")
        self.artifact("provenance/runtime.json", "data", b'{"fixture": true}\n')
        for name in ("runtime", "cxx", "compiler-support", *handoff.COMPONENTS):
            self.artifact("lib/" + name + ".a", "archive", archive_bytes())
        for name in handoff.COMPONENTS:
            self.artifact("provenance/" + name + ".json", "data", b"{}\n")
            self.data["components"][name] = {
                "version": "fixture-1", "runtime_identity_sha256": "",
                "target_triple": self.data["target"]["rust_triple" if name in ("rust_std", "webgpu") else "c_triple"],
                "archives": ["lib/" + name + ".a"], "support_objects": [],
                "provenance": "provenance/" + name + ".json",
            }
        self.identify()

    def artifact(self, name, kind, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o755 if kind == "executable" else 0o644)
        self.data["artifacts"][name] = {"kind": kind, "sha256": digest(content)}

    def identify(self):
        identity = handoff.runtime_identity_sha256(self.data)
        self.data["runtime_identity_sha256"] = identity
        for component in self.data["components"].values():
            component["runtime_identity_sha256"] = identity

    def validate(self):
        return handoff.validate_handoff(self.data, self.root,
                                        expected_repository=REPOSITORY, expected_revision=REVISION)

    def reject(self, text):
        with self.assertRaisesRegex(handoff.HandoffError, text):
            self.validate()

    def test_complete_source_handoff_only_claims_static_identity(self):
        report = self.validate()
        self.assertEqual(report["runtime_kind"], "source-built")
        self.assertEqual(report["verification"], "static-identity-only")
        self.assertFalse(report["toolchain_selected"])
        self.assertFalse(report["runtime_behavior_verified"])

    def test_stock_sdk_is_explicit_and_has_a_different_identity(self):
        original = self.data["runtime_identity_sha256"]
        self.data["runtime"]["kind"] = "stock-sdk"
        self.data["runtime"]["provenance"] = {"version": "4.0.2", "archive_sha256": digest(b"sdk")}
        self.identify()
        self.assertNotEqual(self.data["runtime_identity_sha256"], original)
        self.assertEqual(self.validate()["runtime_kind"], "stock-sdk")

    def test_wrong_source_pin(self):
        self.data["foundation"]["revision"] = "123456789abcdef0" * 2 + "12345678"
        self.reject("selected repository/revision")

    def test_dirty_source(self):
        self.data["foundation"]["clean"] = False
        self.reject("clean pinned tree")

    def test_boolean_is_not_schema_version(self):
        self.data["schema_version"] = True
        self.reject("format/version")

    def test_unknown_keys_are_not_silently_ignored(self):
        self.data["future_runtime"] = {}
        self.reject("expected exactly")

    def test_empty_or_placeholder_identity(self):
        for placeholder in ("", "0" * 64, "TODO"):
            with self.subTest(placeholder=placeholder):
                self.data["foundation"]["source_tree_sha256"] = placeholder
                self.reject("real lowercase")

    def test_unpinned_source_branch(self):
        self.data["runtime"]["provenance"]["revision"] = "main"
        self.reject("full pinned Git revision")

    def test_runtime_identity_changes_with_cxx_abi(self):
        self.artifact("lib/cxx.a", "archive", archive_bytes(b"changed"))
        self.reject("compiler/header/CRT/C\\+\\+ ABI")

    def test_components_cannot_mix_runtime_identities(self):
        for name in handoff.COMPONENTS:
            with self.subTest(component=name):
                original = self.data["components"][name]["runtime_identity_sha256"]
                self.data["components"][name]["runtime_identity_sha256"] = digest(b"other runtime")
                self.reject("different declared runtime")
                self.data["components"][name]["runtime_identity_sha256"] = original

    def test_component_cannot_reuse_a_different_architecture(self):
        self.data["components"]["llvm"]["target_triple"] = "aarch64-linux-cosmo"
        self.reject("target triple differs")

    def test_requested_architecture_is_independent_of_manifest(self):
        self.data["target"]["architecture"] = "aarch64"
        self.reject("requested APE architecture")

    def test_rust_target_is_not_assumed_to_equal_c_target(self):
        self.data["components"]["rust_std"]["target_triple"] = self.data["target"]["c_triple"]
        self.reject("target triple differs")

    def test_artifacts_cannot_be_omitted(self):
        del self.data["artifacts"]["lib/mesa.a"]
        self.reject("lacks a digest")

    def test_cxx_abi_closure_cannot_be_empty(self):
        self.data["runtime"]["cxx_abi_archives"] = []
        self.reject("nonempty list")

    def test_upstream_runtime_provenance_is_bound(self):
        self.artifact("provenance/runtime.json", "data", b'{"fixture": "changed"}\n')
        self.reject("compiler/header/CRT/C\\+\\+ ABI")

    def test_compiler_support_cannot_be_omitted(self):
        self.data["toolchain"]["support_files"] = []
        self.reject("nonempty list")

    def test_missing_artifact(self):
        (self.root / "lib/llvm.a").unlink()
        self.reject("Missing regular artifact")

    def test_mutated_artifact(self):
        (self.root / "lib/mesa.a").write_bytes(b"!<arch>\nmutated")
        self.reject("Artifact SHA256 mismatch")

    def test_thin_archive_rejected_even_with_matching_hash(self):
        self.artifact("lib/mesa.a", "archive", b"!<thin>\n")
        self.reject("complete static archive")

    def test_shared_library_cannot_satisfy_static_archive(self):
        self.artifact("lib/mesa.a", "archive", b"\x7fELFshared fixture")
        self.reject("complete static archive")

    def test_empty_archive_is_rejected(self):
        self.artifact("lib/mesa.a", "archive", b"!<arch>\n")
        self.reject("no object/data members")

    def test_truncated_archive_is_rejected(self):
        self.artifact("lib/mesa.a", "archive", archive_bytes()[:-2])
        self.reject("Truncated archive member")

    def test_malformed_archive_is_rejected(self):
        self.artifact("lib/mesa.a", "archive", b"!<arch>\n" + b"x" * 60)
        self.reject("Malformed archive member header")

    def test_gnu_long_name_requires_a_string_table(self):
        self.artifact("lib/mesa.a", "archive", b"!<arch>\n" + archive_member(b"/999999", b"object"))
        self.reject("Missing/out-of-bounds GNU archive member name")

    def test_gnu_long_name_offset_must_be_in_bounds(self):
        content = (b"!<arch>\n" + archive_member(b"//", b"long-member.o/\n") +
                   archive_member(b"/999999", b"object"))
        self.artifact("lib/mesa.a", "archive", content)
        self.reject("Missing/out-of-bounds GNU archive member name")

    def test_gnu_long_name_must_start_at_an_entry(self):
        content = (b"!<arch>\n" + archive_member(b"//", b"long-member.o/\n") +
                   archive_member(b"/2", b"object"))
        self.artifact("lib/mesa.a", "archive", content)
        self.reject("does not start at an entry")

    def test_valid_gnu_long_name_archive(self):
        content = (b"!<arch>\n" + archive_member(b"//", b"long-member.o/\n") +
                   archive_member(b"/0", b"object"))
        self.artifact("lib/mesa.a", "archive", content)
        self.validate()

    def test_valid_bsd_long_name_archive(self):
        content = b"!<arch>\n" + archive_member(b"#1/16", b"long-member.o\0\0\0" + b"object")
        self.artifact("lib/mesa.a", "archive", content)
        self.validate()

    def test_nonexecutable_tool(self):
        (self.root / "bin/c_compiler").chmod(0o644)
        self.reject("not executable")

    def test_header_content_or_membership_change(self):
        (self.root / "include/new.h").write_text("/* new header */")
        self.reject("header tree SHA256 mismatch")

    def test_empty_header_tree(self):
        (self.root / "include/stdio.h").unlink()
        self.reject("header tree is empty")

    def test_header_symlink(self):
        (self.root / "include/stdio.h").unlink()
        (self.root / "include/stdio.h").symlink_to(self.root / "lib/crt.o")
        self.reject("cannot contain a symlink")

    def test_unreadable_header_subtree_cannot_be_omitted(self):
        blocked = self.root / "include/blocked"
        blocked.mkdir()
        scandir = handoff.os.scandir

        def scan(path):
            if Path(path) == blocked:
                raise PermissionError("fixture: directory is unreadable")
            return scandir(path)

        with mock.patch.object(handoff.os, "scandir", side_effect=scan):
            self.reject("Cannot read complete header tree")

    def test_artifact_symlink(self):
        (self.root / "lib/mesa.a").unlink()
        (self.root / "lib/mesa.a").symlink_to(self.root / "lib/llvm.a")
        self.reject("symlink is not allowed")

    def test_unsafe_artifact_paths(self):
        original = copy.deepcopy(self.data)
        for path in ("../escape", "/absolute", "C:/drive", "lib/../escape", "lib//a", "lib\\a"):
            with self.subTest(path=path):
                self.data = copy.deepcopy(original)
                self.data["artifacts"][path] = {"kind": "data", "sha256": digest(b"data")}
                self.reject("relative POSIX path")

    def test_duplicate_json_keys(self):
        path = self.root / "duplicate.json"
        path.write_text('{"schema_version": 1, "schema_version": 2}')
        with self.assertRaisesRegex(handoff.HandoffError, "Duplicate JSON key"):
            handoff.load_handoff(path)

    def test_json_round_trip(self):
        path = self.root / "handoff.json"
        path.write_text(json.dumps(self.data))
        self.assertEqual(handoff.load_handoff(path), self.data)


if __name__ == "__main__":
    unittest.main()
