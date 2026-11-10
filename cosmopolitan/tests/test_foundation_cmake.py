#!/usr/bin/env python3
"""Compile/link-only checks for the dependency manifest's CMake bridge.

These tests use tiny synthetic C archives and the host compiler, not the
Cosmopolitan SDK or application. No generated executable is ever launched.
They do not load models, exercise WebGPU or consume GPU resources.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import foundation_consumer as consumer
import test_foundation_consumer as fixtures


def program(variable, default):
    return os.environ.get(variable) or shutil.which(default)


TOOLS = {name: program(variable, name) for name, variable in (
    ("cmake", "COSMO_TEST_CMAKE"), ("ninja", "COSMO_TEST_NINJA"),
    ("cc", "COSMO_TEST_CC"), ("ar", "COSMO_TEST_AR"),
)}
BRIDGE = Path(__file__).resolve().parents[1] / "cmake/software-webgpu.cmake"


@unittest.skipUnless(sys.platform.startswith("linux") and all(TOOLS.values()),
                     "Linux host cmake/ninja/cc/ar required; explicit COSMO_TEST_* paths are supported")
class FoundationCMakeTests(unittest.TestCase):
    def setUp(self):
        # Reuse the complete synthetic provenance fixture, never a real SDK.
        self.fixture = fixtures.FoundationConsumerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        temporary = tempfile.TemporaryDirectory(prefix="ed foundation cmake ")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "TEMP", "TMP") if key in os.environ}
        for name, text in {
            "webgpu": "extern int ed_mesa(void); int ed_wgpu(void) { return ed_mesa(); }\n",
            "mesa": "extern int ed_llvm(void); int ed_mesa(void) { return ed_llvm(); }\n",
            "llvm": "int ed_llvm(void) { return 3; }\n",
        }.items():
            obj = self.compile_object(name, text)
            archive = self.work / (name + ".a")
            self.command([TOOLS["ar"], "rcs", archive, obj])
            self.fixture.artifact("deps/lib/" + name + ".a", "archive", archive.read_bytes())
        shim = self.compile_object("runtime", "int ed_shim(void) { return 4; }\n")
        self.fixture.artifact("deps/lib/runtime.o", "object", shim.read_bytes())
        self.fixture.artifact("deps/include/webgpu.h", "data", b"int ed_wgpu(void);\n")
        self.fixture.artifact("deps/runtime/bridge.c", "data",
                              b"#include <webgpu.h>\nextern int ed_shim(void);\n"
                              b"int ed_bridge(void) { return ed_wgpu() + ed_shim(); }\n")
        self.fixture.artifact("deps/provenance/rust_std.json", "data", json.dumps({
            "runtime": self.fixture.upstream, "archive_delivery": {
                "mode": "embedded-in-wgpu",
                "container_sha256": self.fixture.data["artifacts"]["deps/lib/webgpu.a"]["sha256"],
            },
        }).encode())
        # The host fixture does not link the source SDK's libcxx. The real
        # producer retains its own reviewed C++/POSIX closure unchanged.
        self.fixture.link["driver_link_args"].remove("-lcxx")
        self.fixture.link["runtime_link_options"] = ["-Wl,--defsym,ed_alias=ed_bridge"]
        self.fixture.link["file_sha256"] = {
            str(self.fixture.root / name): entry["sha256"]
            for name, entry in self.fixture.data["artifacts"].items()
            if name.startswith("deps/") and name != "deps/LINK.json"
        }
        self.fixture.make_relative()
        self.selected = self.fixture.select()
        self.view = consumer.write_link_view(self.work / "generated", self.selected)
        self.project = self.work / "project"
        self.project.mkdir()
        (self.project / "main.c").write_text("extern int ed_alias(void); int main(void) { return ed_alias(); }\n")
        (self.project / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(manifest_contract LANGUAGES C)\n"
            "include(\"${CONTRACT_BRIDGE}\")\n"
            "add_executable(contract_probe main.c)\n"
            "target_link_libraries(contract_probe PRIVATE cosmo-software-webgpu)\n")

    def command(self, arguments, expected=0):
        result = subprocess.run([str(value) for value in arguments], env=self.env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def compile_object(self, name, content):
        source = self.work / (name + ".c")
        obj = self.work / (name + ".o")
        source.write_text(content)
        self.command([TOOLS["cc"], "-c", source, "-o", obj])
        return obj

    def configure(self, metadata=None, sha=None, expected=0):
        return self.command([
            TOOLS["cmake"], "-S", self.project, "-B", self.work / "build", "-G", "Ninja",
            "-DCMAKE_C_COMPILER=" + TOOLS["cc"], "-DCMAKE_MAKE_PROGRAM=" + TOOLS["ninja"],
            "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY", "-DCONTRACT_BRIDGE=" + str(BRIDGE),
            "-DCOSMO_SOFTWARE_WEBGPU_METADATA=" + str(metadata or self.view),
            "-DCOSMO_FOUNDATION_LINK_SHA256=" + (self.selected["link_view_sha256"] if sha is None else sha),
        ], expected=expected)

    def test_generated_view_compiles_and_links_without_running(self):
        self.configure()
        self.command([TOOLS["cmake"], "--build", self.work / "build", "--target", "contract_probe"])
        self.assertTrue((self.work / "build/contract_probe").is_file())

    def test_relocated_export_paths_with_spaces_compile_and_link(self):
        relocated = self.work / "export with spaces"
        shutil.copytree(self.fixture.root, relocated)
        self.selected = consumer.select_foundation(
            relocated / "HANDOFF.json", relocated, relocated / "sdk", relocated / "deps/LINK.json",
            pin=self.fixture.pin, application_patches=self.fixture.patches)
        self.view = consumer.write_link_view(self.work / "relocated-view", self.selected)
        self.configure()
        self.command([TOOLS["cmake"], "--build", self.work / "build", "--target", "contract_probe"])
        self.assertTrue((self.work / "build/contract_probe").is_file())

    def test_relative_producer_manifest_cannot_skip_consumer(self):
        output = self.configure(self.fixture.root / "deps/LINK.json", sha="", expected=1)
        self.assertIn("require a verified absolute view", output)

    def test_generated_view_requires_bound_digest(self):
        output = self.configure(sha="", expected=1)
        self.assertIn("require the verified view SHA256", output)

    def test_changed_view_is_rejected_before_target_compilation(self):
        self.view.write_bytes(self.view.read_bytes() + b"\n")
        output = self.configure(expected=1)
        self.assertIn("view changed before CMake configuration", output)
        self.assertFalse((self.work / "build/contract_probe").exists())

    def test_malformed_bound_digest_is_rejected(self):
        output = self.configure(sha="not-a-digest", expected=1)
        self.assertIn("Invalid source-runtime dependency view SHA256", output)

    def test_hash_alone_cannot_relabel_original_source_metadata(self):
        data = dict(self.selected["link_view"])
        del data["consumer_view"]
        original = self.work / "original-absolute.json"
        original.write_text(json.dumps(data))
        output = self.configure(original, hashlib.sha256(original.read_bytes()).hexdigest(), expected=1)
        self.assertIn("requires a generated verified consumer view", " ".join(output.split()))

    def changed_view(self, data):
        path = self.work / "changed-view.json"
        path.write_text(json.dumps(data))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_wrong_consumer_marker_type_is_rejected(self):
        data = dict(self.selected["link_view"])
        data["consumer_view"] = []
        output = self.configure(*self.changed_view(data), expected=1)
        self.assertIn("requires a generated verified consumer view", " ".join(output.split()))

    def test_unsupported_consumer_schema_is_rejected(self):
        data = dict(self.selected["link_view"])
        data["consumer_view"] = {"schema_version": 2}
        output = self.configure(*self.changed_view(data), expected=1)
        self.assertIn("Unsupported source-runtime consumer view schema", output)

    def test_missing_consumer_schema_is_rejected(self):
        data = dict(self.selected["link_view"])
        data["consumer_view"] = {}
        output = self.configure(*self.changed_view(data), expected=1)
        self.assertIn("schema_version", output)

    def test_absolute_source_metadata_without_view_or_digest_is_rejected(self):
        data = dict(self.selected["link_view"])
        del data["consumer_view"]
        path, unused = self.changed_view(data)
        output = self.configure(path, sha="", expected=1)
        self.assertIn("require the verified view SHA256", output)

    def test_legacy_absolute_metadata_without_source_binding_still_links(self):
        data = dict(self.selected["link_view"])
        for key in ("consumer_view", "path_mode", "runtime"):
            del data[key]
        legacy = self.work / "legacy-absolute.json"
        legacy.write_text(json.dumps(data))
        self.configure(legacy, sha="")
        self.command([TOOLS["cmake"], "--build", self.work / "build", "--target", "contract_probe"])
        self.assertTrue((self.work / "build/contract_probe").is_file())


if __name__ == "__main__":
    unittest.main()
