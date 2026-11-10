#!/usr/bin/env python3
"""Non-model argument-routing cases; application work and subprocesses are mocked."""

import importlib.util
from contextlib import nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_foundation_consumer as fixtures

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("application_build", ROOT / "build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class FoundationBuildOptionsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = temporary.name
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.application = mock.patch.object(build, "build_application").start()
        self.addCleanup(mock.patch.stopall)
        self.selection = mock.patch.object(build, "source_selection", return_value={"fixture": "selected"}).start()
        # None of these routing cases is allowed to invoke build tools, even
        # when the test suite is eventually run.
        mock.patch.object(build.subprocess, "run", side_effect=AssertionError("unexpected child process")).start()
        mock.patch.object(build.subprocess, "check_output", side_effect=AssertionError("unexpected child process")).start()

    def invoke(self, *arguments):
        with mock.patch.object(sys, "argv", ["build.py", "--out", self.out, *arguments]):
            build.main()

    def source_args(self):
        return ["--foundation-handoff", "/export/HANDOFF.json", "--foundation-root", "/export",
                "--runtime-sdk", "/export/sdk", "--software-webgpu-metadata", "/export/deps/LINK.json"]

    def test_legacy_route_remains_the_default(self):
        self.invoke()
        self.selection.assert_not_called()
        self.assertIsNone(self.application.call_args.args[0].source_selection)

    def test_complete_explicit_selection_is_passed_to_build(self):
        self.invoke(*self.source_args())
        self.selection.assert_called_once()
        self.assertEqual(self.application.call_args.args[0].source_selection, {"fixture": "selected"})

    def test_partial_source_options_fail_before_any_build(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke("--foundation-handoff", "/export/HANDOFF.json")
        self.assertEqual(error.exception.code, 2)
        self.selection.assert_not_called()
        self.application.assert_not_called()

    def test_legacy_staging_cannot_mix_with_a_prebuilt_source_closure(self):
        with self.assertRaises(SystemExit):
            self.invoke(*self.source_args(), "--foundation-source", "/other/source")
        self.application.assert_not_called()

    def test_source_cache_is_not_enabled_without_preprocessing_qualification(self):
        with self.assertRaises(SystemExit):
            self.invoke(*self.source_args(), "--compiler-launcher", "sccache")
        self.selection.assert_not_called()
        self.application.assert_not_called()

    def test_ambient_source_profile_cannot_silently_use_stock_sdk(self):
        os.environ["COSMO_RUNTIME_PROFILE"] = "source-built"
        with self.assertRaises(SystemExit):
            self.invoke()
        self.application.assert_not_called()

    def test_ambient_source_sdk_requires_complete_explicit_selection(self):
        os.environ["COSMO_RUNTIME_SDK"] = "/other/sdk"
        with self.assertRaises(SystemExit):
            self.invoke()
        self.application.assert_not_called()


class FoundationBuildPlumbingTests(unittest.TestCase):
    """Exercise build orchestration without running any compiler or model."""

    def setUp(self):
        self.fixture = fixtures.FoundationConsumerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.make_relative()
        self.selected = self.fixture.select()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.args = SimpleNamespace(source_selection=self.selected, dependencies_only=False,
                                    prepare_only=False, libraries_only=True, cache_replay=False,
                                    software_webgpu_out=None, foundation_source=None, jobs=1)
        self.cache = SimpleNamespace(report={}, launcher=None, phase=lambda name: nullcontext(),
                                     env={"PATH": "/usr/bin", "HOME": str(self.out),
                                          "CC": "/unselected/compiler", "LDFLAGS": "-L/unselected"})
        self.addCleanup(mock.patch.stopall)
        self.host_tools = mock.patch.object(build, "host_tools", return_value=(Path("cmake"), Path("ninja"))).start()
        self.prepare = mock.patch.object(build, "prepare_source", return_value=(
            self.out / "prepared", self.out / "ggml", "fixture-source")).start()
        self.select = mock.patch.object(build, "source_selection", return_value=self.selected).start()
        self.run = mock.patch.object(build, "run").start()
        mock.patch.object(build, "sdk_setup", side_effect=AssertionError("unexpected stock SDK setup")).start()
        mock.patch.object(build.subprocess, "run", side_effect=AssertionError("unexpected child process")).start()
        mock.patch.object(build.subprocess, "check_output", side_effect=AssertionError("unexpected child process")).start()

    def view_path(self):
        return self.out / ("foundation-source-" + self.selected["selection_identity_sha256"]) / "LINK.json"

    def invoke(self):
        # Do not print successful application-build claims from a mocked run.
        with redirect_stdout(io.StringIO()):
            build.build_application(self.args, self.out, self.cache)

    def test_source_build_passes_generated_view_and_digest_to_cmake(self):
        self.invoke()
        self.assertEqual(self.run.call_count, 2)
        command = [str(value) for value in self.run.call_args_list[0].args[0]]
        self.assertIn("-DCOSMO_SOFTWARE_WEBGPU_METADATA=" + str(self.view_path()), command)
        self.assertIn("-DCOSMO_FOUNDATION_LINK_SHA256=" + self.selected["link_view_sha256"], command)
        self.assertNotIn("-DCOSMO_SOFTWARE_WEBGPU_METADATA=" + str(self.selected["link_manifest"]), command)
        environment = self.run.call_args_list[0].kwargs["env"]
        self.assertNotIn("CC", environment)
        self.assertNotIn("LDFLAGS", environment)
        self.select.assert_called_once_with(self.args)

    def test_view_mutation_after_compile_blocks_completion(self):
        def mutate(command, **kwargs):
            if "--build" in command:
                self.view_path().write_bytes(self.view_path().read_bytes() + b"\n")
        self.run.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, "Generated foundation view changed"):
            self.invoke()

    def test_dependencies_only_never_prepares_or_compiles_application(self):
        self.args.dependencies_only = True
        self.invoke()
        self.host_tools.assert_not_called()
        self.prepare.assert_not_called()
        self.run.assert_not_called()
        self.assertFalse(self.view_path().exists())

    def test_prepare_only_does_not_launch_dependency_or_application_build(self):
        self.args.prepare_only = True
        self.invoke()
        self.prepare.assert_called_once_with(self.out)
        self.run.assert_not_called()
        self.assertFalse(self.view_path().exists())


if __name__ == "__main__":
    unittest.main()
