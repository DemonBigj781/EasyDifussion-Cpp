#!/usr/bin/env python3
"""Non-model argument-routing cases; application work and subprocesses are mocked."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
