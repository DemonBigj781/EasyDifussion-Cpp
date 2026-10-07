#!/usr/bin/env python3
"""Offline refusal tests. Archived runtime logs are fixtures, not new GPU proof.

Only the timeout case launches a tiny host Python process tree. No APE, Vulkan
driver, model, Docker image, cloud instance or network operation is launched.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vast_run_test", HERE / "run_test.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ARCHIVE = HERE.parent / "docs/native-vulkan-validation"


class RunnerRefusalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These are the exact archived, trusted parsers. Only pure log/PNG
        # functions are called; source imports never call their main functions.
        pin = json.loads((HERE / "PIN.json").read_text())
        for name in ("verify_runtime.py", "verify_webgpu_device.py", "verify_inference.py"):
            expected = next(row for row in pin["files"] if row["path"] == "artifact/" + name)
            assert runner.file_info(ARCHIVE / "ci/application" / name) == {k: expected[k] for k in ("bytes", "sha256")}
        cls.parsers = runner.load_parsers(ARCHIVE / "ci/application")
        cls.graph = (ARCHIVE / "ci/linux-native/results-native-device/graph.stdout.log").read_text()

    def test_corrupt_payload_rejected_before_parser_import_or_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            artifact.mkdir()
            entries = []
            for name in ("easy-diffusion.exe", "ape-x86_64.elf", "BUILD.json", "PIN.json",
                         "verify_runtime.py", "verify_webgpu_device.py", "verify_inference.py"):
                path = artifact / name
                path.write_bytes(("explicit mocked payload: " + name).encode())
                entries.append({"path": "artifact/" + name, **runner.file_info(path)})
            pin = {"schema": 1, "source_commit": runner.EXPECTED_SOURCE, "run_id": 37568302821,
                   "application": entries[0], "loader": entries[1], "files": entries}
            pin_path = root / "PACKAGE_PIN.json"
            pin_path.write_text(json.dumps(pin))
            metadata = {**pin, "package_pin": {"path": "PACKAGE_PIN.json", **runner.file_info(pin_path)}}
            manifest = root / "PAYLOAD.json"
            manifest.write_text(json.dumps(metadata))
            # Tamper with a parser while keeping the package pin/manifest intact.
            (artifact / "verify_runtime.py").write_bytes(b"raise AssertionError('must never import')")
            with patch.object(runner, "EXPECTED_APP", entries[0]["sha256"]), \
                 patch.object(runner, "EXPECTED_BYTES", entries[0]["bytes"]), \
                 patch.object(runner, "load_parsers") as imports, patch.object(runner.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "Payload hash/size mismatch: verify_runtime.py"):
                    runner.verify_bundle(artifact, manifest)
                imports.assert_not_called()
                launch.assert_not_called()

    def test_xavier_refused_before_payload_import_or_application(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output_dir=Path(temporary) / "result", mode="hardware", timeout=30)
            with patch.object(runner.platform, "system", return_value="Linux"), \
                 patch.object(runner.platform, "machine", return_value="aarch64"), \
                 patch.object(runner, "verify_bundle") as bundle, patch.object(runner.subprocess, "Popen") as launch:
                result = runner.verify(args)
            self.assertEqual(result["status"], "failed")
            self.assertIn("Xavier/aarch64 requires a distinct ARM build", result["error"])
            self.assertFalse(result["hardware_verified"])
            self.assertFalse(result["commands"])
            bundle.assert_not_called()
            launch.assert_not_called()

    def test_software_cannot_pass_hardware_or_continue_to_llama(self):
        parser = self.parsers["verify_webgpu_device"]
        # Even a falsely successful child status cannot make these recorded
        # software counters a physical-hardware result.
        commands = SimpleNamespace(output=Path("/explicit-mocked-output"),
                                   run=Mock(return_value=(self.graph, "")))
        with self.assertRaisesRegex(RuntimeError, "classification/provider"):
            runner.device_commands(commands, ["/mock/loader", "/mock/app"], "native", "auto", True, 30, parser)
        self.assertEqual(commands.run.call_count, 1)
        self.assertIn("--require-hardware", commands.run.call_args.args[1])

    def test_unknown_native_type_cannot_masquerade_as_hardware(self):
        fake = self.graph.replace("software=1", "software=0").replace("adapter_type=3", "adapter_type=0")
        fake = fake.replace("device_kind=software-cpu", "device_kind=unknown").replace("require_hardware=0", "require_hardware=1")
        with self.assertRaisesRegex(RuntimeError, "classification/provider"):
            self.parsers["verify_webgpu_device"].validate_graph(fake, "native", True)

    def test_archived_native_image_contract_and_false_evidence_refusals(self):
        parser = self.parsers["verify_inference"]
        local = ARCHIVE / "local"
        stdout, stderr = (local / "direct.stdout.log").read_text(), (local / "direct.stderr.log").read_text()
        png = parser.inspect_png(local / "image.png", 256, 256)
        device = self.parsers["verify_webgpu_device"].validate_graph(self.graph, "native", False)
        actual = runner.validate_native_image(stdout, stderr, png, device, parser)
        self.assertEqual(actual["sampling"]["matmuls"], 692)
        self.assertTrue(actual["software"])
        for corrupted in (stdout.replace("IMAGE_SAMPLING backend=webgpu first_step=1", "IMAGE_SAMPLING backend=webgpu first_step=0"),
                          stdout.replace("native_loader_opens=1", "native_loader_opens=0"),
                          stdout.replace("IMAGE_DEVICE selector=WebGPU0", "IMAGE_DEVICE selector=WebGPU9")):
            with self.assertRaises(RuntimeError):
                runner.validate_native_image(corrupted, stderr, png, device, parser)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process-group contract")
    def test_timeout_kills_owned_descendants(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "work").mkdir()
            marker = output / "descendant-survived"
            child = "import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(1); pathlib.Path(" + repr(str(marker)) + ").write_text('leaked')"
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); print('child started',flush=True); time.sleep(20)"
            records = []
            commands = runner.Commands(output, {"PATH": "/no-tools"}, time.monotonic() + 10, records)
            with self.assertRaisesRegex(RuntimeError, "bounded deadline"):
                commands.run("timeout-fixture", [sys.executable, "-c", parent], 0.3)
            self.assertTrue(records[0]["timed_out"])
            self.assertIn("process group", records[0]["cleanup"])
            self.assertIn("child started", (output / "timeout-fixture.stdout.log").read_text())
            time.sleep(1.1)
            self.assertFalse(marker.exists(), "Grandchild survived timeout process-group cleanup")

    def test_overall_deadline_refuses_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = runner.Commands(Path(temporary), {}, time.monotonic() - 1, [])
            with patch.object(runner.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "Overall test deadline"):
                    commands.run("never", ["/mock/app"], 5)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
