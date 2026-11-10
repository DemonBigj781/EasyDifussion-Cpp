"""Optional real HTTP regression. SDKIT_TEST_BINARY must be a CPU-only sdkit build.

Uses an empty temporary working directory, no model arguments and explicit CPU.
Never sends an executable generation request or enumerates backend devices.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request


@unittest.skipUnless(os.environ.get("SDKIT_TEST_BINARY"), "Set SDKIT_TEST_BINARY to a CPU-only sdkit binary")
class PromptPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="sdkit-prompt-http-")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}"
        cls.log = open(Path(cls.work.name) / "server.log", "w+")
        cls.process = subprocess.Popen(
            [str(Path(os.environ["SDKIT_TEST_BINARY"]).resolve()), "--backend", "cpu", "--port", str(port)],
            cwd=cls.work.name, stdout=cls.log, stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(100):
                if cls.process.poll() is not None:
                    raise RuntimeError("CPU server exited before readiness")
                try:
                    with urllib.request.urlopen(cls.url + "/v1/internal/ping", timeout=.2) as response:
                        if response.status == 200:
                            return
                except (OSError, urllib.error.URLError):
                    time.sleep(.05)
            raise RuntimeError("CPU server did not become ready")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait()
        cls.log.close()
        cls.work.cleanup()

    def post(self, path, body):
        request = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_invalid_generation_options_fail_before_task_creation(self):
        cases = [("distilled_cfg_scale", value) for value in (None, "3.5", True, 1e100)]
        for route in ("txt2img", "img2img"):
            route_cases = cases + ([("mask_blur", value) for value in (-1, 65, None, "4", True)]
                                   if route == "img2img" else [])
            for index, (field, value) in enumerate(route_cases):
                task = f"options-{route}-{index}"
                with self.subTest(route=route, field=field, value=value):
                    status, body = self.post(f"/v1/sdapi/v1/{route}", {field: value, "force_task_id": task})
                    self.assertEqual(status, 400)
                    self.assertTrue(body["message"].startswith(field + " "))
                    status, _ = self.post("/v1/internal/progress", {"id_task": task})
                    self.assertEqual(status, 404, "Rejected request created a task")

    def test_invalid_prompts_fail_before_task_creation(self):
        for route in ("txt2img", "img2img"):
            for field in ("prompt", "negative_prompt"):
                for index, text in enumerate(("(invalid:.)", "(overflow:" + "9" * 60 + ")",
                                              "((nested:100000000000000000000):100000000000000000000)")):
                    task = f"preflight-{route}-{field}-{index}"
                    with self.subTest(route=route, field=field, text=text):
                        status, body = self.post(f"/v1/sdapi/v1/{route}", {field: text, "force_task_id": task})
                        self.assertEqual(status, 400)
                        self.assertTrue(body["message"].startswith(field + " "))
                        status, _ = self.post("/v1/internal/progress", {"id_task": task})
                        self.assertEqual(status, 404, "Rejected request created a task")


if __name__ == "__main__":
    unittest.main()
