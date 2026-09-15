#!/usr/bin/env python3
"""Checks that the external debug log blocks readable prompt context."""

from __future__ import annotations

import io
import logging
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ui"))

from PIL import Image  # noqa: E402

from easydiffusion.privacy_debug import (  # noqa: E402
    PrivacySafeRotatingFileHandler,
    block_alphabetic_context,
    count_non_finite_values,
    log_non_finite_counts,
    log_result_image_diagnostics,
)
from easydiffusion.backends import common as backend_common  # noqa: E402


class PrivacyDebugLogTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(f"privacy-debug-test-{self.id()}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def tearDown(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def test_prompt_records_are_omitted_and_other_debug_details_remain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "debug.log"
            handler = PrivacySafeRotatingFileHandler(path, maxBytes=1024 * 1024, backupCount=1)
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            self.logger.handlers.append(handler)
            try:
                self.logger.debug("device=cuda:0 free_vram=12GiB")
                self.logger.info("request: {'prompt': 'private-positive-text', 'negative_prompt': 'private-negative-text'}")
                self.logger.warning("positive conditioning contained private-conditioning-text")
            finally:
                handler.close()
                self.logger.handlers.remove(handler)

            contents = path.read_text(encoding="utf-8")
            self.assertIn("device=cuda:0 free_vram=12GiB", contents)
            self.assertIn("███████", contents)
            self.assertNotIn("private-positive-text", contents)
            self.assertNotIn("private-negative-text", contents)
            self.assertNotIn("private-conditioning-text", contents)
            self.assertIn("{'██████': '███████-████████-████'", contents)

    def test_blocking_preserves_prompt_syntax_without_context(self):
        self.assertEqual(
            block_alphabetic_context("cat, (blue eyes:1.25), 猫"),
            "███, (████ ████:1.25), █",
        )

    def test_embedding_filenames_are_not_mistaken_for_prompt_text(self):
        line = "load /models/embeddings/sdxl/negative/deep_negative_pony.safetensors"
        self.assertEqual(backend_common.redact_log_message(line), line)

    def test_non_finite_values_are_counted_separately_without_array_output(self):
        values = [1.0, float("nan"), [float("inf"), float("-inf")]]
        self.assertEqual(count_non_finite_values(values), (1, 2))

        with self.assertLogs(self.logger, level="ERROR") as captured:
            self.assertTrue(log_non_finite_counts(values, self.logger, "test_tensor"))
        message = "\n".join(captured.output)
        self.assertIn("nan_count=1 inf_count=2", message)
        self.assertNotIn(repr(values), message)

    def test_only_supplied_result_images_are_checked_for_brightness(self):
        result_images = [
            Image.new("RGB", (2, 2), "black"),
            Image.new("RGB", (2, 2), (127, 127, 127)),
            Image.new("RGB", (2, 2), "white"),
        ]
        with self.assertLogs(self.logger, level="WARNING") as captured:
            log_result_image_diagnostics(result_images, self.logger)
        message = "\n".join(captured.output)
        self.assertIn("Result image 1 is very dark", message)
        self.assertIn("Result image 3 is very bright", message)
        self.assertNotIn("Result image 2", message)

    def test_native_backend_prompt_is_blocked_and_non_finite_counts_are_retained(self):
        messages = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                messages.append((record.levelno, record.getMessage()))

        backend_logger = logging.getLogger("easydiffusion.native_backend")
        old_level = backend_logger.level
        old_propagate = backend_logger.propagate
        handler = CaptureHandler()
        backend_logger.handlers.append(handler)
        backend_logger.setLevel(logging.DEBUG)
        backend_logger.propagate = False
        try:
            backend_common.read_output(
                io.BytesIO(
                    b"Generating txt2img: prompt='private words', seed=7\n"
                    b"VAE non-finite: nan_count=3 inf_count=2 total_count=99\n"
                ),
                prefix="[native] ",
            )
        finally:
            backend_logger.handlers.remove(handler)
            backend_logger.setLevel(old_level)
            backend_logger.propagate = old_propagate

        rendered = "\n".join(message for _, message in messages)
        self.assertNotIn("private words", rendered)
        self.assertIn("███████ █████", rendered)
        self.assertIn("nan_count=3 inf_count=2 total_count=99", rendered)
        self.assertEqual(messages[-1][0], logging.ERROR)

    def test_terminal_progress_redraws_are_coalesced(self):
        output = "  |=> | 1/49\x1b[K\r  |==> | 2/49\x1b[K\r  |###| 49/49\x1b[K"
        self.assertEqual(backend_common._normalize_backend_output(output), "  |###| 49/49")

    def test_webui_completion_does_not_print_infotext_prompts(self):
        source = (REPO_ROOT / "ui" / "easydiffusion" / "backends" / "webui_common.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('print(json.loads(res["info"])["infotexts"])', source)
        self.assertNotIn('print(f"operation: {operation_to_apply}, args: {args}")', source)


if __name__ == "__main__":
    unittest.main()
