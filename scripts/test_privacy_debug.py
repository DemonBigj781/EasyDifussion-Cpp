import hashlib
import importlib.util
import logging
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "ui" / "easydiffusion" / "privacy_debug.py"
SPEC = importlib.util.spec_from_file_location("privacy_debug_under_test", MODULE_PATH)
privacy_debug = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(privacy_debug)


class TestReportSafeFilenameLogging(unittest.TestCase):
    def test_existing_file_uses_content_md5_without_private_path_or_name(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = pathlib.Path(directory) / "models" / "lora"
            model_dir.mkdir(parents=True)
            model = model_dir / "private descriptive name SDXL.safetensors"
            contents = b"diagnostic model fixture"
            model.write_bytes(contents)

            rendered = privacy_debug.sanitize_log_paths(f"failed to load '{model}'")

            expected_md5 = hashlib.md5(contents).hexdigest()
            self.assertEqual(
                rendered,
                f"failed to load 'models/lora/<md5:{expected_md5}>.safetensors'",
            )
            self.assertNotIn(directory, rendered)
            self.assertNotIn("private descriptive name", rendered)

    def test_unreadable_path_uses_identifier_preserving_redaction(self):
        rendered = privacy_debug.sanitize_log_paths(
            "failed /private/models/lora/private_character_SDXL_v12_lora.safetensors"
        )

        self.assertEqual(
            rendered,
            "failed models/lora/███████_█████████_SDXL_█12_LoRA.safetensors",
        )
        self.assertNotIn("private_character", rendered)

    def test_file_handler_sanitizes_exception_style_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = pathlib.Path(directory) / "report.log"
            model = pathlib.Path(directory) / "models" / "unsafe-name.gguf"
            model.parent.mkdir()
            model.write_bytes(b"gguf fixture")
            handler = privacy_debug.PrivacySafeRotatingFileHandler(log_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            record = logging.LogRecord(
                "test", logging.ERROR, str(model), 9, f"failure at {model}:9", (), None
            )

            rendered = handler.format(record)
            handler.close()

            self.assertIn("models/<md5:", rendered)
            self.assertNotIn(str(model), rendered)
            self.assertNotIn("unsafe-name", rendered)

    def test_http_urls_and_api_routes_are_not_treated_as_files(self):
        message = "GET http://127.0.0.1:7860/v1/progress; route /v1/progress"
        self.assertEqual(privacy_debug.sanitize_log_paths(message), message)


if __name__ == "__main__":
    unittest.main()
