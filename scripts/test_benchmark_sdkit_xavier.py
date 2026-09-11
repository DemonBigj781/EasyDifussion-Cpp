import base64
import binascii
import csv
import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest
import zlib


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "scripts" / "benchmark_sdkit_xavier.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_sdkit_xavier", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_chunk(chunk_type, data):
    payload = chunk_type + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", binascii.crc32(payload) & 0xFFFFFFFF)


def _rgb_png(width, height, pixels):
    rows = []
    stride = width * 3
    for row in range(height):
        rows.append(b"\x00" + pixels[row * stride : (row + 1) * stride])
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(b"".join(rows))),
            _png_chunk(b"IEND", b""),
        )
    )


class TestBenchmarkSdkitXavier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_benchmark_module()

    def test_extracts_and_validates_nonblank_png_without_retaining_base64(self):
        png = _rgb_png(2, 1, bytes((0, 0, 0, 255, 128, 64)))
        response = json.dumps({"images": [base64.b64encode(png).decode("ascii")]}).encode("utf-8")

        result = self.benchmark.extract_and_validate_image(response, 2, 1)

        self.assertTrue(result["valid"])
        self.assertEqual(result["width"], 2)
        self.assertEqual(result["height"], 1)
        self.assertGreater(result["pixel_byte_variance"], 0)
        self.assertNotIn("image", result)
        self.assertNotIn("base64", json.dumps(result).lower())

    def test_default_stress_prompt_is_preserved_exactly(self):
        arguments = self.benchmark.parse_arguments([])

        self.assertEqual(
            arguments.prompt,
            "A blissful painting by Vincent_van_gogh BREAK masterpiece, best_quality, highly_detailed, scenic_landscape, rolling_green_hills, blue_sky, soft_clouds, warm_light, vivid_colors, painterly_brushstrokes, textured_canvas, peaceful_mood, pastoral_scenery, spring_grass, wildflowers, distant_trees, cinematic_composition, atmospheric_perspective, tranquil_countryside, gentle_breeze, storybook_illustration, expressive_color, harmonious_palette, A puppy in the meadows AND under a sunset, calming /red and orange sky/, (a small creek to the left), wearing a dandelion:1.2 on the left ear, Resting in a field of daisies +++",
        )
        self.assertEqual(arguments.steps, 20)

    def test_prompt_budget_places_semantic_tail_beyond_standard_clip(self):
        prompt = "front matter A puppy in the tail"

        result = self.benchmark.validate_prompt_token_budget(
            prompt,
            "A puppy",
            lambda text: 100 if text == prompt else 80,
        )

        self.assertEqual(result["payload_tokens"], 100)
        self.assertEqual(result["context_tokens"], 102)
        self.assertEqual(result["payload_tokens_before_tail"], 80)
        self.assertTrue(result["tail_beyond_standard_clip"])

    def test_prompt_budget_rejects_short_standard_clip_prompt(self):
        with self.assertRaisesRegex(self.benchmark.BenchmarkError, "77-token"):
            self.benchmark.validate_prompt_token_budget(
                "short prompt A puppy",
                "A puppy",
                lambda text: 20 if text.startswith("short") else 10,
            )

    def test_rejects_blank_png(self):
        png = _rgb_png(2, 1, bytes(6))
        response = json.dumps({"images": [base64.b64encode(png).decode("ascii")]}).encode("utf-8")

        with self.assertRaisesRegex(self.benchmark.BenchmarkError, "blank"):
            self.benchmark.extract_and_validate_image(response, 2, 1)

    def test_detects_model_reload_markers(self):
        log = "Model change detected, loading new model\nInitializing SD context with model: test.gguf\n"

        markers = self.benchmark.find_model_reload_markers(log)

        self.assertIn("Model change detected, loading new model", markers)
        self.assertIn("Initializing SD context with model", markers)

    def test_extracts_model_load_and_step_throughput_metrics(self):
        log = "\n".join(
            (
                "[2026-09-11 16:58:23.763] INFO Model change detected, loading new model: test.gguf",
                "[2026-09-11 16:58:24.535] INFO LongCLIP detected for text_model (native context: 248 tokens)",
                "[2026-09-11 16:58:24.552] INFO SD context initialized successfully",
                "[2026-09-11 16:58:26.287] INFO loading tensors completed, taking 0.21s",
                "[2026-09-11 16:58:30.579] INFO loading tensors completed, taking 1.01s",
                "[2026-09-11 16:58:32.100] INFO sampling completed, taking 2.86s",
            )
        )

        metrics = self.benchmark.extract_performance_metrics(log, steps=4, wall_ms=9236)

        self.assertEqual(metrics["model_initialization_ms"], 789)
        self.assertEqual(metrics["tensor_load_seconds"], 1.22)
        self.assertEqual(metrics["longclip_native_context_tokens"], 248)
        self.assertAlmostEqual(metrics["request_steps_per_second"], 4 / 9.236, places=3)
        self.assertAlmostEqual(metrics["sampling_steps_per_second"], 4 / 2.86, places=3)

    def test_warm_metrics_do_not_invent_model_load_time(self):
        metrics = self.benchmark.extract_performance_metrics(
            "[2026-09-11 16:58:35.076] INFO sampling completed, taking 1.43s",
            steps=4,
            wall_ms=2190,
        )

        self.assertIsNone(metrics["model_initialization_ms"])
        self.assertEqual(metrics["tensor_load_seconds"], 0.0)
        self.assertAlmostEqual(metrics["sampling_steps_per_second"], 4 / 1.43, places=3)

    def test_history_contains_all_phases_and_no_image_payload(self):
        record = {
            "run_id": "test-run",
            "timestamp_utc": "2026-09-11T00:00:00Z",
            "status": "passed",
            "timings_ms": {"startup": 100, "cold": 200, "warm": 150},
            "results": {
                "cold": {
                    "valid": True,
                    "png_bytes": 64,
                    "model_reload_detected": True,
                    "request_steps_per_second": 1.0,
                    "sampling_steps_per_second": 2.0,
                    "model_initialization_ms": 750,
                    "tensor_load_seconds": 1.25,
                },
                "warm": {
                    "valid": True,
                    "png_bytes": 64,
                    "model_reload_detected": False,
                    "request_steps_per_second": 3.0,
                    "sampling_steps_per_second": 4.0,
                    "model_initialization_ms": None,
                    "tensor_load_seconds": 0.5,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            jsonl_path = root / "history.jsonl"
            csv_path = root / "history.csv"

            self.benchmark.append_history(jsonl_path, csv_path, record)

            jsonl_text = jsonl_path.read_text(encoding="utf-8")
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertEqual(len(jsonl_text.splitlines()), 1)
            self.assertIn("startup", csv_text)
            self.assertIn("cold", csv_text)
            self.assertIn("warm", csv_text)
            self.assertIn("sampling_steps_per_second", csv_text)
            self.assertIn("model_initialization_ms", csv_text)
            self.assertNotIn("images", jsonl_text)
            self.assertNotIn("base64", jsonl_text.lower())

    def test_history_rejects_image_bearing_records(self):
        record = {"run_id": "unsafe", "images": ["encoded-image"]}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with self.assertRaisesRegex(self.benchmark.BenchmarkError, "image payload"):
                self.benchmark.append_history(root / "history.jsonl", root / "history.csv", record)

    def test_history_rebuilds_an_outdated_csv_schema_from_jsonl(self):
        old_record = {
            "run_id": "old-run",
            "timestamp_utc": "2026-09-11T00:00:00Z",
            "status": "passed",
            "timings_ms": {"startup": 1, "cold": 2, "warm": 3},
            "results": {},
        }
        new_record = {
            "run_id": "new-run",
            "timestamp_utc": "2026-09-11T00:01:00Z",
            "status": "passed",
            "timings_ms": {"startup": 4, "cold": 5, "warm": 6},
            "results": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            jsonl_path = root / "history.jsonl"
            csv_path = root / "history.csv"
            jsonl_path.write_text(json.dumps(old_record) + "\n", encoding="utf-8")
            csv_path.write_text("run_id,phase,wall_ms\nold-run,cold,2\n", encoding="utf-8")

            self.benchmark.append_history(jsonl_path, csv_path, new_record)

            with csv_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames), self.benchmark.CSV_FIELDS)
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["run_id"] for row in rows[:3]], ["old-run"] * 3)
            self.assertEqual([row["run_id"] for row in rows[3:]], ["new-run"] * 3)


if __name__ == "__main__":
    unittest.main()
