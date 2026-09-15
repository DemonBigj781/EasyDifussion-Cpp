#!/usr/bin/env python3
"""Focused tests for the local llama.cpp AI Image Critic service."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ui"))
sys.path.insert(0, str(REPO_ROOT))

from fastapi import HTTPException  # noqa: E402
from ui.plugins.server import ai_image_critic  # noqa: E402


class AiImageCriticTests(unittest.TestCase):
    def test_cpu_mode_disables_every_llama_cpp_gpu_offload_path(self):
        command = ai_image_critic._server_command(
            Path("/opt/llama-server"),
            Path("/models/vision.gguf"),
            Path("/models/mmproj.gguf"),
            12345,
            4096,
            0,
        )
        self.assertIn("--device", command)
        self.assertIn("--no-kv-offload", command)
        self.assertIn("--no-op-offload", command)
        self.assertIn("--no-mmproj-offload", command)
        self.assertIn("--mmproj-device", command)
        self.assertEqual(command[command.index("--device") + 1], "none")
        self.assertEqual(command[command.index("--mmproj-device") + 1], "none")

    def test_gpu_mode_keeps_offload_enabled(self):
        command = ai_image_critic._server_command(
            Path("/opt/llama-server"),
            Path("/models/vision.gguf"),
            Path("/models/mmproj.gguf"),
            12345,
            4096,
            99,
        )
        self.assertNotIn("--device", command)
        self.assertNotIn("--no-mmproj-offload", command)

    def test_server_failure_detail_is_unredacted_for_local_ui(self):
        server = object.__new__(ai_image_critic._VisionServer)
        server.log_lines = ["failed to open /private/local/unfiltered-model.gguf"]
        server.process = mock.Mock()
        self.assertEqual(
            server._failure_detail(),
            "failed to open /private/local/unfiltered-model.gguf",
        )

    def test_bundle_main_model_finds_single_sibling_projector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_root = root / "text"
            vision_root = root / "vision"
            text_root.mkdir()
            bundle = vision_root / "tiny-vlm"
            bundle.mkdir(parents=True)
            main = bundle / "tiny-vlm-q4.gguf"
            projector = bundle / "mmproj-tiny-vlm-f16.gguf"
            main.write_bytes(b"GGUF")
            projector.write_bytes(b"GGUF")
            with (
                mock.patch.object(ai_image_critic, "TEXT_MODEL_DIR", text_root),
                mock.patch.object(ai_image_critic, "VISION_MODEL_DIR", vision_root),
            ):
                resolved = ai_image_critic._resolve_model_pair("", "tiny-vlm/tiny-vlm-q4.gguf")
            self.assertEqual(resolved, (main, projector))

    def test_projector_in_vision_root_uses_selected_text_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_root = root / "text"
            vision_root = root / "vision"
            text_root.mkdir()
            vision_root.mkdir()
            main = text_root / "language.gguf"
            projector = vision_root / "projector-language.gguf"
            main.write_bytes(b"GGUF")
            projector.write_bytes(b"GGUF")
            with (
                mock.patch.object(ai_image_critic, "TEXT_MODEL_DIR", text_root),
                mock.patch.object(ai_image_critic, "VISION_MODEL_DIR", vision_root),
            ):
                resolved = ai_image_critic._resolve_model_pair("language.gguf", "projector-language.gguf")
            self.assertEqual(resolved, (main, projector))

    def test_projector_beside_bundle_uses_its_companion_main_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_root = root / "text"
            vision_root = root / "vision"
            text_root.mkdir()
            bundle = vision_root / "minicpm"
            bundle.mkdir(parents=True)
            unrelated = text_root / "unrelated.gguf"
            main = bundle / "minicpm-q4.gguf"
            projector = bundle / "mmproj-minicpm-f16.gguf"
            unrelated.write_bytes(b"GGUF")
            main.write_bytes(b"GGUF")
            projector.write_bytes(b"GGUF")
            with (
                mock.patch.object(ai_image_critic, "TEXT_MODEL_DIR", text_root),
                mock.patch.object(ai_image_critic, "VISION_MODEL_DIR", vision_root),
            ):
                resolved = ai_image_critic._resolve_model_pair(
                    "unrelated.gguf",
                    "minicpm/mmproj-minicpm-f16.gguf",
                )
            self.assertEqual(resolved, (main, projector))

    def test_text_only_vision_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_root = root / "text"
            vision_root = root / "vision"
            text_root.mkdir()
            vision_root.mkdir()
            (vision_root / "text-only.gguf").write_bytes(b"GGUF")
            with (
                mock.patch.object(ai_image_critic, "TEXT_MODEL_DIR", text_root),
                mock.patch.object(ai_image_critic, "VISION_MODEL_DIR", vision_root),
                self.assertRaises(HTTPException) as raised,
            ):
                ai_image_critic._resolve_model_pair("", "text-only.gguf")
            self.assertIn("no mmproj/projector GGUF", raised.exception.detail)

    def test_model_paths_are_confined_to_configured_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            model_root.mkdir()
            (root / "outside.gguf").write_bytes(b"GGUF")
            with self.assertRaises(HTTPException) as raised:
                ai_image_critic._resolve_model(model_root, "../outside.gguf", "vision GGUF")
            self.assertEqual(raised.exception.status_code, 403)

    def test_inventory_marks_missing_projector_incompatible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_root = root / "text"
            vision_root = root / "vision"
            text_root.mkdir()
            vision_root.mkdir()
            (text_root / "language.gguf").write_bytes(b"GGUF")
            (vision_root / "vision.gguf").write_bytes(b"GGUF")
            with (
                mock.patch.object(ai_image_critic, "TEXT_MODEL_DIR", text_root),
                mock.patch.object(ai_image_critic, "VISION_MODEL_DIR", vision_root),
                mock.patch.object(ai_image_critic, "server_executable", return_value=Path("/bin/true")),
            ):
                inventory = ai_image_critic.model_inventory()
            self.assertTrue(inventory["ready"])
            self.assertEqual(inventory["text_models"], ["language.gguf"])
            self.assertFalse(inventory["vision_models"][0]["usable"])

    def test_inventory_prioritizes_usable_bundles_over_standalone_projectors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_root = root / "text"
            vision_root = root / "vision"
            text_root.mkdir()
            vision_root.mkdir()
            (text_root / "language.gguf").write_bytes(b"GGUF")
            (vision_root / "a-projector.gguf").write_bytes(b"GGUF")
            bundle = vision_root / "judge"
            bundle.mkdir()
            (bundle / "judge-q4.gguf").write_bytes(b"GGUF")
            (bundle / "mmproj-judge-f16.gguf").write_bytes(b"GGUF")
            with (
                mock.patch.object(ai_image_critic, "TEXT_MODEL_DIR", text_root),
                mock.patch.object(ai_image_critic, "VISION_MODEL_DIR", vision_root),
                mock.patch.object(ai_image_critic, "server_executable", return_value=Path("/bin/true")),
            ):
                inventory = ai_image_critic.model_inventory()
            self.assertEqual(inventory["vision_models"][0]["model"], "judge/judge-q4.gguf")
            self.assertEqual(inventory["vision_models"][0]["kind"], "bundle")
            bundled_projector = next(
                entry for entry in inventory["vision_models"] if entry["kind"] == "bundle_projector"
            )
            self.assertTrue(bundled_projector["usable"])
            self.assertIn("judge-q4.gguf", bundled_projector["detail"])

    def test_reasoning_wrappers_and_fences_are_removed_from_json(self):
        report = ai_image_critic._extract_json(
            '<think>private reasoning</think>\n```json\n{"severity":"none","issues":[]}\n```'
        )
        self.assertEqual(report["severity"], "none")

    def test_normalized_report_preserves_prompt_adherence(self):
        report = ai_image_critic._normalize_report(
            {
                "severity": "minor",
                "prompt_adherence": "good",
                "prompt_adherence_summary": "The requested red eyes and solo subject are visible.",
                "issues": [],
            }
        )
        self.assertEqual(report["prompt_adherence"], "good")
        self.assertIn("red eyes", report["prompt_adherence_summary"])

    def test_critic_request_accepts_generation_prompts(self):
        request = ai_image_critic.CriticAnalyzeRequest(
            image="data:image/jpeg;base64,AA==",
            prompt="1girl, solo, red eyes",
            negative_prompt="extra fingers",
            vision_model="vision.gguf",
        )
        self.assertEqual(request.prompt, "1girl, solo, red eyes")
        self.assertEqual(request.negative_prompt, "extra fingers")

    def test_analysis_returns_unredacted_debug_data_to_local_ui(self):
        raw = (
            '{"severity":"none","prompt_adherence":"good",'
            '"prompt_adherence_summary":"matches", "issues":[],'
            '"positive_prompt_additions":"", "negative_prompt_additions":"",'
            '"parameter_suggestions":"", "summary":"clean"}'
        )
        fake_server = mock.Mock()
        fake_server.analyze.return_value = raw
        fake_server.log_lines = ["loading /private/local/critic-model.gguf"]
        request = ai_image_critic.CriticAnalyzeRequest(
            image="data:image/jpeg;base64,AA==",
            vision_model="vision.gguf",
        )
        with (
            mock.patch.object(
                ai_image_critic,
                "_resolve_model_pair",
                return_value=(Path("/private/local/model.gguf"), Path("/private/local/mmproj.gguf")),
            ),
            mock.patch.object(ai_image_critic, "_server_for", return_value=fake_server),
        ):
            result = ai_image_critic.analyze(request)

        self.assertEqual(result["debug"]["raw_response"], raw)
        self.assertEqual(result["debug"]["parsed_response"]["summary"], "clean")
        self.assertEqual(
            result["debug"]["server_log_tail"],
            "loading /private/local/critic-model.gguf",
        )
        self.assertEqual(result["debug"]["model"], "/private/local/model.gguf")

    def test_invalid_json_error_keeps_raw_response_in_ui_debug_data(self):
        raw = "not JSON: /private/local/output and uncensored model text"
        fake_server = mock.Mock()
        fake_server.analyze.return_value = raw
        fake_server.log_lines = ["server detail /private/local/model.gguf"]
        request = ai_image_critic.CriticAnalyzeRequest(
            image="data:image/jpeg;base64,AA==",
            vision_model="vision.gguf",
        )
        with (
            mock.patch.object(
                ai_image_critic,
                "_resolve_model_pair",
                return_value=(Path("/private/local/model.gguf"), Path("/private/local/mmproj.gguf")),
            ),
            mock.patch.object(ai_image_critic, "_server_for", return_value=fake_server),
            self.assertRaises(HTTPException) as raised,
        ):
            ai_image_critic.analyze(request)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["debug"]["raw_response"], raw)
        self.assertIn("did not return valid JSON", raised.exception.detail["message"])


if __name__ == "__main__":
    unittest.main()
