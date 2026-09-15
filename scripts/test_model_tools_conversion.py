#!/usr/bin/env python3
"""Focused checks for Model Tools conversion and GGUF image discovery."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ui"))

from easydiffusion import app  # noqa: E402
from easydiffusion import model_tools  # noqa: E402
from easydiffusion.model_manager import get_model_dirs  # noqa: E402
from easydiffusion.model_manager.list_models import list_models  # noqa: E402
from easydiffusion.utils.model_identifier import identify_model_type  # noqa: E402
from fastapi import HTTPException  # noqa: E402


def write_header(path: Path, tensors: dict[str, list[int]]) -> None:
    header = {key: {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]} for key, shape in tensors.items()}
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


class ModelToolsConversionTests(unittest.TestCase):
    def test_server_mirrors_remain_identical(self):
        pairs = (
            ("ui/easydiffusion/model_tools.py", "ui/plugins/server/model_tools/model_tools.py"),
            ("ui/easydiffusion/model_manager/__init__.py", "ui/plugins/server/model_manager/__init__.py"),
            ("ui/easydiffusion/utils/model_identifier.py", "ui/plugins/server/utils/model_identifier.py"),
        )
        for left, right in pairs:
            with self.subTest(left=left):
                self.assertEqual((REPO_ROOT / left).read_bytes(), (REPO_ROOT / right).read_bytes())

    def test_anima_tensor_signature_is_identified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            write_header(
                source,
                {"model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight": [1, 1]},
            )
            self.assertEqual(identify_model_type(str(source)), "anima")
            self.assertEqual(model_tools._detect_gguf_family(source), "anima")

    def test_gguf_outputs_route_to_architecture_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            with mock.patch.object(app, "MODELS_DIR", str(models_root)):
                cases = {
                    "sd15": "1.5",
                    "sd3": "3.0",
                    "anima": "anima",
                    "sdxl": "sdxl",
                }
                for family, folder in cases.items():
                    with self.subTest(family=family):
                        output = model_tools._resolve_output("example", "q4_1", family)
                        self.assertEqual(output, models_root / "Image_GGUF" / folder / "example.q4_1.gguf")

    def test_image_gguf_is_scanned_with_configured_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            (models_root / "checkpoints").mkdir()
            gguf_dir = models_root / "Image_GGUF" / "sdxl"
            gguf_dir.mkdir(parents=True)
            write_header(models_root / "checkpoints" / "source.safetensors", {})
            (gguf_dir / "converted.q4_1.gguf").write_bytes(b"not-a-real-gguf")
            config = {"directories": {"checkpoints": "checkpoints"}}
            with (
                mock.patch.object(app, "MODELS_DIR", str(models_root)),
                mock.patch.object(app, "getConfig", return_value=config),
            ):
                self.assertEqual(
                    get_model_dirs("stable-diffusion"),
                    [str(models_root / "checkpoints"), str(models_root / "Image_GGUF")],
                )
                names = {item["model"] for item in list_models(["stable-diffusion"])}
                self.assertIn("sdxl/converted.q4_1", names)

    def test_embedding_conversion_uses_restricted_loader_and_preserves_folder(self):
        import torch
        from safetensors.torch import load_file

        source_text = (REPO_ROOT / "ui/easydiffusion/model_tools.py").read_text(encoding="utf-8")
        self.assertIn('weights_only=True', source_text)

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "embeddings" / "sd1.5" / "positive"
            folder.mkdir(parents=True)
            source = folder / "legacy.pt"
            output = folder / "legacy.safetensors"
            torch.save({"string_to_param": {"*": torch.ones((2, 768))}}, source)
            model_tools._JOBS.clear()
            model_tools._JOBS["test"] = {"status": "queued", "log": []}
            model_tools._safetensors_conversion_worker("test", source, output)

            self.assertTrue(output.is_file())
            tensors = load_file(str(output))
            self.assertEqual(set(tensors), {"emb_params"})
            self.assertEqual(tuple(tensors["emb_params"].shape), (2, 768))
            self.assertEqual(model_tools._JOBS["test"]["status"], "completed")

    def test_existing_safetensors_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            folder = models_root / "embeddings" / "sd1.5" / "negitive"
            folder.mkdir(parents=True)
            (folder / "legacy.pt").write_bytes(b"legacy")
            (folder / "legacy.safetensors").write_bytes(b"keep")
            config = {"directories": {"embeddings": "embeddings"}}
            with (
                mock.patch.object(app, "MODELS_DIR", str(models_root)),
                mock.patch.object(app, "getConfig", return_value=config),
            ):
                with self.assertRaises(HTTPException) as raised:
                    model_tools.safetensors_convert({"source": "embeddings/sd1.5/negitive/legacy.pt"})
            self.assertEqual(raised.exception.status_code, 409)
            self.assertEqual((folder / "legacy.safetensors").read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
