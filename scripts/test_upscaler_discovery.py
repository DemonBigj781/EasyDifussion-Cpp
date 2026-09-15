import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ui"))

from easydiffusion import app, model_manager
from easydiffusion.types import convert_legacy_render_req_to_new


class TestUpscalerDiscovery(unittest.TestCase):
    def test_lists_every_native_upscaler_container_format(self):
        supported = {
            "torch-pth.pth": "torch-pth",
            "torch-pt.pt": "torch-pt",
            "safe.safetensors": "safe",
            "short-safe.sft": "short-safe",
            "quantized.gguf": "quantized",
            "legacy.ckpt": "legacy",
            "nested/uppercase.PTH": "nested/uppercase",
        }
        ignored = ("unsupported.onnx", "broken.pth.corrupted", "preview.png")

        with tempfile.TemporaryDirectory() as directory:
            models_root = pathlib.Path(directory)
            upscaler_root = models_root / "upscalers"
            for relative_path in (*supported, *ignored):
                target = upscaler_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()

            config = {"directories": {"realesrgan": "upscalers"}}
            with (
                mock.patch.object(app, "MODELS_DIR", directory),
                mock.patch.object(app, "getConfig", return_value=config),
            ):
                models = model_manager.list_models(["realesrgan"])

        self.assertEqual({model["model"] for model in models}, set(supported.values()))
        self.assertTrue(all(model["tags"] == ["realesrgan"] for model in models))

    def test_custom_filename_routes_through_realesrgan(self):
        converted = convert_legacy_render_req_to_new(
            {"use_upscale": "4x-UltraSharp", "upscale_amount": 4}
        )

        self.assertEqual(converted["model_paths"]["realesrgan"], "4x-UltraSharp")
        self.assertEqual(converted["filters"], ["realesrgan"])
        self.assertEqual(
            converted["filter_params"]["realesrgan"],
            {"upscaler": "4x-UltraSharp", "scale": 4},
        )

    def test_named_builtin_upscaler_keeps_its_existing_route(self):
        converted = convert_legacy_render_req_to_new(
            {"use_upscale": "Lanczos", "upscale_amount": 2}
        )

        self.assertIsNone(converted["model_paths"]["realesrgan"])
        self.assertEqual(converted["model_paths"]["lanczos"], "Lanczos")
        self.assertEqual(converted["filters"], ["lanczos"])


if __name__ == "__main__":
    unittest.main()
