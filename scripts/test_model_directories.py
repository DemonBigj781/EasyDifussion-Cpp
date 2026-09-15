import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ui"))

from easydiffusion import app
from easydiffusion import model_manager


class TestModelDirectories(unittest.TestCase):
    def test_relative_absolute_and_multiple_directories(self):
        config = {
            "directories": {
                "checkpoints": "custom/checkpoints",
                "video": ["SVD", "/srv/video"],
            }
        }
        with mock.patch.object(app, "getConfig", return_value=config):
            self.assertEqual(
                model_manager.get_model_dirs("stable-diffusion", "/models"),
                ["/models/custom/checkpoints"],
            )
            self.assertEqual(
                model_manager.get_model_dirs("video", "/models"),
                ["/models/SVD", "/srv/video"],
            )

    def test_controlnet_lite_uses_standard_controlnet_directories(self):
        config = {
            "directories": {
                "controlnet": "controlnet",
                "controlnet-lite": ["Controlnet_LITE", "controlnet-lite-extra"],
            }
        }
        with mock.patch.object(app, "getConfig", return_value=config):
            self.assertEqual(
                model_manager.get_model_dirs("controlnet", "/models"),
                [
                    "/models/controlnet",
                    "/models/Controlnet_LITE",
                    "/models/controlnet-lite-extra",
                ],
            )

    def test_configured_roots_still_include_existing_mochi_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            models_root = pathlib.Path(directory)
            for relative in ("vae", "text-encoder", "mochi/vae", "mochi/t5xxl"):
                (models_root / relative).mkdir(parents=True)
            config = {
                "directories": {
                    "vae": "vae",
                    "text-encoder": "text-encoder",
                }
            }
            with mock.patch.object(app, "getConfig", return_value=config):
                self.assertEqual(
                    model_manager.get_model_dirs("vae", directory),
                    [str(models_root / "vae"), str(models_root / "mochi/vae")],
                )
                self.assertEqual(
                    model_manager.get_model_dirs("text-encoder", directory),
                    [str(models_root / "text-encoder"), str(models_root / "mochi/t5xxl")],
                )

    def test_automatic_controlnet_sentinel_is_not_resolved_as_a_file(self):
        models_data = types.SimpleNamespace(
            model_paths={"controlnet": model_manager.AUTOMATIC_CONTROLNET_MODEL}
        )
        backend = types.SimpleNamespace(list_controlnet_filters=lambda: [])
        with mock.patch("easydiffusion.backend_manager.backend", backend):
            model_manager.resolve_model_paths(models_data)
        self.assertEqual(
            models_data.model_paths["controlnet"],
            model_manager.AUTOMATIC_CONTROLNET_MODEL,
        )

    def test_directory_config_is_validated_and_deduplicated(self):
        self.assertEqual(
            model_manager.normalize_directory_config(
                {"vae": " VAE ", "video": ["SVD", "", "SVD", "mochi"]}
            ),
            {"vae": "VAE", "video": ["SVD", "mochi"]},
        )
        with self.assertRaisesRegex(ValueError, "Unknown model directory family"):
            model_manager.normalize_directory_config({"unknown": "/tmp/models"})


if __name__ == "__main__":
    unittest.main()
