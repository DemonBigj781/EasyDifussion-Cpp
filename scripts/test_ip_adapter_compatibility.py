import json
import pathlib
import struct
import importlib.util
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ui/easydiffusion/utils/model_identifier.py"
SPEC = importlib.util.spec_from_file_location("model_identifier_under_test", MODULE_PATH)
MODEL_IDENTIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL_IDENTIFIER)
identify_clip_vision_compatibility = MODEL_IDENTIFIER.identify_clip_vision_compatibility
identify_ip_adapter_compatibility = MODEL_IDENTIFIER.identify_ip_adapter_compatibility


def write_header(path, tensors):
    header = {
        name: {"dtype": "F16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


class IPAdapterCompatibilityTests(unittest.TestCase):
    def test_base_adapter_uses_pooled_projection_width(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = pathlib.Path(directory) / "renamed.safetensors"
            clip = pathlib.Path(directory) / "custom.safetensors"
            write_header(
                adapter,
                {
                    "image_proj.proj.weight": [8192, 1280],
                    "image_proj.norm.weight": [2048],
                },
            )
            write_header(
                clip,
                {
                    "vision_model.embeddings.class_embedding": [1664],
                    "visual_projection.weight": [1280, 1664],
                },
            )
            self.assertEqual(
                identify_ip_adapter_compatibility(adapter),
                {"kind": "base", "embedding_dim": 1280},
            )
            self.assertEqual(
                identify_clip_vision_compatibility(clip),
                {"hidden_dim": 1664, "projection_dim": 1280},
            )

    def test_plus_adapter_uses_unpooled_hidden_width(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = pathlib.Path(directory) / "plus.safetensors"
            write_header(
                adapter,
                {
                    "image_proj.latents": [1, 16, 1280],
                    "image_proj.proj_in.weight": [1280, 1280],
                },
            )
            self.assertEqual(
                identify_ip_adapter_compatibility(adapter),
                {"kind": "plus", "embedding_dim": 1280},
            )


if __name__ == "__main__":
    unittest.main()
