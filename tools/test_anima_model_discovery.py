import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_identifier", root / "ui/easydiffusion/utils/model_identifier.py")
assert spec is not None and spec.loader is not None
identifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identifier)


class ModelDiscoveryTests(unittest.TestCase):
    def test_anima_adapter_detected_by_tensors_not_filename(self):
        header = {"__metadata__": {"ip_norm_keys": "False", "ip_inject_before_mlp": "False", "lora_rank": "32"}}
        for block in range(28):
            for name in ["ip_k_proj", "ip_v_proj"]:
                header[f"blocks.{block}.{name}.weight"] = {"shape": [2048, 768]}
            header[f"blocks.{block}.adaln_ip.1.weight"] = {"shape": [2048, 2048]}
        with patch.object(identifier, "read_safetensors_header", return_value=header):
            self.assertEqual(identifier.identify_ip_adapter_compatibility("renamed.safetensors"),
                             {"kind": "anima", "embedding_dim": 768})

    def test_siglip2_not_misclassified_as_clip(self):
        header = {
            "vision_model.embeddings.patch_embedding.weight": {"shape": [768, 3, 16, 16]},
            "vision_model.embeddings.patch_embedding.bias": {"shape": [768]},
            "vision_model.embeddings.position_embedding.weight": {"shape": [1024, 768]},
            "vision_model.encoder.layers.11.layer_norm2.weight": {"shape": [768]},
            "vision_model.post_layernorm.weight": {"shape": [768]},
        }
        with patch.object(identifier, "read_safetensors_header", return_value=header):
            self.assertEqual(identifier.identify_clip_vision_compatibility("renamed.safetensors"),
                             {"hidden_dim": 768, "projection_dim": 0, "kind": "siglip2_base_patch16_512"})

    def test_other_siglip_geometry_is_not_anima_encoder(self):
        header = {"vision_model.embeddings.patch_embedding.weight": {"shape": [768, 3, 16, 16]},
                  "vision_model.embeddings.position_embedding.weight": {"shape": [196, 768]}}
        with patch.object(identifier, "read_safetensors_header", return_value=header):
            result = identifier.identify_clip_vision_compatibility("renamed.safetensors")
            self.assertNotEqual((result or {}).get("kind"), "siglip2_base_patch16_512")

    def test_clip_h_preserves_existing_metadata(self):
        header = {"vision_model.embeddings.class_embedding": {"shape": [1280]},
                  "visual_projection.weight": {"shape": [1024, 1280]}}
        with patch.object(identifier, "read_safetensors_header", return_value=header):
            self.assertEqual(identifier.identify_clip_vision_compatibility("model.safetensors"),
                             {"hidden_dim": 1280, "projection_dim": 1024})


if __name__ == "__main__":
    unittest.main()
