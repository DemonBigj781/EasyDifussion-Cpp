import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("identifier", sys.argv.pop(1))
identifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identifier)

class MetadataTests(unittest.TestCase):
    def classify(self, tensors):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.safetensors"
            data = json.dumps({key: {"shape": shape} for key, shape in tensors.items()}).encode()
            path.write_bytes(struct.pack("<Q", len(data)) + data)
            return identifier.identify_controlnet_lllite_compatibility(path)

    def test_anima_rgb_and_mask_are_distinct(self):
        for channels in (3, 4):
            result = self.classify({"lllite_conditioning1.conv1.weight": [32,channels,4,4],
                "lllite_dit_blocks_0_self_attn_q_proj.down.weight": [64,2048]})
            self.assertEqual(result, {"architecture":"anima","input_channels":channels,"aspp":False})

    def test_unet_and_unknown_are_not_anima(self):
        self.assertEqual(self.classify({"lllite_unet_input_blocks_4_1_to_q.down.0.weight":[32,640]})["architecture"], "unet")
        self.assertIsNone(self.classify({"arbitrary.weight":[64,2048]}))
        self.assertIsNone(self.classify({"lllite_conditioning1.conv1.weight":[32,3,4,4],
            "lllite_dit_blocks_0_self_attn_q_proj.down.weight":[64,3072]}))

    def test_anima_base_prefix_is_recognized(self):
        self.assertEqual(identifier.infer_diffusers_model_type({"net.llm_adapter.blocks.0.cross_attn.q_proj.weight":{"shape":[1024,1024]}}), "anima")

if __name__ == "__main__": unittest.main()
