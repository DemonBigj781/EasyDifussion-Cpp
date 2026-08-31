import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ui"))

from easydiffusion.video_companions import discover_mochi_companions


class TestVideoCompanions(unittest.TestCase):
    def test_mochi_prefers_memory_efficient_sibling_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            mochi = pathlib.Path(directory) / "mochi"
            transformer = mochi / "transformer" / "mochi-q3_k_m.gguf"
            transformer.parent.mkdir(parents=True)
            transformer.write_bytes(b"transformer")

            vae = mochi / "vae"
            vae.mkdir()
            (vae / "mochi_vae.safetensors").write_bytes(b"full")
            (vae / "mochi_vae_scaled.safetensors").write_bytes(b"scaled")
            preferred_vae = vae / "mochi_vae_fp8_e4m3fn.safetensors"
            preferred_vae.write_bytes(b"fp8")

            t5xxl = mochi / "t5xxl"
            t5xxl.mkdir()
            (t5xxl / "t5xxl_fp16.safetensors").write_bytes(b"full")
            preferred_encoder = t5xxl / "t5xxl_fp16-q4_0.gguf"
            preferred_encoder.write_bytes(b"q4")

            companions = discover_mochi_companions(transformer)

            self.assertEqual(companions["vae"], str(preferred_vae.resolve()))
            self.assertEqual(companions["text-encoder"], str(preferred_encoder.resolve()))

    def test_missing_companions_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "mochi" / "transformer" / "model.gguf"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"transformer")
            self.assertEqual(discover_mochi_companions(checkpoint), {})


if __name__ == "__main__":
    unittest.main()
