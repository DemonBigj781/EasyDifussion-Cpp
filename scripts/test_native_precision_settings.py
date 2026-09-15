import pathlib
import unittest


class TestNativePrecisionSettings(unittest.TestCase):
    def setUp(self):
        self.repo_root = pathlib.Path(__file__).resolve().parent.parent
        native_root = self.repo_root / "source" / "sdkit3-port-source"
        self.main_cpp = (native_root / "src" / "main.cpp").read_text(encoding="utf-8")
        self.generator_cpp = (native_root / "src" / "image_generator.cpp").read_text(encoding="utf-8")

    def test_native_backend_accepts_both_no_half_flags(self):
        self.assertIn('arg == "--no-half"', self.main_cpp)
        self.assertIn('arg == "--no-half-vae"', self.main_cpp)
        self.assertIn("server_params.no_half = args.no_half", self.main_cpp)
        self.assertIn("server_params.no_half_vae = args.no_half_vae", self.main_cpp)

    def test_model_no_half_uses_global_f32_override(self):
        self.assertIn("params.wtype = SD_TYPE_F32", self.generator_cpp)

    def test_vae_no_half_uses_selective_f32_override(self):
        self.assertIn("params.tensor_type_rules = FULL_PRECISION_VAE_TENSOR_RULES", self.generator_cpp)
        self.assertIn('"(^|\\\\.)vae\\\\.|first_stage_model=f32"', self.generator_cpp)


if __name__ == "__main__":
    unittest.main()
