import pathlib
import unittest


class TestLoraShapeCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        cls.source = (
            repo_root
            / "source"
            / "sdkit3-port-source"
            / "stable-diffusion.cpp"
            / "src"
            / "model"
            / "adapter"
            / "lora.hpp"
        ).read_text(encoding="utf-8")

    def test_immediate_mode_skips_shape_mismatch_instead_of_aborting(self):
        self.assertNotIn(
            "GGML_ASSERT(ggml_nelements(diff) == ggml_nelements(model_tensor))",
            self.source,
        )
        self.assertIn(
            "if (ggml_nelements(diff) != ggml_nelements(model_tensor))",
            self.source,
        )
        self.assertIn("return nullptr;", self.source)
        self.assertIn("skip incompatible LoRA tensor |%s|", self.source)

    def test_runtime_mode_validates_input_and_final_output_dimensions(self):
        self.assertIn("ggml_tensor* model_weight", self.source)
        self.assertIn("bool compatible = down_in == model_weight->ne[0]", self.source)
        self.assertIn("actual_out_dim != expected_out_dim", self.source)
        self.assertIn(
            "get_out_diff(ctx, backend, x, w, forward_params, prefix + \"weight\")",
            self.source,
        )

    def test_incompatible_tensors_are_reported_separately(self):
        self.assertIn("skipped_incompatible_lora_tensors", self.source)
        self.assertIn("warned_incompatible_model_tensors", self.source)
        self.assertIn("incompatible LoRA tensors have been skipped", self.source)

if __name__ == "__main__":
    unittest.main()
