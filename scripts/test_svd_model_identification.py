import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestSvdIdentification(unittest.TestCase):
    def test_temporal_video_checkpoint_is_not_sd2(self):
        for relative in (
            "ui/easydiffusion/utils/model_identifier.py",
            "ui/plugins/server/utils/model_identifier.py",
        ):
            with self.subTest(source=relative):
                spec = importlib.util.spec_from_file_location("identifier", ROOT / relative)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                image = {module.CHECKPOINT_KEY_NAMES["v2"]: {"shape": [320, 1024]}}
                self.assertEqual(module.infer_diffusers_model_type(image), "sd_v2_base")
                video = {
                    **image,
                    "model.diffusion_model.input_blocks.1.1.time_stack.0.attn1.to_q.weight": {"shape": [320, 320]},
                    "model.diffusion_model.input_blocks.1.1.time_mixer.mix_factor": {"shape": [1]},
                }
                self.assertEqual(module.infer_diffusers_model_type(video), "svd")


if __name__ == "__main__":
    unittest.main()
