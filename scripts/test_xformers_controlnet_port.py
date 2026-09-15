import pathlib
import unittest


class TestXformersControlNetPort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = pathlib.Path(__file__).resolve().parent.parent

    def test_xformers_cuda_dispatch_is_compiled_and_selectable(self):
        cuda_root = (
            self.root
            / "source/sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda"
        )
        implementation = (
            cuda_root / "xformers/xformers-attention.cu"
        ).read_text(encoding="utf-8")
        dispatcher = (cuda_root / "fattn.cu").read_text(encoding="utf-8")
        cmake = (cuda_root / "CMakeLists.txt").read_text(encoding="utf-8")
        main = (
            self.root / "source/sdkit3-port-source/src/main.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("ggml_cuda_xformers_attn(", implementation)
        self.assertIn("ggml_cuda_xformers_attn_supported", implementation)
        self.assertIn('getenv("SD_CUDA_XFORMERS")', dispatcher)
        self.assertIn("BEST_FATTN_KERNEL_XFORMERS", dispatcher)
        self.assertIn("xformers/xformers-attention.cu", cmake)
        self.assertIn('setenv("SD_CUDA_XFORMERS", "1", 1)', main)

    def test_incompatible_controlnet_returns_error_instead_of_asserting(self):
        stable_diffusion = (
            self.root
            / "source/sdkit3-port-source/stable-diffusion.cpp/src/stable-diffusion.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "Uni-ControlNet and ControlNet-LITE spatial adapters require an SD1.x checkpoint",
            stable_diffusion,
        )
        self.assertIn("return false", stable_diffusion)

    def test_lllite_does_not_send_a_second_standard_control_image(self):
        task_manager = (self.root / "ui/media/js/task-manager.js").read_text(encoding="utf-8")
        lllite = (
            self.root
            / "ui/plugins/ui/controlnet_plugin/controlnet-lllite.plugin.js"
        ).read_text(encoding="utf-8")
        self.assertIn("delete task.reqBody.control_image", task_manager)
        self.assertIn("control_net_lllite_image", task_manager)
        self.assertIn("delete event.reqBody.control_image", lllite)


if __name__ == "__main__":
    unittest.main()
