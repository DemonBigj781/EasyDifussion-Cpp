import pathlib
import unittest


class TestNativeSamplerSchedulerOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = pathlib.Path(__file__).resolve().parent.parent
        cls.html = (
            cls.repo_root / "ui" / "plugins" / "ui" / "image_plugin" / "image-settings.plugin.html"
        ).read_text(encoding="utf-8")
        cls.javascript = (
            cls.repo_root / "ui" / "plugins" / "ui" / "image_plugin" / "image-settings.plugin.js"
        ).read_text(encoding="utf-8")
        cls.python_bridge = (
            cls.repo_root / "ui" / "easydiffusion" / "backends" / "webui_common.py"
        ).read_text(encoding="utf-8")
        cls.native_bridge = (
            cls.repo_root / "source" / "sdkit3-port-source" / "src" / "server.cpp"
        ).read_text(encoding="utf-8")
        cls.native_header = (
            cls.repo_root
            / "source"
            / "sdkit3-port-source"
            / "stable-diffusion.cpp"
            / "include"
            / "stable-diffusion.h"
        ).read_text(encoding="utf-8")
        cls.native_denoiser = (
            cls.repo_root
            / "source"
            / "sdkit3-port-source"
            / "stable-diffusion.cpp"
            / "src"
            / "runtime"
            / "denoiser.hpp"
        ).read_text(encoding="utf-8")
        cls.reference = (cls.repo_root / "docs" / "sampler-scheduler-technical-reference.md").read_text(
            encoding="utf-8"
        )

    def test_existing_requested_native_samplers_are_exposed_and_routed(self):
        expected = {
            "dpmpp_2m_sde": ("DPM++ 2M SDE", "dpm++2m_sde"),
            "res_multistep": ("RES Multistep", "res_multistep"),
            "gradient_estimation": ("Gradient Estimation", "euler_ge"),
            "er_sde": ("ER-SDE", "er_sde"),
            "dpmpp_3m_sde": ("DPM++ 3M SDE", "dpm++3m_sde"),
            "unipc": ("UniPC", "unipc"),
            "deis": ("DEIS", "deis"),
        }
        for ui_name, (bridge_name, native_name) in expected.items():
            with self.subTest(ui_name=ui_name):
                self.assertIn(f'<option value="{ui_name}"', self.html)
                self.assertIn("backend_sdkit3", self.html.split(f'<option value="{ui_name}"', 1)[1].split(">", 1)[0])
                self.assertIn(f'"{ui_name}": "{bridge_name}"', self.python_bridge)
                self.assertIn(f'{{"{bridge_name}", "{native_name}"}}', self.native_bridge)

    def test_existing_requested_native_schedulers_are_exposed_and_routed(self):
        expected = {
            "normal": "discrete",
            "beta": "beta",
            "kl_optimal": "kl_optimal",
            "ddim_uniform": "ddim_uniform",
            "linear_quadratic": "linear_quadratic",
        }
        for ui_name, native_name in expected.items():
            with self.subTest(ui_name=ui_name):
                option = self.html.split(f'<option value="{ui_name}"', 1)[1].split(">", 1)[0]
                self.assertIn("backend_sdkit3", option)
                self.assertIn(f'{{"{ui_name}", "{native_name}"}}', self.native_bridge)

    def test_matrix_recommendations_are_advisory_stars(self):
        expected_pairs = {
            "euler": "normal",
            "euler_a": "exponential",
            "dpmpp_2m": "karras",
            "dpmpp_2m_sde": "karras",
            "dpmpp_3m_sde": "linear_quadratic",
            "unipc": "kl_optimal",
            "lcm": "sgm_uniform",
            "deis": "simple",
            "ipndm": "ddim_uniform",
            "res_multistep": "karras",
            "gradient_estimation_cfg_pp": "beta",
            "er_sde": "exponential",
        }
        for sampler, scheduler in expected_pairs.items():
            with self.subTest(sampler=sampler):
                self.assertIn(f'{sampler}: "{scheduler}"', self.javascript)
        self.assertIn('isRecommended ? " *" : ""', self.javascript)
        self.assertIn("* Recommended for the selected sampler", self.html)

    def test_former_gaps_have_distinct_native_implementations(self):
        for supported_enum in (
            "DPMPP3M_SDE_SAMPLE_METHOD",
            "UNIPC_SAMPLE_METHOD",
            "DEIS_SAMPLE_METHOD",
            "DDIM_UNIFORM_SCHEDULER",
            "LINEAR_QUADRATIC_SCHEDULER",
        ):
            with self.subTest(supported_enum=supported_enum):
                self.assertIn(supported_enum, self.native_header)
        for implementation in (
            "sample_dpmpp_3m_sde",
            "sample_unipc",
            "sample_deis",
            "struct DDIMUniformScheduler",
            "struct LinearQuadraticScheduler",
        ):
            with self.subTest(implementation=implementation):
                self.assertIn(implementation, self.native_denoiser)
        for documented_id in (
            "`dpm++3m_sde`",
            "`unipc`",
            "`deis`",
            "`ddim_uniform`",
            "`linear_quadratic`",
        ):
            with self.subTest(documented_id=documented_id):
                self.assertIn(documented_id, self.reference)

    def test_reference_pins_current_and_legacy_python_sources(self):
        self.assertIn("ModelsLab Diffusers++ source audit", self.reference)
        self.assertIn("d1fd977c5ac0cef08c6f965213c20f4460f6c37a", self.reference)
        self.assertIn("Current Hugging Face Diffusers cross-check", self.reference)
        self.assertIn("c419dac0152186060246c93a095bc1bfaea342b3", self.reference)
        self.assertIn("third-order\n  `sde-dpmsolver++` branch", self.reference)


if __name__ == "__main__":
    unittest.main()
