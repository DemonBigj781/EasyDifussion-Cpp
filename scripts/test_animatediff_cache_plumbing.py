"""Execute the native video adapter without loading a GPU or model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]


class CachePlumbingTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "ui/easydiffusion/backends/webui_common.py"
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "generate_video")
        self.requests = []
        def post(url, json):
            self.requests.append((url, json))
            return SimpleNamespace(status_code=200, json=lambda: {"frames": ["frame"], "fps": 8})
        namespace = {"Context": object, "USE_SDKIT3_API": True, "uuid": uuid,
                     "convert_ED_sampler_names": lambda value: value,
                     "print_request": lambda *args: None, "webui_post": post}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        self.generate = namespace["generate_video"]

    def test_each_cache_and_disabled_for_text_and_image_video(self):
        for mode in ("disabled", "easycache", "teacache"):
            for initial in (None, "data:image/png;base64,fixture"):
                with self.subTest(mode=mode, initial=bool(initial)):
                    self.generate(None, motion_module_model="/fixture/motion.safetensors", cache_mode=mode,
                                  cache_threshold=.08, cache_start_percent=1, cache_end_percent=95, init_image=initial)
                    endpoint, request = self.requests[-1]
                    self.assertEqual(endpoint, "/sdapi/v1/img2video" if initial else "/sdapi/v1/txt2video")
                    self.assertEqual(request["motion_module_path"], "/fixture/motion.safetensors")
                    self.assertEqual(request["cache"], {"mode": mode, "threshold": .08,
                                                       "start_percent": .01, "end_percent": .95})

    def test_blank_threshold_uses_native_default(self):
        self.generate(None, cache_mode="teacache")
        self.assertNotIn("threshold", self.requests[-1][1]["cache"])

    def test_invalid_cache_settings_do_not_reach_backend(self):
        for options in ({"cache_mode": "both"}, {"cache_mode": "unknown"},
                        {"cache_threshold": float("nan")}, {"cache_threshold": float("inf")},
                        {"cache_threshold": -.1}, {"cache_start_percent": 95, "cache_end_percent": 15},
                        {"cache_start_percent": float("nan")}, {"cache_end_percent": 101}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.generate(None, **options)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
