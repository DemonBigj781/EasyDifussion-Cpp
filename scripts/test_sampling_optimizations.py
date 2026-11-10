import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))
from easydiffusion.sampling_optimizations import native_sampling_options


class SamplingOptimizationTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(native_sampling_options(), {"tome_ratio":0.0,"negative_guidance_min_sigma":0.0})
        self.assertEqual(native_sampling_options(.5, 1)["tome_ratio"], .5)
        for ratio, sigma in [(-1,0),(.51,0),(float("nan"),0),(0,float("inf")),(0,-1)]:
            with self.subTest(ratio=ratio,sigma=sigma), self.assertRaises(ValueError):
                native_sampling_options(ratio,sigma)

    def test_video_adapter_preserves_both_values_and_zero_default(self):
        source = ROOT / "ui/easydiffusion/backends/webui_common.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node,ast.FunctionDef) and node.name == "generate_video")
        requests = []
        def post(url,json):
            requests.append(json)
            return SimpleNamespace(status_code=200,json=lambda:{"frames":[]})
        namespace = {"Context":object,"USE_SDKIT3_API":True,"uuid":uuid,
                     "convert_ED_sampler_names":lambda x:x,"print_request":lambda *a:None,"webui_post":post,
                     # This isolated AST fixture exercises sampling fields only.
                     # Interposer behavior has its own contract tests.
                     "_add_latent_interposer_options":lambda *a,**k:None,
                     "_add_vae_interposer_options":lambda *a,**k:None}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),"exec"),namespace)
        namespace["generate_video"](None,tome_ratio=.25,negative_guidance_min_sigma=.5)
        self.assertEqual(requests[-1]["tome_ratio"],.25)
        self.assertEqual(requests[-1]["negative_guidance_min_sigma"],.5)
        namespace["generate_video"](None)
        self.assertNotIn("tome_ratio",requests[-1])
        self.assertNotIn("negative_guidance_min_sigma",requests[-1])
        for mode in ("easycache","teacache"):
            with self.assertRaises(ValueError): namespace["generate_video"](None,cache_mode=mode,negative_guidance_min_sigma=1)

    def test_image_and_video_schemas(self):
        spec = importlib.util.spec_from_file_location("optimization_schema", ROOT / "ui/easydiffusion/types.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for schema in (module.GenerateImageRequest,module.VideoGenerationRequest):
            self.assertEqual(schema().tome_ratio,0)
            self.assertEqual(schema(tome_ratio=.25,negative_guidance_min_sigma=1).negative_guidance_min_sigma,1)
            for value in (float("inf"),float("nan"),-1):
                with self.assertRaises(ValueError): schema(negative_guidance_min_sigma=value)

    def test_image_adapter_constructs_native_options_without_affecting_disabled_requests(self):
        from typing import Union, List
        source = ROOT / "ui/easydiffusion/backends/webui_common.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node,ast.FunctionDef) and node.name == "generate_images")
        # Execute the production request-construction prefix, without model
        # resolution, progress threads, or a generation server.
        stop = next(index for index,node in enumerate(function.body)
                    if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id == "init_image")
        function.body = function.body[:stop] + [ast.Return(value=ast.Name(id="cmd",ctx=ast.Load()))]
        tree = ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        namespace = {"Context":object,"USE_SDKIT3_API":True,"uuid":uuid,"Union":Union,"List":List,
                     "convert_ED_sampler_names":lambda x:x,"convert_ED_controlnet_filter_name":lambda x:x}
        exec(compile(tree,str(source),"exec"),namespace)
        command = namespace["generate_images"](None,tome_ratio=.5,negative_guidance_min_sigma=1)
        self.assertEqual(command["tome_ratio"],.5)
        self.assertEqual(command["negative_guidance_min_sigma"],1)
        self.assertNotIn("tome_ratio",namespace["generate_images"](None))
        namespace["USE_SDKIT3_API"] = False
        with self.assertRaises(ValueError): namespace["generate_images"](None,tome_ratio=.5)


if __name__ == "__main__": unittest.main()
