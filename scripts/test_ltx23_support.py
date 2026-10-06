import ast
import importlib.util
import pathlib
import types
import unittest
import uuid


ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestLtx23Support(unittest.TestCase):
    def test_architecture_identification_precedes_original_ltx_video(self):
        for relative in ("ui/easydiffusion/utils/model_identifier.py", "ui/plugins/server/utils/model_identifier.py"):
            spec = importlib.util.spec_from_file_location("identifier", ROOT / relative)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for prefix in ("", "model.diffusion_model."):
                with self.subTest(source=relative, prefix=prefix):
                    tensor = {"shape": [128, 128]}
                    header = {prefix + "patchify_proj.weight": tensor}
                    self.assertEqual(module.infer_diffusers_model_type(header), "ltx_video")
                    header[prefix + "audio_patchify_proj.weight"] = tensor
                    self.assertEqual(module.infer_diffusers_model_type(header), "ltx2")
                    header[prefix + "transformer_blocks.0.attn1.to_gate_logits.weight"] = tensor
                    self.assertEqual(module.infer_diffusers_model_type(header), "ltx2_3")
                    header[prefix + "keyframes_abs_pos_embedding"] = tensor
                    self.assertEqual(module.infer_diffusers_model_type(header), "ltx2_5")

    def test_python_adapter_forwards_both_video_resource_paths(self):
        filename = ROOT / "ui/easydiffusion/backends/webui_common.py"
        tree = ast.parse(filename.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_video")
        requests = []
        response = types.SimpleNamespace(status_code=200, json=lambda: {"frames": []})
        def post(url, json):
            requests.append((url, json))
            return response
        namespace = {"Context": object, "USE_SDKIT3_API": True, "uuid": uuid,
                     "convert_ED_sampler_names": lambda value: value,
                     "print_request": lambda *args: None, "webui_post": post}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(filename), "exec"), namespace)
        namespace["generate_video"](None, audio_vae_model="/models/ltx23_audio.safetensors",
                                    embeddings_connectors_model="/models/ltx23_connectors.safetensors")
        self.assertEqual(requests[-1][1]["audio_vae_path"], "/models/ltx23_audio.safetensors")
        self.assertEqual(requests[-1][1]["embeddings_connectors_path"], "/models/ltx23_connectors.safetensors")
        namespace["generate_video"](None)
        self.assertNotIn("audio_vae_path", requests[-1][1])
        self.assertNotIn("embeddings_connectors_path", requests[-1][1])

    def test_native_loader_keys_and_forwards_the_video_resources(self):
        native = (ROOT / "source/sdkit3-port-source/src/image_generator.cpp").read_text()
        server = (ROOT / "source/sdkit3-port-source/src/server.cpp").read_text()
        key = native.split("const std::string model_configuration_key =", 1)[1].split("});", 1)[0]
        for field in ("audio_vae_path", "embeddings_connectors_path"):
            self.assertIn(field, key, "Changing or removing a resource must invalidate the loaded context")
            self.assertIn(f"params.{field} =", native)
            self.assertIn(f'json_body["{field}"]', server)
            self.assertIn(f"params.{field}", native.split("ImageGenerator::generateVideo", 1)[1].split("std::unique_lock", 1)[0])


if __name__ == "__main__":
    unittest.main()
