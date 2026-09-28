import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))
from easydiffusion import app, sprite_gpt


class SpriteGPTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("unet.ts", "clip.ts", "tokenizer/vocab.json", "tokenizer/merges.txt"):
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.touch()
        self.manifest = self.root / "sprite.sprite-gpt.json"
        self.manifest.write_text(json.dumps({
            "format": sprite_gpt.MODEL_FORMAT, "image_size": 64,
            "unet": "unet.ts", "clip": "clip.ts", "tokenizer": "tokenizer",
        }))
        self.binary = self.root / "fake-native"
        self.binary.write_text(f"#!{sys.executable}\n" + '''
import json, sys, time
from PIL import Image
args = sys.argv[1:]
value = lambda key: args[args.index(key) + 1]
if value('--prompt') == 'fail':
    sys.stderr.write('model load failure')
    sys.exit(9)
if value('--prompt') == 'wait':
    time.sleep(30)
seed = int(value('--seed'))
Image.new('RGB', (64, 64), (seed % 256, 80, 170)).save(value('--output'))
print(json.dumps({'type': 'progress', 'step': int(value('--steps'))}), flush=True)
''')
        self.binary.chmod(0o755)
        self.context = SimpleNamespace(model_paths={"stable-diffusion": str(self.manifest)}, models={},
                                       torch_device="cuda:0", sprite_gpt_options={"output_format": "png"})
        sprite_gpt.load_model(self.context, self.manifest, self.binary)

    def test_manifest_paths_and_missing_artifacts(self):
        self.assertEqual(self.context.sprite_gpt_model["unet"], str(self.root / "unet.ts"))
        (self.root / "clip.ts").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "clip.ts"):
            sprite_gpt.load_manifest(self.manifest)

    def test_reject_wrong_model_format(self):
        data = json.loads(self.manifest.read_text())
        data["image_size"] = 32
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            sprite_gpt.load_manifest(self.manifest)

    def test_device_selection(self):
        self.assertEqual(sprite_gpt.select_device(self.context), "cuda:0")
        self.assertEqual(sprite_gpt.select_device(self.context, "diffusion=CPU,te=CUDA0"), "cpu")
        self.assertEqual(sprite_gpt.select_device(self.context, "diffusion=CUDA1"), "cuda:1")
        with self.assertRaisesRegex(ValueError, "CPU or CUDA"):
            sprite_gpt.select_device(self.context, "diffusion=Vulkan0")

    def test_request_constraints(self):
        with self.assertRaisesRegex(ValueError, "init_image"):
            sprite_gpt.validate_request({"prompt": "ghost", "init_image": "image"}, {})
        with self.assertRaisesRegex(ValueError, "lora"):
            sprite_gpt.validate_request({"prompt": "ghost"}, {"lora": ["adapter"]})
        with self.assertRaisesRegex(ValueError, "guidance"):
            sprite_gpt.validate_request({"prompt": "ghost", "guidance_scale": float("nan")}, {})

    def test_batch_seeds_and_base64_png(self):
        images = sprite_gpt.generate_images(self.context, prompt="ghost $(false)", seed=12, num_outputs=2,
                                            num_inference_steps=2, output_type="base64")
        for index, encoded in enumerate(images):
            with Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as image:
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.getpixel((0, 0))[0], 12 + index)
        self.assertEqual(len(images), 2)
        self.assertIsNone(self.context.sprite_gpt_cancel)

    def test_process_failure_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "model load failure"):
            sprite_gpt.generate_images(self.context, prompt="fail")

    def test_cancellation_terminates_process(self):
        processes = []
        popen = sprite_gpt.subprocess.Popen
        def capture(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        with mock.patch.object(sprite_gpt.subprocess, "Popen", side_effect=capture):
            images = sprite_gpt.generate_images(self.context, prompt="wait",
                callback=lambda *_: sprite_gpt.stop_rendering(self.context))
        self.assertEqual(images, [])
        self.assertIsNotNone(processes[0].poll())
        self.assertIsNone(self.context.sprite_gpt_cancel)

    def test_unload_clears_model(self):
        sprite_gpt.unload_model(self.context)
        self.assertIsNone(self.context.sprite_gpt_model)
        self.assertNotIn("stable-diffusion", self.context.models)

    def test_manifest_is_discovered_and_resolved_as_a_checkpoint(self):
        from easydiffusion import app, model_manager
        from easydiffusion.model_manager.list_models import list_models
        with mock.patch.object(app, "getConfig", return_value={"directories": {"checkpoints": str(self.root)}}), \
             mock.patch.object(app, "MODELS_DIR", str(self.root), create=True):
            models = list_models(["stable-diffusion"])
            entry = next(model for model in models if model["model"] == "sprite")
            self.assertIn("sprite_gpt", entry["tags"])
            self.assertIn("64×64", entry["name"])
            self.assertEqual(model_manager.resolve_model_to_use("sprite", "stable-diffusion"), str(self.manifest))

    def test_prepare_request_records_native_dimensions_and_sampler(self):
        from easydiffusion.types import GenerateImageRequest, ModelsData, RenderTaskData
        request = GenerateImageRequest(prompt="ghost", width=1024, height=768, sampler_name="euler_a")
        models = ModelsData(model_paths={"stable-diffusion": str(self.manifest), "vae": "old-vae"})
        task = RenderTaskData(use_vae_model="old-vae")
        sprite_gpt.prepare_request(request, models, task)
        self.assertEqual((request.width, request.height), (64, 64))
        self.assertEqual(request.sampler_name, "euler")
        self.assertIsNone(models.model_paths["vae"])
        self.assertIsNone(task.use_vae_model)

    def test_backend_switches_between_sprite_and_regular_models(self):
        from easydiffusion import backend_manager
        backend = backend_manager.load_backend_module(str(ROOT / "ui/easydiffusion/backends/sdkit3.py"))
        common = backend.webui_common
        with mock.patch.object(backend, "get_backend_dir", return_value=str(self.root)), \
             mock.patch.object(common, "curr_models", {"stable-diffusion": "previous.safetensors", "vae": None, "text-encoder": None}), \
             mock.patch.object(common, "webui_post") as post, \
             mock.patch.object(backend_manager, "restart_backend") as restart:
            (self.root / "sdkit-sprite-gpt").symlink_to(self.binary)
            backend.load_model(self.context, "stable-diffusion")
            self.assertIsNone(common.curr_models["stable-diffusion"])
            backend.flush_model_changes(self.context)
            restart.assert_called_once()
            self.assertIsNone(post.call_args.kwargs["json"]["sd_model_checkpoint"])
            backend.set_options(self.context, output_format="webp")
            self.assertEqual(self.context.sprite_gpt_options["output_format"], "webp")
            backend.unload_model(self.context, "stable-diffusion")
            with mock.patch.object(common, "load_model") as regular_load:
                self.context.model_paths["stable-diffusion"] = "normal.safetensors"
                backend.load_model(self.context, "stable-diffusion")
                regular_load.assert_called_once()
                self.assertIsNone(self.context.sprite_gpt_model)


if __name__ == "__main__":
    unittest.main()
