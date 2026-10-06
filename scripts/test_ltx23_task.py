import pathlib
import sys
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ui"))

from easydiffusion import app
from easydiffusion import backend_manager, model_manager, runtime, task_manager
from easydiffusion.tasks import render_video
from easydiffusion.types import ModelsData, OutputFormatData, TaskData, VideoGenerationRequest, convert_legacy_render_req_to_new


class TestLtx23Task(unittest.TestCase):
    def task(self, **overrides):
        payload = {
            "use_stable_diffusion_model": "ltx-2.3-distilled",
            "use_vae_model": "ltx-2.3-video-vae",
            "use_text_encoder_model": "gemma-3-12b-it",
            "audio_vae_model": "ltx-2.3-audio-vae",
            "embeddings_connectors_model": "ltx-2.3-connectors",
            **overrides,
        }
        request = convert_legacy_render_req_to_new(payload)
        return render_video.VideoTask(VideoGenerationRequest(**request), TaskData(), ModelsData(**request), OutputFormatData())

    def run_to_backend(self, task, family="ltx2_3"):
        backend = types.SimpleNamespace(generate_video=mock.Mock(return_value={"frames": ["frame"], "fps": 24}))
        with mock.patch.object(backend_manager, "backend", backend), \
             mock.patch.object(runtime, "context", types.SimpleNamespace(), create=True), \
             mock.patch.object(task_manager, "current_state_error", None), \
             mock.patch.object(model_manager, "resolve_model_paths"), \
             mock.patch.object(model_manager, "resolve_model_to_use", side_effect=lambda name, model_type: f"/models/{model_type}/{name}") as resolve, \
             mock.patch.object(model_manager, "reload_models_if_necessary") as reload, \
             mock.patch.object(model_manager, "fail_if_models_did_not_load"), \
             mock.patch.object(render_video, "identify_model_type", return_value=family):
            try:
                task.run()
            except Exception:
                reload.assert_not_called()
                backend.generate_video.assert_not_called()
                raise
            return backend, resolve, reload

    def test_schema_and_task_preserve_and_resolve_resources(self):
        task = self.task()
        backend, resolve, reload = self.run_to_backend(task)
        self.assertEqual(resolve.call_args_list, [mock.call("ltx-2.3-audio-vae", model_type="vae"),
                                                 mock.call("ltx-2.3-connectors", model_type="text-encoder")])
        kwargs = backend.generate_video.call_args.kwargs
        self.assertEqual(kwargs["audio_vae_model"], "/models/vae/ltx-2.3-audio-vae")
        self.assertEqual(kwargs["embeddings_connectors_model"], "/models/text-encoder/ltx-2.3-connectors")
        self.assertEqual(kwargs["scheduler_name"], "ltx2")
        self.assertEqual(kwargs["num_inference_steps"], 8)
        self.assertEqual(kwargs["guidance_scale"], 1)
        reload.assert_called_once()

    def test_missing_auxiliary_resources_fail_before_model_loading(self):
        for field in ("audio_vae_model", "embeddings_connectors_model"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "LTX-2.3 is missing"):
                    self.run_to_backend(self.task(**{field: None}))


if __name__ == "__main__":
    unittest.main()
