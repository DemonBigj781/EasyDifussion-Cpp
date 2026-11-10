import base64
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
from unittest import mock
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
from easydiffusion.video_output import build_animatediff_gif


def frame(color, size=(8, 8)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class GifOutputTests(unittest.TestCase):
    def decode(self, uri):
        self.assertTrue(uri.startswith("data:image/gif;base64,"))
        return Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1], validate=True)))

    def test_valid_gif_decodes_every_frame_and_logs_success(self):
        logger = Mock()
        uri, info = build_animatediff_gif([frame("red"), frame("blue")], 8, logger)
        with self.decode(uri) as image:
            self.assertEqual(image.format, "GIF")
            self.assertEqual(image.n_frames, 2)
            self.assertEqual(image.size, (8, 8))
            duration = 0
            for index in range(image.n_frames):
                image.seek(index); image.load(); duration += image.info["duration"]
            self.assertEqual(duration, 250)
        self.assertEqual(info["frame_count"], 2)
        logger.info.assert_called_once()
        logger.error.assert_not_called()

    def test_identical_frames_may_merge_but_preserve_playback_duration(self):
        uri, info = build_animatediff_gif([frame("red"), frame("red")], 8)
        with self.decode(uri) as image:
            image.load()
            self.assertEqual(image.info["duration"], 250)
        self.assertEqual(info["source_frame_count"], 2)

    def test_corrupt_frames_fail_and_log_errors(self):
        for value in ("not-base64", base64.b64encode(b"not an image").decode(), ""):
            logger = Mock()
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                build_animatediff_gif([value], 8, logger)
            logger.error.assert_called_once()
            logger.info.assert_not_called()

    def test_mismatched_dimensions_fail(self):
        with self.assertRaises(RuntimeError):
            build_animatediff_gif([frame("red"), frame("blue", (4, 4))], 8)

    def test_data_uri_and_single_frame_are_valid(self):
        uri, _ = build_animatediff_gif(["data:image/png;base64," + frame("blue")], 8)
        with self.decode(uri) as image:
            image.load()
            self.assertEqual(image.n_frames, 1)

    def test_uniform_gray_output_is_reported_not_silently_repaired(self):
        logger = Mock()
        uri, _ = build_animatediff_gif([frame((128,128,128))],8,logger)
        self.assertTrue(any("nearly uniform" in call.args[0] for call in logger.warning.call_args_list))
        with self.decode(uri) as image:
            self.assertEqual(image.convert("RGB").getpixel((0,0)),(128,128,128))

    def test_both_task_entrypoints_force_gif_and_report_source_frame_count(self):
        from easydiffusion import app
        from easydiffusion import backend_manager, model_manager, runtime, task_manager
        from easydiffusion.types import ModelsData, OutputFormatData, TaskData, VideoGenerationRequest
        root = Path(__file__).resolve().parents[1]
        for relative in ("ui/easydiffusion/tasks/render_video.py", "ui/plugins/server/tasks/render_video.py"):
            spec = importlib.util.spec_from_file_location("easydiffusion.tasks.gif_fixture", root / relative)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for requested_format in ("jpeg", "png", "webp"):
                with self.subTest(entrypoint=relative, requested_format=requested_format):
                    output_format = OutputFormatData(output_format=requested_format)
                    backend = SimpleNamespace(generate_video=Mock(return_value={"frames": [frame("red")] * 2, "fps": 8}))
                    def generate(_context, callback, **kwargs):
                        callback(None,20,31,False)
                        callback(None,31,31,False,"decoding")
                        return {"frames":[frame("red")]*2,"fps":8}
                    backend.generate_video.side_effect = generate
                    task = module.VideoTask(VideoGenerationRequest(motion_module_model="motion", cache_mode="teacache"),
                                            TaskData(), ModelsData(model_paths={"stable-diffusion": "base"}), output_format)
                    with mock.patch.object(backend_manager, "backend", backend), \
                         mock.patch.object(runtime, "context", SimpleNamespace(), create=True), \
                         mock.patch.object(task_manager, "current_state_error", None), \
                         mock.patch.object(task_manager, "keep_task_alive"), \
                         mock.patch.object(model_manager, "resolve_model_paths"), \
                         mock.patch.object(model_manager, "resolve_model_to_use", return_value="/fixture/motion"), \
                         mock.patch.object(model_manager, "reload_models_if_necessary"), \
                         mock.patch.object(model_manager, "fail_if_models_did_not_load"), \
                         mock.patch.object(module, "identify_model_type", return_value="sd_v1_base"), \
                         mock.patch.object(module, "log", Mock()):
                        task.run()
                    self.assertEqual(backend.generate_video.call_args.kwargs["cache_mode"], "teacache")
                    self.assertEqual(task.response["task_data"]["output_format"], "gif")
                    self.assertEqual(task.response["video"]["frame_count"], 2)
                    self.assertEqual(len(task.response["output"]), 1)
                    progress = __import__("json").loads(task.buffer_queue.get_nowait())
                    self.assertEqual(progress["step"],20)
                    self.assertEqual(progress["total_steps"],31)
                    self.assertEqual(progress["phase"],"sampling")
                    progress = __import__("json").loads(task.buffer_queue.get_nowait())
                    self.assertEqual(progress["phase"],"decoding")
                    self.assertEqual(output_format.output_format, requested_format)
                    with self.decode(task.response["output"][0]["data"]) as image:
                        image.load()
                        self.assertEqual(image.format, "GIF")

    def test_native_video_entrypoints_package_frames_as_one_gif(self):
        from easydiffusion import app
        from easydiffusion import backend_manager, model_manager, runtime, task_manager
        from easydiffusion.types import ModelsData, OutputFormatData, TaskData, VideoGenerationRequest

        root = Path(__file__).resolve().parents[1]
        frames = [frame("red"), frame("blue")]
        for relative in ("ui/easydiffusion/tasks/render_video.py", "ui/plugins/server/tasks/render_video.py"):
            with self.subTest(entrypoint=relative):
                spec = importlib.util.spec_from_file_location("easydiffusion.tasks.native_gif_fixture", root / relative)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                backend = SimpleNamespace(generate_video=Mock(return_value={"frames": frames, "fps": 8}))
                task = module.VideoTask(
                    VideoGenerationRequest(),
                    TaskData(),
                    ModelsData(model_paths={"stable-diffusion": "native-video"}),
                    OutputFormatData(output_format="jpeg"),
                )
                with mock.patch.object(backend_manager, "backend", backend), \
                     mock.patch.object(runtime, "context", SimpleNamespace(), create=True), \
                     mock.patch.object(task_manager, "current_state_error", None), \
                     mock.patch.object(task_manager, "keep_task_alive"), \
                     mock.patch.object(model_manager, "resolve_model_paths"), \
                     mock.patch.object(model_manager, "reload_models_if_necessary"), \
                     mock.patch.object(model_manager, "fail_if_models_did_not_load"), \
                     mock.patch.object(module, "identify_model_type", return_value="ltx_video"), \
                     mock.patch.object(module, "log", Mock()):
                    task.run()

                self.assertEqual(task.response["task_data"]["output_format"], "gif")
                self.assertEqual(task.response["video"]["format"], "gif")
                self.assertTrue(task.response["video"]["is_animated_gif"])
                self.assertEqual(task.response["video"]["frame_count"], 2)
                self.assertEqual(len(task.response["output"]), 1)
                self.assertEqual(task.response["output"][0]["frame_count"], 2)
                with self.decode(task.response["output"][0]["data"]) as image:
                    self.assertEqual(image.n_frames, 2)


if __name__ == "__main__":
    unittest.main()
