import importlib.util
import ast
import json
import queue
import time
import pathlib
import sys
import types
import unittest
from unittest import mock


def _load_webui_common():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    module_path = repo_root / "ui" / "easydiffusion" / "backends" / "webui_common.py"

    log = types.SimpleNamespace(warning=lambda *args, **kwargs: None)
    modules = {
        "sdkit": types.ModuleType("sdkit"),
        "sdkit.utils": types.ModuleType("sdkit.utils"),
        "torchruntime": types.ModuleType("torchruntime"),
        "torchruntime.utils": types.ModuleType("torchruntime.utils"),
        "easydiffusion": types.ModuleType("easydiffusion"),
        "easydiffusion.app": types.ModuleType("easydiffusion.app"),
        "easydiffusion.model_manager": types.ModuleType("easydiffusion.model_manager"),
        "common": types.ModuleType("common"),
    }
    modules["sdkit.utils"].base64_str_to_img = lambda value: value
    modules["sdkit.utils"].img_to_base64_str = lambda value: value
    modules["sdkit.utils"].log = log
    modules["torchruntime.utils"].get_device = lambda *args, **kwargs: None
    modules["easydiffusion.app"].getConfig = lambda: {}
    modules["easydiffusion.model_manager"].get_model_dirs = lambda: {}
    modules["easydiffusion.model_manager"].resolve_model_to_use = lambda value, *_args: value
    modules["common"].kill = lambda *_args, **_kwargs: None

    spec = importlib.util.spec_from_file_location("progress_webui_common", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _NeverStop:
    def is_set(self):
        return False

    def wait(self, _timeout):
        return False


class TestNativeProgressPolling(unittest.TestCase):
    def setUp(self):
        self.webui_common = _load_webui_common()
        self.webui_common.WEBUI_HOST = "127.0.0.1"
        self.webui_common.WEBUI_PORT = "17860"
        self.webui_common.WEBUI_API_PREFIX = "/v1"

    def test_one_session_delivers_distinct_steps_through_completion(self):
        responses = [
            _Response(404),
            _Response(
                200,
                {
                    "id_live_preview": 0,
                    "live_preview": "",
                    "current_step": 0,
                    "total_steps": 0,
                    "progress": 0.0,
                    "completed": False,
                },
            ),
            _Response(
                200,
                {
                    "id_live_preview": 0,
                    "live_preview": "",
                    "current_step": 1,
                    "total_steps": 2,
                    "progress": 0.5,
                    "completed": False,
                },
            ),
            _Response(
                200,
                {
                    "id_live_preview": 0,
                    "live_preview": "",
                    "current_step": 2,
                    "total_steps": 2,
                    "progress": 1.0,
                    "completed": True,
                },
            ),
        ]
        sessions = []

        class FakeSession:
            def __init__(self):
                self.urls = []
                sessions.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def post(self, url, **_kwargs):
                self.urls.append(url)
                return responses.pop(0)

        callbacks = []
        with mock.patch.object(self.webui_common.requests, "Session", FakeSession):
            self.webui_common.image_progress_thread(
                "task-id",
                lambda images, step: callbacks.append((images, step)),
                False,
                1,
                2,
                _NeverStop(),
            )

        self.assertEqual([step for _images, step in callbacks], [0, 1, 2])
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(sessions[0].urls), 4)
        self.assertEqual(set(sessions[0].urls), {"http://127.0.0.1:17860/v1/internal/progress"})

    def test_video_retains_actual_total_and_finalization_state(self):
        responses = [_Response(200, {"id_live_preview":0, "live_preview":"", "current_step":step,
                                    "total_steps":31,"progress":step/31,"completed":done,
                                    "phase":phase,"phase_current":phase_current,"phase_total":32})
                     for step, done, phase, phase_current in ((20,False,"sampling",0),(31,False,"decoding",7),(31,True,"finalizing",32))]
        session = mock.MagicMock()
        session.__enter__.return_value = session
        session.post.side_effect = responses
        callbacks = []
        with mock.patch.object(self.webui_common.requests,"Session",return_value=session):
            self.webui_common.image_progress_thread("video",lambda *args:callbacks.append(args),False,1,20,
                                                   _NeverStop(),with_step_metadata=True,with_phase_metadata=True)
        self.assertEqual([(args[1],args[2],args[3],args[4],args[5],args[6]) for args in callbacks],
                         [(20,31,False,"sampling",0,32),(31,31,False,"decoding",7,32),
                          (31,31,True,"finalizing",32,32)])

    def test_both_image_task_copies_preserve_native_effective_total(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        module = types.ModuleType("easydiffusion.backend_manager"); module.backend = None
        for relative in ("ui/easydiffusion/tasks/render_images.py","ui/plugins/server/tasks/render_images.py"):
            path = root / relative
            function = next(node for node in ast.parse(path.read_text()).body
                            if isinstance(node,ast.FunctionDef) and node.name == "make_step_callback")
            namespace = {"GenerateImageRequest":object,"RenderTaskData":object,"queue":queue,"time":time,"json":json}
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),"exec"),namespace)
            output = queue.Queue()
            req = types.SimpleNamespace(num_inference_steps=40,init_image="fixture",prompt_strength=.75)
            with mock.patch.dict(sys.modules,{"easydiffusion.backend_manager":module}):
                callback = namespace["make_step_callback"](None,req,None,output,[],lambda:None)
                callback(None,20,31,False)
                self.assertEqual(json.loads(output.get())["total_steps"],31)
                callback(None,15)
                self.assertEqual(json.loads(output.get())["total_steps"],30)


if __name__ == "__main__":
    unittest.main()
