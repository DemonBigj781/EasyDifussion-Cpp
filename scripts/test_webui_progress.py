import importlib.util
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


if __name__ == "__main__":
    unittest.main()
