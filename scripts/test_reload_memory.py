import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


def _load_task_manager():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    module_path = repo_root / "ui" / "easydiffusion" / "task_manager.py"

    class Task:
        pass

    log = types.SimpleNamespace(debug=lambda *args, **kwargs: None)
    modules = {
        "easydiffusion": types.ModuleType("easydiffusion"),
        "easydiffusion.device_manager": types.ModuleType("easydiffusion.device_manager"),
        "easydiffusion.tasks": types.ModuleType("easydiffusion.tasks"),
        "easydiffusion.utils": types.ModuleType("easydiffusion.utils"),
        "torchruntime": types.ModuleType("torchruntime"),
        "torchruntime.utils": types.ModuleType("torchruntime.utils"),
        "sdkit": types.ModuleType("sdkit"),
        "sdkit.utils": types.ModuleType("sdkit.utils"),
    }
    modules["easydiffusion"].device_manager = modules["easydiffusion.device_manager"]
    modules["easydiffusion.tasks"].Task = Task
    modules["easydiffusion.utils"].log = log
    for name in ("get_device_count", "get_device", "get_device_name", "get_installed_torch_platform"):
        setattr(modules["torchruntime.utils"], name, lambda *args, **kwargs: None)
    modules["sdkit.utils"].is_cpu_device = lambda device: False
    modules["sdkit.utils"].mem_get_info = lambda device: (0, 0)

    spec = importlib.util.spec_from_file_location("reload_memory_task_manager", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TestReloadSessionMemory(unittest.TestCase):
    def setUp(self):
        self.task_manager = _load_task_manager()

    def test_ping_style_lookups_do_not_allocate_empty_sessions(self):
        for index in range(1_000):
            session = self.task_manager.get_cached_session(
                f"reload-{index}", update_ttl=True, create=False
            )
            self.assertIsNone(session)

        self.assertEqual(self.task_manager.session_cache._base, {})

    def test_work_session_is_created_and_reused(self):
        created = self.task_manager.get_cached_session("working-session")
        reused = self.task_manager.get_cached_session(
            "working-session", update_ttl=True, create=False
        )

        self.assertIs(created, reused)
        self.assertEqual(len(self.task_manager.session_cache._base), 1)


if __name__ == "__main__":
    unittest.main()
