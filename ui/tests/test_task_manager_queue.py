import sys
import types
import unittest


class _Log:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class _Task:
    def __init__(self, task_id):
        self.id = task_id
        self.error = None
        self.render_device = None


fake_device_manager = types.ModuleType("easydiffusion.device_manager")
fake_tasks = types.ModuleType("easydiffusion.tasks")
fake_tasks.Task = _Task
fake_utils = types.ModuleType("easydiffusion.utils")
fake_utils.log = _Log()
fake_torch_utils = types.ModuleType("torchruntime.utils")
fake_torch_utils.get_device_count = lambda: 1
fake_torch_utils.get_device = lambda device: device
fake_torch_utils.get_device_name = lambda device: device
fake_torch_utils.get_installed_torch_platform = lambda: ("cuda", None)
fake_sdkit_utils = types.ModuleType("sdkit.utils")
fake_sdkit_utils.is_cpu_device = lambda device: str(device).startswith("cpu")
fake_sdkit_utils.mem_get_info = lambda _device: (1, 1)

sys.modules["easydiffusion.device_manager"] = fake_device_manager
sys.modules["easydiffusion.tasks"] = fake_tasks
sys.modules["easydiffusion.utils"] = fake_utils
sys.modules.setdefault("torchruntime", types.ModuleType("torchruntime"))
sys.modules["torchruntime.utils"] = fake_torch_utils
sys.modules.setdefault("sdkit", types.ModuleType("sdkit"))
sys.modules["sdkit.utils"] = fake_sdkit_utils

from easydiffusion import runtime, task_manager


class TaskManagerQueueTests(unittest.TestCase):
    def setUp(self):
        task_manager.tasks_queue.clear()
        task_manager.active_tasks.clear()
        task_manager.current_state_error = None
        runtime.bind_context(types.SimpleNamespace(device="cuda:0"))

    def tearDown(self):
        task_manager.tasks_queue.clear()
        task_manager.active_tasks.clear()
        task_manager.current_state_error = None
        runtime.release_context()

    def test_dequeue_is_fifo_and_tracks_dispatched_task(self):
        first = _Task(1)
        second = _Task(2)
        task_manager.tasks_queue.extend((first, second))

        self.assertIs(task_manager.thread_get_next_task(), first)
        self.assertEqual(task_manager.tasks_queue, [second])
        self.assertEqual(task_manager.active_tasks, {first})

    def test_stop_all_marks_queued_and_dispatched_tasks_without_global_error(self):
        queued = _Task(1)
        active = _Task(2)
        task_manager.tasks_queue.append(queued)
        task_manager.active_tasks.add(active)

        task_manager.stop_all_tasks()

        self.assertIsInstance(queued.error, StopAsyncIteration)
        self.assertIsInstance(active.error, StopAsyncIteration)
        self.assertIsNone(task_manager.current_state_error)


if __name__ == "__main__":
    unittest.main()
