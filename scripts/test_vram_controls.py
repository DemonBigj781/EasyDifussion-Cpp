import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ui'))
from easydiffusion import app, backend_manager, server, task_manager
from fastapi import HTTPException

class VRAMControlsTests(unittest.TestCase):
    def setUp(self):
        worker = threading.current_thread()
        self.worker_data = {'ready': True, 'busy': False, 'alive': True}
        for key, value in {'render_threads': [worker], 'weak_thread_data': {worker: self.worker_data},
                           'tasks_queue': [], 'backend_maintenance_active': False}.items():
            patch = mock.patch.object(task_manager, key, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.guard = mock.patch.object(server.training, 'generation_guard')
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.restart = mock.patch.object(backend_manager, 'restart_backend')
        self.restart_mock = self.restart.start()
        self.addCleanup(self.restart.stop)

    def test_clear_idle_models(self):
        def during_restart():
            self.assertFalse(task_manager.backend_is_idle())
            self.assertTrue(task_manager.backend_maintenance_active)
            with self.assertRaisesRegex(ConnectionRefusedError, 'VRAM is being cleared'):
                task_manager.enqueue_task(SimpleNamespace())
        self.restart_mock.side_effect = during_restart
        response = server.clear_vram_internal()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)['status'], 'OK')
        self.restart_mock.assert_called_once_with()
        self.assertTrue(task_manager.backend_is_idle())

    def test_busy_worker_is_not_interrupted(self):
        self.worker_data['busy'] = True
        with self.assertRaises(HTTPException) as raised:
            server.clear_vram_internal()
        self.assertEqual(raised.exception.status_code, 409)
        self.restart_mock.assert_not_called()

    def test_queued_work_is_not_discarded(self):
        queued = object()
        task_manager.tasks_queue.append(queued)
        with self.assertRaises(HTTPException) as raised:
            server.clear_vram_internal()
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(task_manager.tasks_queue, [queued])
        self.restart_mock.assert_not_called()

    def test_failed_restart_releases_reservation(self):
        self.restart_mock.side_effect = RuntimeError('engine startup failed')
        with self.assertRaises(HTTPException) as raised:
            server.clear_vram_internal()
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn('engine startup failed', raised.exception.detail)
        self.assertFalse(task_manager.backend_maintenance_active)

    def test_second_clear_is_rejected(self):
        with task_manager.backend_maintenance():
            with self.assertRaises(HTTPException) as raised:
                server.clear_vram_internal()
            self.assertEqual(raised.exception.status_code, 409)
            self.assertTrue(task_manager.backend_maintenance_active)
        self.restart_mock.assert_not_called()

    def test_dequeued_work_is_already_busy(self):
        from easydiffusion import runtime
        task = SimpleNamespace(render_device='cuda')
        task_manager.tasks_queue.append(task)
        with mock.patch.object(runtime, 'context', SimpleNamespace(device='cuda')):
            self.assertIs(task_manager.thread_get_next_task(), task)
        self.assertTrue(self.worker_data['busy'])
        self.assertFalse(task_manager.backend_is_idle())

    def test_maintenance_blocks_worker_dequeue(self):
        task_manager.backend_maintenance_active = True
        task_manager.tasks_queue.append(object())
        self.assertIsNone(task_manager.thread_get_next_task())
        self.assertEqual(len(task_manager.tasks_queue), 1)

class GenerationErrorPropagationTests(unittest.TestCase):
    def test_native_errors_reach_the_render_task(self):
        with mock.patch.object(sys, "path", [str(ROOT / "ui/easydiffusion/backends"), *sys.path]):
            from easydiffusion.backends import webui_common
        for message in ("Out of VRAM: the GPU ran out of memory.",
                        "Vulkan GPU device lost (ErrorDeviceLost). Use Settings > Clear VRAM.",
                        "Generation failed: Model file not found"):
            response = mock.Mock(status_code=500)
            response.json.return_value = {"message": message}
            with self.subTest(message=message), \
                 mock.patch.object(webui_common, "webui_opts", {}), \
                 mock.patch.object(webui_common, "webui_post", return_value=response), \
                 mock.patch.object(webui_common, "log"):
                with self.assertRaises(Exception) as raised:
                    webui_common.generate_images(SimpleNamespace(model_paths={}), prompt="test")
                self.assertTrue(str(raised.exception).startswith(message))


if __name__ == '__main__':
    unittest.main()
