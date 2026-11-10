"""Mocked Python transport tests: no backend, model, device or network access."""
import threading
import sys
from types import SimpleNamespace, ModuleType
import unittest
import numpy  # Keep dependencies outside the isolated module-import patch.
from PIL import Image
from unittest.mock import Mock, patch

from test_webui_progress import _load_webui_common, _Response


class NativeTaskIdentityTests(unittest.TestCase):
    def setUp(self):
        # Transport is mocked; do not require requests or let a test use a socket.
        requests = ModuleType("requests")
        requests.Session = Mock(side_effect=AssertionError("unexpected network access"))
        exceptions = ModuleType("requests.exceptions")
        for name in ("ConnectTimeout", "ConnectionError", "ReadTimeout"):
            setattr(exceptions, name, type(name, (Exception,), {}))
        with patch.dict(sys.modules, {"requests": requests, "requests.exceptions": exceptions}):
            self.api = _load_webui_common()
        self.api.USE_SDKIT3_API = True
        self.api.webui_opts = {}
        self.api.print_request = lambda *_args: None
        self.api.webui_post = Mock(return_value=_Response(200))

    def test_native_without_owned_task_never_cancels_someone_else(self):
        self.api.stop_rendering(SimpleNamespace())
        self.api.webui_post.assert_not_called()

    def test_external_webui_retains_bodyless_interrupt(self):
        self.api.USE_SDKIT3_API = False
        self.api.stop_rendering(SimpleNamespace())
        self.api.webui_post.assert_called_once_with('/sdapi/v1/interrupt')

    def test_thread_local_context_is_visible_to_other_thread(self):
        context = threading.local()
        with self.api._native_task_scope(context, 'owned', True):
            thread = threading.Thread(target=self.api.stop_rendering, args=(context,))
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.api.webui_post.assert_called_once_with('/sdapi/v1/interrupt', json={'id_task': 'owned'})
        self.assertIsNone(self.api._native_task_id(context))

    def test_delayed_callback_keeps_original_identity(self):
        context = SimpleNamespace()
        callback = self.api._bind_native_task_callback(context, 'old', lambda: self.api.stop_rendering(context))
        with self.api._native_task_scope(context, 'new', True):
            thread = threading.Thread(target=callback)
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(self.api._native_task_id(context), 'new')
        self.api.webui_post.assert_called_once_with('/sdapi/v1/interrupt', json={'id_task': 'old'})

    def test_old_scope_cannot_clear_replacement_identity(self):
        context = SimpleNamespace()
        with self.api._native_task_scope(context, 'old', True):
            with self.api._native_task_lock:
                self.api._native_tasks[id(context)] = (context, 'new')
        self.assertEqual(self.api._native_task_id(context), 'new')

    def test_duplicate_context_rejected_without_replacing_owner(self):
        context = SimpleNamespace()
        with self.api._native_task_scope(context, 'old', True):
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                with self.api._native_task_scope(context, 'new', True):
                    self.fail('second request admitted')
            self.assertEqual(self.api._native_task_id(context), 'old')
        self.assertIsNone(self.api._native_task_id(context))

    def test_callback_binding_unwinds_on_exception(self):
        context = SimpleNamespace()
        def fail():
            raise RuntimeError('callback failed')
        callback = self.api._bind_native_task_callback(context, 'old', fail)
        with self.api._native_task_scope(context, 'new', True):
            with self.assertRaises(RuntimeError):
                callback()
            self.assertEqual(self.api._native_task_id(context), 'new')

    def test_native_refresh_runs_both_indexes_in_one_ordered_worker(self):
        threads = []
        calls = []
        first_entered = threading.Event()
        release_first = threading.Event()
        def make_thread(**kwargs):
            thread = threading.Thread(**kwargs)
            threads.append(thread)
            return thread
        def post(path):
            calls.append(path)
            if len(calls) == 1:
                first_entered.set()
                release_first.wait(timeout=2)
            return _Response(200)
        self.api.webui_post = post
        try:
            with patch.object(self.api, "Thread", make_thread):
                self.api.refresh_models()
            self.assertTrue(first_entered.wait(timeout=2))
            self.assertEqual(len(threads), 1)
            self.assertEqual(calls, ["/sdapi/v1/refresh-checkpoints"])
        finally:
            release_first.set()
            for thread in threads:
                thread.join(timeout=2)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(calls, ["/sdapi/v1/refresh-checkpoints", "/sdapi/v1/refresh-vae-and-text-encoders"])

    def test_external_refresh_keeps_independent_workers(self):
        self.api.USE_SDKIT3_API = False
        workers = []
        class Worker:
            def __init__(self, **kwargs):
                workers.append(kwargs)
            def start(self):
                pass
        with patch.object(self.api, "Thread", Worker):
            self.api.refresh_models()
        self.assertEqual([worker["args"] for worker in workers], [("checkpoints",), ("vae-and-text-encoders",)])

    def test_generation_progress_callback_uses_its_request_id(self):
        for kind in ("image", "video"):
            with self.subTest(kind=kind):
                context = SimpleNamespace(model_paths={})
                captured = {}
                class Poller:
                    def __init__(self, *, target, args, **kwargs):
                        captured["callback"] = args[1]
                    def start(self):
                        pass
                    def join(self, timeout=None):
                        pass
                def post(path, **kwargs):
                    if path.endswith("/interrupt"):
                        captured["interrupt"] = kwargs["json"]["id_task"]
                        return _Response(200)
                    captured["task"] = kwargs["json"]["force_task_id"]
                    thread = threading.Thread(target=captured["callback"])
                    thread.start()
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
                    return _Response(200, {"images": [], "frames": []})
                self.api.webui_post = post
                with patch.object(self.api, "Thread", Poller):
                    generate = self.api.generate_images if kind == "image" else self.api.generate_video
                    generate(context, callback=lambda: self.api.stop_rendering(context))
                self.assertEqual(captured["task"], captured["interrupt"])
                self.assertIsNone(self.api._native_task_id(context))

    def test_actual_image_and_video_request_scopes_cleanup_success_and_failure(self):
        for kind in ('image', 'video'):
            for fail in (False, True):
                with self.subTest(kind=kind, fail=fail):
                    context = SimpleNamespace(model_paths={})
                    seen = []
                    def post(path, **kwargs):
                        task = kwargs['json']['force_task_id']
                        self.assertEqual(self.api._native_task_id(context), task)
                        seen.append(task)
                        if fail:
                            raise RuntimeError('transport failed')
                        return _Response(200, {'images': [], 'frames': [], 'fps': 8})
                    self.api.webui_post = post
                    generate = self.api.generate_images if kind == 'image' else self.api.generate_video
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'transport failed'):
                            generate(context)
                    else:
                        generate(context)
                    self.assertEqual(len(seen), 1)
                    self.assertIsNone(self.api._native_task_id(context))


if __name__ == '__main__':
    unittest.main()
