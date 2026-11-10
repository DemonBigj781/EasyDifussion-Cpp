import threading
import unittest
from types import SimpleNamespace

from easydiffusion import runtime


class RuntimeContextTests(unittest.TestCase):
    def tearDown(self):
        for context in runtime.get_contexts():
            runtime.release_context(context)

    def test_contexts_are_bound_to_their_render_threads(self):
        barrier = threading.Barrier(3)
        observed = {}

        def worker(device):
            context = SimpleNamespace(device=device)
            runtime.bind_context(context)
            barrier.wait()
            observed[device] = runtime.context
            barrier.wait()
            runtime.release_context(context)

        workers = [
            threading.Thread(target=worker, args=("cuda:0",)),
            threading.Thread(target=worker, args=("cuda:1",)),
        ]
        for worker in workers:
            worker.start()
        barrier.wait()
        self.assertEqual(runtime.get_context("cuda:0").device, "cuda:0")
        self.assertEqual({context.device for context in runtime.get_contexts()}, {"cuda:0", "cuda:1"})
        barrier.wait()
        for worker in workers:
            worker.join()

        self.assertEqual(observed["cuda:0"].device, "cuda:0")
        self.assertEqual(observed["cuda:1"].device, "cuda:1")
        self.assertEqual(runtime.get_contexts(), [])

    def test_unbound_thread_cannot_borrow_another_context(self):
        context = SimpleNamespace(device="cuda")

        def worker():
            runtime.bind_context(context)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        with self.assertRaises(RuntimeError):
            runtime.current_context()


if __name__ == "__main__":
    unittest.main()
