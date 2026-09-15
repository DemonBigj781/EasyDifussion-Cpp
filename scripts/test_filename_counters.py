import importlib.util
import pathlib
import re
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAVE_UTILS_PATH = REPO_ROOT / "ui" / "easydiffusion" / "utils" / "save_utils.py"
PLUGIN_SAVE_UTILS_PATH = REPO_ROOT / "ui" / "plugins" / "server" / "utils" / "save_utils.py"


def load_save_utils():
    easydiffusion = types.ModuleType("easydiffusion")
    easydiffusion.__path__ = []
    app = types.ModuleType("easydiffusion.app")
    app.IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
    app.getConfig = lambda: {}
    easydiffusion.app = app

    ed_types = types.ModuleType("easydiffusion.types")
    for name in (
        "GenerateImageRequest",
        "TaskData",
        "RenderTaskData",
        "OutputFormatData",
        "SaveToDiskData",
        "ModelsData",
    ):
        setattr(ed_types, name, type(name, (), {}))

    sdkit = types.ModuleType("sdkit")
    sdkit.__path__ = []
    sdkit_utils = types.ModuleType("sdkit.utils")
    sdkit_utils.save_dicts = lambda *args, **kwargs: None
    sdkit_utils.save_images = lambda *args, **kwargs: None
    sdkit_models = types.ModuleType("sdkit.models")
    sdkit_models.__path__ = []
    model_loader = types.ModuleType("sdkit.models.model_loader")
    model_loader.__path__ = []
    embeddings = types.ModuleType("sdkit.models.model_loader.embeddings")
    embeddings.get_embedding_token = lambda path: path

    stubs = {
        "easydiffusion": easydiffusion,
        "easydiffusion.app": app,
        "easydiffusion.types": ed_types,
        "sdkit": sdkit,
        "sdkit.utils": sdkit_utils,
        "sdkit.models": sdkit_models,
        "sdkit.models.model_loader": model_loader,
        "sdkit.models.model_loader.embeddings": embeddings,
        "regex": re,
    }
    spec = importlib.util.spec_from_file_location("counter_test_save_utils", SAVE_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class TestFilenameCounters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.save_utils = load_save_utils()

    def setUp(self):
        self.save_utils._calculate_img_number.session_img_numbers.clear()

    def test_batch_uses_consecutive_numbers_from_one_lazy_reservation(self):
        request = types.SimpleNamespace(prompt="test prompt", seed=7)
        task = types.SimpleNamespace(session_id="session-a")
        counter = self.save_utils.ImageNumber(lambda: 42)
        make_filename = self.save_utils.make_filename_callback(
            "$n_$p", request, task, counter, now=1_700_000_000
        )

        self.assertEqual(
            [make_filename(index) for index in range(3)],
            ["00042_test_prompt", "00043_test_prompt", "00044_test_prompt"],
        )

    def test_scan_uses_highest_leading_counter_and_reserves_whole_batch(self):
        task = types.SimpleNamespace(session_id="session-b")
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = pathlib.Path(temp_dir)
            (directory / "00007_older.png").touch()
            (directory / "00042_newer.PNG").touch()
            (directory / "image-without-counter.jpg").touch()
            (directory / "99999_not-an-image.txt").touch()
            (directory / "99998_directory.png").mkdir()

            first = self.save_utils.calculate_img_number(str(directory), task, reserve_count=3)
            second = self.save_utils.calculate_img_number(str(directory), task)

            self.assertEqual([first(index) for index in range(3)], [43, 44, 45])
            self.assertEqual(second(), 46)

    def test_same_session_keeps_independent_counters_per_directory(self):
        task = types.SimpleNamespace(session_id="shared-session")
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.save_utils.calculate_img_number(first_dir, task, reserve_count=2)
            second = self.save_utils.calculate_img_number(second_dir, task)

            self.assertEqual(first(), 0)
            self.assertEqual(second(), 0)

    def test_concurrent_reservations_do_not_overlap(self):
        task = types.SimpleNamespace(session_id="concurrent-session")
        with tempfile.TemporaryDirectory() as temp_dir:
            with ThreadPoolExecutor(max_workers=8) as executor:
                starts = list(
                    executor.map(
                        lambda _: self.save_utils._calculate_img_number(temp_dir, task, reserve_count=2),
                        range(12),
                    )
                )

        self.assertEqual(sorted(starts), list(range(0, 24, 2)))

    def test_runtime_and_packaged_save_helpers_stay_identical(self):
        self.assertEqual(
            SAVE_UTILS_PATH.read_text(encoding="utf-8"),
            PLUGIN_SAVE_UTILS_PATH.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
