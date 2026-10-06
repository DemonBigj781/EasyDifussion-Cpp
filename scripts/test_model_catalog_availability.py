import importlib
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ui"))


class TestCatalogAvailability(unittest.TestCase):
    def test_download_placeholders_are_not_installed_models(self):
        for name in (
            "easydiffusion.model_manager.list_models",
            "ui.plugins.server.Browse_files.model_manager.list_models",
        ):
            with self.subTest(module=name):
                module = importlib.import_module(name)
                existing = {"model": "t5xxl_fp16", "tags": ["text-encoder"]}
                absent = {"model": "clip_l", "tags": ["text-encoder"]}
                catalog = [existing]
                module.include_prefilled_models(catalog, [existing, absent])
                self.assertEqual(len(catalog), 2)
                self.assertIs(catalog[0], existing)
                self.assertIsNot(catalog[1], absent)
                self.assertIs(catalog[1]["installed"], False)
                self.assertNotIn("installed", existing)
                self.assertNotIn("installed", absent)


if __name__ == "__main__":
    unittest.main()
