import pathlib
import re
import shutil
import subprocess
import unittest
from urllib.parse import unquote


class TestBundledUIPluginIntegration(unittest.TestCase):
    def setUp(self):
        self.repo_root = pathlib.Path(__file__).resolve().parent.parent
        self.plugin_root = self.repo_root / "ui" / "plugins" / "ui"
        self.loader = (self.repo_root / "ui" / "media" / "js" / "plugins.js").read_text(encoding="utf-8")

    def test_optional_catalog_only_uses_bundled_core_routes(self):
        catalog_source = self.loader.split("const OPTIONAL_UI_PLUGINS", 1)[1].split("])", 1)[0]
        paths = re.findall(r'path:\s*"([^"]+)"', catalog_source)
        self.assertTrue(paths)
        self.assertNotIn("/plugins/user/", catalog_source)
        for url in paths:
            with self.subTest(url=url):
                self.assertTrue(url.startswith("/plugins/core/"))
                filename = unquote(url.removeprefix("/plugins/core/"))
                self.assertTrue((self.plugin_root / filename).is_file(), filename)

    def test_plugin_manager_deduplicates_loads_and_syncs_controls(self):
        self.assertIn("const loadedScriptPromises = new Map()", self.loader)
        self.assertIn("const loadedOptionalUIPluginIds = new Set()", self.loader)
        self.assertIn("data-optional-plugin-id", self.loader)
        self.assertIn("disabled after reload", self.loader)

    def test_spell_tokenizer_uses_the_bundled_csv_and_system_settings(self):
        worker = (self.plugin_root / "spell-tokenizer.worker.js").read_text(encoding="utf-8")
        plugin = (self.plugin_root / "spell-tokenizer.plugin.js").read_text(encoding="utf-8")
        csv_path = self.plugin_root / "merged_2024-12-22_pt2-ia-dd-ed.csv"
        with csv_path.open(encoding="utf-8") as csv_file:
            first_row = csv_file.readline().strip()
        self.assertTrue(first_row.startswith("1girl,0,"))
        self.assertNotIn("tag,category,count", first_row)
        self.assertIn("/plugins/core/merged_2024-12-22_pt2-ia-dd-ed.csv", worker)
        self.assertIn("/plugins/core/spell-tokenizer.worker.js", plugin)
        self.assertIn("getElementById('system-settings-table')", plugin)

    def test_perchance_gallery_exposes_rating_choices(self):
        plugin = (self.plugin_root / "perchance.plugin.js").read_text(encoding="utf-8")
        self.assertIn('id="${ID_PREFIX}-gallery-content-filter"', plugin)
        for rating in ('value="g"', 'value="pg13"', 'value="none"'):
            self.assertIn(rating, plugin)
        self.assertIn("data-custom-rating", plugin)

    def test_rabbit_hole_has_versioned_entry_points(self):
        dispatcher = (self.plugin_root / "rabbithole.plugin.js").read_text(encoding="utf-8")
        implementation = (self.plugin_root / "rabbithole.plugins.js").read_text(encoding="utf-8")
        for version in ("3.5", "4", "4.5"):
            with self.subTest(version=version):
                self.assertTrue((self.plugin_root / f"rabbithole-v{version}.js").is_file())
        self.assertIn("[data-feature-keys]", dispatcher)
        self.assertIn("style?.display !== \"none\"", dispatcher)
        self.assertIn("rabbithole-v${port}.js", dispatcher)
        self.assertTrue((self.plugin_root / "rabbithole.plugin.html").is_file())
        self.assertIn("/plugins/core/rabbithole.plugin.html", implementation)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for JavaScript syntax validation")
    def test_all_bundled_javascript_parses(self):
        scripts = sorted(self.plugin_root.glob("*.js"))
        scripts.append(self.repo_root / "ui" / "media" / "js" / "plugins.js")
        for script in scripts:
            with self.subTest(script=script.name):
                subprocess.run(["node", "--check", str(script)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
