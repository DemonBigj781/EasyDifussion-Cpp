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
        self.assertIn("const optionalUIPluginLoadPromises = new Map()", self.loader)
        self.assertIn("data-optional-plugin-id", self.loader)
        self.assertIn('toggle.className = "input-toggle"', self.loader)
        self.assertIn("switchLabel.htmlFor = checkbox.id", self.loader)
        self.assertIn("setOptionalUIPluginEnabled(plugin, checkbox.checked)", self.loader)
        self.assertIn("disabled after reload", self.loader)
        self.assertNotIn('id: "optional_ui_plugins"', self.loader)
        plugin_tab = (self.plugin_root / "loader_plugin" / "plugins.tab.plugin.js").read_text(encoding="utf-8")
        self.assertIn("createLocalPluginManagerTab()", plugin_tab)
        self.assertIn("loader_plugin/plugins.tab.plugin.html", self.loader)

    def test_native_backend_arguments_use_structured_controls(self):
        parameters = (self.repo_root / "ui" / "media" / "js" / "parameters.js").read_text(encoding="utf-8")
        self.assertIn('id="${parameter.id}" name="${parameter.id}" type="hidden"', parameters)
        self.assertIn('flag: "--vae-tiles"', parameters)
        self.assertIn('flag: "--vae-tiled-overlap"', parameters)
        self.assertIn('flag: "--video-max-vram"', parameters)
        self.assertIn('flag: "--video-offload-to-cpu"', parameters)
        self.assertIn("validateNativeBackendArgumentEditor()", parameters)
        self.assertIn("splitNativeBackendArguments", parameters)
        self.assertIn("native-backend-additional-arguments", parameters)

    def test_beta_channel_is_removed_from_ui_and_backend_config_api(self):
        parameters = (self.repo_root / "ui" / "media" / "js" / "parameters.js").read_text(encoding="utf-8")
        server = (self.repo_root / "ui" / "easydiffusion" / "server.py").read_text(encoding="utf-8")
        app = (self.repo_root / "ui" / "easydiffusion" / "app.py").read_text(encoding="utf-8")
        sample = (self.repo_root / "scripts" / "config.yaml.sample").read_text(encoding="utf-8")
        for source in (parameters, server, app, sample):
            self.assertNotIn("update_branch", source)
        self.assertNotIn("use_beta_channel", parameters)

    def test_huggingface_token_is_saved_with_browser_settings(self):
        plugin = (self.plugin_root / "files_plugin" / "online-model-browser.plugin.js").read_text(encoding="utf-8")
        self.assertIn("huggingfaceToken: hfTokenEl.value.trim()", plugin)
        self.assertIn('hfTokenEl.value = settings.huggingfaceToken || ""', plugin)
        self.assertIn("apiKeyEl, hfTokenEl", plugin)
        self.assertRegex(plugin, re.compile(r"clearHfBtn\.addEventListener.*?saveSettings\(\)", re.S))

    def test_spell_tokenizer_uses_the_bundled_csv_and_system_settings(self):
        prompt_root = self.plugin_root / "prompt_plugin"
        worker = (prompt_root / "spell-tokenizer.worker.js").read_text(encoding="utf-8")
        plugin = (prompt_root / "spell-tokenizer.plugin.js").read_text(encoding="utf-8")
        csv_path = prompt_root / "merged_2024-12-22_pt2-ia-dd-ed.csv"
        with csv_path.open(encoding="utf-8") as csv_file:
            first_row = csv_file.readline().strip()
        self.assertTrue(first_row.startswith("1girl,0,"))
        self.assertNotIn("tag,category,count", first_row)
        self.assertIn("/plugins/core/prompt_plugin/merged_2024-12-22_pt2-ia-dd-ed.csv", worker)
        self.assertIn("/plugins/core/prompt_plugin/spell-tokenizer.worker.js", plugin)
        self.assertIn("getElementById('system-settings-table')", plugin)

    def test_perchance_gallery_exposes_rating_choices(self):
        perchance_root = self.plugin_root / "perchance_plugin"
        plugin = (perchance_root / "perchance-gallery.tab.plugin.js").read_text(encoding="utf-8")
        html = (perchance_root / "perchance-gallery.tab.plugin.html").read_text(encoding="utf-8")
        self.assertIn('id="perchance-generator-gallery-content-filter"', html)
        for rating in ('value="g"', 'value="pg13"', 'value="none"'):
            self.assertIn(rating, html)
        self.assertIn("data-custom-rating", plugin)

    def test_perchance_features_are_independently_toggleable(self):
        self.assertIn('id: "perchance-image"', self.loader)
        self.assertIn('id: "perchance-text"', self.loader)
        self.assertIn('id: "perchance-gallery"', self.loader)
        self.assertIn("perchance_plugin/perchance.plugin.js", self.loader)
        for filename in (
            "perchance-image.plugin",
            "perchance-text.plugin",
            "perchance-gallery.tab.plugin",
        ):
            with self.subTest(filename=filename):
                root = self.plugin_root / "perchance_plugin"
                self.assertTrue((root / f"{filename}.js").is_file())
                self.assertTrue((root / f"{filename}.html").is_file())

    def test_perchance_release_launcher_is_auto_discovered(self):
        backend = (
            self.repo_root / "ui" / "plugins" / "server" / "perchance" / "perchance.py"
        ).read_text(encoding="utf-8")
        self.assertIn('PERCHANCE_RELEASE_TAG = "v1.0.0-rc.1"', backend)
        self.assertIn('Path.home() / "Downloads" / PERCHANCE_APPIMAGE_NAME', backend)
        self.assertIn("PERCHANCE_APPIMAGE_SHA256", backend)

    def test_rabbit_hole_has_versioned_entry_points(self):
        rabbit_root = self.plugin_root / "rabithole_plugin"
        dispatcher = (rabbit_root / "rabbithole.plugin.js").read_text(encoding="utf-8")
        implementation = (rabbit_root / "rabbithole.plugins.js").read_text(encoding="utf-8")
        for version in ("3.5", "4", "4.5"):
            with self.subTest(version=version):
                self.assertTrue((rabbit_root / f"rabbithole-v{version}.js").is_file())
        self.assertIn("[data-feature-keys]", dispatcher)
        self.assertIn("style?.display !== \"none\"", dispatcher)
        self.assertIn("rabbithole-v${port}.js", dispatcher)
        self.assertTrue((rabbit_root / "rabbithole.plugin.html").is_file())
        self.assertIn("/plugins/core/rabithole_plugin/rabbithole.plugin.html", implementation)

    def test_tab_modules_have_html_companions(self):
        pairs = (
            ("main_plugin", "main.tab.plugin"),
            ("ui_plugin", "settings.tab.plugin"),
            ("loader_plugin", "plugins.tab.plugin"),
            ("gallery_plugin", "gallery.tab.plugin"),
            ("draw_plugin", "editor-page.plugin"),
            ("tipo_plugin", "tipo.plugin"),
            ("outpaint_plugin", "outpaint-editor.plugin"),
        )
        for directory, stem in pairs:
            with self.subTest(stem=stem):
                root = self.plugin_root / directory
                self.assertTrue((root / f"{stem}.js").is_file())
                self.assertTrue((root / f"{stem}.html").is_file())

    def test_gallery_can_load_an_image_or_its_configuration(self):
        gallery = (self.plugin_root / "gallery_plugin" / "gallery.tab.plugin.js").read_text(encoding="utf-8")
        self.assertIn("Load image", gallery)
        self.assertIn("Load as config", gallery)
        self.assertIn("loadSetupFromImage", gallery)
        self.assertIn("init_image_preview", gallery)

    def test_server_integrations_are_packaged(self):
        server_root = self.repo_root / "ui" / "plugins" / "server"
        packages = (
            "online_model_browser", "tipo", "model_manager", "easydb", "tasks", "utils",
            "file_parser", "gallery", "model_tools", "native_image_tools", "package_manager",
            "perchance", "wd14_tagger",
        )
        for package in packages:
            with self.subTest(package=package):
                self.assertTrue((server_root / package).is_dir())
                self.assertTrue((server_root / package / "__init__.py").is_file())

    def test_native_checkpoint_conversion_is_exposed(self):
        backend = (self.repo_root / "source" / "sdkit3-port-source" / "src" / "main.cpp").read_text(encoding="utf-8")
        model_tools = (self.repo_root / "ui" / "plugins" / "server" / "model_tools" / "model_tools.py").read_text(encoding="utf-8")
        plugin = (self.plugin_root / "files_plugin" / "model-tools.plugin.js").read_text(encoding="utf-8")
        self.assertIn("--convert-model", backend)
        self.assertIn("--convert-output", backend)
        self.assertIn("--convert-type", backend)
        self.assertIn("_CHECKPOINT_EXTENSIONS", model_tools)
        self.assertIn('router.get("/gguf/sources")', model_tools)
        self.assertIn("Downloaded model source", plugin)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for JavaScript syntax validation")
    def test_all_bundled_javascript_parses(self):
        scripts = sorted(self.plugin_root.rglob("*.js"))
        scripts.append(self.repo_root / "ui" / "media" / "js" / "plugins.js")
        for script in scripts:
            with self.subTest(script=script.name):
                subprocess.run(["node", "--check", str(script)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
