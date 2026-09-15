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

    def test_gif_dependencies_load_in_order_as_required_assets(self):
        gif_library = '"/plugins/core/video_plugin/gif.js"'
        gif_plugin = '"/plugins/core/video_plugin/mads-gifs-cpp.plugin.js"'
        self.assertIn(gif_library, self.loader)
        self.assertIn(gif_plugin, self.loader)
        self.assertLess(self.loader.index(gif_library), self.loader.index(gif_plugin))
        optional_source = self.loader.split("const OPTIONAL_UI_PLUGINS", 1)[1].split("])", 1)[0]
        self.assertNotIn('id: "cpp-gifs"', optional_source)
        plugin = (self.plugin_root / "video_plugin" / "mads-gifs-cpp.plugin.js").read_text(encoding="utf-8")
        library = (self.plugin_root / "video_plugin" / "gif.js").read_text(encoding="utf-8")
        self.assertIn('workerScript: GIF_WORKER_SCRIPT', plugin)
        self.assertIn('option.textContent = "Animated GIF"', plugin)
        self.assertNotIn("function stepAnim", plugin)
        self.assertNotIn("function morph", plugin)
        self.assertIn('gif.on("error", (error) => {', plugin)
        self.assertIn("stopWorkers();", plugin)
        self.assertIn("function addFallbackMotionFrames", plugin)
        self.assertIn("frameCount < 2", plugin)
        self.assertIn('worker.onerror = function(event)', library)

    def test_local_image_actions_are_required_plugins(self):
        for relative in (
            "image_plugin/ai-image-critic.plugin.js",
            "image_plugin/make-very-similar.plugin.js",
        ):
            with self.subTest(relative=relative):
                self.assertIn(f'"/plugins/core/{relative}"', self.loader)
                self.assertTrue((self.plugin_root / relative).is_file())
        critic = (self.plugin_root / "image_plugin" / "ai-image-critic.plugin.js").read_text(encoding="utf-8")
        self.assertIn('fetch(url, options)', critic)
        self.assertIn('"/ai-critic/analyze"', critic)
        self.assertIn("Debug data (full local response)", critic)
        self.assertIn("Raw model response", critic)
        self.assertIn("showError(error)", critic)
        self.assertIn('value="cpu">CPU only', critic)
        self.assertIn('computeMode.value === "cpu"', critic)
        self.assertIn('entry.kind === "bundle"', critic)
        self.assertIn("updateModelPairControls()", critic)
        self.assertNotIn("PerchancePluginCore", critic)
        similar = (self.plugin_root / "image_plugin" / "make-very-similar.plugin.js").read_text(encoding="utf-8")
        self.assertNotIn("NoisePreviouslyAdded", similar)
        self.assertIn('text: "Very Similar"', similar)
        server = (self.repo_root / "ui" / "easydiffusion" / "server.py").read_text(encoding="utf-8")
        self.assertIn('@server_api.get("/ai-critic/models")', server)
        self.assertIn('@server_api.post("/ai-critic/analyze")', server)

    def test_native_backend_arguments_use_structured_controls(self):
        parameters = (self.repo_root / "ui" / "media" / "js" / "parameters.js").read_text(encoding="utf-8")
        self.assertIn('id="${parameter.id}" name="${parameter.id}" type="hidden"', parameters)
        self.assertIn('flag: "--vae-tiles"', parameters)
        self.assertIn('flag: "--vae-tiled-overlap"', parameters)
        self.assertIn('flag: "--video-max-vram"', parameters)
        self.assertIn('flag: "--video-offload-to-cpu"', parameters)
        self.assertIn('flag: "--no-half"', parameters)
        self.assertIn('flag: "--no-half-vae"', parameters)
        self.assertIn('flag: "--image-clip-vision-on-cpu"', parameters)
        self.assertIn('flag: "--image-ip-adapter-on-cpu"', parameters)
        self.assertIn("validateNativeBackendArgumentEditor()", parameters)
        self.assertIn("splitNativeBackendArguments", parameters)
        self.assertIn("native-backend-additional-arguments", parameters)

    def test_model_directory_defaults_preserve_browser_overrides(self):
        parameters = (self.repo_root / "ui" / "media" / "js" / "parameters.js").read_text(encoding="utf-8")
        for directory in ("checkpoints", "vae", "taesdvae", "controlnet-lite", "text-encoder", "video", "tipo"):
            self.assertIn(f'["{directory}",', parameters)
        self.assertIn("MODEL_DIRECTORY_STORAGE_KEY", parameters)
        self.assertIn("Object.prototype.hasOwnProperty.call(browserDirectories, key)", parameters)
        self.assertIn("configDirectories[key]", parameters)

    def test_controlnet_architecture_modes_are_automatic_or_standard(self):
        root = self.plugin_root / "controlnet_plugin"
        html = (root / "controlnet.plugin.html").read_text(encoding="utf-8")
        architectures = (root / "controlnet-architectures.plugin.js").read_text(encoding="utf-8")
        self.assertIn('<option value="auto"', html)
        for manual_mode in ("union", "uni", "lite"):
            self.assertNotIn(f'<option value="{manual_mode}"', html)
            self.assertNotIn(f'key: "{manual_mode}"', architectures)
        self.assertIn('"__automatic_uni_union__"', architectures)

    def test_beta_channel_is_removed_from_ui_and_backend_config_api(self):
        parameters = (self.repo_root / "ui" / "media" / "js" / "parameters.js").read_text(encoding="utf-8")
        server = (self.repo_root / "ui" / "easydiffusion" / "server.py").read_text(encoding="utf-8")
        app = (self.repo_root / "ui" / "easydiffusion" / "app.py").read_text(encoding="utf-8")
        sample = (self.repo_root / "scripts" / "config.yaml.sample").read_text(encoding="utf-8")
        for source in (parameters, server, app, sample):
            self.assertNotIn("update_branch", source)
        self.assertNotIn("use_beta_channel", parameters)
        self.assertNotIn("use_v3_engine", parameters)
        self.assertNotIn("use_v3_engine", server)
        self.assertNotIn("use_v3_engine", app)
        self.assertNotIn('backend: sdkit3', sample)

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

    def test_perchance_gallery_uses_unfiltered_rating(self):
        perchance_root = self.plugin_root / "perchance_plugin"
        plugin = (perchance_root / "perchance-gallery.tab.plugin.js").read_text(encoding="utf-8")
        html = (perchance_root / "perchance-gallery.tab.plugin.html").read_text(encoding="utf-8")
        self.assertNotIn('id="perchance-generator-gallery-content-filter"', html)
        self.assertNotIn("position:sticky", html)
        self.assertIn('content_filter: "none"', plugin)
        self.assertNotIn("fields.cursor", plugin)
        self.assertIn("entry.preview_data_url", plugin)
        self.assertIn('data:image/jpeg;base64,', plugin)
        self.assertNotIn("fetch(requestUrl", plugin)
        self.assertNotIn("entry.imageUrl", plugin)

    def test_editor_background_action_uses_native_u2net_route(self):
        editor = (self.repo_root / "ui" / "media" / "js" / "image-editor.js").read_text(encoding="utf-8")
        self.assertIn('fetch("/image-tools/remove-background"', editor)
        self.assertIn('model: "u2net"', editor)
        self.assertNotIn("const thresholdSquared = threshold * threshold\n    const background", editor)

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
        self.assertIn('Path.home() / "AppImages" / "perchance.AppImage"', backend)
        self.assertIn('Path.home() / "Downloads" / PERCHANCE_APPIMAGE_NAME', backend)
        self.assertIn("PERCHANCE_APPIMAGE_SHA256", backend)

    def test_perchance_image_amount_uses_native_batching(self):
        root = self.plugin_root / "perchance_plugin"
        html = (root / "perchance-image.plugin.html").read_text(encoding="utf-8")
        plugin = (root / "perchance-image.plugin.js").read_text(encoding="utf-8")
        backend = (
            self.repo_root / "ui" / "plugins" / "server" / "perchance" / "perchance.py"
        ).read_text(encoding="utf-8")
        self.assertIn('id="perchance-generator-amount"', html)
        self.assertIn('min="1" max="20"', html)
        self.assertIn("amount: requestedAmount", plugin)
        self.assertIn("data.images", plugin)
        self.assertIn("MAX_IMAGE_AMOUNT = 20", backend)
        self.assertIn('"--count"', backend)
        self.assertIn('"--json"', backend)
        self.assertIn("_parse_image_results", backend)
        self.assertIn('"generated_amount": len(images)', backend)
        self.assertIn("Save to Gallery", plugin)
        self.assertIn('core.requestJson("/perchance/images/save"', plugin)
        self.assertIn("_generated_image_directory", backend)
        self.assertIn("save_generated_image", backend)

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
        self.assertIn("setInitialImageSource", gallery)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for image resolution tests")
    def test_initial_image_resolution_is_aspect_preserving_and_grid_aligned(self):
        main_js = (self.repo_root / "ui" / "media" / "js" / "main.js").read_text(encoding="utf-8")
        start = main_js.index("function getGenerationDimensionsForImage")
        end = main_js.index("\nfunction setInitialImageSource", start)
        function_source = main_js[start:end]
        expression = (
            "console.log(JSON.stringify(["
            "getGenerationDimensionsForImage(1000, 667, 64),"
            "getGenerationDimensionsForImage(1237, 799, 8),"
            "getGenerationDimensionsForImage(768, 1024, 8),"
            "getGenerationDimensionsForImage(3840, 2160, 8)"
            "]))"
        )
        result = subprocess.run(
            ["node", "-e", function_source + "\n" + expression],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            '[{"width":960,"height":640},{"width":1240,"height":800},'
            '{"width":768,"height":1024},{"width":2048,"height":1152}]',
        )

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
