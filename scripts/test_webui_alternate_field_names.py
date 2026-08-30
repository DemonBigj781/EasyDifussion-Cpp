import pathlib
import re
import unittest


class TestTerminologyConsistency(unittest.TestCase):
    def setUp(self):
        self.repo_root = pathlib.Path(__file__).resolve().parent.parent

    def test_ui_label_keeps_guidance_scale_primary_with_cfg_abbr(self):
        image_settings = (self.repo_root / "ui" / "plugins" / "ui" / "image-settings.plugin.html").read_text(encoding="utf-8")
        self.assertIn("Guidance Scale", image_settings)
        self.assertIn("CFG Scale", image_settings)
        self.assertIn("Classifier-Free Guidance", image_settings)

    def test_ui_label_keeps_prompt_strength_primary_with_denoising_hint(self):
        image_settings = (self.repo_root / "ui" / "plugins" / "ui" / "image-settings.plugin.html").read_text(encoding="utf-8")
        self.assertIn("Prompt Strength", image_settings)
        self.assertIn("Denoising Strength", image_settings)

    def test_task_summary_uses_same_labels(self):
        main_js = (self.repo_root / "ui" / "media" / "js" / "main.js").read_text(encoding="utf-8")
        self.assertIn("Guidance Scale", main_js)
        self.assertIn("CFG", main_js)
        self.assertIn("Classifier-Free Guidance", main_js)
        self.assertIn("Prompt Strength", main_js)
        self.assertIn("Denoising Strength", main_js)

    def test_text_import_accepts_both_label_variants(self):
        dnd_js = (self.repo_root / "ui" / "media" / "js" / "dnd.js").read_text(encoding="utf-8")
        self.assertIn('guidance_scale: ["Guidance Scale", "CFG Scale"]', dnd_js)
        self.assertIn('prompt_strength: ["Prompt Strength", "Denoising Strength"]', dnd_js)

    def test_metadata_export_prefers_beginner_friendly_labels(self):
        save_utils_py = (self.repo_root / "ui" / "easydiffusion" / "utils" / "save_utils.py").read_text(encoding="utf-8")
        self.assertIn('"guidance_scale": "Guidance Scale"', save_utils_py)
        self.assertIn('"prompt_strength": "Prompt Strength"', save_utils_py)

    def test_abbr_has_dotted_underline_styling(self):
        css = (self.repo_root / "ui" / "media" / "css" / "main.css").read_text(encoding="utf-8")
        self.assertRegex(css, re.compile(r"abbr\[title\][^{]*\{[^}]*text-decoration\s*:\s*underline\s+dotted", re.S))

    def test_lora_controls_are_owned_by_required_standalone_panel(self):
        image_settings = (self.repo_root / "ui" / "plugins" / "ui" / "image-settings.plugin.html").read_text(encoding="utf-8")
        lora_settings = (self.repo_root / "ui" / "plugins" / "ui" / "lora-settings.plugin.html").read_text(encoding="utf-8")
        index = (self.repo_root / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('id="lora_model"', image_settings)
        self.assertIn('id="lora_model"', lora_settings)
        self.assertLess(index.index("/plugins/core/lora-settings.plugin.js"), index.index("media/js/main.js"))

    def test_image_checkpoint_can_be_left_unselected(self):
        main_js = (self.repo_root / "ui" / "media" / "js" / "main.js").read_text(encoding="utf-8")
        self.assertRegex(
            main_js,
            re.compile(
                r'new ModelDropdown\(\s*document\.querySelector\("#stable_diffusion_model"\),'
                r'\s*"stable-diffusion",\s*"None"',
                re.S,
            ),
        )

    def test_video_options_uses_shared_checkpoint_and_independent_companions(self):
        video_html = (self.repo_root / "ui" / "plugins" / "ui" / "native-video.plugin.html").read_text(encoding="utf-8")
        video_js = (self.repo_root / "ui" / "plugins" / "ui" / "native-video.plugin.js").read_text(encoding="utf-8")
        image_html = (self.repo_root / "ui" / "plugins" / "ui" / "image-settings.plugin.html").read_text(encoding="utf-8")
        self.assertIn('class="collapsible">Video Options', video_html)
        self.assertNotIn('id="native-video-model"', video_html)
        self.assertIn("Options", image_html)
        self.assertIn('<label for="stable_diffusion_model">Model:</label>', image_html)
        self.assertIn('id="native-video-vae"', video_html)
        self.assertIn('id="native-video-text-encoder"', video_html)
        self.assertIn('const modelInput = byId("stable_diffusion_model")', video_js)
        self.assertNotIn('new ModelDropdown(modelInput, "video"', video_js)
        self.assertIn('new ModelDropdown(vaeInput, "vae", "None / embedded")', video_js)
        self.assertIn('new ModelDropdown(textEncoderInput, "text-encoder", "None / embedded")', video_js)
        self.assertIn('PLUGINS.TASK_BUILD.push', video_js)
        self.assertNotIn('event.reqBody.use_stable_diffusion_model =', video_js)
        self.assertIn('event.reqBody.use_vae_model = videoVae.value || null', video_js)
        self.assertIn('event.reqBody.use_text_encoder_model = videoTextEncoder.value || null', video_js)
        self.assertIn('event.reqBody.sampler_name = "euler"', video_js)
        self.assertIn('companions[model]', video_js)

    def test_sdkit3_receives_absolute_video_checkpoint_paths(self):
        webui_common_py = (
            self.repo_root / "ui" / "easydiffusion" / "backends" / "webui_common.py"
        ).read_text(encoding="utf-8")
        model_manager_cpp = (
            self.repo_root / "source" / "sdkit3-port-source" / "src" / "model_manager.cpp"
        ).read_text(encoding="utf-8")
        self.assertRegex(
            webui_common_py,
            re.compile(r'if USE_SDKIT3_API:\s+.*?model_path = os\.path\.realpath\(model_path\)', re.S),
        )
        self.assertIn('type == ModelType::CHECKPOINT || type == ModelType::CONTROLNET', model_manager_cpp)
        self.assertIn('inside_models && type == ModelType::CHECKPOINT', model_manager_cpp)

    def test_native_video_model_index_scans_shared_store_layouts(self):
        model_manager_py = (
            self.repo_root / "ui" / "easydiffusion" / "model_manager" / "__init__.py"
        ).read_text(encoding="utf-8")
        list_models_py = (
            self.repo_root / "ui" / "easydiffusion" / "model_manager" / "list_models.py"
        ).read_text(encoding="utf-8")
        server_py = (self.repo_root / "ui" / "easydiffusion" / "server.py").read_text(encoding="utf-8")
        for folder in ("checkpoints/video", "SVD", "lighttricks", "Wan", "mochi", "diffusion_models"):
            self.assertIn(f'"{folder}"', model_manager_py)
        self.assertIn('"vae": ("mochi/vae", "Mochi/vae")', model_manager_py)
        self.assertIn('"mochi/t5xxl"', model_manager_py)
        self.assertIn('seen_real_paths = set()', list_models_py)
        self.assertIn('if "/mochi/vae/" in f"/{rel_path}"', list_models_py)
        self.assertIn('LISTABLE_MODEL_TYPES = KNOWN_MODEL_TYPES + ["video"]', model_manager_py)
        self.assertIn('relative_to = app.MODELS_DIR if model_type == "video" else None', list_models_py)
        self.assertIn("model_manager.LISTABLE_MODEL_TYPES", server_py)

    def test_image_model_dropdown_renders_valid_nested_folders_first(self):
        searchable_models_js = (self.repo_root / "ui" / "media" / "js" / "searchable-models.js").read_text(
            encoding="utf-8"
        )
        searchable_models_css = (
            self.repo_root / "ui" / "media" / "css" / "searchable-models.css"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'createElement("li", { "data-folder-path": childFolderPath.substring(1) }, ["model-folder"]',
            searchable_models_js,
        )
        self.assertIn("const allModelElements = [...folderElements, ...modelElements]", searchable_models_js)
        self.assertIn(".model-list li.model-folder > ul", searchable_models_css)

    def test_model_scanner_follows_directory_symlinks_with_cycle_protection(self):
        list_models_py = (
            self.repo_root / "ui" / "easydiffusion" / "model_manager" / "list_models.py"
        ).read_text(encoding="utf-8")
        self.assertIn("entry.is_dir(follow_symlinks=True)", list_models_py)
        self.assertIn("if real_dir in ancestor_targets", list_models_py)

    def test_mochi_video_requires_its_own_companion_models(self):
        render_video_py = (
            self.repo_root / "ui" / "easydiffusion" / "tasks" / "render_video.py"
        ).read_text(encoding="utf-8")
        self.assertIn('model_class == "mochi_v1_preview"', render_video_py)
        self.assertIn('(("Video VAE", "vae"), ("Video Text Encoder", "text-encoder"))', render_video_py)
        self.assertIn('self.request.scheduler_name = "mochi"', render_video_py)
        self.assertIn('self.request.sampler_name = "euler"', render_video_py)
        self.assertIn("Native Mochi currently supports text-to-video only", render_video_py)

    def test_native_video_cpu_offload_flags_are_scoped_to_video_requests(self):
        main_cpp = (self.repo_root / "source" / "sdkit3-port-source" / "src" / "main.cpp").read_text()
        generator_cpp = (
            self.repo_root / "source" / "sdkit3-port-source" / "src" / "image_generator.cpp"
        ).read_text()
        config = (self.repo_root / "config.yaml").read_text()

        self.assertIn('arg == "--video-clip-on-cpu"', main_cpp)
        self.assertIn('arg == "--video-vae-on-cpu"', main_cpp)
        self.assertIn('arg == "--video-offload-to-cpu"', main_cpp)
        self.assertIn('arg == "--video-max-vram"', main_cpp)
        self.assertIn('arg == "--video-stream-layers"', main_cpp)
        self.assertIn('arg == "--image-clip-on-cpu"', main_cpp)
        self.assertIn('arg == "--image-vae-on-cpu"', main_cpp)
        self.assertNotIn('arg == "--clip-on-cpu"', main_cpp)
        self.assertNotIn('arg == "--vae-on-cpu"', main_cpp)
        self.assertIn("!native_video_request && image_clip_on_cpu_", generator_cpp)
        self.assertIn("!native_video_request && image_vae_on_cpu_", generator_cpp)
        self.assertIn("native_video_request && video_clip_on_cpu_", generator_cpp)
        self.assertIn("native_video_request && video_vae_on_cpu_", generator_cpp)
        self.assertIn("native_video_request && video_offload_to_cpu_", generator_cpp)
        self.assertIn("native_video_request && video_mmap_weights_", generator_cpp)
        self.assertIn("native_video_request && !video_max_vram_.empty()", generator_cpp)
        self.assertIn("native_video_request && video_stream_layers_", generator_cpp)
        self.assertIn("--video-clip-on-cpu", config)
        self.assertIn("--video-vae-on-cpu", config)
        self.assertIn("--video-offload-to-cpu", config)
        self.assertIn("--video-max-vram", config)
        self.assertIn("--video-stream-layers", config)
        self.assertIn("g_callback_data.video_generation   = true", generator_cpp)
        self.assertIn("Video sampling step %d/%d", generator_cpp)

    def test_explicit_no_image_checkpoint_does_not_fall_back_to_config(self):
        types_py = (self.repo_root / "ui" / "easydiffusion" / "types.py").read_text(encoding="utf-8")
        model_manager_py = (
            self.repo_root / "ui" / "easydiffusion" / "model_manager" / "__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn('None if selected_checkpoint == "" else selected_checkpoint', types_py)
        self.assertRegex(
            model_manager_py,
            re.compile(r'if model_paths\[model_type\] is None:\s+continue'),
        )
        self.assertIn('None if selected_vae == "" else selected_vae', types_py)
        self.assertIn('None if selected_text_encoder == "" else selected_text_encoder', types_py)
        self.assertIn('and configured_models[model_type] is None', model_manager_py)


if __name__ == "__main__":
    unittest.main()
