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


if __name__ == "__main__":
    unittest.main()
