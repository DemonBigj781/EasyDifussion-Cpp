import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ui"))

from easydiffusion.prompt_utils import combine_negative_prompt_for_clip, combine_positive_prompt_for_clip


class TestHiddenNegativePrompt(unittest.TestCase):
    def test_hidden_positive_embeddings_are_appended_last(self):
        self.assertEqual(
            combine_positive_prompt_for_clip("portrait, studio light", "detail_slider, add_detail,"),
            "portrait, studio light, detail_slider, add_detail,",
        )

    def test_hidden_embeddings_are_appended_last(self):
        self.assertEqual(
            combine_negative_prompt_for_clip("blurry, low quality", "EasyNegative, badhandv4,"),
            "blurry, low quality, EasyNegative, badhandv4,",
        )

    def test_hidden_embeddings_work_without_visible_negative_text(self):
        self.assertEqual(
            combine_negative_prompt_for_clip("", "EasyNegative,"),
            "EasyNegative,",
        )

    def test_empty_hidden_prompt_does_not_change_visible_negative_text(self):
        self.assertEqual(
            combine_negative_prompt_for_clip("blurry, low quality", ""),
            "blurry, low quality",
        )

    def test_existing_visible_trailing_comma_is_not_duplicated(self):
        self.assertEqual(
            combine_negative_prompt_for_clip("blurry,", "EasyNegative,"),
            "blurry, EasyNegative,",
        )

    def test_negative_embedding_picker_targets_the_hidden_field_with_a_comma(self):
        source = (REPO_ROOT / "ui" / "media" / "js" / "main.js").read_text(encoding="utf-8")
        self.assertIn('insertAtCursor(hiddenNegativePromptField, text.replace(/,+\\s*$/, "") + ",")', source)
        self.assertIn('insertAtCursor(hiddenPositivePromptField, text.replace(/,+\\s*$/, "") + ",")', source)
        self.assertIn('${text.replace(/,+\\s*$/, "")}, `', source)
        self.assertIn('hiddenNegativePromptField.dispatchEvent(new Event("input", { bubbles: true }))', source)
        self.assertIn('hiddenPositivePromptField.dispatchEvent(new Event("input", { bubbles: true }))', source)

    def test_backend_payload_combines_then_removes_the_private_field(self):
        task_files = (
            REPO_ROOT / "ui" / "easydiffusion" / "tasks" / "render_images.py",
            REPO_ROOT / "ui" / "plugins" / "server" / "tasks" / "render_images.py",
        )
        for task_file in task_files:
            with self.subTest(task_file=task_file):
                source = task_file.read_text(encoding="utf-8")
                positive_at = source.index('backend_req["prompt"] = combine_positive_prompt_for_clip(')
                negative_at = source.index('backend_req["negative_prompt"] = combine_negative_prompt_for_clip(')
                remove_positive_at = source.index('backend_req.pop("hidden_positive_prompt", None)')
                remove_negative_at = source.index('backend_req.pop("hidden_negative_prompt", None)')
                generate_at = source.index("images = backend.generate_images", remove_negative_at)
                self.assertLess(positive_at, remove_positive_at)
                self.assertLess(negative_at, remove_negative_at)
                self.assertLess(remove_positive_at, generate_at)
                self.assertLess(remove_negative_at, generate_at)

    def test_embedding_sets_store_and_restore_both_sides(self):
        source = (REPO_ROOT / "ui" / "media" / "js" / "main.js").read_text(encoding="utf-8")
        self.assertIn('positive: parseEmbeddingField(hiddenPositivePromptField.value)', source)
        self.assertIn('negative: parseEmbeddingField(hiddenNegativePromptField.value)', source)
        self.assertIn('hiddenPositivePromptField.value = formatEmbeddingField', source)
        self.assertIn('hiddenNegativePromptField.value = formatEmbeddingField', source)

    def test_loading_legacy_settings_clears_stale_hidden_embeddings(self):
        source = (REPO_ROOT / "ui" / "media" / "js" / "dnd.js").read_text(encoding="utf-8")
        self.assertIn('!("hidden_negative_prompt" in task.reqBody)', source)
        self.assertIn('hiddenNegativePromptField.value = ""', source)
        self.assertIn('!("hidden_positive_prompt" in task.reqBody)', source)
        self.assertIn('hiddenPositivePromptField.value = ""', source)


if __name__ == "__main__":
    unittest.main()
