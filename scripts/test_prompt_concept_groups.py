import json
import pathlib
import shutil
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT_GROUPS_JS = REPO_ROOT / "ui" / "media" / "js" / "prompt-groups.js"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for prompt-group JavaScript tests")
class TestPromptConceptGroups(unittest.TestCase):
    def run_helper(self, prompt, start, end):
        script = """
const helper = require(process.argv[1]);
const result = helper.isolateConceptSelection(
    process.argv[2],
    Number(process.argv[3]),
    Number(process.argv[4])
);
process.stdout.write(JSON.stringify(result));
"""
        result = subprocess.run(
            ["node", "-e", script, str(PROMPT_GROUPS_JS), prompt, str(start), str(end)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_selected_attributes_become_one_isolated_group(self):
        prompt = "portrait, red-haired woman, green dress, blue robot, studio"
        selected = "red-haired woman, green dress"
        start = prompt.index(selected)
        result = self.run_helper(prompt, start, start + len(selected))

        self.assertEqual(
            result["value"],
            "portrait, BREAK red-haired woman, green dress BREAK, blue robot, studio",
        )
        self.assertEqual(
            result["value"][result["selectionStart"] : result["selectionEnd"]],
            selected,
        )

    def test_whitespace_is_kept_outside_the_selected_group(self):
        prompt = "cat,   red fur, green eyes   , dog"
        start = prompt.index("   red")
        end = prompt.index("   , dog") + 3
        result = self.run_helper(prompt, start, end)

        self.assertEqual(result["value"], "cat,   BREAK red fur, green eyes BREAK   , dog")
        self.assertEqual(
            result["value"][result["selectionStart"] : result["selectionEnd"]],
            "red fur, green eyes",
        )

    def test_existing_boundaries_are_not_duplicated(self):
        prompt = "scene BREAK red fox BREAK blue bird"
        selected = "red fox"
        start = prompt.index(selected)
        result = self.run_helper(prompt, start, start + len(selected))

        self.assertEqual(result["value"], prompt)

    def test_empty_multiline_and_nested_groups_are_rejected(self):
        self.assertEqual(self.run_helper("cat dog", 3, 3), {"error": "empty-selection"})
        self.assertEqual(
            self.run_helper("cat\ndog", 0, len("cat\ndog")),
            {"error": "multi-line-selection"},
        )
        self.assertEqual(
            self.run_helper("cat BREAK dog", 0, len("cat BREAK dog")),
            {"error": "contains-boundary"},
        )

    def test_ui_loads_and_wires_the_prompt_group_helper(self):
        index = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
        main = (REPO_ROOT / "ui" / "media" / "js" / "main.js").read_text(encoding="utf-8")

        helper_at = index.index("media/js/prompt-groups.js")
        main_at = index.index("media/js/main.js")
        self.assertLess(helper_at, main_at)
        self.assertIn('id="isolate-concept-button"', index)
        self.assertIn("PromptGroups.isolateConceptSelection", main)
        self.assertIn('dispatchEvent(new Event("input", { bubbles: true }))', main)


if __name__ == "__main__":
    unittest.main()
