import unittest

from easydiffusion.prompt_interpreter import interpret_prompt, normalize_prompt_mode


class PromptInterpreterTests(unittest.TestCase):
    def test_natural_language_is_unchanged(self):
        prompt = "A white fox AND a blue sky: at 16:9"
        self.assertEqual(interpret_prompt(prompt, "natural"), prompt)

    def test_attention_shorthand_is_available_in_natural_language(self):
        self.assertEqual(
            interpret_prompt("bright xyz:1.2 and soft abc++ at 16:9", "natural"),
            "bright (xyz:1.2) and soft (abc:1.21) at 16:9",
        )

    def test_relational_tag_groups_are_fused(self):
        self.assertEqual(
            interpret_prompt("red:fur green:eyes red:orange:chest", "tags"),
            "red fur, green eyes, red orange chest",
        )

    def test_parenthesized_relations_keep_attention_grouping(self):
        self.assertEqual(
            interpret_prompt(
                "(white:color:fur), (white:color:chest) , (furry:female)",
                "tags",
            ),
            "(white color fur), (white color chest), (furry female)",
        )

    def test_numeric_suffix_becomes_explicit_attention(self):
        self.assertEqual(interpret_prompt("xyz:1.2", "tags"), "(xyz:1.2)")
        self.assertEqual(interpret_prompt("(xyz:1.2)", "tags"), "(xyz:1.2)")

    def test_plus_suffix_compounds_ten_percent_attention(self):
        self.assertEqual(interpret_prompt("xyz++", "tags"), "(xyz:1.21)")

    def test_and_fuses_clauses_and_break_remains_a_hard_boundary(self):
        self.assertEqual(
            interpret_prompt("red:fur AND green:eyes BREAK blue:background", "tags"),
            "red fur, green eyes BREAK blue background",
        )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_prompt_mode("automatic")


if __name__ == "__main__":
    unittest.main()
