import unittest

from training import trainer


class TextEncoderRateTests(unittest.TestCase):
    def spec(self, **values):
        return dict(kind="lora", architecture="sd15", output_name="cat", **values)

    def test_frozen_default(self):
        self.assertEqual(trainer.validate_training(self.spec())["text_encoder_learning_rate"], 0)

    def test_separate_rate(self):
        spec = trainer.validate_training(self.spec(learning_rate=5e-5, text_encoder_learning_rate=5e-6, steps=1200))
        self.assertEqual(spec["learning_rate"], 5e-5)
        self.assertEqual(spec["text_encoder_learning_rate"], 5e-6)
        self.assertEqual(spec["steps"], 1200)

    def test_invalid_rates(self):
        for value in (-1, float("nan"), float("inf"), True, "0.001", .2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trainer.validate_training(self.spec(text_encoder_learning_rate=value))

    def test_unsupported_architecture_does_not_ignore_rate(self):
        for architecture in ("anima", "sdxl"):
            spec = self.spec(text_encoder_learning_rate=5e-6)
            spec["architecture"] = architecture
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                trainer.validate_training(spec)


if __name__ == "__main__":
    unittest.main()
