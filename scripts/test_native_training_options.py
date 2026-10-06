import unittest
import tempfile
from pathlib import Path

from training import trainer
from training.service import TrainingService


class NativeTrainingOptionsTests(unittest.TestCase):
    def spec(self, **values):
        return dict(kind="lora", architecture="sd15", output_name="cat", **values)

    def test_epoch_steps_use_images_and_repeats(self):
        values = trainer.native_epoch_options(self.spec(epochs=10, dataset_repeats=150), 1)
        self.assertEqual(values["steps"], 1500)
        self.assertEqual(values["steps_per_epoch"], 150)
        for bad in (True, 0, -1, 1.5, "2", 100001):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                trainer.native_epoch_options(self.spec(epochs=bad), 2)

    def test_python_backend_can_train_sd15_clip(self):
        spec = trainer.validate_training(self.spec(backend="python", text_encoder_learning_rate=5e-5))
        command = trainer.build_command({**spec, "model": "/model.safetensors"},
            Path("/backend"), Path("/python"), Path("/job"))
        self.assertIn("--text_encoder_lr=5e-05", command)
        self.assertIn("--unet_lr=0.0001", command)
        self.assertNotIn("--network_train_unet_only", command)

    def test_backend_selection_validation(self):
        self.assertEqual(trainer.validate_training(self.spec())["backend"], "native")
        self.assertEqual(trainer.validate_training(self.spec(backend="python"))["backend"], "python")
        with self.assertRaises(ValueError):
            trainer.validate_training(self.spec(backend="invalid"))
        with self.assertRaises(ValueError):
            trainer.validate_training(dict(kind="lora", architecture="sdxl", output_name="x", backend="native"))

    def test_caption_is_preserved_without_optional_trigger(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "original"
            source.mkdir()
            (source / "image.png").write_bytes(b"fixture")
            text = "anthropomorphic cat, white armor\nrunning, blue eyes"
            (source / "image.txt").write_text(text)
            service = TrainingService(root, lambda _: [], lambda: True)
            service.stage_native_dataset(source, root / "staged", "")
            self.assertEqual((root / "staged/000000.txt").read_text(), text + "\n")
            self.assertEqual((source / "image.txt").read_text(), text)

    def test_legacy_defaults_follow_rank(self):
        for alpha in ({}, {"network_alpha": None}):
            spec = trainer.validate_training(self.spec(rank=32, **alpha))
            self.assertEqual(spec["network_alpha"], 32)
            self.assertEqual(spec["lr_scheduler"], "constant")
            self.assertEqual(spec["lr_warmup_steps"], 0)
            self.assertEqual(spec["lr_scheduler_num_cycles"], 1)

    def test_independent_alpha_and_cosine_restarts(self):
        spec = trainer.validate_training(self.spec(rank=32, network_alpha=16,
            steps=1500, lr_scheduler="cosine_with_restarts", lr_warmup_steps=100,
            lr_scheduler_num_cycles=3))
        self.assertEqual(spec["network_alpha"], 16)
        command = trainer.build_command({**spec, "model": "/model.safetensors"},
            Path("/backend"), Path("/python"), Path("/job"))
        for option in ("--network_alpha=16", "--lr_scheduler=cosine_with_restarts",
                       "--lr_warmup_steps=100", "--lr_scheduler_num_cycles=3"):
            self.assertIn(option, command)

    def test_invalid_alpha(self):
        for value in (0, -1, float("nan"), float("inf"), True, "16", 129):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trainer.validate_training(self.spec(network_alpha=value))

    def test_invalid_schedule(self):
        for values in ({"lr_scheduler": "typo"}, {"lr_warmup_steps": -1},
                       {"lr_warmup_steps": True}, {"lr_warmup_steps": 1.5},
                       {"lr_scheduler_num_cycles": 0}, {"lr_scheduler_num_cycles": True},
                       {"lr_scheduler_num_cycles": 1.5}, {"lr_scheduler_num_cycles": 11}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                trainer.validate_training(self.spec(steps=10, **values))
        with self.assertRaises(ValueError):
            trainer.validate_training(self.spec(steps=10, lr_scheduler="cosine_with_restarts", lr_warmup_steps=10))

    def test_constant_rejects_ignored_schedule_options(self):
        for values in ({"lr_warmup_steps": 1}, {"lr_scheduler_num_cycles": 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                trainer.validate_training(self.spec(**values))


if __name__ == "__main__":
    unittest.main()
