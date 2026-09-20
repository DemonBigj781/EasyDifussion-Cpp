"""Run with training/runtime/.venv/bin/python; validates actual sd-scripts parsers.

No checkpoint is loaded and no GPU training is started.
"""
import importlib
from pathlib import Path
import sys
import tempfile
import threading
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import trainer

backend, python = trainer.settings(ROOT)
sys.path.insert(0, str(backend))
import library.config_util as config_util

with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    for (kind, architecture), script in trainer.SCRIPTS.items():
        spec = trainer.validate_training({"kind": kind, "architecture": architecture,
            "output_name": "test", "model": "/tmp/base.safetensors", "trigger": "mything"})
        job = directory / (kind + architecture)
        job.mkdir()
        data = job / "source"
        data.mkdir()
        (data / "one.png").touch()
        spec["dataset"] = str(data)
        trainer.stage_dataset(spec, job, threading.Event())
        module = importlib.import_module(script.removesuffix(".py"))
        command = trainer.build_command(spec, backend, python, job)
        index = command.index(str(backend / script))
        parsed = module.setup_parser().parse_args(command[index + 1:])
        assert parsed.save_model_as == "safetensors"
        config = tomllib.loads((job / "dataset.toml").read_text())
        config_util.ConfigSanitizer(True, True, kind == "lora", True).sanitize_user_config(config)
        print(f"PASS: {script} arguments and dataset config")
