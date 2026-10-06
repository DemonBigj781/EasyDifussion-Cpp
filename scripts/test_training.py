"""Trainer filesystem, protocol, job lifecycle and HTTP boundary tests."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import trainer
from training.service import Conflict, TrainingService


def spec(**overrides):
    return trainer.validate_training({"kind": "lora", "architecture": "sd15",
        "output_name": "example", "model": "/tmp/base.safetensors", "trigger": "mything", **overrides})


class TrainerTests(unittest.TestCase):
    def test_traversal_and_symlink_escapes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "datasets"
            root.mkdir()
            (root / "escape").symlink_to(Path(temp), target_is_directory=True)
            for path in ("../x", "", "/etc/passwd", "escape/x"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    trainer.inside(root, path)

    def test_snapshot_includes_trigger_without_changing_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "original"
            dataset.mkdir()
            (dataset / "photo.png").write_bytes(b"image-data")
            (dataset / "photo.txt").write_text("blue sky")
            job = root / "job"
            job.mkdir()
            trainer.stage_dataset(spec(dataset=str(dataset)), job, threading.Event())
            self.assertEqual((dataset / "photo.txt").read_text(), "blue sky")
            self.assertEqual((job / "data/000000.txt").read_text(), "mything, blue sky\n")
            config = tomllib.loads((job / "dataset.toml").read_text())
            self.assertEqual(config["datasets"][0]["subsets"][0]["image_dir"], str(job / "data"))
            prompts = json.loads((job / "sample-prompts.json").read_text())
            self.assertEqual(prompts[0]["prompt"], "mything, blue sky")
            self.assertEqual(prompts[0]["seed"], 42)

    def test_colliding_images_and_symlink_captions_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "same.jpg").touch()
            (root / "same.png").touch()
            with self.assertRaises(ValueError):
                trainer.dataset_images(root)
            (root / "same.jpg").unlink()
            (root / "same.txt").symlink_to(root / "other.txt")
            with self.assertRaises(ValueError):
                trainer.dataset_images(root)

    def test_training_parameter_validation(self):
        for changes in ({"steps": -1}, {"steps": 1.5}, {"rank": True},
                        {"learning_rate": float("nan")}, {"output_name": "../file"},
                        {"resolution": 513}, {"kind": "embedding", "trigger": "two words"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                spec(**changes)

    def test_four_backends_have_matching_training_flags(self):
        for (kind, architecture), script in trainer.SCRIPTS.items():
            with self.subTest(kind=kind, architecture=architecture):
                command = trainer.build_command(spec(kind=kind, architecture=architecture,
                    qwen3="/models/qwen3.safetensors", vae="/models/qwen-image.safetensors"),
                    Path("/backend"), Path("/venv/bin/python"), Path("/job"))
                self.assertIn(f"/backend/{script}", command)
                self.assertIn("--save_model_as=safetensors", command)
                self.assertEqual("--network_module=networks.lora" in command, kind == "lora" and architecture != "anima")
                self.assertEqual("--token_string=mything" in command, kind == "embedding")
                self.assertEqual("--no_half_vae" in command, architecture == "sdxl")
                self.assertEqual("--cache_text_encoder_outputs" in command, architecture == "anima")

    def test_cancellation_terminates_backend_process_group(self):
        cancel = threading.Event()
        pids = []
        def event(name, **values):
            if name == "started":
                pids.append(values["pid"])
                cancel.set()
        with self.assertRaises(trainer.Cancelled):
            trainer.run_child([sys.executable, "-c", "import time; time.sleep(300)"], ROOT, cancel, event)
        with self.assertRaises(ProcessLookupError):
            os.kill(pids[0], 0)

    def test_autotag_jsonl_protocol_and_no_overwrite(self):
        received = []
        class Tagger(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"tags":"blue sky"}')
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Tagger)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                dataset = Path(temp)
                (dataset / "one.png").write_bytes(b"first-image")
                (dataset / "two.png").write_bytes(b"second-image")
                (dataset / "two.txt").write_text("handwritten caption")
                binary = ROOT / "training/dist/sdkit-trainer/sdkit-trainer"
                commands = [[sys.executable, str(ROOT / "training/trainer.py")]]
                if binary.is_file():
                    commands.append([str(binary)])
                for command in commands:
                    (dataset / "one.txt").unlink(missing_ok=True)
                    process = subprocess.Popen(command + ["--root", str(ROOT)], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    request = {"command": "autotag", "dataset": temp, "trigger": "mything", "port": server.server_port}
                    process.stdin.write(json.dumps(request) + "\n")
                    process.stdin.flush()
                    events = [json.loads(line) for line in process.stdout]
                    self.assertEqual(process.wait(timeout=10), 0, process.stderr.read())
                    process.stdin.close()
                    process.stdout.close()
                    process.stderr.close()
                    self.assertEqual(events[-1]["event"], "completed")
                    self.assertEqual(events[-1]["written"], 1)
                    self.assertEqual(events[-1]["skipped"], 1)
                    self.assertEqual((dataset / "one.txt").read_text(), "mything, blue sky\n")
                    self.assertEqual((dataset / "two.txt").read_text(), "handwritten caption")
                self.assertEqual(base64.b64decode(received[0]["image"]), b"first-image")
        finally:
            server.shutdown()
            server.server_close()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = TrainingService(self.root, lambda category: [str(self.root / "models" / category)], lambda: True)
        self.dataset = self.root / "training/datasets/example"
        self.dataset.mkdir(parents=True)
        (self.dataset / "photo.png").touch()
        self.model = self.root / "models/stable-diffusion/base.safetensors"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"model")

    def test_caption_conflicts_and_path_validation(self):
        self.service.save_caption("example", "photo.png", "first", "")
        with self.assertRaises(Conflict):
            self.service.save_caption("example", "photo.png", "second", "")
        with self.assertRaises(ValueError):
            self.service.save_caption("example", "../../config.yaml", "bad", "")
        self.assertEqual((self.dataset / "photo.txt").read_text(), "first\n")

    def test_generation_guard_and_busy_backend(self):
        self.service.jobs["busy"] = {"status": "running"}
        with self.assertRaises(Conflict), self.service.generation_guard():
            self.fail("Generation must be blocked")
        self.service.jobs.clear()
        self.service.idle = lambda: False
        with self.assertRaises(Conflict):
            self.service.start({"dataset": "example", "command": "autotag"})

    def test_model_path_restrictions(self):
        outside = self.root / "external.safetensors"
        outside.touch()
        with self.assertRaises(ValueError):
            self.service.resolve_model(str(outside))

    def wait_finished(self, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = self.service.get(job_id)
            if job.get("finished"):
                return job
            time.sleep(.02)
        self.fail("Job did not terminate")

    def test_process_job_publication_and_persisted_history(self):
        # Exercise the real pipe reader, terminal validation, export and reload.
        code = (
            "import json,sys,pathlib; r=json.loads(sys.stdin.readline()); "
            "p=pathlib.Path(r['job_dir'])/'output'; p.mkdir(); "
            "(p/(r['output_name']+'.safetensors')).write_bytes(b'test weights'); "
            "print(json.dumps({'event':'completed'}),flush=True)"
        )
        with patch.object(self.service, "command", return_value=[sys.executable, "-c", code]), \
             patch.object(self.service, "readiness", return_value={"python_runtime": {"ready": True}}):
            started = self.service.start(spec(architecture="sdxl", dataset="example", model=str(self.model)))
            job = self.wait_finished(started["id"])
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(Path(job["output"]).read_bytes(), b"test weights")
        reloaded = TrainingService(self.root, self.service.model_dirs, lambda: True)
        self.assertEqual(reloaded.get(job["id"])["status"], "completed")
        with self.assertRaises(FileExistsError):
            self.service._publish(job)

    def test_failure_and_cancellation_are_not_success(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                code = "import sys,time; sys.stdin.readline(); " + ("time.sleep(300)" if cancel else "sys.exit(3)")
                with patch.object(self.service, "command", return_value=[sys.executable, "-c", code]):
                    started = self.service.start({"dataset": "example", "command": "autotag"})
                    if cancel:
                        self.service.cancel(started["id"])
                    job = self.wait_finished(started["id"])
                self.assertEqual(job["status"], "cancelled" if cancel else "failed")

    def test_resume_uses_complete_state_and_original_dataset_snapshot(self):
        old_id = "a" * 32
        old_dir = self.service.job_root / old_id
        snapshot = old_dir / "data"
        snapshot.mkdir(parents=True)
        (snapshot / "one.png").touch()
        settings = spec(architecture="sdxl", dataset=str(self.dataset), model=str(self.model))
        self.service.jobs[old_id] = {"id": old_id, "status": "interrupted", "spec": settings,
                                     "kind": "lora", "created": 0, "log": []}
        complete = old_dir / "output" / "example-step00000100-state"
        complete.mkdir(parents=True)
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
            (complete / name).write_bytes(b"test-state")
        (complete / "train_state.json").write_text('{"current_step": 100}')
        incomplete = old_dir / "output" / "example-step00000200-state"
        incomplete.mkdir()
        (incomplete / "optimizer.bin").write_bytes(b"partial")
        with patch.object(self.service, "readiness", return_value={"python_runtime": {"ready": True}}), \
             patch("training.service.threading.Thread.start"):
            result = self.service.start({}, resume_id=old_id)
        self.assertEqual(result["spec"]["resume_state"], str(complete))
        self.assertEqual(result["spec"]["dataset"], str(snapshot))
        self.assertEqual(result["spec"]["command"], "resume")

    def test_server_restart_marks_unfinished_jobs_interrupted(self):
        old_id = "b" * 32
        directory = self.service.job_root / old_id
        directory.mkdir(parents=True)
        (directory / "job.json").write_text(json.dumps({"id": old_id, "status": "running", "created": 0}))
        reloaded = TrainingService(self.root, self.service.model_dirs, lambda: True)
        self.assertEqual(reloaded.get(old_id)["status"], "interrupted")

    def test_native_job_passes_and_persists_both_learning_rates(self):
        (self.dataset / "photo.txt").write_text("cat")
        fake = self.root / "fake-trainer"
        captured = self.root / "arguments.json"
        fake.write_text(
            f"#!{sys.executable}\n"
            "import json,sys,pathlib\n"
            "args=dict(zip(sys.argv[1::2],sys.argv[2::2]))\n"
            f"pathlib.Path({str(captured)!r}).write_text(json.dumps(args))\n"
            "pathlib.Path(args['--output']).write_bytes(b'test weights')\n"
            "print(json.dumps({'event':'completed'}),flush=True)\n"
        )
        fake.chmod(0o700)
        with patch.object(self.service, "native_sd15_binary", return_value=fake):
            job = self.service.start(spec(dataset="example", model=str(self.model),
                learning_rate=5e-5, text_encoder_learning_rate=5e-6, steps=1200,
                rank=32, network_alpha=16, lr_scheduler="cosine_with_restarts",
                lr_warmup_steps=20, lr_scheduler_num_cycles=3))
            final = self.wait_finished(job["id"])
        self.assertEqual(final["status"], "completed", final)
        command = json.loads(captured.read_text())
        self.assertEqual(float(command["--learning-rate"]), 5e-5)
        self.assertEqual(float(command["--text-encoder-learning-rate"]), 5e-6)
        self.assertEqual(command["--steps"], "1200")
        self.assertEqual(float(command["--network-alpha"]), 16)
        self.assertEqual(command["--lr-scheduler"], "cosine_with_restarts")
        self.assertEqual(command["--lr-warmup-steps"], "20")
        self.assertEqual(command["--lr-scheduler-num-cycles"], "3")
        restored = TrainingService(self.root, self.service.model_dirs, lambda: True).get(job["id"])
        self.assertEqual(restored["spec"]["text_encoder_learning_rate"], 5e-6)
        self.assertEqual(restored["spec"]["network_alpha"], 16)
        self.assertEqual(restored["spec"]["lr_scheduler_num_cycles"], 3)

    def test_native_optimizer_does_not_silently_substitute(self):
        with self.assertRaisesRegex(ValueError, "AdamW8bit is not implemented"):
            self.service.start(spec(dataset="example", model=str(self.model), optimizer_type="AdamW8bit"))

    def test_explicit_python_backend_bypasses_native_sd15(self):
        with patch.object(self.service, "readiness", return_value={"python_runtime": {"ready": True}}), \
             patch.object(self.service, "native_sd15_binary", side_effect=AssertionError("must not select native")), \
             patch("training.service.threading.Thread"):
            job = self.service.start(spec(dataset="example", model=str(self.model), backend="python", batch_size=2))
        self.assertEqual(job["spec"]["backend"], "python")
        self.assertFalse(job["spec"]["native_cpp"])
        self.assertEqual(job["spec"]["batch_size"], 2)

    def test_api_rejects_bad_requests_and_maps_conflicts(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ui.plugins.server import training as api
        app = FastAPI()
        app.include_router(api.router, prefix="/training")
        with patch.object(api, "service", return_value=self.service), TestClient(app) as client:
            response = client.post("/training/dataset/scan", json={"dataset": "../outside"})
            self.assertEqual(response.status_code, 400)
            response = client.post("/training/jobs", json={"dataset": "example", "kind": "code"})
            self.assertEqual(response.status_code, 422)
            response = client.post("/training/dataset/caption", json={"dataset": "example", "image": "photo.png", "caption": "new", "previous": "stale"})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(client.get("/training/jobs/missing").status_code, 404)
            response = client.post("/training/jobs", json={"dataset": "example", "model": str(self.model),
                "output_name": "example", "text_encoder_learning_rate": -1})
            self.assertEqual(response.status_code, 422)

    def test_api_preserves_alpha_and_scheduler_fields(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ui.plugins.server import training as api
        app = FastAPI()
        app.include_router(api.router, prefix="/training")
        payload = {"dataset": "example", "model": str(self.model), "output_name": "example",
                   "rank": 32, "network_alpha": 16, "lr_scheduler": "cosine_with_restarts",
                   "lr_warmup_steps": 20, "lr_scheduler_num_cycles": 3}
        with patch.object(api, "service", return_value=self.service), TestClient(app) as client:
            with patch.object(self.service, "start", return_value={"id": "not-a-real-job"}) as start:
                self.assertEqual(client.post("/training/jobs", json=payload).status_code, 200)
                submitted = start.call_args.args[0]
                for key in ("rank", "network_alpha", "lr_scheduler", "lr_warmup_steps", "lr_scheduler_num_cycles"):
                    self.assertEqual(submitted[key], payload[key])
            for invalid in ({"network_alpha": 0}, {"network_alpha": True}, {"lr_warmup_steps": True},
                            {"lr_scheduler": "typo"}, {"lr_scheduler_num_cycles": 1.5}):
                with self.subTest(invalid=invalid):
                    self.assertEqual(client.post("/training/jobs", json={**payload, **invalid}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
