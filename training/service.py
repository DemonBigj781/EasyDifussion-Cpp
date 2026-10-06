"""Persistent jobs and filesystem boundaries for the web application's trainer."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from training import trainer

ACTIVE = {"queued", "running", "cancelling"}


class Conflict(ValueError):
    pass


class TrainingService:
    def __init__(self, root, model_dirs, idle):
        self.root = Path(root).resolve()
        self.dataset_root = self.root / "training" / "datasets"
        self.job_root = self.root / "bucket" / "training"
        self.model_dirs = model_dirs
        self.idle = idle
        self.lock = threading.RLock()
        self.jobs = {}
        self.processes = {}
        for saved in self.job_root.glob("*/job.json"):
            try:
                job = json.loads(saved.read_text())
                if job["id"] != saved.parent.name:
                    continue
                if job["status"] in ACTIVE:
                    job.update(status="interrupted", error="Server stopped before this job finished")
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                continue

    def command(self):
        binary = self.root / "training" / "dist" / "sdkit-trainer" / "sdkit-trainer"
        if binary.is_file():
            return [str(binary), "--root", str(self.root)]
        return [sys.executable, str(self.root / "training" / "trainer.py"), "--root", str(self.root)]

    def native_sd15_binary(self):
        configured = os.getenv("ED_SD15_NATIVE_TRAINER")
        candidates = ([Path(configured).expanduser()] if configured else []) + [
            self.root / "training" / "native" / "sd15-sycl" / "sdkit-sd15-trainer",
            self.root / "source" / "sdkit3-port-source" / "build" / "local-linux-x64-sycl" / "bin" / "sdkit-sd15-trainer",
        ]
        return next((path.resolve() for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)

    def native_environment(self):
        env = trainer.child_env()
        oneapi = Path(os.getenv("ONEAPI_ROOT", "/opt/intel/oneapi"))
        library_dirs = []
        for product in ("compiler", "mkl", "tbb", "umf"):
            versions = sorted((oneapi / product).glob("*/"), reverse=True)
            for version in versions[:1]:
                for suffix in ("lib", "lib/x64", "opt/compiler/lib", "lib/intel64/gcc4.8"):
                    candidate = version / suffix
                    if candidate.is_dir():
                        library_dirs.append(str(candidate))
        existing = [item for item in env.get("LD_LIBRARY_PATH", "").split(":") if item]
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(library_dirs + existing))
        # These optimizations are still unstable in the native backward graph.
        env["GGML_SYCL_ENABLE_FUSION"] = "0"
        env["GGML_SYCL_ENABLE_GRAPH"] = "0"
        return env

    def stage_native_dataset(self, source, destination, trigger):
        images = trainer.dataset_images(Path(source))
        destination.mkdir(parents=True, exist_ok=False)
        for index, image in enumerate(images):
            caption = trainer.read_caption(image)
            tags = [tag.strip() for tag in caption.split(",") if tag.strip()]
            if trigger and trigger not in tags:
                tags.insert(0, trigger)
            if not tags:
                raise ValueError(f"Image has no caption or trigger tag: {image.name}")
            target = destination / f"{index:06d}{image.suffix.lower()}"
            shutil.copyfile(image, target)
            target.with_suffix(".txt").write_text(", ".join(tags) + "\n", encoding="utf-8")
        return len(images)

    def readiness(self):
        native_binary = self.native_sd15_binary()
        native = {"ready": native_binary is not None,
                  "detail": "Native C++ SD1.5 LoRA trainer ready on SYCL1" if native_binary else
                            "Native C++ SD1.5 trainer is not built; run source/Build.sh --sycl --target sdkit-sd15-trainer --deploy training/native/sd15-sycl",
                  "binary": str(native_binary) if native_binary else "", "device": "SYCL1"}
        try:
            result = subprocess.run(self.command() + ["probe"], capture_output=True,
                                    text=True, timeout=75)
            if result.returncode:
                python_runtime = {"ready": False, "detail": (result.stderr or result.stdout)[-3000:]}
            else:
                python_runtime = json.loads(result.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            python_runtime = {"ready": False, "detail": "Python training runtime check timed out"}
        except (OSError, ValueError, IndexError) as exc:
            python_runtime = {"ready": False, "detail": str(exc)}
        return {"ready": native["ready"] or python_runtime.get("ready", False),
                "detail": native["detail"] if native["ready"] else
                          python_runtime.get("detail", "Python training runtime unavailable"),
                "native_cpp": native, "python_runtime": python_runtime}

    def dataset(self, value):
        return trainer.inside(self.dataset_root, value)

    def create_dataset(self, value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
            raise ValueError("Dataset name must use 1–80 letters, digits, underscores or hyphens")
        destination = self.dataset(value)
        destination.mkdir(parents=True, exist_ok=False)
        return {"dataset": value, "path": str(destination)}

    def scan(self, value):
        folder = self.dataset(value)
        images = trainer.dataset_images(folder)
        return {"dataset": value, "count": len(images), "items": [
            {"image": str(image.relative_to(folder)), "caption": trainer.read_caption(image)}
            for image in images
        ]}

    def save_caption(self, dataset, image, caption, previous):
        with self.lock:
            self.check_idle_job()
            folder = self.dataset(dataset)
            path = trainer.inside(folder, image)
            if path not in trainer.dataset_images(folder):
                raise ValueError("Image is not part of this dataset")
            if len(caption.encode("utf-8")) > 65536:
                raise ValueError("Caption exceeds 64 KiB")
            if trainer.read_caption(path) != previous:
                raise Conflict("Caption changed since scanning; refresh before saving")
            destination = path.with_suffix(".txt")
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            with temporary.open("x", encoding="utf-8") as output:
                output.write(caption.strip() + "\n")
            temporary.replace(destination)
            return {"caption": caption.strip()}

    def models(self):
        result = []
        seen = set()
        for folder in self.model_dirs("stable-diffusion"):
            root = Path(folder).resolve()
            for path in root.rglob("*.safetensors"):
                resolved = path.resolve()
                if resolved.is_relative_to(root) and resolved not in seen:
                    seen.add(resolved)
                    result.append({"path": str(resolved), "name": str(path.relative_to(root))})
        return sorted(result, key=lambda item: item["name"].lower())

    def training_assets(self):
        result = {"text_encoders": [], "vaes": []}
        for key, output_key, extensions, include_model_dirs in (
            ("text-encoder", "text_encoders", {".safetensors"}, True),
            ("vae", "vaes", {".safetensors", ".pth"}, False),
        ):
            seen = set()
            seen_dirs = set()
            for folder in self.model_dirs(key):
                root = Path(folder).resolve()
                if not root.is_dir():
                    continue
                for path in root.rglob("*"):
                    if path.is_symlink():
                        continue
                    resolved = path.resolve()
                    if not resolved.is_relative_to(root) or resolved in seen:
                        continue
                    if path.is_file() and path.suffix.lower() in extensions:
                        seen.add(resolved)
                        result[output_key].append({"path": str(resolved), "name": str(path.relative_to(root))})
                        if include_model_dirs and path.parent != root:
                            directory = path.parent.resolve()
                            if directory not in seen_dirs:
                                seen_dirs.add(directory)
                                result[output_key].append({
                                    "path": str(directory), "name": f"{path.parent.relative_to(root)} (folder)",
                                })
            result[output_key].sort(key=lambda item: item["name"].lower())
        return result

    def resolve_model(self, value):
        path = Path(value).resolve()
        if path.suffix.lower() != ".safetensors" or not path.is_file():
            raise ValueError("Choose a full SD1.5/SDXL .safetensors checkpoint")
        if not any(path.is_relative_to(Path(root).resolve()) for root in self.model_dirs("stable-diffusion")):
            raise ValueError("Model must be inside a configured checkpoint directory")
        return path

    def resolve_training_asset(self, value, directory_key, extensions, *, allow_directory=False):
        path = Path(value).expanduser().resolve()
        roots = [Path(root).resolve() for root in self.model_dirs(directory_key)]
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f"Asset must be inside a configured {directory_key} directory")
        if allow_directory and path.is_dir():
            return path
        if not path.is_file() or path.suffix.lower() not in extensions:
            allowed = ", ".join(sorted(extensions))
            raise ValueError(f"Choose a file with one of these extensions: {allowed}")
        return path

    def check_idle_job(self):
        if any(job["status"] in ACTIVE for job in self.jobs.values()):
            raise Conflict("A training or autotag job is already active")

    @contextmanager
    def generation_guard(self):
        with self.lock:
            self.check_idle_job()
            yield

    def _save(self, job):
        destination = self.job_root / job["id"] / "job.json"
        temp = destination.with_suffix(".tmp")
        temp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temp.replace(destination)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("Training job not found")
            return json.loads(json.dumps(self.jobs[job_id]))

    def history(self):
        with self.lock:
            return [{key: value for key, value in job.items() if key not in {"log", "spec"}}
                    for job in sorted(self.jobs.values(), key=lambda job: job["created"], reverse=True)[:100]]

    def start(self, payload, *, port=9000, resume_id=None):
        with self.lock:
            self.check_idle_job()
            if not self.idle():
                raise Conflict("Wait for generation to finish and empty the queue first")
            spec = dict(payload)
            if resume_id:
                old = self.get(resume_id)
                if old["spec"].get("native_cpp"):
                    raise ValueError("Native C++ SD1.5 jobs do not support resume yet; start a new job")
                if old["status"] not in {"failed", "cancelled", "interrupted"} or old["spec"].get("kind") != "lora":
                    raise ValueError("Resume supports interrupted LoRA jobs with saved state")
                spec = dict(old["spec"])
                spec["backend"] = "python"
                states = []
                for state in (self.job_root / resume_id / "output").glob("*-state"):
                    required = ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "train_state.json")
                    if not all((state / name).is_file() and (state / name).stat().st_size for name in required):
                        continue
                    try:
                        step = json.loads((state / "train_state.json").read_text())["current_step"]
                        if isinstance(step, int) and 0 < step < spec["steps"]:
                            states.append((step, state))
                    except (ValueError, KeyError):
                        continue
                if not states:
                    raise ValueError("No saved training state yet; start a new job")
                spec["resume_state"] = str(max(states, key=lambda entry: entry[0])[1])
                # Reuse the original dataset snapshot, even if source captions changed.
                spec["dataset"] = str(self.job_root / resume_id / "data")
            else:
                spec["dataset"] = str(self.dataset(spec.get("dataset", "")))
            images = trainer.dataset_images(Path(spec["dataset"]))
            if spec.get("command") == "autotag":
                spec["port"] = port
                for name, default in (("threshold", .35), ("character_threshold", .85)):
                    value = spec.get(name, default)
                    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                        raise ValueError("Tag thresholds must be between 0 and 1")
                if len(spec.get("trigger", "")) > 200:
                    raise ValueError("Trigger is too long")
            else:
                requested_backend = spec.get("backend") or ("native" if spec.get("architecture") == "sd15" else "python")
                if requested_backend == "native":
                    spec = trainer.native_epoch_options(spec, len(images))
                elif spec.get("epochs") is not None or spec.get("dataset_repeats", 1) != 1:
                    raise ValueError("Epoch sampling settings currently apply only to native SD1.5")
                spec = trainer.validate_training(spec)
                spec["model"] = str(self.resolve_model(spec.get("model", "")))
                if spec["architecture"] == "anima":
                    spec["qwen3"] = str(self.resolve_training_asset(
                        spec.get("qwen3", ""), "text-encoder", {".safetensors"}, allow_directory=True))
                    spec["vae"] = str(self.resolve_training_asset(
                        spec.get("vae", ""), "vae", {".safetensors", ".pth"}))
                spec["native_cpp"] = spec["backend"] == "native"
                if spec["native_cpp"]:
                    if spec.get("optimizer_type", "AdamW") != "AdamW":
                        raise ValueError("Native C++ SD1.5 training supports AdamW only; AdamW8bit is not implemented")
                    if spec["kind"] != "lora":
                        raise ValueError("Native C++ SD1.5 training currently supports LoRA only")
                    if spec["batch_size"] != 1:
                        raise ValueError("Native C++ SD1.5 training currently requires batch size 1")
                    if spec["resolution"] > 1024:
                        raise ValueError("Native C++ SD1.5 training supports resolutions up to 1024")
                    binary = self.native_sd15_binary()
                    if binary is None:
                        raise ValueError("Native C++ SD1.5 trainer is not built. Build with source/Build.sh --sycl --target sdkit-sd15-trainer --deploy training/native/sd15-sycl")
                    spec["native_cpp"] = True
                    spec["native_binary"] = str(binary)
                    spec["device"] = "SYCL1"
                    spec["threads"] = 10
                else:
                    ready = self.readiness().get("python_runtime", {"ready": False, "detail": "Python trainer unavailable"})
                    if not ready["ready"]:
                        raise ValueError(ready["detail"])
                spec["command"] = "resume" if resume_id else f"train-{spec['kind']}"
            job_id = uuid4().hex
            directory = self.job_root / job_id
            directory.mkdir(parents=True)
            spec["job_dir"] = str(directory)
            job = {"id": job_id, "status": "queued", "created": time.time(),
                   "spec": spec, "kind": spec.get("kind", "autotag"), "log": [], "resume_of": resume_id}
            self.jobs[job_id] = job
            self._save(job)
            thread = threading.Thread(target=self._worker, args=(job_id,), daemon=True)
            thread.start()
            return self.get(job_id)

    def _publish(self, job):
        spec = job["spec"]
        source = self.job_root / job["id"] / "output" / f"{spec['output_name']}.safetensors"
        if not source.is_file() or not source.stat().st_size:
            raise ValueError("Training did not create its expected output")
        category = "lora" if spec["kind"] == "lora" else "embeddings"
        root = Path(self.model_dirs(category)[0]).resolve()
        output_dir = trainer.inside(root, "trained")
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{spec['output_name']}-{job['id'][:8]}.safetensors"
        temp = output_dir / f".{job['id']}.partial"
        # Publish atomically with no replacement of any existing model.
        with source.open("rb") as src, temp.open("xb") as dest:
            shutil.copyfileobj(src, dest)
        try:
            os.link(temp, output)
        finally:
            temp.unlink(missing_ok=True)
        return str(output)

    def _worker(self, job_id):
        job = self.jobs[job_id]
        process = None
        backend_pid = None
        terminal = None
        final_status = "failed"
        try:
            with self.lock:
                if job["status"] == "cancelling":
                    raise trainer.Cancelled()
                if job["spec"].get("native_cpp"):
                    spec = job["spec"]
                    job_dir = self.job_root / job_id
                    staged_dataset = job_dir / "data"
                    count = self.stage_native_dataset(spec["dataset"], staged_dataset, spec["trigger"])
                    output = job_dir / "output" / f"{spec['output_name']}.safetensors"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    command = [spec["native_binary"], "--model", spec["model"],
                               "--dataset", str(staged_dataset), "--output", str(output),
                               "--trigger", spec["trigger"], "--device", spec["device"],
                               "--resolution", str(spec["resolution"]), "--steps", str(spec["steps"]),
                               "--save-every", str(spec["save_every"]), "--rank", str(spec["rank"]),
                               "--network-alpha", str(spec["network_alpha"]),
                               "--lr-scheduler", spec["lr_scheduler"],
                               "--lr-warmup-steps", str(spec["lr_warmup_steps"]),
                               "--lr-scheduler-num-cycles", str(spec["lr_scheduler_num_cycles"]),
                               "--threads", str(spec["threads"]), "--learning-rate", str(spec["learning_rate"]),
                               "--text-encoder-learning-rate", str(spec["text_encoder_learning_rate"]),
                               "--seed", str(spec["seed"])]
                    command += ["--dataset-repeats", str(spec.get("dataset_repeats", 1))]
                    process = subprocess.Popen(command, cwd=self.root, env=self.native_environment(),
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, errors="replace", bufsize=1, start_new_session=True)
                    backend_pid = process.pid
                    job["log"].append({"event": "dataset_staged", "images": count})
                else:
                    process = subprocess.Popen(self.command(), cwd=self.root, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                        start_new_session=True)
                self.processes[job_id] = process
                job["status"] = "running"
                self._save(job)
                if process.stdin is not None:
                    process.stdin.write(json.dumps(job["spec"]) + "\n")
                    process.stdin.flush()
            last_saved = 0
            with (self.job_root / job_id / "events.jsonl").open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError()
                    except ValueError:
                        event = {"event": "log", "message": line.strip()}
                    with self.lock:
                        if event.get("event") == "started":
                            backend_pid = event.get("pid")
                        if event.get("event") == "progress":
                            job["progress"] = event
                        if event.get("event") in {"completed", "failed", "cancelled"}:
                            terminal = event
                        job["log"].append(event)
                        job["log"] = job["log"][-200:]
                        if time.monotonic() - last_saved >= 1:
                            self._save(job)
                            last_saved = time.monotonic()
            code = process.wait()
            with self.lock:
                if job["status"] == "cancelling" or terminal and terminal["event"] == "cancelled":
                    raise trainer.Cancelled()
                if code or not terminal or terminal["event"] != "completed":
                    raise RuntimeError((terminal or {}).get("error", f"Trainer exited unexpectedly ({code})"))
                if job["spec"]["command"] != "autotag":
                    job["output"] = self._publish(job)
                final_status = "completed"
        except trainer.Cancelled:
            final_status = "cancelled"
        except Exception as exc:
            with self.lock:
                job["error"] = str(exc)
        finally:
            if process:
                if process.stdin is not None:
                    process.stdin.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
                # The controller normally cleans up; cover controller crashes too.
                if backend_pid and (not terminal or terminal["event"] == "failed"):
                    try:
                        os.killpg(backend_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            with self.lock:
                self.processes.pop(job_id, None)
                job["status"] = final_status
                job["finished"] = time.time()
                self._save(job)

    def cancel(self, job_id):
        with self.lock:
            self.get(job_id)
            job = self.jobs[job_id]
            if job["status"] in ACTIVE:
                job["status"] = "cancelling"
                process = self.processes.get(job_id)
                if process and process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                self._save(job)
            return self.get(job_id)

    def shutdown(self):
        with self.lock:
            for job_id in list(self.processes):
                self.cancel(job_id)
