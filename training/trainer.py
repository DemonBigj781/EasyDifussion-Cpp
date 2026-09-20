"""Dependency-free trainer controller. One JSON request in; JSONL events out.

The application launches one controller per job. A subsequent {"command":"cancel"}
on stdin, EOF from the owning application, or SIGTERM cancels the process tree.
Heavy imports only happen inside the separate uv-managed training runtime.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

BACKEND_REVISION = "4e624302e0088e39933b31cbc71f24212e900f5f"
SCRIPTS = {
    ("lora", "sd15"): "train_network.py",
    ("lora", "sdxl"): "sdxl_train_network.py",
    ("embedding", "sd15"): "train_textual_inversion.py",
    ("embedding", "sdxl"): "sdxl_train_textual_inversion.py",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGES = 10000


class Cancelled(Exception):
    pass


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def inside(root: Path, value: str, *, relative=True) -> Path:
    """Reject traversal and symlink escapes before reading or writing."""
    root = root.resolve()
    raw = Path(value)
    if not value or (relative and raw.is_absolute()) or ".." in raw.parts:
        raise ValueError("Use a non-empty path relative to the permitted directory")
    result = (root / raw).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Path leaves the permitted directory")
    return result


def dataset_images(dataset: Path) -> list[Path]:
    if not dataset.is_dir():
        raise ValueError("Dataset directory does not exist")
    found = []
    captions = set()
    for directory, dirs, files in os.walk(dataset, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if any((Path(directory) / d).is_symlink() for d in dirs):
            raise ValueError("Dataset directories must not contain symlinks")
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            caption = path.with_suffix(".txt")
            if path.is_symlink() or caption.is_symlink():
                raise ValueError("Dataset images and captions must not be symlinks")
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError(f"Image exceeds 64 MiB: {name}")
            if caption in captions:
                raise ValueError(f"Images share a caption filename: {caption.name}")
            captions.add(caption)
            found.append(path)
            if len(found) > MAX_IMAGES:
                raise ValueError(f"Dataset exceeds {MAX_IMAGES} images")
    if not found:
        raise ValueError("Dataset contains no supported images")
    return found


def read_caption(image: Path) -> str:
    path = image.with_suffix(".txt")
    if path.is_symlink():
        raise ValueError("Caption must not be a symlink")
    if not path.exists():
        return ""
    if path.stat().st_size > 65536:
        raise ValueError(f"Caption exceeds 64 KiB: {path.name}")
    return path.read_text(encoding="utf-8").strip()


def settings(root: Path):
    config_file = root / "training" / "local.json"
    config = json.loads(config_file.read_text()) if config_file.is_file() else {}
    backend = Path(os.getenv("ED_TRAINER_BACKEND") or config.get("backend_dir")
                   or root / "scripts" / "sd-scripts").expanduser().resolve()
    # Preserve the interpreter symlink: resolving it would leave the venv.
    python = Path(os.getenv("ED_TRAINER_PYTHON") or config.get("python")
                  or root / "training" / "runtime" / ".venv" / "bin" / "python").absolute()
    return backend, python


def child_env():
    env = os.environ.copy()
    # PyInstaller's loader path must not leak into system Python/uv children.
    if getattr(sys, "frozen", False):
        if "LD_LIBRARY_PATH_ORIG" in env:
            env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH_ORIG"]
        else:
            env.pop("LD_LIBRARY_PATH", None)
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def probe(root: Path):
    backend, python = settings(root)
    result = {"ready": False, "backend": str(backend), "python": str(python),
              "expected_revision": BACKEND_REVISION, "revision": None}
    if not all((backend / name).is_file() for name in SCRIPTS.values()):
        return {**result, "detail": "Set ED_TRAINER_BACKEND to the sd-scripts checkout"}
    try:
        result["revision"] = subprocess.check_output(
            ["git", "-C", str(backend), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=5, env=child_env()).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    if result["revision"] != BACKEND_REVISION:
        return {**result, "detail": f"Backend revision must be {BACKEND_REVISION}; checkout was not changed"}
    if not python.is_file():
        return {**result, "detail": "Training runtime is missing. Run python training/trainer.py setup"}
    code = (
        "import json, torch, accelerate, transformers, diffusers, safetensors; "
        "import train_network, sdxl_train_network, train_textual_inversion, sdxl_train_textual_inversion; "
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    try:
        checked = subprocess.run([str(python), "-c", code], cwd=backend, env=child_env(),
                                 capture_output=True, text=True, timeout=60)
        if checked.returncode:
            return {**result, "detail": (checked.stderr or checked.stdout)[-3000:]}
        result.update(json.loads(checked.stdout.strip().splitlines()[-1]))
        result["ready"] = result["cuda"]
        result["detail"] = "Training runtime ready" if result["ready"] else "A CUDA GPU is required by this runtime"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["detail"] = str(exc)
    return result


def validate_training(payload):
    spec = dict(payload)
    if (spec.get("kind"), spec.get("architecture")) not in SCRIPTS:
        raise ValueError("Choose LoRA or embedding and SD1.5 or SDXL")
    name = spec.get("output_name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Output name must use 1–80 letters, digits, underscores or hyphens")
    for key, default, low, high in (
        ("steps", 1000, 1, 100000), ("batch_size", 1, 1, 16),
        ("resolution", 512 if spec["architecture"] == "sd15" else 1024, 256, 1536),
        ("rank", 16, 1, 128), ("vectors", 4, 1, 16),
        ("save_every", 100, 1, 10000), ("seed", 42, 0, 2147483647),
    ):
        value = spec.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
        spec[key] = value
    if spec["resolution"] % 64:
        raise ValueError("Resolution must be a multiple of 64")
    rate = spec.get("learning_rate", 0.0001 if spec["kind"] == "lora" else 0.0005)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 0.1:
        raise ValueError("Learning rate must be positive and at most 0.1")
    spec["learning_rate"] = rate
    spec.setdefault("precision", "fp16")
    if spec["precision"] not in {"fp16", "bf16", "no"}:
        raise ValueError("Unsupported precision")
    trigger = spec.get("trigger", "").strip()
    if len(trigger) > 200 or any(c in trigger for c in "\n\r\x00"):
        raise ValueError("Trigger must be a short, single-line phrase")
    if spec["kind"] == "embedding" and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,79}", trigger):
        raise ValueError("Embedding token must be a unique word (2–80 letters, digits, _ or -)")
    spec["trigger"] = trigger
    spec.setdefault("init_word", "object")
    if not isinstance(spec["init_word"], str) or not 1 <= len(spec["init_word"]) <= 100:
        raise ValueError("Embedding initialization word is required")
    return spec


def build_command(spec, backend: Path, python: Path, job_dir: Path):
    spec = validate_training(spec)
    command = [str(python), "-m", "accelerate.commands.launch", "--num_processes=1",
               "--num_machines=1", "--num_cpu_threads_per_process=2",
               f"--mixed_precision={spec['precision']}", "--dynamo_backend=no",
               str(backend / SCRIPTS[(spec["kind"], spec["architecture"])])]
    args = {
        "pretrained_model_name_or_path": spec["model"],
        "dataset_config": job_dir / "dataset.toml", "output_dir": job_dir / "output",
        "output_name": spec["output_name"], "save_model_as": "safetensors",
        "max_train_steps": spec["steps"], "save_every_n_steps": spec["save_every"],
        "learning_rate": spec["learning_rate"], "optimizer_type": "AdamW",
        "lr_scheduler": "constant", "mixed_precision": spec["precision"],
        "seed": spec["seed"], "max_data_loader_n_workers": 0,
    }
    if spec["kind"] == "lora":
        args.update(network_module="networks.lora", network_dim=spec["rank"],
                    network_alpha=spec["rank"], training_comment=f"Easy Diffusion; trigger: {spec['trigger']}")
        command += ["--network_train_unet_only", "--save_state", "--save_state_on_train_end"]
    else:
        args.update(token_string=spec["trigger"], init_word=spec["init_word"], num_vectors_per_token=spec["vectors"])
    if spec.get("resume_state"):
        args["resume"] = spec["resume_state"]
    command += [f"--{key}={value}" for key, value in args.items()]
    command += ["--sdpa", "--gradient_checkpointing", "--cache_latents"]
    if spec["architecture"] == "sdxl":
        command += ["--no_half_vae"]
    return command


def stage_dataset(spec, job_dir: Path, cancel: threading.Event):
    images = dataset_images(Path(spec["dataset"]))
    target = job_dir / "data"
    target.mkdir()
    for index, path in enumerate(images):
        if cancel.is_set():
            raise Cancelled()
        caption = read_caption(path)
        if spec["trigger"] and spec["trigger"] not in caption.split(", "):
            caption = ", ".join(filter(None, [spec["trigger"], caption]))
        if not caption:
            raise ValueError(f"Missing caption for {path.name}; autotag or add a trigger first")
        dest = target / f"{index:06d}{path.suffix.lower()}"
        shutil.copyfile(path, dest)
        dest.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
    config = (
        '[general]\ncaption_extension = ".txt"\nshuffle_caption = false\n'
        f'[[datasets]]\nresolution = {spec["resolution"]}\nbatch_size = {spec["batch_size"]}\n'
        'enable_bucket = true\nmin_bucket_reso = 256\nmax_bucket_reso = 1536\n'
        f'[[datasets.subsets]]\nimage_dir = {json.dumps(str(target), ensure_ascii=False)}\nnum_repeats = 1\n'
    )
    (job_dir / "dataset.toml").write_text(config, encoding="utf-8")
    emit("dataset", images=len(images))


def run_child(command, cwd, cancel, callback=emit):
    process = subprocess.Popen(command, cwd=cwd, env=child_env(), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               errors="replace", bufsize=1, start_new_session=True)
    lines = queue.Queue(maxsize=1024)

    def read_output():
        try:
            for line in process.stdout:
                lines.put(line.rstrip()[:8192])
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    callback("started", pid=process.pid)
    try:
        ended = False
        while not ended:
            if cancel.is_set():
                raise Cancelled()
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                ended = True
            elif line:
                callback("log", message=line)
                match = re.search(r"\b(\d+)/(\d+)\s*\[", line)
                if match:
                    callback("progress", step=int(match[1]), total=int(match[2]))
        code = process.wait()
        if code:
            raise RuntimeError(f"Backend exited with status {code}")
    finally:
        # Kill the entire backend group, including dataloader/accelerate workers.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()


def autotag(spec, cancel):
    # Only local Easy Diffusion is contacted; arbitrary URLs are not accepted.
    port = int(spec.get("port", 9000))
    if not 1 <= port <= 65535:
        raise ValueError("Invalid local server port")
    images = dataset_images(Path(spec["dataset"]))
    written = skipped = 0
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for index, path in enumerate(images):
        if cancel.is_set():
            raise Cancelled()
        caption_file = path.with_suffix(".txt")
        if caption_file.exists():
            skipped += 1
        else:
            payload = {"image": base64.b64encode(path.read_bytes()).decode("ascii"),
                       "model": spec.get("tagger", "wd-v1-4-moat-tagger-v2"),
                       "threshold": spec.get("threshold", 0.35),
                       "character_threshold": spec.get("character_threshold", 0.85),
                       "exclude_tags": spec.get("exclude_tags", "")}
            request = urllib.request.Request(f"http://127.0.0.1:{port}/tag",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with opener.open(request, timeout=60) as response:
                tags = json.load(response)["tags"]
            if cancel.is_set():
                raise Cancelled()
            caption = ", ".join(filter(None, [spec.get("trigger", "").strip(), tags]))
            if not caption:
                raise ValueError(f"No tags found for {path.name}; lower the threshold")
            # Exclusive creation preserves existing and concurrently edited captions.
            try:
                with caption_file.open("x", encoding="utf-8") as output:
                    output.write(caption + "\n")
                written += 1
            except FileExistsError:
                skipped += 1
        emit("progress", step=index + 1, total=len(images), written=written, skipped=skipped)
    return {"written": written, "skipped": skipped}


def train(root, payload, cancel):
    spec = validate_training(payload)
    backend, python = settings(root)
    readiness = probe(root)
    if not readiness["ready"]:
        raise ValueError(readiness["detail"])
    job_dir = Path(spec["job_dir"])
    stage_dataset(spec, job_dir, cancel)
    (job_dir / "output").mkdir()
    command = build_command(spec, backend, python, job_dir)
    (job_dir / "manifest.json").write_text(json.dumps({"spec": spec, "runtime": readiness, "command": command}, indent=2))
    run_child(command, backend, cancel)
    output = job_dir / "output" / f"{spec['output_name']}.safetensors"
    if not output.is_file() or not output.stat().st_size:
        raise RuntimeError("Backend finished without producing the expected safetensors output")
    return {"output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["probe", "setup", "build", "run"], default="run")
    default_root = Path(sys.executable).resolve().parents[3] if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    parser.add_argument("--root", type=Path, default=default_root)
    args = parser.parse_args()
    root = args.root.resolve()
    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    try:
        if args.action == "probe":
            emit("result", **probe(root))
            return 0
        if args.action in {"setup", "build"}:
            uv = shutil.which("uv")
            if not uv:
                raise ValueError("Install uv first: https://docs.astral.sh/uv/")
            project = root / "training"
            if args.action == "setup":
                command = [uv, "sync", "--locked", "--project", str(project / "runtime")]
            else:
                command = [uv, "run", "--locked", "--project", str(project), "--group", "build",
                           "pyinstaller", "--noconfirm", "--distpath", str(project / "dist"),
                           "--workpath", str(project / "build"), str(project / "trainer.spec")]
            run_child(command, root, cancel)
            emit("completed")
            return 0
        request = json.loads(sys.stdin.readline(1024 * 1024))

        def control():
            for line in sys.stdin:
                try:
                    if json.loads(line).get("command") == "cancel":
                        cancel.set()
                except (ValueError, AttributeError):
                    pass
            cancel.set()  # owning server disappeared

        threading.Thread(target=control, daemon=True).start()
        command = request.get("command")
        if command == "autotag":
            result = autotag(request, cancel)
        elif command in {"train-lora", "train-embedding", "resume"}:
            result = train(root, request, cancel)
        else:
            raise ValueError("Unknown trainer command")
        if cancel.is_set():
            raise Cancelled()
        emit("completed", **result)
        return 0
    except Cancelled:
        emit("cancelled")
        return 130
    except Exception as exc:
        emit("failed", error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
