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
    ("lora", "anima"): "anima_train_network.py",
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


def normalize_anima_caption(caption: str) -> str:
    """Use Anima's tag spelling: underscores only remain in @artist tags."""
    tags = []
    for tag in caption.split(","):
        tag = tag.strip()
        if not tag.startswith("@"):
            tag = tag.replace("_", " ")
        if tag:
            tags.append(tag)
    return ", ".join(tags)


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
        "import train_network, sdxl_train_network, anima_train_network, train_textual_inversion, sdxl_train_textual_inversion; "
        "cuda=torch.cuda.is_available(); xpu=hasattr(torch,'xpu') and torch.xpu.is_available(); "
        "gpu=(torch.cuda.get_device_name(0) if cuda else (torch.xpu.get_device_name(0) if xpu else None)); "
        "print(json.dumps({'torch':torch.__version__,'cuda':cuda,'xpu':xpu,'gpu':gpu}))"
    )
    try:
        checked = subprocess.run([str(python), "-c", code], cwd=backend, env=child_env(),
                                 capture_output=True, text=True, timeout=60)
        if checked.returncode:
            return {**result, "detail": (checked.stderr or checked.stdout)[-3000:]}
        result.update(json.loads(checked.stdout.strip().splitlines()[-1]))
        result["ready"] = result["cuda"] or result["xpu"]
        result["detail"] = "Training runtime ready" if result["ready"] else "A CUDA or Intel XPU GPU is required by this runtime"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["detail"] = str(exc)
    return result


def native_epoch_options(payload, image_count):
    spec = dict(payload)
    repeats = spec.get("dataset_repeats", 1)
    epochs = spec.get("epochs")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 100000:
        raise ValueError("Dataset repeats must be an integer from 1 to 100000")
    if epochs is not None and (isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100000):
        raise ValueError("Epochs must be an integer from 1 to 100000")
    if not 1 <= image_count * repeats <= 100000:
        raise ValueError("Steps per epoch must be from 1 to 100000")
    spec["dataset_repeats"] = repeats
    spec["steps_per_epoch"] = image_count * repeats
    if epochs is not None:
        spec["steps"] = spec["steps_per_epoch"] * epochs
        if spec["steps"] > 100000:
            raise ValueError("Total training steps must not exceed 100000")
    return spec


def validate_training(payload):
    spec = dict(payload)
    if (spec.get("kind"), spec.get("architecture")) not in SCRIPTS:
        raise ValueError("Choose LoRA or embedding with SD1.5/SDXL, or LoRA with Anima")
    if spec["architecture"] == "anima" and spec["kind"] != "lora":
        raise ValueError("Anima currently supports LoRA training only")
    backend = spec.get("backend") or ("native" if spec["architecture"] == "sd15" else "python")
    if backend not in {"native", "python"}:
        raise ValueError("Choose the native C++ or legacy Python training backend")
    if backend == "native" and spec["architecture"] != "sd15":
        raise ValueError("Native C++ training supports SD1.5 only")
    spec["backend"] = backend
    name = spec.get("output_name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Output name must use 1–80 letters, digits, underscores or hyphens")
    for key, default, low, high in (
        ("steps", 1000, 1, 100000), ("batch_size", 1, 1, 16),
        ("resolution", 512 if spec["architecture"] in {"sd15", "anima"} else 1024, 256, 1536),
        ("rank", 16, 1, 128), ("vectors", 4, 1, 16),
        ("save_every", 100, 1, 10000), ("seed", 42, 0, 2147483647),
    ):
        value = spec.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
        spec[key] = value
    if spec["resolution"] % 64:
        raise ValueError("Resolution must be a multiple of 64")
    alpha = spec.get("network_alpha")
    if alpha is None:
        alpha = spec["rank"]
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or not 0 < alpha <= 128):
        raise ValueError("LoRA alpha must be finite, positive and at most 128")
    spec["network_alpha"] = alpha
    spec.setdefault("lr_scheduler", "constant")
    if spec["lr_scheduler"] not in {"constant", "cosine_with_restarts"}:
        raise ValueError("Unsupported learning rate scheduler")
    for key, default, low, high in (
        ("lr_warmup_steps", 0, 0, spec["steps"] - 1),
        ("lr_scheduler_num_cycles", 1, 1, spec["steps"]),
    ):
        value = spec.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
        spec[key] = value
    if spec["lr_scheduler_num_cycles"] > spec["steps"] - spec["lr_warmup_steps"]:
        raise ValueError("Scheduler cycles must not exceed post-warmup steps")
    if spec["lr_scheduler"] == "constant" and (spec["lr_warmup_steps"] or spec["lr_scheduler_num_cycles"] != 1):
        raise ValueError("Constant scheduling requires zero warmup and one cycle")
    if spec["architecture"] == "anima":
        if spec.get("checkpointing", "standard") not in {"off", "standard", "cpu_offload", "unsloth"}:
            raise ValueError("Unsupported Anima gradient checkpointing strategy")
        blocks = spec.get("blocks_to_swap", 0)
        if isinstance(blocks, bool) or not isinstance(blocks, int) or not 0 <= blocks <= 40:
            raise ValueError("Anima block swap must be an integer from 0 to 40")
        if blocks and spec.get("checkpointing") in {"cpu_offload", "unsloth"}:
            raise ValueError("Block swapping cannot be combined with CPU-offloaded checkpointing")
        if spec.get("optimizer_type", "AdamW") not in {"AdamW", "AdamW8bit"}:
            raise ValueError("Unsupported Anima optimizer")
    rate = spec.get("learning_rate", 0.0001 if spec["kind"] == "lora" else 0.0005)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 0.1:
        raise ValueError("Learning rate must be positive and at most 0.1")
    spec["learning_rate"] = rate
    text_rate = spec.get("text_encoder_learning_rate", 0.0)
    if (isinstance(text_rate, bool) or not isinstance(text_rate, (int, float))
            or not math.isfinite(text_rate) or not 0 <= text_rate <= 0.1):
        raise ValueError("Text encoder learning rate must be finite, non-negative and at most 0.1")
    if text_rate and (spec["kind"], spec["architecture"]) != ("lora", "sd15"):
        raise ValueError("Text encoder training is currently supported only for SD1.5 LoRA")
    spec["text_encoder_learning_rate"] = text_rate
    spec.setdefault("precision", "fp16")
    if spec["precision"] not in {"fp16", "bf16", "no"}:
        raise ValueError("Unsupported precision")
    trigger = spec.get("trigger", "").strip()
    if spec["architecture"] == "anima":
        if not trigger.startswith("@"):
            trigger = trigger.replace("_", " ")
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
        "learning_rate": spec["learning_rate"],
        "optimizer_type": spec.get("optimizer_type", "AdamW") if spec["architecture"] == "anima" else "AdamW",
        "lr_scheduler": spec["lr_scheduler"], "mixed_precision": spec["precision"],
        "seed": spec["seed"], "max_data_loader_n_workers": 0,
        "sample_every_n_epochs": 1, "sample_prompts": job_dir / "sample-prompts.json",
        "sample_sampler": "ddim",
    }
    if spec["lr_scheduler"] == "cosine_with_restarts":
        args.update(lr_warmup_steps=spec["lr_warmup_steps"],
                    lr_scheduler_num_cycles=spec["lr_scheduler_num_cycles"])
    if spec["kind"] == "lora":
        network_module = "networks.lora_anima" if spec["architecture"] == "anima" else "networks.lora"
        args.update(network_module=network_module, network_dim=spec["rank"],
                    network_alpha=spec["network_alpha"], training_comment=f"Easy Diffusion; trigger: {spec['trigger']}")
        if spec["text_encoder_learning_rate"]:
            args.update(unet_lr=spec["learning_rate"], text_encoder_lr=spec["text_encoder_learning_rate"])
        else:
            command.append("--network_train_unet_only")
        command += ["--save_state", "--save_state_on_train_end"]
    else:
        args.update(token_string=spec["trigger"], init_word=spec["init_word"], num_vectors_per_token=spec["vectors"])
    if spec.get("resume_state"):
        args["resume"] = spec["resume_state"]
    command += [f"--{key}={value}" for key, value in args.items()]
    command += ["--sdpa", "--cache_latents"]
    if spec["architecture"] != "anima":
        command.append("--gradient_checkpointing")
    elif spec.get("checkpointing", "standard") != "off":
        checkpointing = spec.get("checkpointing", "standard")
        command.append("--gradient_checkpointing")
        if checkpointing == "cpu_offload":
            command.append("--cpu_offload_checkpointing")
        elif checkpointing == "unsloth":
            command.append("--unsloth_offload_checkpointing")
    if spec["architecture"] == "anima":
        if not spec.get("qwen3") or not spec.get("vae"):
            raise ValueError("Anima training requires Qwen3 text encoder and Qwen-Image VAE paths")
        args_for_anima = [f"--qwen3={spec['qwen3']}", f"--vae={spec['vae']}"]
        # Insert Anima-specific flags before the script arguments' common optimization tail.
        command += args_for_anima
        if spec.get("cache_text_encoder_outputs", True):
            command.append("--cache_text_encoder_outputs")
        command.extend(["--vae_chunk_size=64", "--qwen_image_vae_2d"])
        if spec.get("vae_disable_cache"):
            command.append("--vae_disable_cache")
        blocks_to_swap = spec.get("blocks_to_swap", 0)
        if blocks_to_swap:
            command.append(f"--blocks_to_swap={blocks_to_swap}")
    if spec["architecture"] == "sdxl":
        command += ["--no_half_vae"]
    return command


def stage_dataset(spec, job_dir: Path, cancel: threading.Event):
    images = dataset_images(Path(spec["dataset"]))
    selected = spec.get("images")
    if selected is not None:
        if not isinstance(selected, list) or not selected or any(
            not isinstance(name, str) or Path(name).name != name for name in selected
        ):
            raise ValueError("Autotag image selection is invalid")
        selected_names = set(selected)
        images = [image for image in images if image.name in selected_names]
        if {image.name for image in images} != selected_names:
            raise ValueError("One or more selected images are missing from the autotag dataset")
    target = job_dir / "data"
    target.mkdir()
    for index, path in enumerate(images):
        if cancel.is_set():
            raise Cancelled()
        caption = read_caption(path)
        if spec.get("architecture") == "anima":
            caption = normalize_anima_caption(caption)
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
    first_caption = next(target.glob("*.txt")).read_text(encoding="utf-8").strip()
    (job_dir / "sample-prompts.json").write_text(json.dumps([{
        "prompt": first_caption, "negative_prompt": "", "seed": spec["seed"],
        "width": spec["resolution"], "height": spec["resolution"],
        "sample_steps": 25, "scale": 7.5,
    }], ensure_ascii=False), encoding="utf-8")
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
        replace_existing = bool(spec.get("replace_existing", False))
        if caption_file.exists() and not replace_existing:
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
            if replace_existing and caption_file.exists():
                job_name = Path(spec.get("job_dir", "tagging")).name
                backup = caption_file.with_name(f"{caption_file.name}.{job_name}.original")
                shutil.copy2(caption_file, backup)
                temporary = caption_file.with_name(f".{caption_file.name}.{job_name}.tmp")
                temporary.write_text(caption + "\n", encoding="utf-8")
                temporary.replace(caption_file)
                written += 1
            else:
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
