"""Load Sprite-GPT manifests and run the isolated native 64x64 generator."""

import base64
from io import BytesIO
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from threading import Event

from PIL import Image

MODEL_EXTENSION = ".sprite-gpt.json"
MODEL_FORMAT = "sdkit-sprite-gpt-torchscript-v1"


def is_sprite_model(path):
    return isinstance(path, (str, Path)) and str(path).lower().endswith(MODEL_EXTENSION)


def load_manifest(path):
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != MODEL_FORMAT or data.get("image_size") != 64:
        raise ValueError("Expected a 64x64 Sprite-GPT model manifest")
    bundle = {}
    for key in ("unet", "clip", "tokenizer"):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Sprite-GPT manifest is missing {key}")
        bundle[key] = str((manifest_path.parent / value).resolve())
    for path in (Path(bundle["unet"]), Path(bundle["clip"]),
                 Path(bundle["tokenizer"]) / "vocab.json", Path(bundle["tokenizer"]) / "merges.txt"):
        if not path.is_file():
            raise FileNotFoundError(f"Sprite-GPT model artifact is missing: {path}")
    return bundle


def validate_request(request, model_paths):
    unsupported = [key for key in (
        "init_image", "init_image_mask", "ref_images", "control_image",
        "control_net_lllite_image", "control_net_lllite_model", "ip_adapter_image",
        "ip_adapter_model", "latent_interposer_enabled", "latent_interposer_encode_enabled",
        "latent_interposer_decode_enabled", "tiling",
    ) if request.get(key)]
    unsupported.extend(key for key in ("lora", "hypernetwork", "embeddings", "controlnet")
                       if model_paths.get(key))
    if unsupported:
        raise ValueError("Sprite-GPT supports 64x64 text-to-image only; disable: " + ", ".join(unsupported))
    if not str(request.get("prompt", "")).strip():
        raise ValueError("Sprite-GPT requires a prompt")
    if not 1 <= int(request.get("num_inference_steps", 50)) <= 1000:
        raise ValueError("Sprite-GPT steps must be between 1 and 1000")
    if not 1 <= int(request.get("num_outputs", 1)) <= 100:
        raise ValueError("Sprite-GPT batch size must be between 1 and 100")
    guidance = float(request.get("guidance_scale", 5))
    if not math.isfinite(guidance) or not 0 <= guidance <= 100:
        raise ValueError("Sprite-GPT guidance must be finite and between 0 and 100")


def prepare_request(request, models_data, task_data):
    if not is_sprite_model(models_data.model_paths.get("stable-diffusion")):
        return
    validate_request(request.dict(), models_data.model_paths)
    request.width = request.height = 64
    request.sampler_name = "euler"
    request.scheduler_name = "simple"
    # Sprite-GPT has its own CLIP encoder and generates pixels without a VAE.
    for key in ("vae", "text-encoder"):
        models_data.model_paths[key] = None
    task_data.use_vae_model = task_data.use_text_encoder_model = None
    task_data.clip_skip = task_data.enable_vae_tiling = False


def select_device(context, assignment=""):
    selected = str(getattr(context, "device", None) or getattr(context, "torch_device", "cpu"))
    for item in str(assignment or "").split(","):
        key, separator, value = item.strip().partition("=")
        if separator and key in ("diffusion", "all"):
            selected = value
        elif not separator and key:
            selected = key
    normalized = selected.lower()
    if normalized in ("cpu", "cpu:0", "cpu:none"):
        return "cpu"
    match = re.fullmatch(r"cuda(?::)?(\d*)", normalized)
    if match:
        return "cuda:" + (match.group(1) or "0")
    raise ValueError(f"Sprite-GPT supports CPU or CUDA; selected device: {selected}")


def load_model(context, manifest, binary):
    context.sprite_gpt_model = None
    context.models.pop("stable-diffusion", None)
    bundle = load_manifest(manifest)
    if not Path(binary).is_file():
        raise FileNotFoundError("Sprite-GPT native generator is not installed; build and deploy sdkit-sprite-gpt")
    bundle["binary"] = str(binary)
    context.sprite_gpt_model = bundle
    context.models["stable-diffusion"] = {"format": MODEL_FORMAT}


def unload_model(context):
    context.sprite_gpt_model = None
    context.models.pop("stable-diffusion", None)


def stop_rendering(context):
    event = getattr(context, "sprite_gpt_cancel", None)
    if event is not None:
        event.set()


def generate_images(context, callback=None, output_type="pil", **request):
    validate_request(request, context.model_paths)
    bundle = context.sprite_gpt_model
    if not bundle:
        raise RuntimeError("Sprite-GPT model is not loaded")
    device = select_device(context, request.get("backend_assignment", ""))
    steps = int(request.get("num_inference_steps", 50))
    count = int(request.get("num_outputs", 1))
    seed = int(request.get("seed", 42))
    if seed < 0 or seed + count - 1 > 2**63 - 1:
        raise ValueError("Sprite-GPT seeds must be nonnegative signed 64-bit integers")
    context.sprite_gpt_cancel = Event()
    images = []
    try:
        with tempfile.TemporaryDirectory(prefix="ed-sprite-gpt-") as directory:
            directory = Path(directory)
            for index in range(count):
                if context.sprite_gpt_cancel.is_set():
                    break
                output = directory / f"{index}.png"
                command = [bundle["binary"], "--unet", bundle["unet"], "--clip", bundle["clip"],
                           "--tokenizer", bundle["tokenizer"], "--prompt", request["prompt"],
                           "--negative-prompt", str(request.get("negative_prompt") or ""),
                           "--output", str(output), "--steps", str(steps),
                           "--guidance", str(request.get("guidance_scale", 5)),
                           "--seed", str(seed + index), "--device", device, "--progress"]
                # Files avoid pipe deadlocks, including on verbose LibTorch failures.
                with (directory / "stdout").open("w+") as stdout, (directory / "stderr").open("w+") as stderr:
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                    try:
                        with (directory / "stdout").open() as progress:
                            step = 0
                            while process.poll() is None:
                                for line in progress:
                                    try:
                                        event = json.loads(line)
                                    except ValueError:
                                        continue
                                    if event.get("type") == "progress":
                                        step = max(0, min(steps - 1, int(event["step"]) - 1))
                                if callback:
                                    callback(None, int((index * steps + step) / count))
                                if context.sprite_gpt_cancel.wait(0.25):
                                    break
                        if context.sprite_gpt_cancel.is_set():
                            break
                        if process.returncode:
                            stderr.seek(0)
                            detail = stderr.read()[-4000:]
                            raise RuntimeError(f"Sprite-GPT generation failed ({process.returncode}): {detail}")
                        with Image.open(output) as image:
                            if image.size != (64, 64):
                                raise RuntimeError("Sprite-GPT returned an unexpected image size")
                            images.append(image.convert("RGB"))
                        if callback:
                            callback(list(images), max(0, int((index + 1) * steps / count) - 1))
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
    finally:
        context.sprite_gpt_cancel = None
    if output_type == "pil":
        return images
    if output_type != "base64":
        raise ValueError(f"Unsupported Sprite-GPT output type: {output_type}")
    options = getattr(context, "sprite_gpt_options", {})
    output_format = str(options.get("output_format", "png")).lower()
    if output_format not in ("png", "jpeg", "webp"):
        output_format = "png"
    encoded = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format=output_format.upper(), quality=int(options.get("output_quality", 75)),
                   lossless=bool(options.get("output_lossless", False)))
        encoded.append(f"data:image/{output_format};base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return encoded
