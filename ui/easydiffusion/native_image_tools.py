"""Bounded wrappers for the short-lived native vision and inpaint helpers."""

import base64
import binascii
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from easydiffusion import app


_PROCESS_LOCK = threading.Lock()
_MAX_IMAGE_BYTES = 32 * 1024 * 1024


class NativeDetectionRequest(BaseModel):
    image: str
    model: str = "objects"
    confidence: float = 0.25
    iou: float = 0.45
    max_results: int = 300


class TextMaskRequest(BaseModel):
    image: str
    sensitivity: float = 0.55
    padding: int = 3


def _decode_image(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("image must be a non-empty base64 string")
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    if len(encoded) > (_MAX_IMAGE_BYTES * 4 // 3 + 16):
        raise ValueError("image is larger than the 32 MiB tool limit")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image is not valid base64") from exc
    if not decoded or len(decoded) > _MAX_IMAGE_BYTES:
        raise ValueError("image is empty or larger than the 32 MiB tool limit")
    return decoded


def _backend_tool(name: str, override: str) -> Path:
    if os.getenv(override):
        candidates = [Path(os.environ[override])]
    else:
        backend_dir = None
        try:
            from easydiffusion.backends.sdkit3 import get_backend_dir

            backend_dir = Path(get_backend_dir())
        except Exception:
            pass
        sdkit_root = Path(app.ROOT_DIR) / "backends" / "sdkit3"
        candidates = ([backend_dir / name] if backend_dir else [])
        candidates.extend(sorted(sdkit_root.glob(f"*/{name}")))
        candidates.append(Path(app.ROOT_DIR) / "backends" / "tools" / name)
    for candidate in candidates:
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate
    raise FileNotFoundError(f"Native helper is not installed: {name}")


def _detector_model(kind: str) -> Path:
    filenames = {
        "objects": "yolov8s.torchscript",
        "face": "face_yolov8n.torchscript",
    }
    if kind not in filenames:
        raise ValueError("model must be 'objects' or 'face'")
    filename = filenames[kind]
    roots = []
    if os.getenv("SDKIT_VISION_MODELS"):
        roots.append(Path(os.environ["SDKIT_VISION_MODELS"]))
    roots.extend(
        [
            Path(app.MODELS_DIR) / "Ultralytics" / "cpp",
            Path(app.MODELS_DIR) / "ultralytics-cpp",
            Path(app.ROOT_DIR) / "models" / "ultralytics-cpp",
        ]
    )
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"TorchScript detector is not installed: {filename}")


def _run(command, timeout: int) -> Dict[str, Any]:
    # Serializing helpers prevents two transient LibTorch processes from
    # doubling RAM use on machines already close to their OOM limit.
    with _PROCESS_LOCK:
        completed = subprocess.run(
            [str(value) for value in command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "native helper failed"
        raise RuntimeError(detail[-2000:])
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("native helper returned invalid JSON") from exc


def detect(request: NativeDetectionRequest) -> Dict[str, Any]:
    if not 0 <= request.confidence <= 1 or not 0 <= request.iou <= 1:
        raise ValueError("confidence and iou must be in [0,1]")
    if not 1 <= request.max_results <= 1000:
        raise ValueError("max_results must be in [1,1000]")
    image = _decode_image(request.image)
    binary = _backend_tool("sdkit-vision", "SDKIT_VISION_BINARY")
    model = _detector_model(request.model)
    with tempfile.NamedTemporaryFile(suffix=".img") as source:
        source.write(image)
        source.flush()
        return _run(
            [
                binary,
                "--model", model,
                "--image", source.name,
                "--conf", request.confidence,
                "--iou", request.iou,
                "--max", request.max_results,
                "--threads", min(4, os.cpu_count() or 1),
            ],
            timeout=120,
        )


def text_mask(request: TextMaskRequest) -> Dict[str, Any]:
    if not 0 <= request.sensitivity <= 1:
        raise ValueError("sensitivity must be in [0,1]")
    if not 0 <= request.padding <= 64:
        raise ValueError("padding must be in [0,64]")
    image = _decode_image(request.image)
    binary = _backend_tool("sdkit-image-tools", "SDKIT_IMAGE_TOOLS_BINARY")
    with tempfile.NamedTemporaryFile(suffix=".img") as source, tempfile.NamedTemporaryFile(suffix=".png") as output:
        source.write(image)
        source.flush()
        result = _run(
            [
                binary,
                "text-mask",
                "--image", source.name,
                "--output", output.name,
                "--sensitivity", request.sensitivity,
                "--padding", request.padding,
            ],
            timeout=30,
        )
        output.seek(0)
        result["mask"] = "data:image/png;base64," + base64.b64encode(output.read()).decode("ascii")
        return result
