"""Bounded wrappers for the short-lived native vision and inpaint helpers."""

import base64
import binascii
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageFilter
from pydantic import BaseModel

from easydiffusion import app


_PROCESS_LOCK = threading.Lock()
_BACKGROUND_REMOVAL_SESSION = None
_OBJECT_REMOVAL_SESSION = None
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_BACKGROUND_MODELS = {"u2net": "u2net.onnx"}


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


class BackgroundRemovalRequest(BaseModel):
    image: str
    model: str = "u2net"
    alpha_matting: bool = False


class ObjectRemovalRequest(BaseModel):
    image: str
    mask: str
    feather: int = 4


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


def _png_data_url(value: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(value).decode("ascii")


def _source_model_directory(name: str) -> Path:
    return Path(app.ROOT_DIR) / "source" / "models" / name


def _background_removal_model(model: str) -> Path:
    if model not in _BACKGROUND_MODELS:
        raise ValueError(f"unsupported background-removal model: {model}")
    explicit = os.getenv("SDKIT_BACKGROUND_REMOVAL_MODEL")
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend(
        [
            _source_model_directory("remove_background_onnx") / _BACKGROUND_MODELS[model],
            Path(app.MODELS_DIR) / "background-removal" / _BACKGROUND_MODELS[model],
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{_BACKGROUND_MODELS[model]} is not installed in "
        f"{_source_model_directory('remove_background_onnx')}"
    )


def _background_removal_session(model: str):
    global _BACKGROUND_REMOVAL_SESSION
    with _PROCESS_LOCK:
        if _BACKGROUND_REMOVAL_SESSION is not None:
            return _BACKGROUND_REMOVAL_SESSION
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        _BACKGROUND_REMOVAL_SESSION = ort.InferenceSession(
            str(_background_removal_model(model)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return _BACKGROUND_REMOVAL_SESSION


def remove_background(request: BackgroundRemovalRequest) -> Dict[str, Any]:
    image_bytes = _decode_image(request.image)
    with Image.open(BytesIO(image_bytes)) as opened:
        original = opened.convert("RGB")
    original_size = original.size

    import numpy as np

    resized = original.resize((320, 320), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32)
    pixels /= max(float(np.max(pixels)), 1e-6)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    tensor = ((pixels - mean) / std).transpose((2, 0, 1))[None, ...].astype(np.float32)

    session = _background_removal_session(request.model)
    input_name = session.get_inputs()[0].name
    with _PROCESS_LOCK:
        prediction = np.asarray(session.run(None, {input_name: tensor})[0])
    if prediction.ndim != 4 or prediction.shape[0] != 1:
        raise RuntimeError(f"U2Net returned unexpected shape {prediction.shape}")
    mask_array = prediction[0, 0]
    minimum = float(np.min(mask_array))
    maximum = float(np.max(mask_array))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise RuntimeError("U2Net returned non-finite mask values")
    if maximum > minimum:
        mask_array = (mask_array - minimum) / (maximum - minimum)
    else:
        mask_array = np.zeros_like(mask_array)
    mask = Image.fromarray((np.clip(mask_array, 0, 1) * 255).astype(np.uint8), mode="L")
    mask = mask.resize(original_size, Image.Resampling.LANCZOS)
    if request.alpha_matting:
        mask = mask.filter(ImageFilter.GaussianBlur(1.0))
    result = original.convert("RGBA")
    result.putalpha(mask)
    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return {
        "image": _png_data_url(buffer.getvalue()),
        "width": original_size[0],
        "height": original_size[1],
        "model": request.model,
    }


def _object_removal_model() -> Path:
    candidates = []
    if os.getenv("DEEP_OBJECT_REMOVAL_MODEL"):
        candidates.append(Path(os.environ["DEEP_OBJECT_REMOVAL_MODEL"]))
    candidates.extend(
        [
            _source_model_directory("remove_object_onnx") / "deep-object-removal-400.onnx",
            Path(app.MODELS_DIR) / "deep-object-removal" / "deep-object-removal-400.onnx",
            Path(app.ROOT_DIR) / "models" / "deep-object-removal" / "deep-object-removal-400.onnx",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Deep Object Removal ONNX model is not installed in "
        f"{_source_model_directory('remove_object_onnx')}"
    )


def _object_removal_session():
    global _OBJECT_REMOVAL_SESSION
    with _PROCESS_LOCK:
        if _OBJECT_REMOVAL_SESSION is not None:
            return _OBJECT_REMOVAL_SESSION
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        _OBJECT_REMOVAL_SESSION = ort.InferenceSession(
            str(_object_removal_model()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return _OBJECT_REMOVAL_SESSION


def _editable_mask(value: bytes, size) -> Image.Image:
    with Image.open(BytesIO(value)) as opened:
        rgba = opened.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
    import numpy as np

    pixels = np.asarray(rgba, dtype=np.uint8)
    alpha = pixels[:, :, 3]
    # The editor normally stores white strokes on transparent pixels. Imported
    # black/white masks are commonly fully opaque, in which case luminance is
    # the intended selection channel instead.
    if alpha.min() < 255:
        selected = alpha
    else:
        selected = np.max(pixels[:, :, :3], axis=2)
    return Image.fromarray(selected, mode="L")


def remove_object(request: ObjectRemovalRequest) -> Dict[str, Any]:
    if not 0 <= request.feather <= 32:
        raise ValueError("feather must be in [0,32]")
    image_bytes = _decode_image(request.image)
    mask_bytes = _decode_image(request.mask)
    with Image.open(BytesIO(image_bytes)) as opened:
        original = opened.convert("RGB")
    original_size = original.size
    mask = _editable_mask(mask_bytes, original_size)
    if mask.getbbox() is None:
        raise ValueError("object-removal mask is empty")

    import numpy as np

    network_image = original.resize((400, 400), Image.Resampling.LANCZOS)
    network_mask = mask.resize((400, 400), Image.Resampling.LANCZOS)
    image_array = np.asarray(network_image, dtype=np.float32) / 255.0
    mask_array = np.asarray(network_mask, dtype=np.float32) / 255.0
    masked = image_array * (1.0 - mask_array[:, :, None])

    session = _object_removal_session()
    input_name = session.get_inputs()[0].name
    with _PROCESS_LOCK:
        reconstructed = session.run(None, {input_name: masked[None, ...]})[0]
    reconstructed = np.asarray(reconstructed)[0]
    if reconstructed.shape != (400, 400, 3):
        raise RuntimeError(f"Deep Object Removal returned unexpected shape {reconstructed.shape}")
    reconstructed = np.clip(reconstructed * 255.0, 0, 255).astype(np.uint8)
    replacement = Image.fromarray(reconstructed, mode="RGB").resize(original_size, Image.Resampling.LANCZOS)

    blend_mask = mask
    if request.feather:
        kernel = request.feather * 2 + 1
        blend_mask = blend_mask.filter(ImageFilter.MaxFilter(kernel)).filter(
            ImageFilter.GaussianBlur(request.feather)
        )
    result = Image.composite(replacement, original, blend_mask)
    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return {
        "image": _png_data_url(buffer.getvalue()),
        "width": original_size[0],
        "height": original_size[1],
        "model": "VPanjeta/Deep-Object-Removal",
    }


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


def _torch_library_directory() -> Path:
    library_names = ("libtorch.so", "libtorch.dylib", "torch.dll")
    python_paths = sysconfig.get_paths()
    candidates = []
    for path_name in ("platlib", "purelib"):
        package_root = python_paths.get(path_name)
        if package_root:
            candidate = Path(package_root) / "torch" / "lib"
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if any((candidate / library_name).is_file() for library_name in library_names):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates) or "the active Python environment"
    raise RuntimeError(f"LibTorch runtime libraries are missing from {searched}")


def _runtime_environment(command) -> Dict[str, str]:
    environment = os.environ.copy()
    if not command or Path(str(command[0])).name != "sdkit-vision":
        return environment

    torch_library_directory = str(_torch_library_directory())
    if os.name == "nt":
        library_path_name = "PATH"
    elif sys.platform == "darwin":
        library_path_name = "DYLD_LIBRARY_PATH"
    else:
        library_path_name = "LD_LIBRARY_PATH"
    current = environment.get(library_path_name, "")
    environment[library_path_name] = os.pathsep.join(
        value for value in (torch_library_directory, current) if value
    )
    return environment


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
            env=_runtime_environment(command),
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
