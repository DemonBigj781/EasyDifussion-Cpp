"""Built-in WD14 image tagging support for Easy Diffusion."""

import base64
import binascii
import csv
import io
import os
import threading

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field


DEFAULT_MODEL = "wd-v1-4-moat-tagger-v2"
MAX_IMAGE_BYTES = 64 * 1024 * 1024

_cache_lock = threading.Lock()
_session_cache = {}
_labels_cache = {}


class WD14TagRequest(BaseModel):
    image: str
    model: str = DEFAULT_MODEL
    threshold: float = Field(0.35, ge=0.0, le=1.0)
    character_threshold: float = Field(0.85, ge=0.0, le=1.0)
    exclude_tags: str = ""
    replace_underscore: bool = False
    trailing_comma: bool = False


def _decode_image(encoded: str) -> Image.Image:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image is required")
    if encoded.startswith("data:"):
        marker = encoded.find(",")
        if marker < 0:
            raise ValueError("invalid image data URL")
        encoded = encoded[marker + 1 :]
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3) + 16:
        raise ValueError("image payload is too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image is not valid base64") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("decoded image is too large")
    with Image.open(io.BytesIO(raw)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def _resolve_files(model_name: str):
    from easydiffusion.model_manager import resolve_model_to_use

    model_name = os.path.splitext(os.path.basename(model_name))[0]
    model_path = resolve_model_to_use(model_name, "wd14-tagger")
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"WD14 model not found: {model_name}")
    csv_path = os.path.splitext(model_path)[0] + ".csv"
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"WD14 tag CSV not found: {csv_path}")
    return model_name, model_path, csv_path


def _providers(ort):
    available = set(ort.get_available_providers())
    selected = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
    return selected or list(available)


def _get_session(model_path: str):
    import onnxruntime as ort

    with _cache_lock:
        session = _session_cache.get(model_path)
        if session is None:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            session = ort.InferenceSession(model_path, sess_options=options, providers=_providers(ort))
            _session_cache[model_path] = session
        return session


def _get_labels(csv_path: str):
    with _cache_lock:
        cached = _labels_cache.get(csv_path)
        if cached is not None:
            return cached
        labels = []
        categories = []
        with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not {"name", "category"}.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"WD14 CSV is missing required columns: {csv_path}")
            for row in reader:
                labels.append(row["name"])
                categories.append(int(row["category"]))
        cached = (labels, categories)
        _labels_cache[csv_path] = cached
        return cached


def _prepare_image(image: Image.Image, size: int) -> np.ndarray:
    ratio = float(size) / max(image.size)
    resized_size = tuple(max(1, int(axis * ratio)) for axis in image.size)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(resized, ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2))
    pixels = np.asarray(square, dtype=np.float32)
    return np.expand_dims(pixels[:, :, ::-1], axis=0)


def _format_tag(tag: str, replace_underscore: bool) -> str:
    if replace_underscore:
        tag = tag.replace("_", " ")
    return tag.replace("(", "\\(").replace(")", "\\)")


def tag_image(request: WD14TagRequest):
    model_name, model_path, csv_path = _resolve_files(request.model)
    session = _get_session(model_path)
    labels, categories = _get_labels(csv_path)
    input_info = session.get_inputs()[0]
    size = input_info.shape[1]
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"WD14 model has unsupported input shape: {input_info.shape}")
    probabilities = session.run(
        [session.get_outputs()[0].name],
        {input_info.name: _prepare_image(_decode_image(request.image), size)},
    )[0][0]
    if len(probabilities) != len(labels):
        raise ValueError(f"WD14 model/CSV mismatch: model returned {len(probabilities)} scores, CSV has {len(labels)} tags")

    excluded = {tag.strip().lower() for tag in request.exclude_tags.split(",") if tag.strip()}
    ratings = []
    general = []
    characters = []
    for label, category, score_value in zip(labels, categories, probabilities):
        score = float(score_value)
        rendered = _format_tag(label, request.replace_underscore)
        item = {"tag": rendered, "raw_tag": label, "score": score}
        if category == 9:
            ratings.append(item)
        elif category == 4 and score > request.character_threshold and label.lower() not in excluded and rendered.lower() not in excluded:
            characters.append(item)
        elif category == 0 and score > request.threshold and label.lower() not in excluded and rendered.lower() not in excluded:
            general.append(item)

    matches = characters + general
    tags = "".join(item["tag"] + ", " for item in matches) if request.trailing_comma else ", ".join(item["tag"] for item in matches)
    ratings.sort(key=lambda item: item["score"], reverse=True)
    return {"model": model_name, "tags": tags, "matches": matches, "ratings": ratings, "providers": session.get_providers()}
