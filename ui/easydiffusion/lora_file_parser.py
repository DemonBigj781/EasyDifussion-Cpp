"""Built-in LoRA safetensors metadata parser for Easy Diffusion."""

import json
import os
from pathlib import Path

from safetensors import safe_open


def lora_dir() -> Path:
    configured = os.environ.get("ED_LORA_DIR") or os.environ.get("LORA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    from easydiffusion.model_manager import get_model_dirs

    directories = [Path(path).resolve() for path in get_model_dirs("lora")]
    return next((path for path in directories if path.is_dir()), directories[0])


def resolve_lora_path(filepath: str) -> str:
    if not filepath:
        raise ValueError("No filepath provided")
    base = lora_dir()
    path = Path(filepath)
    path = (path if path.is_absolute() else base / path).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"LoRA path must be under {base}") from exc
    if not path.exists() and path.suffix.lower() != ".safetensors":
        candidate = Path(f"{path}.safetensors")
        if candidate.exists():
            path = candidate
    return str(path)


def list_lora_files():
    return sorted(str(path.resolve()) for path in lora_dir().rglob("*.safetensors") if path.is_file())


def list_checkpoint_files():
    from easydiffusion.model_manager import MODEL_EXTENSIONS, get_model_dirs

    extensions = {extension.lower() for extension in MODEL_EXTENSIONS["stable-diffusion"]}
    files = {
        str(path.resolve())
        for directory in get_model_dirs("stable-diffusion")
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }
    return sorted(files)


def extract_checkpoint_metadata(filepath):
    from easydiffusion.model_manager import get_model_dirs

    path = Path(filepath).resolve()
    roots = [Path(directory).resolve() for directory in get_model_dirs("stable-diffusion")]
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError("Checkpoint path must be under an Easy Diffusion checkpoint directory")
    result = {"model_name": path.name, "model": path.stem, "meta": {}}
    if path.suffix.lower() == ".safetensors":
        try:
            with safe_open(str(path), framework="numpy") as file:
                result["meta"] = file.metadata() or {}
        except Exception as exc:
            result["error"] = str(exc)
    return result


def scan_checkpoint_metadata():
    return [extract_checkpoint_metadata(filepath) for filepath in list_checkpoint_files()]


def _collect_tag_frequencies(value, frequencies):
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if isinstance(item, dict):
            _collect_tag_frequencies(item, frequencies)
            continue
        try:
            count = float(item)
        except (TypeError, ValueError):
            continue
        tag = str(key).strip()
        if tag:
            frequencies[tag] = frequencies.get(tag, 0) + count


def _metadata_words(value):
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except Exception:
            decoded = None
        return _metadata_words(decoded) if decoded is not None and decoded != value else [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [word for item in value for word in _metadata_words(item)]
    return [str(value).strip()]


def extract_lora_metadata(filepath, include_metadata=False):
    filepath = resolve_lora_path(filepath)
    result = {"model_name": os.path.basename(filepath), "model": None, "trigger_words": []}
    if not os.path.exists(filepath):
        result["error"] = f"File not found: {filepath}"
        return result
    try:
        relative = Path(filepath).resolve().relative_to(lora_dir())
        result["model"] = relative.with_suffix("").as_posix() if relative.suffix.lower() == ".safetensors" else relative.as_posix()
        with safe_open(filepath, framework="numpy") as file:
            metadata = file.metadata() or {}
        tag_frequency = metadata.get("ss_tag_frequency")
        if isinstance(tag_frequency, str):
            try:
                tag_frequency = json.loads(tag_frequency)
            except Exception:
                tag_frequency = {}
        frequencies = {}
        _collect_tag_frequencies(tag_frequency, frequencies)
        frequency_words = sorted(frequencies, key=lambda tag: (-frequencies[tag], tag.casefold()))
        explicit_words = []
        for key in ("modelspec.trigger_phrase", "ss_trigger_words", "trigger_words"):
            explicit_words.extend(_metadata_words(metadata.get(key)))
        seen = set()
        for word in explicit_words + frequency_words:
            normalized = word.casefold()
            if normalized not in seen:
                seen.add(normalized)
                result["trigger_words"].append(word)
        if include_metadata:
            result["meta"] = metadata
    except Exception as exc:
        result["error"] = str(exc)
    return result


def scan_lora_metadata():
    return [extract_lora_metadata(filepath) for filepath in list_lora_files()]
