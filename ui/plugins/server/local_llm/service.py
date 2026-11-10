"""Server-persisted local LLM settings and prompt processing."""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict

from easydiffusion import app as easy_app
from easydiffusion.llama_server import LlamaServer, backend_readiness
from easydiffusion.utils import log as _logger
from fastapi import HTTPException
from pydantic import BaseModel


DEFAULTS = {
    "model_path": "",
    "device": "cpu",
    "gpu_layers": 0,
    "context_length": 4096,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.05,
}
SETTING_KEYS = frozenset(DEFAULTS)
SYSTEM_PROMPT = (
    "You are an expert image-generation prompt editor. Follow the requested editing instruction "
    "while preserving useful visual details, artist-intent qualifiers, and prompt syntax. Return "
    "only the final processed image prompt, with no analysis, preamble, quotation marks, or labels."
)


class ProcessRequest(BaseModel):
    prompt: str
    instruction: str = "Improve this image-generation prompt while preserving its intent."


_settings_lock = threading.RLock()
_model_lock = threading.RLock()
_current_model: Dict[str, Any] = {"key": None, "instance": None}


def _models_root() -> Path:
    config = easy_app.getConfig()
    configured = config.get("models_dir") or getattr(easy_app, "MODELS_DIR", None)
    if not configured:
        raise HTTPException(status_code=503, detail="models_dir is not configured.")
    return Path(configured).expanduser().resolve()


def _configured_settings() -> Dict[str, Any]:
    configured = easy_app.getConfig().get("local_llm") or {}
    if not isinstance(configured, dict):
        raise HTTPException(status_code=503, detail="local_llm configuration must be a mapping.")
    return {**DEFAULTS, **configured}


def _number(value: Any, name: str, minimum: float, maximum: float, integer: bool = False):
    expected = int if integer else float
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{name} must be a number.")
    try:
        converted = expected(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"{name} must be a number.") from error
    if not minimum <= converted <= maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between {minimum:g} and {maximum:g}.",
        )
    return converted


def _validate_model_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail="Choose a local LLM GGUF model in Settings.")
    root = _models_root()
    candidate = Path(value.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink():
        raise HTTPException(status_code=400, detail="Local LLM model symlinks are not allowed.")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise HTTPException(
            status_code=403,
            detail=f"Local LLM models must be inside {root}.",
        ) from error
    if path.suffix.lower() != ".gguf" or not path.is_file():
        raise HTTPException(status_code=400, detail=f"Local LLM model does not exist: {path}")
    return path


def _validate_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(raw) - SETTING_KEYS)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown local LLM settings: {', '.join(unknown)}.",
        )
    device = str(raw.get("device", DEFAULTS["device"])).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="Local LLM device must be cpu or cuda.")
    model_path = _validate_model_path(raw.get("model_path"))
    gpu_layers = _number(raw.get("gpu_layers"), "gpu_layers", 0, 999, integer=True)
    if device == "cpu":
        gpu_layers = 0
    return {
        "model_path": str(model_path),
        "device": device,
        "gpu_layers": gpu_layers,
        "context_length": _number(
            raw.get("context_length"), "context_length", 512, 131072, integer=True
        ),
        "max_tokens": _number(raw.get("max_tokens"), "max_tokens", 1, 4096, integer=True),
        "temperature": _number(raw.get("temperature"), "temperature", 0, 2),
        "top_p": _number(raw.get("top_p"), "top_p", 0, 1),
        "top_k": _number(raw.get("top_k"), "top_k", 0, 200, integer=True),
        "repeat_penalty": _number(
            raw.get("repeat_penalty"), "repeat_penalty", 0.5, 2
        ),
    }


def _available_models():
    root = _models_root()
    search_root = root / "gguf" / "text"
    if not search_root.is_dir():
        search_root = root
    models = []
    for path in search_root.rglob("*.gguf"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        models.append(
            {
                "path": str(path.resolve()),
                "relative_path": path.resolve().relative_to(root).as_posix(),
                "name": path.name,
                "size": stat.st_size,
            }
        )
    return sorted(models, key=lambda item: item["relative_path"].lower())


def _cuda_available() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def get_settings() -> Dict[str, Any]:
    raw = _configured_settings()
    error = None
    try:
        settings = _validate_settings(raw)
    except HTTPException as exc:
        settings = raw
        error = str(exc.detail)
    readiness = backend_readiness()
    return {
        **settings,
        "valid": error is None and readiness["ready"],
        "error": error,
        "backend": readiness,
        "cuda_available": _cuda_available(),
        "models_root": str(_models_root()),
        "models": _available_models(),
        "loaded": _current_model["instance"] is not None,
    }


def save_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Local LLM settings must be an object.")
    with _settings_lock:
        current = _configured_settings()
        settings = _validate_settings({**current, **payload})
        config = easy_app.getConfig()
        changed = config.get("local_llm") != settings
        config["local_llm"] = settings
        easy_app.setConfig(config)
    if changed:
        shutdown()
    return get_settings()


def health() -> Dict[str, Any]:
    readiness = backend_readiness()
    try:
        settings = _validate_settings(_configured_settings())
        if settings["device"] == "cuda" and not _cuda_available():
            raise HTTPException(status_code=503, detail="CUDA is selected but no NVIDIA device is available.")
        detail = readiness["detail"]
        ready = readiness["ready"]
    except HTTPException as exc:
        settings = _configured_settings()
        detail = str(exc.detail)
        ready = False
    return {
        "status": "ok" if ready else "error",
        "ready": ready,
        "detail": detail,
        "backend": readiness["backend"],
        "executable": readiness["executable"],
        "model_path": settings.get("model_path", ""),
        "device": settings.get("device", "cpu"),
        "gpu_layers": settings.get("gpu_layers", 0),
        "loaded": _current_model["instance"] is not None,
    }


def _load_model(settings: Dict[str, Any]) -> LlamaServer:
    if settings["device"] == "cuda" and not _cuda_available():
        raise HTTPException(status_code=503, detail="CUDA is selected but no NVIDIA device is available.")
    gpu_layers = settings["gpu_layers"] if settings["device"] == "cuda" else 0
    key = (settings["model_path"], settings["context_length"], gpu_layers)
    if key == _current_model["key"] and _current_model["instance"] is not None:
        return _current_model["instance"]
    old_model = _current_model["instance"]
    _current_model["key"] = None
    _current_model["instance"] = None
    if old_model is not None:
        old_model.close()
    _logger.info(
        f"Loading local prompt LLM: {settings['model_path']} "
        f"(device={settings['device']}, gpu_layers={gpu_layers})"
    )
    try:
        model = LlamaServer(
            Path(settings["model_path"]),
            settings["context_length"],
            gpu_layers,
            label="Local prompt processor",
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    _current_model["key"] = key
    _current_model["instance"] = model
    return model


def _processed_text(result: Dict[str, Any]) -> str:
    choices = result.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        raise HTTPException(status_code=502, detail="The local LLM did not return text.")
    text = text.strip()
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if not text:
        raise HTTPException(status_code=502, detail="The local LLM returned an empty prompt.")
    return text


def process_prompt(req: ProcessRequest) -> Dict[str, Any]:
    prompt = req.prompt.strip() if isinstance(req.prompt, str) else ""
    instruction = req.instruction.strip() if isinstance(req.instruction, str) else ""
    if not prompt:
        raise HTTPException(status_code=400, detail="Enter an image prompt before processing it.")
    if len(prompt) > 20000:
        raise HTTPException(status_code=400, detail="Image prompts are limited to 20,000 characters.")
    if not instruction:
        raise HTTPException(status_code=400, detail="Enter a prompt-processing instruction.")
    if len(instruction) > 4000:
        raise HTTPException(status_code=400, detail="Instructions are limited to 4,000 characters.")
    try:
        settings = _validate_settings(_configured_settings())
    except HTTPException as exc:
        raise HTTPException(status_code=503, detail=str(exc.detail)) from exc
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Instruction:\n{instruction}\n\n"
                f"Original image prompt:\n{prompt}\n\n"
                "Processed prompt:"
            ),
        },
    ]
    with _model_lock:
        model = _load_model(settings)
        try:
            result = model.create_chat_completion(
                messages,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                max_tokens=settings["max_tokens"],
                repeat_penalty=settings["repeat_penalty"],
                chat_template_kwargs={"enable_thinking": False},
            )
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
    return {
        "processed_prompt": _processed_text(result),
        "model_path": settings["model_path"],
        "device": settings["device"],
    }


def shutdown() -> None:
    with _model_lock:
        model = _current_model["instance"]
        _current_model["key"] = None
        _current_model["instance"] = None
        if model is not None:
            model.close()
