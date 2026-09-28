"""Filesystem listing helpers for configured model directories."""

import os
from pathlib import Path


def list_model_files(model_type: str) -> list[str]:
    from .model_manager import MODEL_EXTENSIONS, get_model_dirs

    extensions = tuple(extension.lower() for extension in MODEL_EXTENSIONS.get(model_type, ()))
    if not extensions:
        return []
    files = set()
    for directory in get_model_dirs(model_type):
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and candidate.name.lower().endswith(extensions):
                files.add(str(resolved))
    return sorted(files, key=str.casefold)


def lora_dir() -> Path:
    configured = os.environ.get("ED_LORA_DIR") or os.environ.get("LORA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    from .model_manager import get_model_dirs

    directories = [Path(path).expanduser().resolve() for path in get_model_dirs("lora")]
    if not directories:
        raise FileNotFoundError("No LoRA model directory is configured")
    return next((path for path in directories if path.is_dir()), directories[0])


def list_lora_files() -> list[str]:
    return list_model_files("lora")


def list_checkpoint_files() -> list[str]:
    return list_model_files("stable-diffusion")


def list_vae_files() -> list[str]:
    return list_model_files("vae")
