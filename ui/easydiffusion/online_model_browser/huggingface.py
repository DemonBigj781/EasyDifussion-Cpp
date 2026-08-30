"""Hugging Face Hub search and download helpers for the Online Model Browser."""
from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, urlparse

import requests

from . import service as civ


HF_API_ROOT = "https://huggingface.co/api"
HF_WEB_ROOT = "https://huggingface.co"
_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
_MODEL_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".sft",
)
_IMAGE_SUFFIXES = (".avif", ".jpeg", ".jpg", ".png", ".webp")
_SORT_FIELDS = {
    "Most Downloaded": "downloads",
    "Most Liked": "likes",
    "Recently Updated": "lastModified",
}


def _validate_repo_id(repo_id: str) -> str:
    clean = str(repo_id or "").strip()
    if not _REPO_ID_RE.fullmatch(clean):
        raise ValueError("Invalid Hugging Face repository ID")
    return clean


def _validate_repo_file(filename: str) -> str:
    clean = str(filename or "").strip().replace("\\", "/")
    path = PurePosixPath(clean)
    if not clean or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Invalid Hugging Face repository filename")
    if not clean.lower().endswith(_MODEL_SUFFIXES):
        raise ValueError("The selected Hugging Face file is not a supported model file")
    return clean


def _headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "Easy-Diffusion-Online-Model-Browser"}
    clean_token = str(token or "").strip()
    if clean_token:
        headers["Authorization"] = f"Bearer {clean_token}"
    return headers


def _base_model(tags: list[Any], pipeline_tag: str) -> str:
    values = [str(tag).split(":", 1)[1] for tag in tags if str(tag).startswith("base_model:")]
    value = next((item for item in values if "/" in item), values[0] if values else "")
    lowered = " ".join(str(tag).lower() for tag in tags) + " " + value.lower()
    if "pony" in lowered or "sdxl" in lowered:
        return "SDXL / Pony"
    if "stable-diffusion-3.5" in lowered or "sd 3.5" in lowered:
        return "SD 3.5"
    if "stable-diffusion-3" in lowered or "sd3" in lowered:
        return "SD 3"
    if "stable-diffusion-2" in lowered or "sd2" in lowered:
        return "SD 2"
    if "stable-diffusion" in lowered or pipeline_tag == "text-to-image":
        return "Stable Diffusion"
    return value or pipeline_tag or "Hugging Face"


def _file_type(filename: str) -> str:
    lowered = filename.lower()
    if "vae" in lowered:
        return "VAE"
    if any(part in lowered for part in ("text_encoder", "text-encoder", "clip_l", "clip_g", "t5")):
        return "Text Encoder"
    if any(part in lowered for part in ("unet", "diffusion_model")):
        return "Diffusion Model"
    if "controlnet" in lowered:
        return "ControlNet"
    if "lora" in lowered:
        return "LoRA"
    return Path(filename).suffix.lower().lstrip(".").upper() or "Model"


def _preview_url(repo_id: str, revision: str, siblings: list[Dict[str, Any]]) -> Optional[str]:
    image = next(
        (
            str(item.get("rfilename") or "")
            for item in siblings
            if str(item.get("rfilename") or "").lower().endswith(_IMAGE_SUFFIXES)
        ),
        "",
    )
    if not image:
        return None
    return (
        f"{HF_WEB_ROOT}/{quote(repo_id, safe='/')}/resolve/"
        f"{quote(revision or 'main', safe='')}/{quote(image, safe='/')}"
    )


def _trim_model(item: Dict[str, Any]) -> Dict[str, Any]:
    repo_id = _validate_repo_id(item.get("id") or item.get("modelId"))
    revision = str(item.get("sha") or "main")
    siblings = item.get("siblings") if isinstance(item.get("siblings"), list) else []
    files = []
    for sibling in siblings:
        filename = str(sibling.get("rfilename") or "")
        if not filename.lower().endswith(_MODEL_SUFFIXES):
            continue
        size = sibling.get("size") or (sibling.get("lfs") or {}).get("size")
        files.append(
            {
                "name": filename,
                "type": _file_type(filename),
                "sizeKB": (float(size) / 1024.0) if isinstance(size, (int, float)) else None,
                "repoId": repo_id,
                "revision": revision,
            }
        )

    tags = item.get("tags") if isinstance(item.get("tags"), list) else []
    pipeline_tag = str(item.get("pipeline_tag") or "model")
    preview = _preview_url(repo_id, revision, siblings)
    author = str(item.get("author") or repo_id.split("/", 1)[0])
    return {
        "id": repo_id,
        "name": repo_id.split("/", 1)[1],
        "type": pipeline_tag,
        "provider": "huggingface",
        "pageUrl": f"{HF_WEB_ROOT}/{quote(repo_id, safe='/')}",
        "creator": {"username": author},
        "downloads": item.get("downloads") or 0,
        "likes": item.get("likes") or 0,
        "gated": item.get("gated") or False,
        "private": bool(item.get("private")),
        "tags": [str(tag) for tag in tags[:40]],
        "modelVersion": {
            "id": revision,
            "baseModel": _base_model(tags, pipeline_tag),
            "files": files,
            "images": [{"url": preview, "thumbnailUrl": preview}] if preview else [],
        },
    }


def search(
    query: str,
    sort: str,
    page: int,
    limit: int,
    token: Optional[str],
) -> Dict[str, Any]:
    page = max(1, int(page or 1))
    limit = max(1, min(50, int(limit or 20)))
    params: Dict[str, Any] = {
        "limit": limit,
        "skip": (page - 1) * limit,
        "sort": _SORT_FIELDS.get(sort, "downloads"),
        "direction": -1,
        "full": "true",
    }
    clean_query = str(query or "").strip()
    if clean_query:
        params["search"] = clean_query

    response = requests.get(
        f"{HF_API_ROOT}/models",
        params=params,
        headers=_headers(token),
        timeout=60,
    )
    if response.status_code in {401, 403}:
        raise PermissionError("Hugging Face rejected the token or repository access")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected Hugging Face search response")
    return {
        "ok": True,
        "items": [_trim_model(item) for item in payload if isinstance(item, dict)],
        "metadata": {
            "page": page,
            "limit": limit,
            "hasNext": len(payload) == limit,
            "hasPrevious": page > 1,
        },
    }


def _destination(target: Dict[str, Any], filename: str) -> Path:
    mv = target.get("modelVersion") if isinstance(target.get("modelVersion"), dict) else {}
    base_model = str(mv.get("baseModel") or "")
    pipeline = str(target.get("type") or "").lower()
    repo_id = _validate_repo_id(target.get("id"))
    lowered = filename.lower()

    if pipeline.startswith("text-") or pipeline in {"conversational", "feature-extraction"}:
        relative = Path("LLM") / civ._safe_slug(repo_id.replace("/", "--"))
    elif "vae" in lowered:
        relative = Path("VAE") / civ._base_model_bucket(base_model)
    elif "lora" in lowered:
        relative = Path("Lora") / civ._base_model_bucket(base_model)
    elif any(part in lowered for part in ("text_encoder", "text-encoder", "clip_l", "clip_g", "t5")):
        relative = Path("Text-encoder") / civ._base_model_bucket(base_model)
    elif any(part in lowered for part in ("unet", "diffusion_model")):
        relative = Path("DiffusionModels") / civ._base_model_bucket(base_model)
    else:
        relative = Path("checkpoints") / civ._base_model_bucket(base_model)
    return civ._resolve_download_dir(str(relative))


def download_target(target: Dict[str, Any], file_choice: Dict[str, Any]) -> Tuple[str, str, str]:
    repo_id = _validate_repo_id(file_choice.get("repoId") or target.get("id"))
    filename = _validate_repo_file(file_choice.get("name"))
    revision = str(file_choice.get("revision") or "main").strip()
    if not revision or ".." in PurePosixPath(revision).parts or not re.fullmatch(r"[A-Za-z0-9._/-]+", revision):
        raise ValueError("Invalid Hugging Face revision")
    url = (
        f"{HF_WEB_ROOT}/{quote(repo_id, safe='/')}/resolve/"
        f"{quote(revision, safe='/')}/{quote(filename, safe='/')}"
    )
    local_name = civ._safe_slug(PurePosixPath(filename).name, "model.bin")
    return url, local_name, str(_destination(target, filename))


def download_file(
    url: str,
    dest_dir: str,
    filename: str,
    token: Optional[str],
    progress_callback: Optional[Any] = None,
) -> Tuple[str, int]:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ValueError("Download URL must use the official Hugging Face HTTPS host")

    resolved_dir = civ._resolve_download_dir(dest_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = civ._safe_slug(Path(filename).name, "model.bin")
    dest_path = resolved_dir / safe_filename
    temp_path = dest_path.with_name(f"{dest_path.name}.part")
    resumed_bytes = temp_path.stat().st_size if temp_path.exists() else 0
    headers = _headers(token)
    headers["Accept"] = "application/octet-stream"
    if resumed_bytes:
        headers["Range"] = f"bytes={resumed_bytes}-"

    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        if response.status_code in {401, 403}:
            raise PermissionError("Hugging Face download requires access approval and a valid token")
        if response.status_code == 416 and resumed_bytes:
            content_range = str(response.headers.get("content-range") or "")
            total_match = re.search(r"\*/(\d+)$", content_range)
            if total_match and int(total_match.group(1)) == resumed_bytes:
                os.replace(temp_path, dest_path)
                return str(dest_path), resumed_bytes
        response.raise_for_status()
        is_resuming = resumed_bytes > 0 and response.status_code == 206
        if resumed_bytes and not is_resuming:
            resumed_bytes = 0

        size_header = response.headers.get("content-length")
        response_bytes = int(size_header) if size_header and size_header.isdigit() else None
        total_bytes = resumed_bytes + response_bytes if response_bytes is not None else None
        total = resumed_bytes
        with temp_path.open("ab" if is_resuming else "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                total += len(chunk)
                if progress_callback:
                    progress_callback(total, total_bytes)

        if total_bytes is not None and total != total_bytes:
            raise IOError(f"Incomplete download: received {total} of {total_bytes} bytes")
        os.replace(temp_path, dest_path)
    return str(dest_path), total
