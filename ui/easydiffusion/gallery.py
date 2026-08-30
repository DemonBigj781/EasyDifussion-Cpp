"""Filesystem-backed image gallery built into Easy Diffusion."""

from pathlib import Path
from typing import Optional
from urllib.parse import quote
import hashlib
import json
import mimetypes
import os
import threading
import time

from easydiffusion import app
from fastapi import HTTPException

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DEFAULT_PAGE_SIZE = 60
MAX_PAGE_SIZE = 120
MINIMUM_FILE_AGE_NS = 2_000_000_000
THUMBNAIL_DIR = Path(app.BUCKET_DIR) / "gallery-thumbnails"
LEGACY_PLUGIN_DIR = Path(app.ROOT_DIR) / "plugins" / "server" / "gallery_plugin"
LEGACY_CONFIG_PATH = LEGACY_PLUGIN_DIR / "config.json"
LEGACY_CONFIG_BACKUP_PATH = Path(app.BUCKET_DIR) / "gallery-settings.legacy.json"
CONFIG_LOCK = threading.Lock()
THUMBNAIL_LOCK = threading.Lock()


def _default_directory() -> Path:
    configured = app.getConfig().get("force_save_path")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return (Path(app.ROOT_DIR) / "outputs").resolve()


def _legacy_directory() -> str:
    for config_path in (LEGACY_CONFIG_PATH, LEGACY_CONFIG_BACKUP_PATH):
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            directory = str(data.get("gallery_directory", "")).strip()
            if directory:
                return directory
    return ""


def _configured_value(config: dict) -> str:
    gallery_config = config.get("gallery", {})
    if isinstance(gallery_config, dict):
        value = str(gallery_config.get("directory", "")).strip()
        if value:
            return value
    value = str(config.get("gallery_directory", "")).strip()
    return value or _legacy_directory()


def configured_directory() -> Path:
    raw = _configured_value(app.getConfig())
    if not raw:
        return _default_directory()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(app.ROOT_DIR) / path
    return path.resolve()


def _write_config(directory: Path) -> None:
    with CONFIG_LOCK:
        config = app.getConfig()
        gallery_config = config.get("gallery")
        if not isinstance(gallery_config, dict):
            gallery_config = {}
        gallery_config["directory"] = str(directory)
        config["gallery"] = gallery_config
        config.pop("gallery_directory", None)
        app.setConfig(config)


def migrate_legacy_settings() -> None:
    """Move the old plugin setting into config.yaml on the first core start."""

    config = app.getConfig()
    gallery_config = config.get("gallery", {})
    if isinstance(gallery_config, dict) and str(gallery_config.get("directory", "")).strip():
        return
    legacy = _legacy_directory()
    if legacy:
        path = Path(legacy).expanduser()
        if not path.is_absolute():
            path = Path(app.ROOT_DIR) / path
        _write_config(path.resolve())


def get_settings() -> dict:
    directory = configured_directory()
    return {
        "gallery_directory": str(directory),
        "exists": directory.is_dir(),
        "default_directory": str(_default_directory()),
    }


def save_settings(payload: dict) -> dict:
    raw = str(payload.get("gallery_directory", "")).strip() if isinstance(payload, dict) else ""
    if not raw:
        directory = _default_directory()
    else:
        directory = Path(raw).expanduser()
        if not directory.is_absolute():
            directory = Path(app.ROOT_DIR) / directory
        directory = directory.resolve()
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {directory}")
    _write_config(directory)
    return {"gallery_directory": str(directory), "exists": True}


def _resolve_gallery_path(relative_path: str, require_ready: bool) -> Path:
    root = configured_directory()
    unresolved = root / relative_path
    if unresolved.is_symlink():
        raise HTTPException(status_code=404, detail="Gallery image not found.")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HTTPException(status_code=403, detail="Path is outside the gallery directory.") from error
    if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Gallery image not found.")
    try:
        stat = candidate.stat()
    except OSError as error:
        raise HTTPException(status_code=404, detail="Gallery image not found.") from error
    if require_ready and (stat.st_size == 0 or time.time_ns() - stat.st_mtime_ns < MINIMUM_FILE_AGE_NS):
        raise HTTPException(status_code=409, detail="Gallery image is still being written; retry shortly.")
    return candidate


def resolve_gallery_file(relative_path: str) -> Path:
    return _resolve_gallery_path(relative_path, require_ready=True)


def relative_gallery_path(path) -> Optional[str]:
    root = configured_directory()
    candidate = Path(path).expanduser().resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return relative.as_posix()


def thumbnail_path(source: Path) -> Path:
    stat = source.stat()
    cache_key = f"{source}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    return THUMBNAIL_DIR / f"{hashlib.sha256(cache_key).hexdigest()}.jpg"


def create_thumbnail(source: Path) -> Path:
    if Image is None or ImageOps is None:
        return source
    try:
        thumbnail = thumbnail_path(source)
    except OSError:
        return source
    if thumbnail.is_file():
        return thumbnail

    with THUMBNAIL_LOCK:
        if thumbnail.is_file():
            return thumbnail
        temporary = thumbnail.with_suffix(".tmp")
        try:
            THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened)
                if getattr(image, "is_animated", False):
                    image.seek(0)
                image = image.convert("RGB")
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image.thumbnail((384, 384), resampling)
                image.save(temporary, format="JPEG", quality=82, optimize=True)
            os.replace(temporary, thumbnail)
            return thumbnail
        except (OSError, ValueError):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return source


def list_images(page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> dict:
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), MAX_PAGE_SIZE)
    root = configured_directory()
    if not root.is_dir():
        return {
            "directory": str(root), "exists": False, "images": [], "total": 0,
            "page": 1, "page_size": page_size, "total_pages": 1,
            "has_previous": False, "has_next": False,
        }

    files = []
    for directory, child_directories, filenames in os.walk(root, followlinks=False):
        child_directories[:] = [name for name in child_directories if not (Path(directory) / name).is_symlink()]
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix.lower() not in IMAGE_EXTENSIONS or path.is_symlink():
                continue
            try:
                stat = path.stat()
                relative = path.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            if stat.st_size == 0 or time.time_ns() - stat.st_mtime_ns < MINIMUM_FILE_AGE_NS:
                continue
            files.append((stat.st_mtime_ns, stat.st_size, relative))

    files.sort(key=lambda item: item[0], reverse=True)
    total = len(files)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    page_files = files[start : start + page_size]
    images = [
        {
            "id": relative,
            "filename": Path(relative).name,
            "relative_path": relative,
            "mtime": mtime_ns / 1_000_000,
            "size": size,
            "url": f"/gallery/file/{quote(relative, safe='/')}",
            "thumbnail_url": f"/gallery/thumb/{quote(relative, safe='/')}",
        }
        for mtime_ns, size, relative in page_files
    ]
    return {
        "directory": str(root), "exists": True, "images": images, "total": total,
        "page": page, "page_size": page_size, "total_pages": total_pages,
        "has_previous": page > 1, "has_next": page < total_pages,
        "start_index": start + 1 if total else 0, "end_index": start + len(images),
    }


def delete_file(relative_path: str) -> dict:
    # A generated result is only returned after its save has completed, so a
    # DELETE does not need the two-second read stabilization delay used by GET.
    path = _resolve_gallery_path(relative_path, require_ready=False)
    try:
        thumbnail = thumbnail_path(path)
    except OSError:
        thumbnail = None
    try:
        path.unlink()
    except OSError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    if thumbnail:
        try:
            thumbnail.unlink(missing_ok=True)
        except OSError:
            pass
    deleted_sidecars = []
    for suffix in (".json", ".txt"):
        sidecar = path.with_suffix(suffix)
        if sidecar.is_symlink() or not sidecar.is_file():
            continue
        try:
            sidecar.unlink()
            deleted_sidecars.append(sidecar.name)
        except OSError:
            pass
    return {"deleted": relative_path, "deleted_sidecars": deleted_sidecars}


def file_media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
