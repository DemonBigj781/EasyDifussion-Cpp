"""Filesystem-backed image gallery for Easy Diffusion."""

from pathlib import Path
from urllib.parse import quote
import hashlib
import json
import mimetypes
import os
import threading
import time

from easydiffusion import app as easy_app
from easydiffusion.server import server_api
from fastapi import HTTPException, Query, Request
from starlette.responses import FileResponse, JSONResponse

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None


PLUGIN_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PLUGIN_DIR / "config.json"
THUMBNAIL_DIR = PLUGIN_DIR / ".thumbnails"
ED_ROOT = PLUGIN_DIR.parents[2]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DEFAULT_PAGE_SIZE = 60
MAX_PAGE_SIZE = 120
MINIMUM_FILE_AGE_NS = 2_000_000_000
CONFIG_LOCK = threading.Lock()
THUMBNAIL_LOCK = threading.Lock()
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _route_exists(path: str) -> bool:
    return any(getattr(route, "path", None) == path for route in server_api.routes)


def _default_directory() -> Path:
    configured = easy_app.getConfig().get("force_save_path")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return (ED_ROOT / "outputs").resolve()


def _read_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _configured_directory() -> Path:
    raw = str(_read_config().get("gallery_directory", "")).strip()
    if not raw:
        return _default_directory()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ED_ROOT / path
    return path.resolve()


def _write_config(directory: Path) -> None:
    payload = {"gallery_directory": str(directory)}
    temporary = CONFIG_PATH.with_suffix(".json.tmp")
    with CONFIG_LOCK:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, CONFIG_PATH)


def _resolve_gallery_file(relative_path: str) -> Path:
    root = _configured_directory()
    candidate = (root / relative_path).resolve()
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
    if stat.st_size == 0 or time.time_ns() - stat.st_mtime_ns < MINIMUM_FILE_AGE_NS:
        raise HTTPException(status_code=409, detail="Gallery image is still being written; retry shortly.")
    return candidate


def _thumbnail_path(source: Path) -> Path:
    stat = source.stat()
    cache_key = f"{source}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    return THUMBNAIL_DIR / f"{hashlib.sha256(cache_key).hexdigest()}.jpg"


def _create_thumbnail(source: Path) -> Path:
    if Image is None or ImageOps is None:
        return source

    try:
        thumbnail = _thumbnail_path(source)
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
                image.save(
                    temporary,
                    format="JPEG",
                    quality=82,
                    optimize=True,
                )
            os.replace(temporary, thumbnail)
            return thumbnail
        except (OSError, ValueError):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return source


def _list_images(root: Path, page: int, page_size: int) -> dict:
    if not root.is_dir():
        return {
            "directory": str(root),
            "exists": False,
            "images": [],
            "total": 0,
            "page": 1,
            "page_size": page_size,
            "total_pages": 1,
            "has_previous": False,
            "has_next": False,
        }

    files = []
    for directory, child_directories, filenames in os.walk(root, followlinks=False):
        child_directories[:] = [
            name for name in child_directories
            if not (Path(directory) / name).is_symlink()
        ]
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix.lower() not in IMAGE_EXTENSIONS or path.is_symlink():
                continue
            try:
                stat = path.stat()
                relative = path.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            # sdkit saves directly to the final filename. Do not expose a new
            # image until its size and metadata writes have had time to finish,
            # otherwise FileResponse can advertise an early Content-Length and
            # then stream more bytes as the file grows.
            if stat.st_size == 0 or time.time_ns() - stat.st_mtime_ns < MINIMUM_FILE_AGE_NS:
                continue
            files.append((stat.st_mtime_ns, stat.st_size, relative))

    files.sort(key=lambda item: item[0], reverse=True)
    total = len(files)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(page, 1), total_pages)
    start = (page - 1) * page_size
    page_files = files[start:start + page_size]
    images = [
        {
            "id": relative,
            "filename": Path(relative).name,
            "relative_path": relative,
            "mtime": mtime_ns / 1_000_000,
            "size": size,
            "url": f"/gallery-plugin/file/{quote(relative, safe='/')}",
            "thumbnail_url": f"/gallery-plugin/thumb/{quote(relative, safe='/')}",
        }
        for mtime_ns, size, relative in page_files
    ]
    return {
        "directory": str(root),
        "exists": True,
        "images": images,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
        "start_index": start + 1 if total else 0,
        "end_index": start + len(images),
    }


if not _route_exists("/gallery-plugin/settings"):

    @server_api.get("/gallery-plugin/settings", include_in_schema=False)
    def gallery_plugin_get_settings():
        directory = _configured_directory()
        return JSONResponse(
            {
                "gallery_directory": str(directory),
                "exists": directory.is_dir(),
                "default_directory": str(_default_directory()),
            },
            headers=NO_CACHE_HEADERS,
        )

    @server_api.post("/gallery-plugin/settings", include_in_schema=False)
    async def gallery_plugin_save_settings(request: Request):
        payload = await request.json()
        raw = str(payload.get("gallery_directory", "")).strip() if isinstance(payload, dict) else ""
        if not raw:
            directory = _default_directory()
        else:
            directory = Path(raw).expanduser()
            if not directory.is_absolute():
                directory = ED_ROOT / directory
            directory = directory.resolve()
        if not directory.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {directory}",
            )
        _write_config(directory)
        return JSONResponse(
            {"gallery_directory": str(directory), "exists": True},
            headers=NO_CACHE_HEADERS,
        )


if not _route_exists("/gallery-plugin/images"):

    @server_api.get("/gallery-plugin/images", include_in_schema=False)
    def gallery_plugin_images(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(
            default=DEFAULT_PAGE_SIZE,
            ge=1,
            le=MAX_PAGE_SIZE,
        ),
    ):
        return JSONResponse(
            _list_images(_configured_directory(), page, page_size),
            headers=NO_CACHE_HEADERS,
        )


if not _route_exists("/gallery-plugin/file/{relative_path:path}"):

    @server_api.get("/gallery-plugin/file/{relative_path:path}", include_in_schema=False)
    def gallery_plugin_file(relative_path: str):
        path = _resolve_gallery_file(relative_path)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Cache-Control": "private, max-age=60"},
        )

    @server_api.delete("/gallery-plugin/file/{relative_path:path}", include_in_schema=False)
    def gallery_plugin_delete_file(relative_path: str):
        path = _resolve_gallery_file(relative_path)
        try:
            thumbnail = _thumbnail_path(path)
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
        return JSONResponse({"deleted": relative_path}, headers=NO_CACHE_HEADERS)


if not _route_exists("/gallery-plugin/thumb/{relative_path:path}"):

    @server_api.get("/gallery-plugin/thumb/{relative_path:path}", include_in_schema=False)
    def gallery_plugin_thumbnail(relative_path: str):
        source = _resolve_gallery_file(relative_path)
        thumbnail = _create_thumbnail(source)
        media_type = (
            "image/jpeg"
            if thumbnail != source
            else mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        )
        return FileResponse(
            thumbnail,
            media_type=media_type,
            headers={"Cache-Control": "private, max-age=86400"},
        )
