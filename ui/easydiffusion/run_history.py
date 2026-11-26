"""Opt-in private render archive. Never store prompts or arbitrary paths."""
import base64
import binascii
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import tempfile
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/v1/runs", include_in_schema=False)
HEADERS = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}
TOKEN_RE = re.compile(rb"[0-9a-fA-F]{64}")
RUN_RE = re.compile(r"[0-9a-f]{32}")
ASSET_RE = re.compile(r"output-[0-9]{3}\.(gif|png|webp|jpg)|diagnostics\.json")
MIME = {".gif": "image/gif", ".png": "image/png", ".webp": "image/webp",
        ".jpg": "image/jpeg", ".json": "application/json"}
MAX_BYTES = 128 * 1024 * 1024
NUMERIC = ("width", "height", "seed", "fps", "frame_count", "encoded_frame_count",
           "num_inference_steps", "guidance_scale", "elapsed_seconds")
CHOICES = ("sampler_name", "scheduler_name", "cache_mode", "engine_build")


def _private_stat(path, predicate):
    try:
        s = path.lstat()
        owner = getattr(os, "getuid", lambda: None)()
        return predicate(s.st_mode) and owner == s.st_uid and not s.st_mode & 0o077
    except OSError:
        return False


def is_private_path(path):
    """Keep private bytes out of older file-serving routes, even while disabled."""
    candidate = Path(path).resolve()
    default_root = Path.home() / ".local/state/easy-diffusion-history"
    default_token = Path.home() / ".config/easy-diffusion-history/token"
    root = Path(os.environ.get("EASY_DIFFUSION_RUN_HISTORY_DIR") or default_root)
    token = Path(os.environ.get("EASY_DIFFUSION_RUN_HISTORY_TOKEN_FILE") or default_token)
    try:
        return (root.is_absolute() and candidate.is_relative_to(root.resolve())
                or token.is_absolute() and candidate == token.resolve()
                or candidate.is_relative_to(default_root.resolve())
                or candidate == default_token.resolve())
    except (OSError, RuntimeError):
        return True


def _configuration():
    if os.environ.get("EASY_DIFFUSION_RUN_HISTORY") != "1":
        return None
    root = Path(os.environ.get("EASY_DIFFUSION_RUN_HISTORY_DIR", ""))
    token_file = Path(os.environ.get("EASY_DIFFUSION_RUN_HISTORY_TOKEN_FILE", ""))
    if not root.is_absolute() or not token_file.is_absolute():
        return None
    if not _private_stat(root, stat.S_ISDIR) or not _private_stat(token_file, stat.S_ISREG):
        return None
    try:
        root = root.resolve(strict=True)
        token_file = token_file.resolve(strict=True)
        # The token is not part of the archive. Refuse to overlap either tree.
        if token_file.is_relative_to(root):
            return None
        with token_file.open("rb") as stream:
            token = stream.read(67).rstrip(b"\n\r")
    except OSError:
        return None
    return (root, token) if TOKEN_RE.fullmatch(token) else None


def authorize(authorization):
    conf = _configuration()
    if conf is None:
        raise HTTPException(404, "Run history is unavailable", headers=HEADERS)
    scheme, _, value = (authorization or "").partition(" ")
    supplied = value.encode("utf-8")
    if scheme.lower() != "bearer" or not TOKEN_RE.fullmatch(supplied) or not hmac.compare_digest(conf[1], supplied):
        raise HTTPException(401, "Bearer token required",
                            headers={**HEADERS, "WWW-Authenticate": "Bearer"})
    return conf[0]


def _media(data):
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    raise ValueError("Unsupported history artifact")


def _metadata(fields):
    safe = {}
    for field in NUMERIC:
        value = fields.get(field)
        if type(value) in (int, float) and -1e30 < value < 1e30:
            safe[field] = value
    for field in CHOICES:
        value = fields.get(field)
        if type(value) is str and re.fullmatch(r"[A-Za-z0-9_. -]{1,96}", value):
            safe[field] = value
    return safe


def _save(root, artifacts, fields, diagnostics=None):
    run_id = secrets.token_hex(16)
    tmp = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    names = []
    size = 0
    try:
        for i, data in enumerate(artifacts):
            size += len(data)
            if i >= 513 or size > MAX_BYTES:
                raise ValueError("Run history artifact limit exceeded")
            name = f"output-{i:03d}.{_media(data)}"
            (tmp / name).write_bytes(data)
            names.append(name)
        if not names:
            raise ValueError("No history artifacts")
        if diagnostics:
            (tmp / "diagnostics.json").write_text(json.dumps(_metadata(diagnostics)), encoding="utf-8")
            names.append("diagnostics.json")
        row = dict(id=run_id, created_at=datetime.now(timezone.utc).isoformat(),
                   settings=_metadata(fields), artifacts=names)
        (tmp / "manifest.json").write_text(json.dumps(row, allow_nan=False), encoding="utf-8")
        for f in tmp.iterdir():
            f.chmod(0o600)
        tmp.rename(root / run_id)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return run_id


def _decode_outputs(output):
    total = 0
    for item in output:
        data = item["data"]
        if data.startswith("data:"):
            header, _, data = data.partition(",")
            if not header.endswith(";base64"):
                raise ValueError("Unsupported history data URI")
        total += len(data)
        if total > ((MAX_BYTES + 2) // 3) * 4 + 1024:
            raise ValueError("Run history artifact limit exceeded")
        try:
            yield base64.b64decode(data, validate=True)
        except binascii.Error as exc:
            raise ValueError("Invalid history artifact") from exc


def capture_video(output, fields):
    conf = _configuration()
    if conf is None:
        return None
    fields = {**fields, "engine_build": os.environ.get("EASY_DIFFUSION_RUN_HISTORY_BUILD")}
    return _save(conf[0], _decode_outputs(output), fields)


def import_saved(root, files, fields, diagnostics=None):
    """Local operator import, never an HTTP path selection input."""
    root = Path(root)
    if not root.is_absolute() or not _private_stat(root, stat.S_ISDIR):
        raise ValueError("Archive directory must be absolute and private")

    def read():
        remaining = MAX_BYTES
        for file in files:
            with Path(file).open("rb") as stream:
                data = stream.read(remaining + 1)
            remaining -= len(data)
            if remaining < 0:
                raise ValueError("Run history artifact limit exceeded")
            yield data

    return _save(root, read(), fields, diagnostics)


def _not_found():
    return HTTPException(404, "Run not found", headers=HEADERS)


def _manifest(root, run_id):
    if not RUN_RE.fullmatch(run_id):
        raise _not_found()
    folder = root / run_id
    f = folder / "manifest.json"
    if not _private_stat(folder, stat.S_ISDIR) or not _private_stat(f, stat.S_ISREG):
        raise _not_found()
    try:
        with f.open(encoding="utf-8") as stream:
            row = json.loads(stream.read(65537))
        if row["id"] == run_id and isinstance(row["created_at"], str):
            datetime.fromisoformat(row["created_at"])
            assets = row["artifacts"]
            if not isinstance(assets, list):
                raise ValueError("Invalid artifact manifest")
            return {"id": run_id, "created_at": row["created_at"],
                    "settings": _metadata(row["settings"]), "artifacts": [
                        name for name in assets if isinstance(name, str) and ASSET_RE.fullmatch(name)
                    ]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    raise _not_found()


@router.get("")
def list_runs(request: Request):
    root = authorize(request.headers.get("Authorization"))
    try:
        limit = int(request.query_params.get("limit", 20))
        offset = int(request.query_params.get("offset", 0))
    except ValueError as exc:
        raise HTTPException(400, "Invalid pagination", headers=HEADERS) from exc
    rows = []
    for folder in root.iterdir():
        if not RUN_RE.fullmatch(folder.name):
            continue
        try:
            row = _manifest(root, folder.name)
            rows.append({k: row[k] for k in ("id", "created_at", "settings")})
        except (HTTPException, KeyError):
            continue
    rows.sort(key=lambda value: (value["created_at"], value["id"]), reverse=True)
    offset = max(0, offset)
    return JSONResponse({"runs": rows[offset:offset + max(1, min(limit, 100))]},
                        headers=HEADERS)


@router.get("/{run_id}")
def get_run(request: Request, run_id: str):
    root = authorize(request.headers.get("Authorization"))
    row = _manifest(root, run_id)
    return JSONResponse({**row, "artifacts": [
        {"name": name, "url": f"/v1/runs/{run_id}/artifacts/{name}"}
        for name in row["artifacts"] if ASSET_RE.fullmatch(name)
    ]}, headers=HEADERS)


@router.get("/{run_id}/artifacts/{name}")
def get_artifact(request: Request, run_id: str, name: str):
    root = authorize(request.headers.get("Authorization"))
    row = _manifest(root, run_id)
    if not ASSET_RE.fullmatch(name) or name not in row["artifacts"]:
        raise _not_found()
    f = root / run_id / name
    if not _private_stat(f, stat.S_ISREG):
        raise _not_found()
    return FileResponse(f, filename=name, media_type=MIME[f.suffix], headers=HEADERS)
