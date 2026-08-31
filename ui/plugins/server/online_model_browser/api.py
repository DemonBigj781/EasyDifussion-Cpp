"""Native Easy Diffusion routes for online model browsing and downloads."""
from __future__ import annotations

import os
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from . import huggingface as hf
from . import service as civ
from easydiffusion import gallery
from easydiffusion.utils import log as _logger

APP = APIRouter()
HUGGINGFACE_APP = APIRouter()

_download_jobs: Dict[str, Dict[str, Any]] = {}
_download_jobs_lock = threading.Lock()


def _safe_error(error: Any) -> str:
    return civ._sanitize_error(error)


def _set_download_job(download_id: str, **fields: Any) -> Dict[str, Any]:
    with _download_jobs_lock:
        if download_id not in _download_jobs and len(_download_jobs) >= 512:
            completed = sorted(
                (
                    (key, value)
                    for key, value in _download_jobs.items()
                    if value.get("status") in {"completed", "failed"}
                ),
                key=lambda item: item[1].get("completedAt", 0),
            )
            for key, _ in completed[: max(1, len(_download_jobs) - 511)]:
                _download_jobs.pop(key, None)
        job = _download_jobs.setdefault(download_id, {})
        job.update(fields)
        return dict(job)


def _get_download_job(download_id: str) -> Optional[Dict[str, Any]]:
    with _download_jobs_lock:
        job = _download_jobs.get(download_id)
        return dict(job) if job else None


def _download_worker(
    download_id: str,
    target: Dict[str, Any],
    file_choice: Dict[str, Any],
    payload: Dict[str, Any],
    api_key: str,
) -> None:
    start = time.perf_counter()
    file_override = file_choice if isinstance(file_choice, dict) else None
    try:
        url = payload.get("fileUrl")
        fname = payload.get("filename")
        dest_dir_override = payload.get("destDir")

        if not url or not fname or not dest_dir_override:
            url_auto, fname_auto, dest_dir_auto = civ._pick_download(target, file_override)
            url = url or url_auto
            fname = fname or fname_auto
            dest_dir_override = dest_dir_override or dest_dir_auto

        if not url:
            mv = target.get("modelVersion") or {}
            mv_id = mv.get("id")
            if mv_id:
                url = f"https://civitai.com/api/download/models/{mv_id}"

        if not url or not fname or not dest_dir_override:
            raise ValueError("Unable to resolve download target")

        _set_download_job(
            download_id,
            status="downloading",
            filename=fname,
            destDir=dest_dir_override,
            startedAt=time.time(),
        )

        def on_progress(downloaded_bytes: int, total_bytes: Optional[int]) -> None:
            percent = None
            if total_bytes:
                percent = min(100.0, round((downloaded_bytes / total_bytes) * 100.0, 1))
            _set_download_job(
                download_id,
                status="downloading",
                downloadedBytes=downloaded_bytes,
                totalBytes=total_bytes,
                percent=percent,
            )

        dest_path, size = civ._download_file(
            url,
            dest_dir_override,
            fname,
            api_key,
            progress_callback=on_progress,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _set_download_job(
            download_id,
            status="completed",
            path=dest_path,
            size=size,
            downloadedBytes=size,
            completedAt=time.time(),
            percent=100.0,
        )
        _logger.info("POST /civitai/download -> 200 (%.1fms)", elapsed_ms)
    except Exception as exc:
        message = _safe_error(exc)
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 5:
            message = "Input/output error while writing the downloaded file. Check the mounted drive and destination path."
        civ._log(f"download error: {exc}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _set_download_job(
            download_id,
            status="failed",
            error=message,
            completedAt=time.time(),
        )
        _logger.error("POST /civitai/download -> 500 (%.1fms): %s", elapsed_ms, message)


def _huggingface_download_worker(
    download_id: str,
    target: Dict[str, Any],
    file_choice: Dict[str, Any],
    token: Optional[str],
) -> None:
    start = time.perf_counter()
    try:
        url, filename, dest_dir = hf.download_target(target, file_choice)
        _set_download_job(
            download_id,
            status="downloading",
            filename=filename,
            destDir=dest_dir,
            startedAt=time.time(),
        )

        def on_progress(downloaded_bytes: int, total_bytes: Optional[int]) -> None:
            percent = None
            if total_bytes:
                percent = min(100.0, round((downloaded_bytes / total_bytes) * 100.0, 1))
            _set_download_job(
                download_id,
                status="downloading",
                downloadedBytes=downloaded_bytes,
                totalBytes=total_bytes,
                percent=percent,
            )

        dest_path, size = hf.download_file(
            url,
            dest_dir,
            filename,
            token,
            progress_callback=on_progress,
        )
        _set_download_job(
            download_id,
            status="completed",
            path=dest_path,
            size=size,
            downloadedBytes=size,
            completedAt=time.time(),
            percent=100.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("POST /huggingface/download -> 200 (%.1fms)", elapsed_ms)
    except Exception as exc:
        message = _safe_error(exc)
        _set_download_job(
            download_id,
            status="failed",
            error=message,
            completedAt=time.time(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.error("POST /huggingface/download -> 500 (%.1fms): %s", elapsed_ms, message)


def _resolve_key(request: Request, body: Optional[Dict[str, Any]] = None) -> Optional[str]:
    key = civ._env_key() or civ._load_config_key()
    if key:
        return key

    hdr = request.headers.get("x-civitai-key")
    if hdr:
        return hdr

    if body and isinstance(body, dict):
        bk = body.get("apiKey")
        if bk:
            return bk

    return None


def _resolve_search_key(request: Request) -> Optional[str]:
    return (
        os.environ.get("CIVITAI_SEARCH_KEY")
        or request.headers.get("x-civitai-search-key")
    )


def _resolve_huggingface_token(request: Request, body: Optional[Dict[str, Any]] = None) -> Optional[str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or request.headers.get("x-huggingface-token")
    )
    if token:
        return token
    if body and isinstance(body, dict):
        value = body.get("token")
        if value:
            return str(value)
    return None


def _created_at_range(request: Request):
    qp = request.query_params
    created_from = qp.get("createdFrom")
    created_to = qp.get("createdTo")
    combined = qp.get("createdAt")
    if combined and ":" in combined:
        range_from, range_to = combined.split(":", 1)
        created_from = created_from or range_from
        created_to = created_to or range_to
    return created_from, created_to


def _integer_csv(value: str) -> list[int]:
    values = []
    for part in str(value or "").split(","):
        clean = part.strip()
        if clean:
            values.append(int(clean))
    return values


@APP.get("/health")
def health():
    return {"status": "ok", "mode": "built-in", "standalone_port": False}


def _resolve_imported_gallery_file(relative_path: str) -> Path:
    root = gallery.configured_directory()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HTTPException(status_code=403, detail="Path is outside the gallery directory") from error
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.suffix.lower() not in gallery.IMAGE_EXTENSIONS
    ):
        raise HTTPException(status_code=404, detail="Imported image not found")
    return candidate


@APP.get("/civitai/imported/{relative_path:path}")
async def civitai_imported_image(relative_path: str):
    path = _resolve_imported_gallery_file(relative_path)
    return FileResponse(path, media_type=gallery.file_media_type(path))


@APP.post("/civitai/import-image")
async def civitai_import_image(request: Request):
    payload = await request.json()
    image = payload.get("image") if isinstance(payload, dict) else None
    if not isinstance(image, dict):
        raise HTTPException(status_code=400, detail="Missing CivitAI image payload")

    source_url = str(image.get("url") or "").strip()
    parsed = urlparse(source_url)
    allowed_hosts = {"image.civitai.com", "civitai.com", "www.civitai.com"}
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise HTTPException(status_code=400, detail="Image URL must use an official CivitAI HTTPS host")

    headers = {"Accept": "image/*"}
    api_key = _resolve_key(request, payload)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    root = gallery.configured_directory()
    root.mkdir(parents=True, exist_ok=True)
    image_id = str(image.get("id") or uuid4().hex)
    safe_id = "".join(character for character in image_id if character.isalnum() or character in "-_")[:80]
    if not safe_id:
        safe_id = uuid4().hex

    temporary = root / f".civitai-{safe_id}-{uuid4().hex}.part"
    try:
        with civ.requests.get(source_url, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            final = urlparse(response.url)
            if final.scheme != "https" or final.hostname not in allowed_hosts:
                raise ValueError("CivitAI image redirected to an unsupported host")
            content_type = str(response.headers.get("content-type") or "").split(";", 1)[0].lower()
            extension = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
                "image/bmp": ".bmp",
            }.get(content_type)
            if extension is None:
                raise ValueError(f"Unsupported CivitAI image type: {content_type or 'unknown'}")
            total = 0
            maximum = 100 * 1024 * 1024
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > maximum:
                        raise ValueError("CivitAI image exceeds the 100 MB import limit")
                    output.write(chunk)

        if gallery.Image is not None:
            with gallery.Image.open(temporary) as opened:
                opened.verify()

        destination = root / f"civitai-{safe_id}{extension}"
        if destination.exists():
            destination = root / f"civitai-{safe_id}-{uuid4().hex[:8]}{extension}"
        os.replace(temporary, destination)
        relative = destination.relative_to(root).as_posix()
        sidecar = destination.with_suffix(destination.suffix + ".civitai.json")
        sidecar.write_text(
            json.dumps({"source": "civitai", "image": image}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "relativePath": relative,
            "galleryUrl": f"/gallery/file/{quote(relative, safe='/')}",
            "editorUrl": f"/civitai-api/civitai/imported/{quote(relative, safe='/')}",
            "metadata": image.get("meta") or {},
            "size": destination.stat().st_size,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=_safe_error(error)) from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


@APP.get("/civitai/me")
async def civitai_me(request: Request):
    start = time.perf_counter()
    try:
        api_key = _resolve_key(request)
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="CivitAI API token is required.",
            )

        result = civ._me_civitai(api_key)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("GET /civitai/me -> 200 (%.1fms)", elapsed_ms)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        civ._log(f"token verification error: {exc}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        upstream_response = getattr(exc, "response", None)
        upstream_status = getattr(upstream_response, "status_code", None)
        if upstream_status in (401, 403):
            _logger.warning(
                "GET /civitai/me -> %s (%.1fms)",
                upstream_status,
                elapsed_ms,
            )
            raise HTTPException(
                status_code=upstream_status,
                detail="CivitAI API token was rejected.",
            )
        message = _safe_error(exc)
        _logger.error("GET /civitai/me -> 500 (%.1fms): %s", elapsed_ms, message)
        raise HTTPException(status_code=500, detail=message)


@APP.get("/civitai/tags")
async def civitai_tags(request: Request):
    start = time.perf_counter()
    try:
        qp = request.query_params
        result = civ._tags_civitai(
            limit=int(qp.get("limit", "20") or "20"),
            page=max(1, int(qp.get("page", "1") or "1")),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("GET /civitai/tags -> 200 (%.1fms)", elapsed_ms)
        return result
    except Exception as exc:
        civ._log(f"tags error: {exc}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        message = _safe_error(exc)
        _logger.error("GET /civitai/tags -> 500 (%.1fms): %s", elapsed_ms, message)
        raise HTTPException(status_code=500, detail=message)


@APP.get("/civitai/creators")
async def civitai_creators(request: Request):
    start = time.perf_counter()
    try:
        qp = request.query_params
        result = civ._creators_civitai(
            query=qp.get("query"),
            limit=int(qp.get("limit", "20") or "20"),
            page=max(1, int(qp.get("page", "1") or "1")),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("GET /civitai/creators -> 200 (%.1fms)", elapsed_ms)
        return result
    except Exception as exc:
        civ._log(f"creators error: {exc}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.error(
            "GET /civitai/creators -> 500 (%.1fms): %s",
            elapsed_ms,
            _safe_error(exc),
        )
        raise HTTPException(status_code=500, detail=_safe_error(exc))


@APP.get("/civitai/enums")
async def civitai_enums():
    start = time.perf_counter()
    try:
        result = civ._enums_civitai()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("GET /civitai/enums -> 200 (%.1fms)", elapsed_ms)
        return result
    except Exception as exc:
        civ._log(f"enums error: {exc}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        message = _safe_error(exc)
        _logger.error("GET /civitai/enums -> 500 (%.1fms): %s", elapsed_ms, message)
        raise HTTPException(status_code=500, detail=message)


@APP.get("/civitai/search")
async def civitai_search(request: Request):
    start = time.perf_counter()
    try:
        qp = request.query_params

        query = qp.get("query", "") or ""
        page = int(qp.get("page", "1") or "1")
        cursor = qp.get("cursor")

        nsfw = civ._to_bool(qp.get("nsfw"))
        sort = qp.get("sort")
        period = qp.get("period")
        limit = int(qp.get("limit", "20") or "20")
        types = qp.get("types")
        base_models = qp.get("baseModels")

        model_id = civ._to_int(qp.get("modelId"))
        mvid = civ._to_int(qp.get("modelVersionId"))
        vid = civ._to_int(qp.get("versionId"))
        model_version_id = mvid if mvid is not None else vid
        model_hash = qp.get("hash")

        urn_model_id, urn_ver_id, _, _ = civ._parse_urn(query)
        if urn_model_id is not None:
            model_id = urn_model_id
            model_version_id = urn_ver_id
            query = ""

        api_key = _resolve_key(request)

        result = civ._search_civitai(
            query=query,
            page=max(1, page),
            api_key=api_key,
            cursor=cursor,
            nsfw=nsfw,
            sort=sort,
            period=period,
            limit=limit,
            types=types,
            base_models=base_models,
            model_id=model_id,
            model_version_id=model_version_id,
            model_hash=model_hash,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("GET /civitai/search -> 200 (%.1fms)", elapsed_ms)
        return result

    except Exception as exc:
        civ._log(f"search error: {exc}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        message = _safe_error(exc)
        _logger.error("GET /civitai/search -> 500 (%.1fms): %s", elapsed_ms, message)
        raise HTTPException(status_code=500, detail=message)


@APP.get("/civitai/lookup/hash/{model_hash}")
async def civitai_lookup_hash(model_hash: str, request: Request):
    try:
        result = civ._get_model_version_by_hash(
            model_hash,
            _resolve_key(request),
        )
        return {"ok": True, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_safe_error(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error(exc))


@APP.get("/civitai/lookup/mini/{model_version_id}")
async def civitai_lookup_mini(model_version_id: int, request: Request):
    try:
        result = civ._get_model_version_mini(
            model_version_id,
            _resolve_key(request),
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error(exc))


@APP.post("/civitai/lookup/hashes")
async def civitai_lookup_hashes(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        hashes = payload.get("hashes") or []
        if not isinstance(hashes, list):
            raise ValueError("hashes must be a JSON array")
        result = civ._model_versions_by_hashes(
            [str(value) for value in hashes],
            ids_only=bool(payload.get("idsOnly")),
            api_key=_resolve_key(request),
        )
        return {"ok": True, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_safe_error(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error(exc))


@APP.get("/civitai/collections")
async def civitai_collections(request: Request):
    try:
        qp = request.query_params
        result = civ._collections_civitai(
            query=qp.get("query"),
            sort=qp.get("sort"),
            limit=int(qp.get("limit", "20") or "20"),
            page=max(1, int(qp.get("page", "1") or "1")),
            api_key=_resolve_key(request),
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error(exc))


@APP.get("/civitai/collections/{collection_id}")
async def civitai_collection(collection_id: int, request: Request):
    try:
        result = civ._collection_by_id(
            collection_id,
            _resolve_key(request),
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error(exc))


@APP.get("/civitai/permissions")
async def civitai_permissions(request: Request):
    try:
        qp = request.query_params
        entity_ids = _integer_csv(qp.get("entityIds", "") or "")
        user_id = civ._to_int(qp.get("userId"))
        result = civ._permissions_check(
            entity_ids,
            user_id=user_id,
            api_key=_resolve_key(request),
        )
        return {"ok": True, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_safe_error(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error(exc))


@APP.get("/civitai/images")
async def civitai_images(request: Request):
    start = time.perf_counter()
    try:
        qp = request.query_params

        query = qp.get("query", "") or ""
        page = int(qp.get("page", "1") or "1")
        cursor = qp.get("cursor")
        limit = int(qp.get("limit", "50") or "50")

        nsfw = civ._to_bool(qp.get("nsfw"))
        sort = qp.get("sort")
        model_sort = qp.get("modelSort")
        period = qp.get("period")

        model_id = civ._to_int(qp.get("modelId"))
        mvid = civ._to_int(qp.get("modelVersionId"))
        vid = civ._to_int(qp.get("versionId"))
        model_version_id = mvid if mvid is not None else vid

        post_id = civ._to_int(qp.get("postId"))
        username = qp.get("username")

        urn_model_id, urn_ver_id, _, _ = civ._parse_urn(query)
        if urn_model_id is not None:
            model_id = urn_model_id
            model_version_id = urn_ver_id

        api_key = _resolve_key(request)

        result = civ._images_civitai(
            query=query,
            api_key=api_key,
            limit=limit,
            page=page,
            cursor=cursor,
            nsfw=nsfw,
            sort=sort,
            model_sort=model_sort,
            period=period,
            model_id=model_id,
            model_version_id=model_version_id,
            post_id=post_id,
            username=username,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("GET /civitai/images -> 200 (%.1fms)", elapsed_ms)
        return result

    except Exception as exc:
        civ._log(f"images error: {exc}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        message = _safe_error(exc)
        _logger.error("GET /civitai/images -> 500 (%.1fms): %s", elapsed_ms, message)
        raise HTTPException(status_code=500, detail=message)


@APP.get("/civitai/global-images")
async def civitai_global_images(request: Request):
    start = time.perf_counter()
    try:
        qp = request.query_params
        search_key = _resolve_search_key(request)

        created_from, created_to = _created_at_range(request)
        query = qp.get("query") or qp.get("q") or ""
        limit = int(qp.get("limit", "51") or "51")
        page = max(1, int(qp.get("page", "1") or "1"))
        nsfw = civ._to_bool(qp.get("nsfw"))
        sort = qp.get("sort")
        period = qp.get("period")
        post_id = civ._to_int(qp.get("postId"))
        username = qp.get("username") or qp.get("user") or qp.get("users")

        # The public images API is authoritative for a normal gallery browse and
        # implements the image-specific sort and period controls itself.  Use
        # images_v6 only when its full-text/faceted search is actually needed.
        index_filters = {
            "query": query,
            "tag": qp.get("tag") or qp.get("tags"),
            "technique": qp.get("technique") or qp.get("techniques"),
            "tool": qp.get("tool") or qp.get("tools"),
            "aspect_ratio": qp.get("aspectRatio"),
            "base_model": qp.get("baseModel"),
            "media_type": qp.get("mediaType") or qp.get("type"),
            "created_from": created_from,
            "created_to": created_to,
        }
        try:
            if any(str(value or "").strip() for value in index_filters.values()):
                result = civ._global_images_civitai(
                    query=query,
                    search_key=search_key,
                    limit=limit,
                    page=page,
                    nsfw=nsfw,
                    tag=index_filters["tag"],
                    technique=index_filters["technique"],
                    tool=index_filters["tool"],
                    aspect_ratio=index_filters["aspect_ratio"],
                    base_model=index_filters["base_model"],
                    media_type=index_filters["media_type"],
                    username=username,
                    created_from=created_from,
                    created_to=created_to,
                    sort=sort,
                    period=period,
                )
            else:
                result = civ._images_civitai(
                    query="",
                    api_key=_resolve_key(request),
                    limit=limit,
                    page=page,
                    nsfw=nsfw,
                    sort=sort,
                    period=period,
                    post_id=post_id,
                    username=username,
                )
                result.setdefault("metadata", {})["searchBackend"] = "public-images-api"
        except Exception as search_error:
            _logger.warning(
                "CivitAI images_v6 search unavailable; using public API fallback: %s",
                _safe_error(search_error),
            )
            fallback_query = query or qp.get("tag") or qp.get("technique") or qp.get("tool") or ""
            result = civ._images_civitai(
                query=fallback_query,
                api_key=_resolve_key(request),
                limit=limit,
                page=page,
                nsfw=nsfw,
                sort=sort,
                model_sort=civ._map_image_sort_to_model_sort(sort),
                period=period,
                post_id=post_id,
                username=username,
            )
            result.setdefault("metadata", {})["searchBackend"] = "public-api-fallback"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("GET /civitai/global-images -> 200 (%.1fms)", elapsed_ms)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        civ._log(f"global image search error: {exc}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.error(
            "GET /civitai/global-images -> 500 (%.1fms): %s",
            elapsed_ms,
            _safe_error(exc),
        )
        raise HTTPException(status_code=500, detail=_safe_error(exc))


@HUGGINGFACE_APP.get("/search")
async def huggingface_search(request: Request):
    qp = request.query_params
    try:
        return hf.search(
            query=qp.get("query") or "",
            sort=qp.get("sort") or "Most Downloaded",
            page=max(1, int(qp.get("page") or 1)),
            limit=max(1, min(50, int(qp.get("limit") or 20))),
            token=_resolve_huggingface_token(request),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=_safe_error(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_safe_error(exc)) from exc
    except Exception as exc:
        _logger.error("GET /huggingface/search -> 502: %s", _safe_error(exc))
        raise HTTPException(status_code=502, detail=_safe_error(exc)) from exc


@HUGGINGFACE_APP.post("/download")
async def huggingface_download(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    target = payload.get("model") or {}
    file_choice = payload.get("file") or {}
    if not isinstance(target, dict) or not isinstance(file_choice, dict) or not target or not file_choice:
        raise HTTPException(status_code=400, detail="model and file payloads are required")

    try:
        # Resolve and validate the destination before scheduling the background job.
        hf.download_target(target, file_choice)
        download_id = uuid4().hex
        _set_download_job(
            download_id,
            status="queued",
            downloadedBytes=0,
            totalBytes=None,
            percent=0.0,
            createdAt=time.time(),
        )
        worker = threading.Thread(
            target=_huggingface_download_worker,
            args=(download_id, target, file_choice, _resolve_huggingface_token(request, payload)),
            daemon=True,
        )
        worker.start()
        return {"ok": True, "downloadId": download_id, "status": "queued"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_safe_error(exc)) from exc
    except Exception as exc:
        _logger.error("POST /huggingface/download -> 500: %s", _safe_error(exc))
        raise HTTPException(status_code=500, detail=_safe_error(exc)) from exc


@HUGGINGFACE_APP.get("/download/{download_id}")
async def huggingface_download_status(download_id: str):
    job = _get_download_job(download_id)
    if not job:
        raise HTTPException(status_code=404, detail="download not found")
    return {"ok": True, "downloadId": download_id, **job}


@APP.post("/civitai/download")
async def civitai_download(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    api_key = _resolve_key(request, payload)

    target = payload.get("model") or {}
    file_choice = payload.get("file") or {}
    if not target:
        _logger.warning("POST /civitai/download -> 400 (missing model payload)")
        raise HTTPException(status_code=400, detail="model payload required")

    try:
        if not api_key:
            raise HTTPException(status_code=400, detail="CivitAI API key required")
        download_id = uuid4().hex
        _set_download_job(
            download_id,
            status="queued",
            downloadedBytes=0,
            totalBytes=None,
            percent=0.0,
            createdAt=time.time(),
        )
        worker = threading.Thread(
            target=_download_worker,
            args=(download_id, target, file_choice, payload, api_key),
            daemon=True,
        )
        worker.start()
        return {"ok": True, "downloadId": download_id, "status": "queued"}

    except HTTPException:
        raise
    except Exception as exc:
        civ._log(f"download error: {exc}")
        message = _safe_error(exc)
        _logger.error("POST /civitai/download -> 500: %s", message)
        raise HTTPException(status_code=500, detail=message)


@APP.get("/civitai/download/{download_id}")
async def civitai_download_status(download_id: str):
    job = _get_download_job(download_id)
    if not job:
        raise HTTPException(status_code=404, detail="download not found")
    return {"ok": True, "downloadId": download_id, **job}


router = APP
huggingface_router = HUGGINGFACE_APP
