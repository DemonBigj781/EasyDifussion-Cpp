"""CivitAI downloader server helper module for Easy Diffusion.

Framework-agnostic helpers:
- search models (cursor pagination when query is provided)
- images (cursor pagination with full generation metadata)
- download

Also supports URN lookup:
  urn:air:sdxl:lora:civitai:MODELID@VERSIONID

Important behaviors (matches current CivitAI API):
- When `query` is non-empty, CivitAI rejects `page`; you must use cursor pagination.
- `sort` values include spaces (e.g. "Most Reactions"); do NOT strip spaces.
"""
from __future__ import annotations

import os
import re
import time
import copy
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs

import requests
from requests.exceptions import RequestException, Timeout, HTTPError

PLUGIN_NAME = "civitai"
PLUGIN_VERSION = "0.4.0"

ALLOWED_SORT_MODELS = {"Highest Rated", "Most Downloaded", "Newest"}
ALLOWED_SORT_IMAGES = {"Most Reactions", "Most Comments", "Newest"}

# plugins/server/civitai/plugin_entry.py -> parents[3] is /easy-diffusion
ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config.yaml"

_CONFIG_KEY: Optional[str] = None
_REQUEST_CACHE_TTL_SECONDS = 120
_REQUEST_CACHE_MAX_ENTRIES = 512
_REQUEST_CACHE: Dict[Tuple[str, str, Tuple[Tuple[str, str], ...], Optional[str]], Tuple[float, Dict[str, Any]]] = {}
_REQUEST_CACHE_LOCK = threading.Lock()
_BRIDGED_IMAGE_CACHE_TTL_SECONDS = 180
_BRIDGED_IMAGE_CACHE_MAX_ENTRIES = 128
_BRIDGED_IMAGE_CACHE: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = {}
_BRIDGED_IMAGE_CACHE_LOCK = threading.Lock()

_URN_RE = re.compile(
    r"^urn:air:(?P<base>[^:]+):(?P<mtype>[^:]+):civitai:(?P<model>\d+)(?:@(?P<ver>\d+))?$",
    re.IGNORECASE,
)

# Base models that are distributed as a diffusion/transformer component rather
# than a self-contained Stable Diffusion checkpoint. CivitAI still labels many
# of these model pages as "Checkpoint", so model type alone is not sufficient
# to choose the install directory.
_MODULAR_BASE_MODEL_PREFIXES = (
    "ace audio",
    "anima",
    "auraflow",
    "boogu",
    "chroma",
    "cogvideox",
    "ernie",
    "flux",
    "grok",
    "happyhorse",
    "hidream",
    "hunyuan",
    "ideogram",
    "imagen",
    "kling",
    "kolors",
    "krea",
    "lens",
    "ltxv",
    "lumina",
    "mageflow",
    "mai",
    "minimax",
    "mochi",
    "nano banana",
    "odor",
    "openai",
    "pixart",
    "polygen",
    "qwen",
    "reve",
    "seedance",
    "seedream",
    "sora",
    "svd",
    "tripo",
    "veo",
    "vidu",
    "wan ",
    "zimage",
)


def _env_key() -> Optional[str]:
    return (
        os.environ.get("CIVITAI_TOKEN")
        or os.environ.get("CIVIT_API_KEY")
        or os.environ.get("CIVITAI_KEY")
    )


def _log(msg: str) -> None:
    print(f"[CivitAI] {msg}", flush=True)


def _safe_slug(text: str, fallback: str = "model") -> str:
    text = text or ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return slug or fallback


def _load_config_key() -> Optional[str]:
    global _CONFIG_KEY
    env_key = _env_key()
    if env_key:
        return env_key
    if _CONFIG_KEY is not None:
        return _CONFIG_KEY

    key = None
    if CONFIG_PATH.exists():
        try:
            import yaml  # type: ignore
        except Exception:
            yaml = None
        try:
            if yaml:
                with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
                key = cfg.get("civitai_api_key") or cfg.get("civitai-key")
            else:
                with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip().startswith("civitai_api_key"):
                            _, val = line.split(":", 1)
                            key = val.strip().strip('"').strip("'")
                            break
        except Exception as err:
            _log(f"Failed to read config.yaml for civitai_api_key: {err}")

    _CONFIG_KEY = key
    return key


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_cursor(c: Optional[str]) -> Optional[str]:
    if not c:
        return None
    if isinstance(c, str) and c.startswith("http"):
        parsed = urlparse(c)
        qs = parse_qs(parsed.query)
        return qs.get("cursor", [None])[0] or None
    return c


def _next_page_number(next_page: Any, current_page: int) -> Optional[int]:
    if next_page is None:
        return None
    if isinstance(next_page, int):
        return next_page if next_page > current_page else None
    if isinstance(next_page, str):
        s = next_page.strip()
        if not s:
            return None
        if s.isdigit():
            n = int(s)
            return n if n > current_page else None
        if s.startswith("http"):
            try:
                parsed = urlparse(s)
                qs = parse_qs(parsed.query)
                p = qs.get("page", [None])[0]
                if p and str(p).isdigit():
                    n = int(str(p))
                    return n if n > current_page else None
            except Exception:
                return None
    return None


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _image_sort_key(item: Dict[str, Any], sort: Optional[str]) -> tuple[float, float]:
    stats = item.get("stats") or {}
    if sort == "Most Reactions":
        val = (
            _to_float(stats.get("reactionCount"))
            + _to_float(stats.get("likeCount"))
            + _to_float(stats.get("heartCount"))
            + _to_float(stats.get("laughCount"))
            + _to_float(stats.get("cryCount"))
        )
        return (val, _to_float(item.get("id")))
    if sort == "Most Comments":
        val = _to_float(stats.get("commentCount"))
        return (val, _to_float(item.get("id")))

    # Default / Newest fallback.
    created = item.get("createdAt")
    if isinstance(created, str):
        try:
            ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            return (ts, _to_float(item.get("id")))
        except Exception:
            pass
    return (_to_float(item.get("id")), 0.0)


def _sort_unified_images(items: list[Dict[str, Any]], sort: Optional[str]) -> list[Dict[str, Any]]:
    if not items:
        return items
    if sort not in ALLOWED_SORT_IMAGES:
        sort = "Newest"
    return sorted(items, key=lambda x: _image_sort_key(x, sort), reverse=True)


def _map_image_sort_to_model_sort(sort: Optional[str]) -> Optional[str]:
    # Model and image sort enums differ; use closest model sort for bridge query.
    if sort == "Most Reactions":
        return "Highest Rated"
    if sort == "Most Comments":
        return "Most Downloaded"
    if sort == "Newest":
        return "Newest"
    return None


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _cache_key(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]],
    api_key: Optional[str],
) -> Tuple[str, str, Tuple[Tuple[str, str], ...], Optional[str]]:
    frozen_params: Tuple[Tuple[str, str], ...] = tuple(
        sorted((str(k), str(v)) for k, v in (params or {}).items())
    )
    return (method.upper(), url, frozen_params, api_key)


def _cache_get(
    key: Tuple[str, str, Tuple[Tuple[str, str], ...], Optional[str]]
) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _REQUEST_CACHE_LOCK:
        entry = _REQUEST_CACHE.get(key)
        if not entry:
            return None
        ts, data = entry
        if now - ts > _REQUEST_CACHE_TTL_SECONDS:
            _REQUEST_CACHE.pop(key, None)
            return None
        return copy.deepcopy(data)


def _cache_set(
    key: Tuple[str, str, Tuple[Tuple[str, str], ...], Optional[str]],
    data: Dict[str, Any],
) -> None:
    now = time.time()
    with _REQUEST_CACHE_LOCK:
        _REQUEST_CACHE[key] = (now, copy.deepcopy(data))
        if len(_REQUEST_CACHE) <= _REQUEST_CACHE_MAX_ENTRIES:
            return
        # Evict oldest entries first.
        for k, _ in sorted(_REQUEST_CACHE.items(), key=lambda item: item[1][0])[
            : max(1, len(_REQUEST_CACHE) - _REQUEST_CACHE_MAX_ENTRIES)
        ]:
            _REQUEST_CACHE.pop(k, None)


def _bridge_cache_key(
    *,
    query: Optional[str],
    api_key: Optional[str],
    limit: int,
    page: int,
    cursor: Optional[str],
    nsfw: Optional[bool],
    sort: Optional[str],
    model_sort: Optional[str],
    period: Optional[str],
    model_id: Optional[int],
    model_version_id: Optional[int],
    post_id: Optional[int],
    username: Optional[str],
) -> Tuple[Any, ...]:
    return (
        (query or "").strip(),
        api_key,
        int(limit),
        int(page),
        _normalize_cursor(cursor),
        nsfw,
        sort,
        model_sort,
        period,
        model_id,
        model_version_id,
        post_id,
        username or "",
    )


def _bridge_cache_get(key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _BRIDGED_IMAGE_CACHE_LOCK:
        entry = _BRIDGED_IMAGE_CACHE.get(key)
        if not entry:
            return None
        ts, data = entry
        if now - ts > _BRIDGED_IMAGE_CACHE_TTL_SECONDS:
            _BRIDGED_IMAGE_CACHE.pop(key, None)
            return None
        return copy.deepcopy(data)


def _bridge_cache_set(key: Tuple[Any, ...], data: Dict[str, Any]) -> None:
    now = time.time()
    with _BRIDGED_IMAGE_CACHE_LOCK:
        _BRIDGED_IMAGE_CACHE[key] = (now, copy.deepcopy(data))
        if len(_BRIDGED_IMAGE_CACHE) <= _BRIDGED_IMAGE_CACHE_MAX_ENTRIES:
            return
        for k, _ in sorted(_BRIDGED_IMAGE_CACHE.items(), key=lambda item: item[1][0])[
            : max(1, len(_BRIDGED_IMAGE_CACHE) - _BRIDGED_IMAGE_CACHE_MAX_ENTRIES)
        ]:
            _BRIDGED_IMAGE_CACHE.pop(k, None)


def _request_with_retries(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    json_body: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> Any:
    cacheable = method.upper() == "GET"
    key = _cache_key(method, url, params, api_key) if cacheable else None
    if cacheable and key is not None:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    headers = _auth_headers(api_key)
    if extra_headers:
        headers.update(extra_headers)
    for attempt in range(max_retries + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=json_body,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json() or {}
            if cacheable and key is not None:
                _cache_set(key, data)
            return data

        except Timeout:
            if attempt < max_retries:
                wait_time = min(2**attempt, 30)
                _log(f"Request timeout, retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise

        except HTTPError as err:
            status_code = err.response.status_code if err.response is not None else None
            if status_code == 429 and attempt < max_retries:
                retry_after = int(err.response.headers.get("Retry-After", 60))
                _log(f"Rate limited, retrying in {retry_after}s... ({attempt+1}/{max_retries})")
                time.sleep(retry_after)
                continue
            if status_code is not None and status_code >= 500 and attempt < max_retries:
                wait_time = min(2**attempt, 30)
                _log(f"Upstream {status_code}, retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise

        except RequestException:
            if attempt < max_retries:
                wait_time = min(2**attempt, 30)
                _log(f"Request failed, retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise


def _to_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _to_bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    v = str(x).strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return None


def _parse_urn(s: str) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    if not isinstance(s, str):
        return (None, None, None, None)
    m = _URN_RE.match(s.strip())
    if not m:
        return (None, None, None, None)
    model_id = int(m.group("model"))
    ver = m.group("ver")
    version_id = int(ver) if ver else None
    return (model_id, version_id, m.group("base"), m.group("mtype"))


def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        out[k] = v
    return out


# ----------------------------
# CivitAI API helpers
# ----------------------------

def _trim_model_item(item: Dict[str, Any], version_id: Optional[int] = None) -> Dict[str, Any]:
    versions = item.get("modelVersions") or []

    chosen_ver: Dict[str, Any] = {}
    if version_id is not None:
        for v in versions:
            try:
                if int(v.get("id")) == int(version_id):
                    chosen_ver = v
                    break
            except Exception:
                continue
    if not chosen_ver:
        chosen_ver = versions[0] if versions else {}

    files = chosen_ver.get("files") or []
    images = chosen_ver.get("images") or []

    thumb_url = None
    full_url = None
    if images:
        thumb_url = images[0].get("thumbnailUrl") or images[0].get("url")
        full_url = images[0].get("url")

    creator = item.get("creator") or item.get("user") or item.get("createdBy") or None

    mv_id = chosen_ver.get("id")
    mv_base = chosen_ver.get("baseModel")

    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "type": item.get("type"),
        "nsfw": item.get("nsfw"),
        "stats": item.get("stats"),
        "creator": creator,
        "modelVersion": {
            "id": mv_id,
            "name": chosen_ver.get("name"),
            "baseModel": mv_base,
            "description": chosen_ver.get("description"),
            "files": [
                {
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "type": f.get("type"),
                    "sizeKB": f.get("sizeKB"),
                    "downloadUrl": f.get("downloadUrl"),
                    "primary": f.get("primary"),
                }
                for f in files
            ],
            "images": [{"url": full_url, "thumbnailUrl": thumb_url}] if thumb_url else [],
        },
    }


def _trim_image_item(img: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": img.get("id"),
        "url": img.get("url"),
        "hash": img.get("hash"),
        "width": img.get("width"),
        "height": img.get("height"),
        "nsfw": img.get("nsfw"),
        "createdAt": img.get("createdAt"),
        "postId": img.get("postId"),
        "stats": img.get("stats") or {},
        "meta": img.get("meta") or {},
        "username": img.get("username"),
    }


def _global_image_urls(hit: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    media_id = str(hit.get("url") or "").strip()
    if not media_id:
        return (None, None)
    if media_id.startswith(("http://", "https://")):
        return (media_id, media_id)

    safe_media_id = re.sub(r"[^A-Za-z0-9-]", "", media_id)
    if not safe_media_id:
        return (None, None)
    mime_type = str(hit.get("mimeType") or "image/jpeg").lower()
    extension = {
        "image/avif": "avif",
        "image/gif": "gif",
        "image/jpeg": "jpeg",
        "image/png": "png",
        "image/webp": "webp",
        "video/mp4": "mp4",
        "video/webm": "webm",
    }.get(mime_type, "jpeg")
    base = (
        "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/"
        f"{safe_media_id}"
    )
    filename = f"{safe_media_id}.{extension}"
    original_url = f"{base}/original=true/{filename}"
    if mime_type.startswith("video/"):
        return (original_url, original_url)
    return (
        original_url,
        f"{base}/width=450/{filename}",
    )


def _global_images_civitai(
    *,
    query: str,
    search_key: str,
    limit: int = 51,
    page: int = 1,
    nsfw: Optional[bool] = None,
    tag: Optional[str] = None,
    technique: Optional[str] = None,
    tool: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    base_model: Optional[str] = None,
    media_type: Optional[str] = None,
    username: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
) -> Dict[str, Any]:
    clean_search_key = str(search_key or "").strip()
    if not clean_search_key:
        raise ValueError("Global image search key is required")

    limit = max(1, min(int(limit or 51), 100))
    page = max(1, int(page or 1))
    filters: list[Any] = []

    def exact_filter(field: str, value: Optional[str]) -> None:
        clean_value = str(value or "").strip()
        if clean_value:
            escaped = clean_value.replace("\\", "\\\\").replace('"', '\\"')
            filters.append(f'"{field}"="{escaped}"')

    exact_filter("tagNames", tag)
    exact_filter("techniqueNames", technique)
    exact_filter("toolNames", tool)
    exact_filter("aspectRatio", aspect_ratio)
    exact_filter("baseModel", base_model)
    exact_filter("type", media_type)
    exact_filter("user.username", username)

    def unix_seconds(value: Optional[str]) -> Optional[int]:
        try:
            timestamp = float(str(value or "").strip())
        except (TypeError, ValueError):
            return None
        if timestamp <= 0:
            return None
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return int(timestamp)

    created_from_unix = unix_seconds(created_from)
    created_to_unix = unix_seconds(created_to)
    if created_from_unix is not None:
        filters.append(f"createdAtUnix >= {created_from_unix}")
    if created_to_unix is not None:
        filters.append(f"createdAtUnix <= {created_to_unix}")

    safety_filter = (
        "(poi != true OR user.username = DemonBigj781) AND "
        "(minor != true) AND "
        "(NOT (nsfwLevel IN [4, 8, 16, 32] AND baseModel IN "
        "['SD 3', 'SD 3.5', 'SD 3.5 Medium', 'SD 3.5 Large', "
        "'SD 3.5 Large Turbo', 'SDXL Turbo', 'SVD', 'SVD XT', "
        "'Stable Cascade', 'Ideogram 4.0']))"
    )
    if nsfw:
        safety_filter += (
            " AND (nsfwLevel=1 OR nsfwLevel=2 OR nsfwLevel=4 "
            "OR nsfwLevel=8 OR nsfwLevel=16)"
        )
    else:
        safety_filter += " AND (nsfwLevel=1 OR nsfwLevel=2)"
    filters.append(safety_filter)

    payload = {
        "queries": [
            {
                "q": str(query or ""),
                "indexUid": "images_v6",
                "facets": [
                    "aspectRatio",
                    "baseModel",
                    "createdAtUnix",
                    "tagNames",
                    "techniqueNames",
                    "toolNames",
                    "type",
                    "user.username",
                ],
                "attributesToHighlight": [],
                "highlightPreTag": "__ais-highlight__",
                "highlightPostTag": "__/ais-highlight__",
                "limit": limit,
                "offset": (page - 1) * limit,
                "filter": filters,
            }
        ]
    }
    data = _request_with_retries(
        "POST",
        "https://search-new.civitai.com/multi-search",
        api_key=clean_search_key,
        json_body=payload,
        extra_headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://civitai.red",
            "Referer": "https://civitai.red/",
            "X-Meilisearch-Client": (
                "Meilisearch instant-meilisearch (v0.13.5) ; "
                "Meilisearch JavaScript (v0.34.0)"
            ),
        },
    )
    results = data.get("results", []) or []
    first = results[0] if results and isinstance(results[0], dict) else {}
    hits = first.get("hits", []) or []
    items = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        full_url, thumbnail_url = _global_image_urls(hit)
        user = hit.get("user") or {}
        items.append(
            {
                "id": hit.get("id"),
                "url": full_url,
                "thumbnailUrl": thumbnail_url,
                "hash": hit.get("hash"),
                "width": hit.get("width"),
                "height": hit.get("height"),
                "nsfw": hit.get("nsfwLevel"),
                "createdAt": hit.get("createdAt"),
                "postId": hit.get("postId"),
                "stats": hit.get("stats") or {},
                "meta": hit.get("metadata") or {},
                "username": user.get("username") if isinstance(user, dict) else None,
                "mediaType": hit.get("type"),
                "mimeType": hit.get("mimeType"),
                "modelVersionId": hit.get("modelVersionId"),
            }
        )

    estimated_total = first.get("estimatedTotalHits")
    try:
        total = max(0, int(estimated_total))
    except (TypeError, ValueError):
        total = len(items)
    pages = max(1, (total + limit - 1) // limit)
    return {
        "ok": True,
        "items": items,
        "metadata": {
            "page": page,
            "pages": pages,
            "total": total,
            "returned": len(items),
            "nextPage": page + 1 if page < pages else None,
            "prevPage": page - 1 if page > 1 else None,
            "processingTimeMs": first.get("processingTimeMs"),
            "facets": first.get("facetDistribution") or {},
        },
    }


def _me_civitai(api_key: str) -> Dict[str, Any]:
    clean_api_key = str(api_key or "").strip()
    if not clean_api_key:
        raise ValueError("CivitAI API token is required")

    data = _request_with_retries(
        "GET",
        "https://civitai.com/api/v1/me",
        api_key=clean_api_key,
    )
    if not isinstance(data, dict):
        raise ValueError("Unexpected CivitAI account response")

    return {
        "ok": True,
        "user": {
            "id": data.get("id"),
            "username": data.get("username"),
            "name": data.get("name"),
            "image": data.get("image"),
        },
    }


def _tags_civitai(limit: int = 20, page: int = 1) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 20), 200))
    page = max(1, int(page or 1))
    data = _request_with_retries(
        "GET",
        "https://civitai.com/api/v1/tags",
        params={"limit": limit, "page": page},
        api_key=None,
    )
    items = []
    for raw in data.get("items", []) or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if name:
            items.append({"name": name, "link": raw.get("link")})

    meta = data.get("metadata", {}) or {}
    return {
        "ok": True,
        "items": items,
        "metadata": {
            "page": meta.get("currentPage", page),
            "pages": meta.get("totalPages"),
            "total": meta.get("totalItems"),
            "nextPage": meta.get("nextPage"),
            "prevPage": meta.get("prevPage"),
        },
    }


def _creators_civitai(
    query: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 20), 200))
    page = max(1, int(page or 1))
    params = _clean_params(
        {
            "query": str(query or "").strip(),
            "limit": limit,
            "page": page,
        }
    )
    data = _request_with_retries(
        "GET",
        "https://civitai.com/api/v1/creators",
        params=params,
        api_key=None,
    )
    items = []
    for raw in data.get("items", []) or []:
        if not isinstance(raw, dict):
            continue
        username = str(raw.get("username") or "").strip()
        if username:
            items.append(
                {
                    "username": username,
                    "modelCount": raw.get("modelCount"),
                    "link": raw.get("link"),
                }
            )

    meta = data.get("metadata", {}) or {}
    return {
        "ok": True,
        "items": items,
        "metadata": {
            "page": meta.get("currentPage", page),
            "pages": meta.get("totalPages"),
            "total": meta.get("totalItems"),
            "hasMore": meta.get("hasMore"),
        },
    }


def _enums_civitai() -> Dict[str, Any]:
    data = _request_with_retries(
        "GET",
        "https://civitai.com/api/v1/enums",
        api_key=None,
    )
    if not isinstance(data, dict):
        raise ValueError("Unexpected enums response")
    return {
        "ok": True,
        "modelTypes": data.get("ModelType") or [],
        "baseModels": data.get("BaseModel") or [],
        "activeBaseModels": data.get("ActiveBaseModel") or [],
        "baseModelTypes": data.get("BaseModelType") or [],
    }


def _get_model_by_id(model_id: int, api_key: Optional[str]) -> Dict[str, Any]:
    url = f"https://civitai.com/api/v1/models/{int(model_id)}"
    data = _request_with_retries("GET", url, api_key=api_key)
    if not isinstance(data, dict):
        raise ValueError("Unexpected model response")
    return data


def _get_model_version_by_id(model_version_id: int, api_key: Optional[str]) -> Dict[str, Any]:
    url = f"https://civitai.com/api/v1/model-versions/{int(model_version_id)}"
    data = _request_with_retries("GET", url, api_key=api_key)
    if not isinstance(data, dict):
        raise ValueError("Unexpected model-version response")
    return data


def _normalize_sha256(value: str) -> str:
    model_hash = str(value or "").strip().upper()
    if not re.fullmatch(r"[A-F0-9]{64}", model_hash):
        raise ValueError("A complete 64-character SHA256 is required")
    return model_hash


def _get_model_version_by_hash(
    model_hash: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    normalized_hash = _normalize_sha256(model_hash)
    url = (
        "https://civitai.com/api/v1/model-versions/by-hash/"
        f"{normalized_hash}"
    )
    data = _request_with_retries("GET", url, api_key=api_key)
    if not isinstance(data, dict):
        raise ValueError("Unexpected hash lookup response")
    return data


def _get_model_version_mini(
    model_version_id: int,
    api_key: Optional[str],
) -> Dict[str, Any]:
    url = (
        "https://civitai.com/api/v1/model-versions/mini/"
        f"{int(model_version_id)}"
    )
    data = _request_with_retries("GET", url, api_key=api_key)
    if not isinstance(data, dict):
        raise ValueError("Unexpected mini model-version response")
    return data


def _model_versions_by_hashes(
    hashes: list[str],
    *,
    ids_only: bool,
    api_key: Optional[str],
) -> Any:
    normalized = [_normalize_sha256(value) for value in hashes]
    if not normalized:
        raise ValueError("At least one SHA256 is required")
    if len(normalized) > 100:
        raise ValueError("At most 100 hashes may be checked at once")
    suffix = "/ids" if ids_only else ""
    return _request_with_retries(
        "POST",
        f"https://civitai.com/api/v1/model-versions/by-hash{suffix}",
        api_key=api_key,
        json_body=normalized,
        extra_headers={"Content-Type": "application/json"},
    )


def _collections_civitai(
    *,
    query: Optional[str],
    sort: Optional[str],
    limit: int,
    page: int,
    api_key: Optional[str],
) -> Dict[str, Any]:
    params = _clean_params(
        {
            "query": str(query or "").strip(),
            "sort": str(sort or "").strip(),
            "limit": max(1, min(int(limit or 20), 100)),
            "page": max(1, int(page or 1)),
        }
    )
    data = _request_with_retries(
        "GET",
        "https://civitai.com/api/v1/collections",
        params=params,
        api_key=api_key,
    )
    if not isinstance(data, dict):
        raise ValueError("Unexpected collections response")
    return data


def _collection_by_id(
    collection_id: int,
    api_key: Optional[str],
) -> Dict[str, Any]:
    data = _request_with_retries(
        "GET",
        f"https://civitai.com/api/v1/collections/{int(collection_id)}",
        api_key=api_key,
    )
    if not isinstance(data, dict):
        raise ValueError("Unexpected collection response")
    return data


def _permissions_check(
    entity_ids: list[int],
    *,
    user_id: Optional[int],
    api_key: Optional[str],
) -> Any:
    normalized_ids = [int(value) for value in entity_ids]
    if not normalized_ids:
        raise ValueError("At least one entity ID is required")
    if len(normalized_ids) > 100:
        raise ValueError("At most 100 entity IDs may be checked at once")
    params: Dict[str, Any] = {
        "entityIds": ",".join(str(value) for value in normalized_ids),
    }
    if user_id is not None:
        params["userId"] = int(user_id)
    return _request_with_retries(
        "GET",
        "https://civitai.com/api/v1/permissions/check",
        params=params,
        api_key=api_key,
    )


def _search_civitai(
    *,
    query: str,
    page: int,
    api_key: Optional[str],
    cursor: Optional[str] = None,
    nsfw: Optional[bool] = None,
    sort: Optional[str] = None,
    period: Optional[str] = None,
    limit: int = 20,
    types: Optional[str] = None,
    base_models: Optional[str] = None,
    model_id: Optional[int] = None,
    model_version_id: Optional[int] = None,
    model_hash: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_version: Optional[Dict[str, Any]] = None
    if model_hash:
        resolved_version = _get_model_version_by_hash(model_hash, api_key)
        model_version_id = _to_int(str(resolved_version.get("id") or ""))
        if model_version_id is None:
            raise ValueError("Hash lookup did not include a model-version ID")

    # Direct lookup: version
    if model_version_id is not None:
        ver = resolved_version or _get_model_version_by_id(model_version_id, api_key)
        parent_id = ver.get("modelId") or ver.get("model") or ver.get("model_id")
        if parent_id is None:
            raise ValueError("modelVersionId lookup did not include modelId")
        model = _get_model_by_id(int(parent_id), api_key)
        item = _trim_model_item(model, version_id=int(model_version_id))
        return {"ok": True, "items": [item], "metadata": {"page": 1, "pages": 1, "total": 1}}

    # Direct lookup: model
    if model_id is not None:
        model = _get_model_by_id(model_id, api_key)
        item = _trim_model_item(model, version_id=None)
        return {"ok": True, "items": [item], "metadata": {"page": 1, "pages": 1, "total": 1}}

    url = "https://civitai.com/api/v1/models"

    params: Dict[str, Any] = {"limit": max(1, min(int(limit or 20), 100))}

    # CivitAI: query => cursor pagination; browsing (no query) => page ok
    norm_cursor = _normalize_cursor(cursor)
    if query:
        params["query"] = query
        if norm_cursor:
            params["cursor"] = norm_cursor
        # DO NOT send page when query is present
    else:
        params["page"] = max(1, int(page or 1))

    if nsfw is not None:
        params["nsfw"] = str(bool(nsfw)).lower()
    if sort:
        if sort in ALLOWED_SORT_MODELS:
            params["sort"] = sort
        else:
            _log(f"Ignoring unsupported model sort: {sort}")
    if period:
        params["period"] = period
    if types:
        params["types"] = types
    if base_models:
        params["baseModels"] = base_models

    params = _clean_params(params)

    try:
        data = _request_with_retries(
            "GET",
            url,
            params=params,
            api_key=api_key,
        )
    except HTTPError as err:
        status_code = err.response.status_code if err.response is not None else None
        if status_code is None or status_code < 500:
            raise

        # CivitAI can intermittently return 5xx for valid model-search URLs.
        # Retry once with relaxed optional filters to reduce backend failures.
        relaxed = dict(params)
        relaxed.pop("sort", None)
        relaxed.pop("period", None)
        relaxed.pop("nsfw", None)
        _log("Model search upstream 5xx; retrying with relaxed filters (no sort/period/nsfw).")
        data = _request_with_retries(
            "GET",
            url,
            params=relaxed,
            api_key=api_key,
        )

    items = [_trim_model_item(x) for x in data.get("items", [])]
    meta = data.get("metadata", {}) or {}

    return {
        "ok": True,
        "items": items,
        "metadata": {
            "page": meta.get("currentPage", page),
            "pages": meta.get("totalPages"),
            "total": meta.get("totalItems"),
            "nextPage": meta.get("nextPage"),
            "prevPage": meta.get("prevPage"),
            "nextCursor": meta.get("nextCursor") or meta.get("nextPageCursor"),
            "prevCursor": meta.get("prevCursor") or meta.get("prevPageCursor"),
        },
    }


def _images_civitai(
    *,
    query: Optional[str],
    api_key: Optional[str],
    limit: int = 50,
    page: int = 1,
    cursor: Optional[str] = None,
    nsfw: Optional[bool] = None,
    sort: Optional[str] = None,
    model_sort: Optional[str] = None,
    period: Optional[str] = None,
    model_id: Optional[int] = None,
    model_version_id: Optional[int] = None,
    post_id: Optional[int] = None,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    url = "https://civitai.com/api/v1/images"

    limit = max(1, min(int(limit or 50), 200))
    bridge_key = _bridge_cache_key(
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
    cached = _bridge_cache_get(bridge_key)
    if cached is not None:
        return cached

    # When no text query needs resolving, use the public images endpoint
    # directly. This covers selected models, posts, and unfiltered traversal.
    if not str(query or "").strip():
        image_params: Dict[str, Any] = {
            "limit": limit,
            "withMeta": "true",
        }
        if model_id is not None:
            image_params["modelId"] = int(model_id)
        norm_cursor = _normalize_cursor(cursor)
        if norm_cursor:
            image_params["cursor"] = norm_cursor
        if model_version_id is not None:
            image_params["modelVersionId"] = int(model_version_id)
        if post_id is not None:
            image_params["postId"] = int(post_id)
        if username:
            image_params["username"] = username
        if nsfw is not None:
            image_params["nsfw"] = str(bool(nsfw)).lower()
        if sort in ALLOWED_SORT_IMAGES:
            image_params["sort"] = sort
        if period:
            image_params["period"] = period

        data = _request_with_retries(
            "GET",
            url,
            params=_clean_params(image_params),
            api_key=api_key,
        )
        items = [_trim_image_item(raw) for raw in data.get("items", []) or []]
        meta = data.get("metadata", {}) or {}
        result = {
            "ok": True,
            "items": items,
            "metadata": {
                "page": meta.get("currentPage", page),
                "pages": meta.get("totalPages"),
                "total": meta.get("totalItems"),
                "returned": len(items),
                "modelsMatched": 1 if model_id is not None else None,
                "nextPage": meta.get("nextPage"),
                "prevPage": meta.get("prevPage"),
                "nextCursor": meta.get("nextCursor") or meta.get("nextPageCursor"),
                "prevCursor": meta.get("prevCursor") or meta.get("prevPageCursor"),
            },
        }
        _bridge_cache_set(bridge_key, result)
        return result

    # Always bridge image searches through model-search, then unify images by modelId.
    # This keeps image discovery consistent with model relevance/search controls.
    mapped_model_sort = model_sort or _map_image_sort_to_model_sort(sort) or "Newest"
    model_search = _search_civitai(
        query=(str(query or "").strip()),
        page=max(1, int(page or 1)),
        api_key=api_key,
        cursor=cursor,
        nsfw=nsfw,
        sort=mapped_model_sort,
        period=period,
        limit=20,
        model_id=model_id,
        model_version_id=model_version_id,
    )
    model_items = model_search.get("items", []) or []

    merged: list[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    max_total_images = 4000

    for model in model_items:
        mid = model.get("id")
        if mid is None:
            continue
        try:
            model_id_int = int(mid)
        except Exception:
            continue

        image_cursor: Optional[str] = None
        seen_image_cursors: set[str] = set()
        while True:
            image_params: Dict[str, Any] = {
                "limit": 200,
                "modelId": model_id_int,
                "withMeta": "true",
            }
            if image_cursor:
                image_params["cursor"] = image_cursor
            if model_version_id is not None:
                image_params["modelVersionId"] = int(model_version_id)
            if post_id is not None:
                image_params["postId"] = int(post_id)
            if username:
                image_params["username"] = username
            if nsfw is not None:
                image_params["nsfw"] = str(bool(nsfw)).lower()
            if sort in ALLOWED_SORT_IMAGES:
                image_params["sort"] = sort
            if period:
                image_params["period"] = period

            data = _request_with_retries("GET", url, params=_clean_params(image_params), api_key=api_key)
            items = data.get("items", []) or []

            for raw in items:
                trimmed = _trim_image_item(raw)
                iid = trimmed.get("id")
                if isinstance(iid, int):
                    if iid in seen_ids:
                        continue
                    seen_ids.add(iid)
                merged.append(trimmed)

            if len(merged) >= max_total_images:
                break

            meta = data.get("metadata", {}) or {}
            next_image_cursor = _normalize_cursor(
                meta.get("nextCursor") or meta.get("nextPageCursor")
            )
            if (
                not next_image_cursor
                or next_image_cursor in seen_image_cursors
            ):
                break
            seen_image_cursors.add(next_image_cursor)
            image_cursor = next_image_cursor

        if len(merged) >= max_total_images:
            break

    merged = _sort_unified_images(merged, sort)
    unified_total = len(merged)
    merged = merged[:limit]

    result = {
        "ok": True,
        "items": merged,
        "metadata": {
            "page": model_search.get("metadata", {}).get("page", page),
            "pages": model_search.get("metadata", {}).get("pages"),
            "total": unified_total,
            "returned": len(merged),
            "modelsMatched": len(model_items),
            "nextPage": model_search.get("metadata", {}).get("nextPage"),
            "prevPage": model_search.get("metadata", {}).get("prevPage"),
            "nextCursor": model_search.get("metadata", {}).get("nextCursor"),
            "prevCursor": model_search.get("metadata", {}).get("prevCursor"),
        },
    }
    _bridge_cache_set(bridge_key, result)
    return result


# ----------------------------
# Download helpers
# ----------------------------

def _normalized_model_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _is_modular_base_model(base_model: str) -> bool:
    bm = _normalized_model_label(base_model)
    return any(bm.startswith(prefix) for prefix in _MODULAR_BASE_MODEL_PREFIXES)


def _model_type_dir(
    model_type: str,
    file_type: str = "",
    base_model: str = "",
) -> str:
    """Choose the model root using both page and selected-file metadata."""
    t = _normalized_model_label(model_type)
    ft = _normalized_model_label(file_type)

    # Explicit component file types take precedence over the model-page type.
    if ft == "text encoder":
        return "Text-encoder"
    if ft in ("vision encoder", "clipvision"):
        return "clip_vision"
    if ft == "vae":
        return "VAE"
    if ft in ("unet", "diffusion model"):
        return "DiffusionModels"
    if ft == "controlnet":
        return "ControlNet"
    if ft == "workflow":
        return "workflows"
    if ft == "upscaler":
        return "upscale_models"
    if ft == "enhancement lora":
        return "Lora"
    if ft == "training data":
        return "training_data"

    # Adapter pages usually describe their safetensors file simply as "Model".
    if t in ("lora", "locon", "dora"):
        return "Lora"
    if "control" in t:
        return "ControlNet"
    if "textual" in t or "embedding" in t or "aesthetic gradient" in t:
        return "embeddings"
    if "hyper" in t:
        return "hypernetworks"
    if t == "vae":
        return "VAE"
    if t == "textencoder":
        return "Text-encoder"
    if t == "clipvision":
        return "clip_vision"
    if t == "clip":
        return "clip"
    if t == "unet":
        return "DiffusionModels"
    if t in ("workflow", "workflows", "comfyworkflows"):
        return "workflows"
    if t == "motionmodule":
        return "animatediff_models"
    if t == "upscaler":
        return "upscale_models"
    if t == "poses":
        return "poses"
    if t == "detection":
        return "Ultralytics"
    if t in ("visionlanguage", "llm"):
        return "LLM"
    if t == "wildcards":
        return "wildcards"
    if t == "checkpoint" and _is_modular_base_model(base_model):
        return "DiffusionModels"
    if t == "other":
        return "other"
    return "checkpoints"


def _base_model_bucket(base_model: str) -> str:
    """Return a stable, filesystem-safe bucket for a CivitAI base model."""
    bm = _normalized_model_label(base_model)
    if not bm or bm in ("other", "upscaler"):
        return "misc"
    if "cascade" in bm:
        return "cascade"
    if bm.startswith("sd 3 5") or bm == "3 5":
        return "3.5"
    if bm.startswith("sd 3") or bm == "3 0":
        return "3.0"
    if bm.startswith("flux"):
        if bm.startswith("flux 1 d") or bm in ("flux1d", "flux dev"):
            return "fluxd"
        if bm.startswith("flux 1 s") or bm in ("flux1s", "flux schnell"):
            return "fluxs"
        if bm.startswith("flux 1 krea"):
            return "fluxkrea"
        if bm.startswith("flux 1 kontext"):
            return "fluxkontext"
        if bm.startswith("flux 2 d"):
            return "flux2d"
        if bm.startswith("flux 2 klein 9b"):
            return "flux2klein9b"
        if bm.startswith("flux 2 klein 4b"):
            return "flux2klein4b"
        if bm.startswith("flux 3 video"):
            return "flux3video"
        return "flux"
    if "sdxl" in bm or "pony" in bm or "noobai" in bm or "illustrious" in bm:
        return "sdxl"
    if bm.startswith("sd 2") or bm in ("2 0", "2 1"):
        return "2.0"
    if bm.startswith("sd 1") or bm in ("1 4", "1 5"):
        return "1.5"
    if bm.startswith("hidream"):
        return "hidream"
    if bm.startswith("hunyuan video"):
        return "hunyuan-video"
    if bm.startswith("hunyuan"):
        return "hunyuan"
    if bm.startswith("ltxv"):
        return "ltxv"
    if bm.startswith("pixart"):
        return "pixart"
    if bm.startswith("qwen"):
        return "qwen"
    if bm.startswith("wan "):
        return "wan"
    if bm.startswith("zimage"):
        return "zimage"

    # Unknown future families get their own deterministic bucket instead of
    # silently accumulating in misc.
    return _safe_slug(base_model, "misc").lower()


def _base_model_dir(base_model: str) -> str:
    return _base_model_bucket(base_model)


def _lora_base_model_dir(base_model: str) -> str:
    return _base_model_bucket(base_model)


def _pick_download(target: Dict[str, Any], file_override: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str]:
    mv = target.get("modelVersion") or {}
    files = mv.get("files") or []
    base_model = mv.get("baseModel") or ""
    model_type = target.get("type") or ""

    chosen_file = file_override or None
    if not chosen_file:
        if not files:
            raise ValueError("No downloadable files in selection")
        chosen_file = next((f for f in files if f.get("primary")), None) or files[0]

    subdir = _model_type_dir(
        model_type,
        chosen_file.get("type") or "",
        base_model,
    )
    if subdir in ("Lora", "checkpoints", "DiffusionModels"):
        standard = _lora_base_model_dir(base_model)
    else:
        standard = _base_model_dir(base_model)
    dest_dir = MODELS_DIR / subdir / standard
    _ensure_dir(dest_dir)

    url = chosen_file.get("downloadUrl")
    if not url:
        mv_id = mv.get("id")
        if mv_id:
            url = f"https://civitai.com/api/download/models/{mv_id}"

    fname = _safe_slug(chosen_file.get("name") or f"{target.get('name') or 'model'}.safetensors")
    return url, fname, str(dest_dir)


def _download_file(
    url: str,
    dest_dir: str,
    filename: str,
    api_key: Optional[str],
    progress_callback: Optional[Any] = None,
) -> Tuple[str, int]:
    headers: Dict[str, str] = {}
    clean_key = (api_key or "").strip()

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"
        qs.pop("token", None)
        qs["token"] = [clean_key]
        url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_bytes_header = r.headers.get("content-length")
        try:
            total_bytes = int(total_bytes_header) if total_bytes_header else None
        except (TypeError, ValueError):
            total_bytes = None
        cd = r.headers.get("content-disposition") or ""
        if "filename=" in cd:
            try:
                fname_part = cd.split("filename=", 1)[1].strip().strip('"')
                filename = _safe_slug(fname_part, filename)
            except Exception:
                pass

        dest_path = Path(dest_dir) / filename
        _ensure_dir(dest_path.parent)
        temp_path = dest_path.with_name(f"{dest_path.name}.part")
        if temp_path.exists():
            temp_path.unlink()

        total = 0
        try:
            with temp_path.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)
                    if progress_callback:
                        progress_callback(total, total_bytes)

            os.replace(temp_path, dest_path)
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            raise

    return str(dest_path), total
