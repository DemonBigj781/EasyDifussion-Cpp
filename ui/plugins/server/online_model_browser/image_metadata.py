"""Inspect delivered image bytes without merging website metadata or re-encoding."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

from PIL import ExifTags, Image
from . import service as civ

MAX_IMAGE_BYTES = 100 * 1024 * 1024
ALLOWED_HOSTS = {"image.civitai.com", "civitai.com", "www.civitai.com"}
IMAGE_EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}


def _json_value(value):
    if isinstance(value, bytes):
        result = {"base64": base64.b64encode(value).decode("ascii")}
        prefix, text = value[:8], value[8:]
        if prefix == b"ASCII\0\0\0":
            result["text"] = text.decode("ascii", errors="replace").rstrip("\0")
        elif prefix == b"UNICODE\0":
            encoding = "utf-16" if text.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-16-be"
            result["text"] = text.decode(encoding, errors="replace").rstrip("\0")
        else:
            try:
                result["text"] = value.decode("utf-8")
            except UnicodeDecodeError:
                pass
        return result
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def inspect_image_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with Image.open(path) as image:
        if image.format not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {image.format}")
        image.load()  # Includes PNG text chunks located after image data.
        exif = image.getexif()
        fields = {ExifTags.TAGS.get(tag, str(tag)): _json_value(value) for tag, value in exif.items()}
        for tag, name in ((34665, "ExifIFD"), (34853, "GPSInfo")):
            if tag in exif:
                names = ExifTags.GPSTAGS if tag == 34853 else ExifTags.TAGS
                fields[name] = {names.get(key, str(key)): _json_value(value) for key, value in exif.get_ifd(tag).items()}
        return {
            "sizeBytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "embedded": {"info": _json_value(image.info), "exif": fields},
        }


def download_image(source_url: str, destination: Path, api_key=None):
    url = source_url
    headers = {"Accept": "image/*"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for _ in range(6):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError("Image URL must use an official Civitai HTTPS host")
        with civ.requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=False) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Image redirect has no Location")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > MAX_IMAGE_BYTES:
                raise ValueError("CivitAI image exceeds the 100 MB limit")
            total = 0
            try:
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            raise ValueError("CivitAI image exceeds the 100 MB limit")
                        output.write(chunk)
            finally:
                civ._log(f"Image response status={response.status_code} received_bytes={total} content_length={declared or 'unknown'}")
            result = inspect_image_file(destination)
            result["sourceUrl"] = source_url
            result["resolvedUrl"] = url
            result["contentType"] = response.headers.get("Content-Type")
            return result
    raise ValueError("Too many CivitAI image redirects")
