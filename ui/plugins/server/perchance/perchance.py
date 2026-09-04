"""Built-in Perchance image, text, and public-gallery integration.

Perchance runs outside Easy Diffusion's render queue. A non-blocking lock keeps
the bundled Camoufox runtime to one operation at a time.
"""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import asyncio
import base64
import concurrent.futures
import json
import math
import os
import re
import signal
import threading
import time
from io import BytesIO

from easydiffusion import app as easy_app
from fastapi import HTTPException
from PIL import Image, ImageOps


ED_ROOT = Path(easy_app.ROOT_DIR)
PERCHANCE_RELEASE_TAG = "v1.0.0-rc.1"
PERCHANCE_RELEASE_NAME = "Perchance 1.0.0 RC 1"
PERCHANCE_APPIMAGE_NAME = "Perchance-1.0.0-x86_64.AppImage"
PERCHANCE_APPIMAGE_SHA256 = "2a90bffff3883a704f0b57f6239a430578082628c4b7a2daae896b7e31c4fc01"
PERCHANCE_RELEASE_URL = "https://github.com/DemonBigj781/perchance/releases/tag/v1.0.0-rc.1"
PERCHANCE_DOWNLOAD_URL = (
    "https://github.com/DemonBigj781/perchance/releases/download/"
    f"{PERCHANCE_RELEASE_TAG}/{PERCHANCE_APPIMAGE_NAME}"
)


def _launcher_path() -> Path:
    configured = easy_app.getConfig().get("perchance", {})
    configured = configured if isinstance(configured, dict) else {}
    explicit = os.environ.get("PERCHANCE_LAUNCHER") or configured.get("launcher")
    if explicit:
        return Path(str(explicit)).expanduser()

    candidates = (
        ED_ROOT / "bin" / "perchance",
        Path.home() / ".local" / "bin" / "perchance",
        Path.home() / "bin" / "perchance",
        ED_ROOT / "tools" / "perchance" / PERCHANCE_APPIMAGE_NAME,
        ED_ROOT / "perchance" / PERCHANCE_APPIMAGE_NAME,
        ED_ROOT / PERCHANCE_APPIMAGE_NAME,
        Path.home() / "AppImages" / "perchance.AppImage",
        Path.home() / "AppImages" / PERCHANCE_APPIMAGE_NAME,
        Path.home() / "Downloads" / PERCHANCE_APPIMAGE_NAME,
    )
    executable = next(
        (candidate for candidate in candidates if candidate.is_file() and os.access(candidate, os.X_OK)),
        None,
    )
    if executable is not None:
        return executable
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


PERCHANCE_LOCK = threading.Lock()
SETTINGS_LOCK = threading.Lock()
IMAGE_TIMEOUT_SECONDS = 15 * 60
MAX_IMAGE_AMOUNT = 20
TEXT_TIMEOUT_SECONDS = 10 * 60
# Opening the official nested gallery frame can consume about two minutes by
# itself. Allow room for slow links and optional image downloads while still
# bounding an upstream stall.
GALLERY_TIMEOUT_SECONDS = 10 * 60
MAX_PROMPT_LENGTH = 20_000
MAX_CAPTURE_LENGTH = 1_000_000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PERCHANCE_IMAGE_NAME_PATTERN = re.compile(
    r"^[0-9a-f]{64}(?:-[0-9]+)?\.(?:png|jpe?g|webp|gif)$",
    re.IGNORECASE,
)
PERCHANCE_GALLERY_CACHE_DIRNAME = "perchance-gallery-cache"
PERCHANCE_GALLERY_IMAGE_ORIGIN = "https://aigc.uploads.dev"
PERCHANCE_GALLERY_IMAGE_PATTERN = re.compile(
    r"^/image/(?P<filename>[0-9a-f]{64}\.(?:png|jpe?g|webp))$",
    re.IGNORECASE,
)
GALLERY_IMAGE_TIMEOUT_SECONDS = 45
GALLERY_IMAGE_DOWNLOAD_ATTEMPTS = 3
MAX_GALLERY_IMAGE_BYTES = 50 * 1024 * 1024
GALLERY_PREVIEW_MAX_DIMENSION = 768
MAX_GALLERY_PREVIEW_BYTES = 1024 * 1024
MAX_GALLERY_SOURCE_PIXELS = 64 * 1024 * 1024
GALLERY_LAUNCH_ATTEMPTS = 3
GALLERY_TRANSIENT_ERROR_MARKERS = (
    "ns_error_unknown_host",
    "official generator did not open its public gallery frame",
    "public gallery frame did not finish loading structured data",
    "page.goto: timeout",
)
DEFAULT_GALLERY_CHANNEL = "ai-text-to-image-generator"
CHANNEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
FUSE_ERROR_MARKERS = (
    "cannot mount appimage",
    "fusermount",
    "fuse: device not found",
)


def _output_directory() -> Path:
    configured = easy_app.getConfig().get("force_save_path")
    if configured:
        directory = Path(str(configured)).expanduser()
        if not directory.is_absolute():
            directory = ED_ROOT / directory
    else:
        directory = ED_ROOT / "outputs"
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _require_payload(payload) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object.")
    return payload


def _prompt_from(payload: dict) -> str:
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt must be at most {MAX_PROMPT_LENGTH} characters.",
        )
    return prompt


def _integer(value, name: str, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"{name} must be an integer.") from error


def _finite_number(value, name: str, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"{name} must be a number.") from error
    if not math.isfinite(result):
        raise HTTPException(status_code=400, detail=f"{name} must be finite.")
    return result


def _boolean(value, name: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise HTTPException(status_code=400, detail=f"{name} must be a boolean.")


def _bounded_string(value, name: str, maximum: int, allow_empty: bool = False) -> str:
    result = str(value or "").strip()
    if not result and not allow_empty:
        raise HTTPException(status_code=400, detail=f"{name} is required.")
    if len(result) > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be at most {maximum} characters.",
        )
    return result


def _gallery_channel(value=None) -> str:
    if value in (None, ""):
        value = get_settings()["channel"]
    channel = _bounded_string(value, "channel", 128)
    if not CHANNEL_PATTERN.fullmatch(channel):
        raise HTTPException(
            status_code=400,
            detail="channel may contain only ASCII letters, digits, underscores, and hyphens.",
        )
    return channel


def get_settings() -> dict:
    section = easy_app.getConfig().get("perchance", {})
    if not isinstance(section, dict):
        section = {}
    gallery_id = str(section.get("gallery_id", "") or "").strip()
    channel = str(section.get("channel", DEFAULT_GALLERY_CHANNEL) or DEFAULT_GALLERY_CHANNEL).strip()
    if not CHANNEL_PATTERN.fullmatch(channel):
        channel = DEFAULT_GALLERY_CHANNEL
    launcher = str(section.get("launcher", "") or "").strip()
    return {"gallery_id": gallery_id, "channel": channel, "launcher": launcher}


def save_settings(payload) -> dict:
    payload = _require_payload(payload)
    with SETTINGS_LOCK:
        current = get_settings()
        gallery_id = current["gallery_id"]
        channel = current["channel"]
        if "gallery_id" in payload:
            gallery_id = _bounded_string(
                payload.get("gallery_id"),
                "gallery_id",
                2_048,
                allow_empty=True,
            )
        if "channel" in payload:
            raw_channel = str(payload.get("channel") or "").strip()
            channel = _gallery_channel(raw_channel or DEFAULT_GALLERY_CHANNEL)

        config = easy_app.getConfig()
        config["perchance"] = {
            "gallery_id": gallery_id,
            "channel": channel,
            "launcher": current.get("launcher", ""),
        }
        easy_app.setConfig(config)
    return {"gallery_id": gallery_id, "channel": channel}


def _trim_capture(value: str) -> str:
    if len(value) <= MAX_CAPTURE_LENGTH:
        return value
    return value[-MAX_CAPTURE_LENGTH:]


async def _terminate(process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
        return
    except asyncio.TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    await process.wait()


async def _invoke(
    launcher: Path,
    arguments: list[str],
    timeout_seconds: int,
    environment=None,
) -> dict:
    try:
        process = await asyncio.create_subprocess_exec(
            str(launcher),
            *arguments,
            cwd=str(ED_ROOT),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not start {launcher}: {error}",
        ) from error

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as error:
        await _terminate(process)
        raise HTTPException(
            status_code=504,
            detail=f"Perchance exceeded its {timeout_seconds}-second time limit.",
        ) from error

    return {
        "returncode": process.returncode,
        "stdout": _trim_capture(stdout_bytes.decode("utf-8", errors="replace")),
        "stderr": _trim_capture(stderr_bytes.decode("utf-8", errors="replace")),
    }


async def _run_perchance(arguments: list[str], timeout_seconds: int) -> dict:
    launcher = _launcher_path()
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise HTTPException(
            status_code=500,
            detail=f"Perchance launcher is missing or not executable: {launcher}",
        )

    result = await _invoke(launcher, arguments, timeout_seconds)
    combined_error = f"{result['stdout']}\n{result['stderr']}".lower()
    if result["returncode"] != 0 and any(
        marker in combined_error for marker in FUSE_ERROR_MARKERS
    ):
        environment = os.environ.copy()
        environment["APPIMAGE_EXTRACT_AND_RUN"] = "1"
        result = await _invoke(launcher, arguments, timeout_seconds, environment)

    if result["returncode"] != 0:
        detail = result["stderr"].strip() or result["stdout"].strip()
        raise HTTPException(
            status_code=502,
            detail=detail or f"Perchance exited with status {result['returncode']}.",
        )
    return result


def _acquire_perchance() -> None:
    if not PERCHANCE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Perchance is busy. Wait for the current generation to finish.",
        )


def _saved_image_from(stdout: str, output_directory: Path) -> tuple[Path, str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=502, detail="Perchance did not report an image path.")

    saved_path = Path(lines[-1]).expanduser()
    if not saved_path.is_absolute():
        saved_path = ED_ROOT / saved_path
    saved_path = saved_path.resolve()
    try:
        relative_path = saved_path.relative_to(output_directory)
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Perchance saved outside the Easy Diffusion output directory: {saved_path}",
        ) from error
    if not saved_path.is_file() or saved_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=502,
            detail=f"Perchance did not create the expected image: {saved_path}",
        )
    return saved_path, relative_path.as_posix()


def _parse_json_object(stdout: str, description: str) -> dict:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise HTTPException(
        status_code=502,
        detail=f"Perchance {description} output did not contain the expected JSON object.",
    )


def _parse_image_results(stdout: str, output_directory: Path) -> list[dict]:
    parsed = None
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, (dict, list)):
            parsed = candidate
            break
    if parsed is None:
        raise HTTPException(status_code=502, detail="Perchance image output did not contain JSON.")

    records = parsed if isinstance(parsed, list) else [parsed]
    images = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise HTTPException(status_code=502, detail="Perchance returned invalid image metadata.")
        saved_path, relative_path = _saved_image_from(record["path"], output_directory)
        images.append(
            {
                **record,
                "path": str(saved_path),
                "relative_path": relative_path,
                "url": f"/perchance/file/{quote(relative_path, safe='/')}",
            }
        )
    return images


def recent_images(limit=8) -> dict:
    """Return recent local Perchance outputs so the UI can recover a lost response."""
    limit = _integer(limit, "limit", 8)
    if limit < 1 or limit > MAX_IMAGE_AMOUNT:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be between 1 and {MAX_IMAGE_AMOUNT}.",
        )

    output_directory = _output_directory()
    candidates = [
        path
        for path in output_directory.iterdir()
        if path.is_file() and PERCHANCE_IMAGE_NAME_PATTERN.fullmatch(path.name)
    ]
    selected = sorted(
        candidates,
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:limit]
    selected.reverse()
    images = []
    for path in selected:
        relative_path = path.relative_to(output_directory).as_posix()
        images.append(
            {
                "path": str(path),
                "relative_path": relative_path,
                "url": f"/perchance/file/{quote(relative_path, safe='/')}",
            }
        )
    return {
        "images": images,
        "generated_amount": len(images),
        "output_directory": str(output_directory),
    }


def _run_locked(arguments: list[str], timeout_seconds: int):
    async def run():
        _acquire_perchance()
        try:
            return await _run_perchance(arguments, timeout_seconds)
        finally:
            PERCHANCE_LOCK.release()

    return run()


def _run_gallery_locked(arguments: list[str]):
    async def run():
        _acquire_perchance()
        try:
            for attempt in range(GALLERY_LAUNCH_ATTEMPTS):
                try:
                    return await _run_perchance(arguments, GALLERY_TIMEOUT_SECONDS)
                except HTTPException as error:
                    detail = str(error.detail).lower()
                    transient = error.status_code == 502 and any(
                        marker in detail for marker in GALLERY_TRANSIENT_ERROR_MARKERS
                    )
                    if not transient or attempt + 1 >= GALLERY_LAUNCH_ATTEMPTS:
                        raise
                    await asyncio.sleep(attempt + 1)
        finally:
            PERCHANCE_LOCK.release()

    return run()


def status() -> dict:
    launcher = _launcher_path()
    try:
        resolved = str(launcher.resolve(strict=True))
    except OSError:
        resolved = ""
    return {
        "ready": launcher.is_file() and os.access(launcher, os.X_OK),
        "busy": PERCHANCE_LOCK.locked(),
        "launcher": str(launcher),
        "resolved_launcher": resolved,
        "output_directory": str(_output_directory()),
        "settings": get_settings(),
        "release": {
            "tag": PERCHANCE_RELEASE_TAG,
            "name": PERCHANCE_RELEASE_NAME,
            "url": PERCHANCE_RELEASE_URL,
            "download_url": PERCHANCE_DOWNLOAD_URL,
            "filename": PERCHANCE_APPIMAGE_NAME,
            "sha256": PERCHANCE_APPIMAGE_SHA256,
        },
    }


async def generate_image(payload) -> dict:
    payload = _require_payload(payload)
    prompt = _prompt_from(payload)
    shape = str(payload.get("shape", "square")).strip().lower()
    if shape not in {"portrait", "square", "landscape"}:
        raise HTTPException(
            status_code=400,
            detail="shape must be portrait, square, or landscape.",
        )
    seed = _integer(payload.get("seed"), "seed", -1)
    guidance_scale = _finite_number(payload.get("guidance_scale"), "guidance_scale", 7)
    amount = _integer(payload.get("amount"), "amount", 1)
    if amount < 1 or amount > MAX_IMAGE_AMOUNT:
        raise HTTPException(
            status_code=400,
            detail=f"amount must be between 1 and {MAX_IMAGE_AMOUNT}.",
        )
    negative_prompt = str(payload.get("negative_prompt", "")).strip()
    output_directory = _output_directory()
    arguments = [
        "image",
        "-o",
        str(output_directory),
        "--count",
        str(amount),
        "--shape",
        shape,
        "--seed",
        str(seed),
        "--guidance-scale",
        str(guidance_scale),
        "--json",
    ]
    if negative_prompt:
        arguments.extend(["--negative-prompt", negative_prompt])
    arguments.append(prompt)
    result = await _run_locked(arguments, IMAGE_TIMEOUT_SECONDS)
    images = _parse_image_results(result["stdout"], output_directory)
    if len(images) != amount:
        raise HTTPException(
            status_code=502,
            detail=f"Perchance returned {len(images)} images after {amount} were requested.",
        )

    first_image = images[0]
    return {
        **first_image,
        "images": images,
        "requested_amount": amount,
        "generated_amount": len(images),
        "output_directory": str(output_directory),
    }


async def generate_text(payload) -> dict:
    payload = _require_payload(payload)
    prompt = _prompt_from(payload)
    start_with = str(payload.get("start_with", ""))
    raw_stops = payload.get("stop", [])
    if isinstance(raw_stops, str):
        stop_sequences = [raw_stops] if raw_stops else []
    elif isinstance(raw_stops, list):
        stop_sequences = [str(value) for value in raw_stops if str(value)]
    else:
        raise HTTPException(status_code=400, detail="stop must be a string or list.")
    if len(stop_sequences) > 100:
        raise HTTPException(status_code=400, detail="stop accepts at most 100 values.")

    arguments = ["text", "--json"]
    if start_with:
        arguments.extend(["--start-with", start_with])
    for stop_sequence in stop_sequences:
        arguments.extend(["--stop", stop_sequence])
    timeout_ms = payload.get("timeout_ms")
    if timeout_ms not in (None, ""):
        timeout_ms = _integer(timeout_ms, "timeout_ms", 0)
        if timeout_ms <= 0:
            raise HTTPException(status_code=400, detail="timeout_ms must be positive.")
        arguments.extend(["--timeout", str(timeout_ms)])
    arguments.append(prompt)

    result = await _run_locked(arguments, TEXT_TIMEOUT_SECONDS)
    parsed = _parse_json_object(result["stdout"], "text")
    if not isinstance(parsed.get("text"), str):
        raise HTTPException(status_code=502, detail="Perchance returned invalid text output.")
    return {"text": parsed["text"]}


def _gallery_cache_directory() -> Path:
    directory = (ED_ROOT / "tmp" / PERCHANCE_GALLERY_CACHE_DIRNAME).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _gallery_image_filename(image_url: str) -> str:
    try:
        parsed = urlsplit(image_url)
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Perchance returned an invalid gallery image URL.") from error
    match = (
        PERCHANCE_GALLERY_IMAGE_PATTERN.fullmatch(parsed.path)
        if parsed.scheme == "https"
        and parsed.netloc == urlsplit(PERCHANCE_GALLERY_IMAGE_ORIGIN).netloc
        and not parsed.query
        and not parsed.fragment
        else None
    )
    if match is None:
        raise HTTPException(
            status_code=502,
            detail="Perchance returned a gallery image outside the supported image origin.",
        )
    return match.group("filename").lower()


def _download_gallery_image(image_url: str, directory: Path) -> Path:
    filename = _gallery_image_filename(image_url)
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.is_file():
        return destination

    last_error = None
    for attempt in range(GALLERY_IMAGE_DOWNLOAD_ATTEMPTS):
        partial = directory / f".{filename}.{os.getpid()}.{threading.get_ident()}.part"
        try:
            request = Request(
                image_url,
                headers={
                    "Accept": "image/*",
                    "User-Agent": "Easy-Diffusion-Perchance/1.0",
                },
            )
            with urlopen(request, timeout=GALLERY_IMAGE_TIMEOUT_SECONDS) as response:
                final_filename = _gallery_image_filename(response.geturl())
                if final_filename != filename:
                    raise ValueError("Gallery image download redirected to a different image.")
                content_type = str(response.headers.get("Content-Type", "")).lower()
                if not content_type.startswith("image/"):
                    raise ValueError("Gallery image download returned a non-image response.")
                raw_length = response.headers.get("Content-Length")
                if raw_length is not None and int(raw_length) > MAX_GALLERY_IMAGE_BYTES:
                    raise ValueError("Gallery image exceeds the maximum supported size.")

                total = 0
                with partial.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_GALLERY_IMAGE_BYTES:
                            raise ValueError("Gallery image exceeds the maximum supported size.")
                        output.write(chunk)
                if total == 0:
                    raise ValueError("Gallery image download was empty.")
                os.replace(partial, destination)
                return destination
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, HTTPException) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if destination.is_file():
                return destination
            if attempt + 1 < GALLERY_IMAGE_DOWNLOAD_ATTEMPTS:
                time.sleep(0.25 * (attempt + 1))

    raise HTTPException(
        status_code=502,
        detail=f"Could not download Perchance gallery image after {GALLERY_IMAGE_DOWNLOAD_ATTEMPTS} attempts: {last_error}",
    )


def _decorate_download(
    item: dict,
    root_directory: Path | None = None,
    route: str = "/perchance/file",
) -> dict:
    file_path = item.get("filePath")
    if not isinstance(file_path, str) or not file_path:
        return item
    root_directory = (root_directory or _output_directory()).resolve()
    resolved = Path(file_path).expanduser().resolve()
    try:
        relative_path = resolved.relative_to(root_directory).as_posix()
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Perchance saved outside its selected gallery directory: {resolved}",
        ) from error
    if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=502, detail=f"Perchance did not create {resolved}.")
    item["relative_path"] = relative_path
    item["local_url"] = f"{route}/{quote(relative_path, safe='/')}"
    return item


def _decorate_lazy_cache(item: dict) -> dict:
    filename = _gallery_image_filename(str(item.get("imageUrl", "")))
    item["local_url"] = f"/perchance/gallery/cache/{quote(filename)}"
    return item


def _gallery_preview_data_url(image_path: Path) -> str:
    try:
        with Image.open(image_path) as source:
            if source.width * source.height > MAX_GALLERY_SOURCE_PIXELS:
                raise ValueError("image dimensions exceed the gallery preview limit")
            source.seek(0)
            preview = ImageOps.exif_transpose(source)
            preview.thumbnail(
                (GALLERY_PREVIEW_MAX_DIMENSION, GALLERY_PREVIEW_MAX_DIMENSION),
                Image.Resampling.LANCZOS,
            )
            if "A" in preview.getbands():
                rgba = preview.convert("RGBA")
                flattened = Image.new("RGB", rgba.size, "white")
                flattened.paste(rgba, mask=rgba.getchannel("A"))
                preview = flattened
            elif preview.mode != "RGB":
                preview = preview.convert("RGB")

            encoded = BytesIO()
            preview.save(encoded, format="JPEG", quality=82, optimize=True)
            preview_bytes = encoded.getvalue()
    except (Image.DecompressionBombError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not prepare a Perchance gallery preview: {error}",
        ) from error

    if not preview_bytes or len(preview_bytes) > MAX_GALLERY_PREVIEW_BYTES:
        raise HTTPException(
            status_code=502,
            detail="Perchance gallery preview exceeded the embedded-image size limit.",
        )
    payload = base64.b64encode(preview_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _attach_gallery_preview(item: dict) -> dict:
    try:
        file_path = item.get("filePath")
        if isinstance(file_path, str) and file_path:
            image_path = Path(file_path).expanduser().resolve()
        else:
            filename = _gallery_image_filename(str(item.get("imageUrl", "")))
            image_path = resolve_gallery_cache_file(filename)
        item["preview_data_url"] = _gallery_preview_data_url(image_path)
        item.pop("preview_error", None)
    except HTTPException as error:
        item.pop("preview_data_url", None)
        item["preview_error"] = str(error.detail)
    return item


async def _attach_gallery_previews(entries: list[dict]) -> list[dict]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_attach_gallery_preview, item) for item in entries]
        while any(not future.done() for future in futures):
            await asyncio.sleep(0.05)
        return [future.result() for future in futures]


def _save_gallery_image(item: dict) -> dict:
    output_directory = _output_directory()
    target = output_directory / "perchance-gallery"
    saved_path = _download_gallery_image(str(item.get("imageUrl", "")), target)
    item["filePath"] = str(saved_path)
    return _decorate_download(item, output_directory, "/perchance/file")


async def _save_gallery_images(entries: list[dict]) -> list[dict]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_save_gallery_image, item) for item in entries]
        while any(not future.done() for future in futures):
            await asyncio.sleep(0.05)
        return [future.result() for future in futures]


def _gallery_common(payload: dict) -> tuple[str, str, bool, bool]:
    channel = _gallery_channel(payload.get("channel"))
    content_filter = _bounded_string(
        payload.get("content_filter", "g"),
        "content_filter",
        128,
    )
    download = _boolean(payload.get("download"), "download")
    visible = _boolean(payload.get("visible"), "visible")
    return channel, content_filter, download, visible


async def gallery_list(payload) -> dict:
    payload = _require_payload(payload)
    channel, content_filter, download, visible = _gallery_common(payload)
    limit = _integer(payload.get("limit"), "limit", 20)
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")
    sort = str(payload.get("sort", "recent")).strip().lower()
    if sort not in {"recent", "top", "trending"}:
        raise HTTPException(status_code=400, detail="sort must be recent, top, or trending.")

    arguments = [
        "gallery",
        "list",
        "--channel",
        channel,
        "--content-filter",
        content_filter,
        "--limit",
        str(limit),
        "--sort",
        sort,
    ]
    cursor = _bounded_string(payload.get("cursor"), "cursor", 2_048, allow_empty=True)
    time_range = _bounded_string(
        payload.get("time_range"),
        "time_range",
        64,
        allow_empty=True,
    )
    if cursor:
        arguments.extend(["--cursor", cursor])
    if time_range:
        arguments.extend(["--time-range", time_range])
    if visible:
        arguments.append("--visible")

    result = await _run_gallery_locked(arguments)
    parsed = _parse_json_object(result["stdout"], "gallery list")
    entries = parsed.get("entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise HTTPException(status_code=502, detail="Perchance returned an invalid gallery page.")
    if download:
        entries = await _save_gallery_images(entries)
    else:
        entries = [_decorate_lazy_cache(item) for item in entries]
    parsed["entries"] = await _attach_gallery_previews(entries)
    parsed["channel"] = channel
    return parsed


async def gallery_get(payload) -> dict:
    payload = _require_payload(payload)
    channel, content_filter, download, visible = _gallery_common(payload)
    saved_id = get_settings()["gallery_id"]
    gallery_id = _bounded_string(
        payload.get("gallery_id", payload.get("id_or_url", saved_id)),
        "gallery_id",
        2_048,
    )
    arguments = [
        "gallery",
        "get",
        gallery_id,
        "--channel",
        channel,
        "--content-filter",
        content_filter,
    ]
    if visible:
        arguments.append("--visible")

    result = await _run_gallery_locked(arguments)
    parsed = _parse_json_object(result["stdout"], "gallery item")
    parsed["channel"] = channel
    if download:
        entries = await _save_gallery_images([parsed])
    else:
        entries = [_decorate_lazy_cache(parsed)]
    return (await _attach_gallery_previews(entries))[0]


def resolve_output_file(relative_path: str) -> Path:
    output_directory = _output_directory()
    image_path = (output_directory / relative_path).resolve()
    try:
        image_path.relative_to(output_directory)
    except ValueError as error:
        raise HTTPException(
            status_code=403,
            detail="Path is outside the Easy Diffusion output directory.",
        ) from error
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Generated image not found.")
    return image_path


def resolve_gallery_cache_file(relative_path: str) -> Path:
    cache_directory = _gallery_cache_directory()
    image_path = (cache_directory / relative_path).resolve()
    try:
        image_path.relative_to(cache_directory)
    except ValueError as error:
        raise HTTPException(status_code=403, detail="Path is outside the Perchance cache.") from error
    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
        return image_path
    match = PERCHANCE_GALLERY_IMAGE_PATTERN.fullmatch(f"/image/{image_path.name}")
    if image_path.parent == cache_directory and match is not None:
        image_url = f"{PERCHANCE_GALLERY_IMAGE_ORIGIN}/image/{match.group('filename').lower()}"
        return _download_gallery_image(image_url, cache_directory)
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Cached Perchance image not found.")
    return image_path
