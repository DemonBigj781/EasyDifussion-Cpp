"""On-disk diagnostics that never retain readable prompt context."""

from __future__ import annotations

import logging
import hashlib
import math
import os
import re
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from numbers import Real
from pathlib import Path


_PRIVACY_BLOCK = "█"
_PRIVATE_CONTEXT_PATTERN = re.compile(
    r"\b(?:hidden[ _-]+)?(?:positive[ _-]+|negative[ _-]+)?prompt\b"
    r"|\b(?:positive|negative)[ _-]+conditioning\b"
    r"|\b(?:positive|negative)\s*[:=]",
    re.IGNORECASE,
)

_QUOTED_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:[\\/]|/)[^'\"\r\n]+)(?P=quote)"
)
_UNQUOTED_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_:/])(?P<path>(?:[A-Za-z]:[\\/]|/)[^\s'\"<>|]+)"
)
_TECHNICAL_IDENTIFIERS = (
    "ControlNet",
    "SDXL",
    "SD15",
    "LoRA",
    "UNet",
    "CLIP",
    "GGUF",
    "VAE",
    "SD3",
    "SD",
)
_TECHNICAL_IDENTIFIER_PATTERN = re.compile(
    "|".join(sorted(map(re.escape, _TECHNICAL_IDENTIFIERS), key=len, reverse=True)),
    re.IGNORECASE,
)
_NORMALIZED_IDENTIFIERS = {value.lower(): value for value in _TECHNICAL_IDENTIFIERS}
_SAFE_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".json",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
_SAFE_RESOURCE_CATEGORIES = {
    "clip",
    "clip-vision",
    "controlnet",
    "embeddings",
    "lora",
    "models",
    "outputs",
    "unet",
    "vae",
}


def block_alphabetic_context(value: object) -> str:
    """Preserve prompt syntax while removing readable alphabetic context."""

    return "".join(_PRIVACY_BLOCK if character.isalpha() else character for character in str(value))


def redact_log_message(message: object, force: bool = False) -> str:
    rendered = str(message)
    if force or _PRIVATE_CONTEXT_PATTERN.search(rendered):
        return block_alphabetic_context(rendered)
    return rendered


@lru_cache(maxsize=512)
def _cached_file_md5(path_text: str, size: int, modified_ns: int) -> str:
    """Hash a file once for a stable diagnostic identity.

    Size and modification time are cache-key inputs so a changed file is read
    again. MD5 is an identifier here, not a security or integrity guarantee.
    """

    del size, modified_ns
    digest = hashlib.md5()
    with open(path_text, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_md5(path: Path) -> str | None:
    try:
        details = path.stat()
        if not path.is_file():
            return None
        return _cached_file_md5(str(path), details.st_size, details.st_mtime_ns)
    except (OSError, ValueError):
        return None


def _safe_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    return extension if extension in _SAFE_EXTENSIONS else ""


def _redact_filename(filename: str) -> str:
    """Block alphabetic context while retaining known model identifiers."""

    extension = _safe_extension(filename)
    stem = filename[: -len(extension)] if extension else filename
    output: list[str] = []
    cursor = 0
    for match in _TECHNICAL_IDENTIFIER_PATTERN.finditer(stem):
        output.append(block_alphabetic_context(stem[cursor : match.start()]))
        output.append(_NORMALIZED_IDENTIFIERS[match.group(0).lower()])
        cursor = match.end()
    output.append(block_alphabetic_context(stem[cursor:]))
    return "".join(output) + extension


def _relative_resource_category(path_text: str) -> str:
    parts = [part for part in path_text.replace("\\", "/").split("/") if part]
    lowered = [part.lower() for part in parts]
    for index, part in enumerate(lowered[:-1]):
        if part == "models":
            category = ["models"]
            if index + 1 < len(parts) - 1 and lowered[index + 1] in _SAFE_RESOURCE_CATEGORIES:
                category.append(lowered[index + 1])
            return "/".join(category)
        if part in {"outputs", "output"}:
            return "outputs"
    return "resource"


def _looks_like_filesystem_path(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/")
    lowered = normalized.lower()
    if re.match(r"^[a-z]:/", lowered):
        return True
    if any(
        lowered.startswith(prefix)
        for prefix in ("/home/", "/users/", "/workspace/", "/tmp/", "/var/", "/opt/", "/mnt/", "/media/", "/srv/", "/root/")
    ):
        return True
    parts = {part for part in lowered.split("/") if part}
    if parts.intersection(_SAFE_RESOURCE_CATEGORIES):
        return True
    if re.search(r"\.[a-z0-9]{1,12}$", lowered):
        return True
    try:
        return Path(path_text).expanduser().exists()
    except (OSError, ValueError):
        return False


def report_safe_path(path_text: str) -> str:
    """Return a shareable path label without its directory or basename."""

    suffix_match = re.match(r"^(.*?)(:\d+)?([.,;)]+)?$", path_text)
    normalized = suffix_match.group(1) if suffix_match else path_text
    trailing = "" if suffix_match is None else "".join(
        part or "" for part in suffix_match.groups()[1:]
    )
    if not _looks_like_filesystem_path(normalized):
        return path_text
    filename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    extension = _safe_extension(filename)
    digest = _file_md5(Path(normalized).expanduser())
    identity = f"<md5:{digest}>{extension}" if digest else _redact_filename(filename)
    return f"{_relative_resource_category(normalized)}/{identity}{trailing}"


def sanitize_log_paths(message: object) -> str:
    """Remove absolute paths and sensitive filenames from exported log text."""

    rendered = str(message)

    def replace_quoted(match: re.Match) -> str:
        quote = match.group("quote")
        return f"{quote}{report_safe_path(match.group('path'))}{quote}"

    rendered = _QUOTED_ABSOLUTE_PATH_PATTERN.sub(replace_quoted, rendered)
    return _UNQUOTED_ABSOLUTE_PATH_PATTERN.sub(
        lambda match: report_safe_path(match.group("path")), rendered
    )


def count_non_finite_values(value: object) -> tuple[int, int]:
    """Return NaN and infinity counts without rendering container contents."""

    nan_count = 0
    inf_count = 0
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            pending.extend(current)
        elif isinstance(current, Real) and not isinstance(current, (bool, int)):
            numeric = float(current)
            if math.isnan(numeric):
                nan_count += 1
            elif math.isinf(numeric):
                inf_count += 1
    return nan_count, inf_count


def log_non_finite_counts(value: object, logger: logging.Logger, stage: str) -> bool:
    nan_count, inf_count = count_non_finite_values(value)
    if nan_count == 0 and inf_count == 0:
        return False
    logger.error(
        "Non-finite values detected at %s: nan_count=%d inf_count=%d "
        "(possible invalid settings, precision failure, or broken weights)",
        stage,
        nan_count,
        inf_count,
    )
    return True


def log_result_image_diagnostics(
    images: list,
    logger: logging.Logger,
    image_decoder=None,
    dark_threshold: float = 20.0,
    bright_threshold: float = 235.0,
) -> None:
    """Inspect only caller-supplied result images; inputs are never discovered here."""

    from PIL import ImageStat

    for index, encoded_image in enumerate(images, start=1):
        try:
            image = image_decoder(encoded_image) if image_decoder is not None else encoded_image
            red, green, blue = ImageStat.Stat(image.convert("RGB")).mean
            mean_luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
        except Exception as error:
            logger.warning(
                "Result image %d could not be decoded for diagnostics: error_type=%s",
                index,
                type(error).__name__,
            )
            continue

        if not math.isfinite(mean_luminance):
            log_non_finite_counts([mean_luminance], logger, f"result_image_{index}_luminance")
        elif mean_luminance <= dark_threshold:
            logger.warning(
                "Result image %d is very dark: mean_luminance=%.2f/255 threshold=%.2f",
                index,
                mean_luminance,
                dark_threshold,
            )
        elif mean_luminance >= bright_threshold:
            logger.warning(
                "Result image %d is very bright: mean_luminance=%.2f/255 threshold=%.2f",
                index,
                mean_luminance,
                bright_threshold,
            )


class PrivacySafeRotatingFileHandler(RotatingFileHandler):
    """Redact only this handler's formatted copy; console logging is unchanged."""

    privacy_safe_debug_handler = True

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        rendered = redact_log_message(
            rendered, force=bool(getattr(record, "privacy_sensitive", False))
        )
        return sanitize_log_paths(rendered)


def install_privacy_debug_logger(log_path: str, log_format: str, date_format: str = "%X") -> Path:
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "privacy_safe_debug_handler", False):
            return Path(handler.baseFilename)

    handler = PrivacySafeRotatingFileHandler(
        path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(handler)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path
