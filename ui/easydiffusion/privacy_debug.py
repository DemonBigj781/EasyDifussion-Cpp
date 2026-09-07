"""On-disk diagnostics that never retain readable prompt context."""

from __future__ import annotations

import logging
import math
import os
import re
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


def block_alphabetic_context(value: object) -> str:
    """Preserve prompt syntax while removing readable alphabetic context."""

    return "".join(_PRIVACY_BLOCK if character.isalpha() else character for character in str(value))


def redact_log_message(message: object, force: bool = False) -> str:
    rendered = str(message)
    if force or _PRIVATE_CONTEXT_PATTERN.search(rendered):
        return block_alphabetic_context(rendered)
    return rendered


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
        return redact_log_message(rendered, force=bool(getattr(record, "privacy_sensitive", False)))


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
