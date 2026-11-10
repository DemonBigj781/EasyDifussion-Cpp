"""Helpers for validating and packaging generated video frames."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def _decode_frame(frame: str) -> Image.Image:
    if not isinstance(frame, str) or not frame:
        raise ValueError("AnimateDiff returned an empty frame")
    encoded = frame.split(",", 1)[1] if frame.startswith("data:") and "," in frame else frame
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("AnimateDiff returned invalid base64 frame data") from exc
    with Image.open(BytesIO(payload)) as opened:
        opened.load()
        if opened.width < 1 or opened.height < 1:
            raise ValueError("AnimateDiff returned a frame with invalid dimensions")
        return opened.convert("RGB")


def build_animatediff_gif(frames: list[str], fps: int, logger=None) -> tuple[str, dict]:
    """Encode PNG/JPEG frame data as a validated animated GIF data URI."""

    decoded = []
    try:
        if not frames:
            raise ValueError("AnimateDiff returned no frames")
        if not 1 <= fps <= 60:
            raise ValueError("AnimateDiff FPS must be between 1 and 60")
        for frame in frames:
            decoded.append(_decode_frame(frame))
        if any(frame.size != decoded[0].size for frame in decoded):
            raise ValueError("AnimateDiff frame dimensions do not match")
        if logger is not None:
            from easydiffusion.privacy_debug import log_result_image_diagnostics
            log_result_image_diagnostics(decoded, logger)
        buffer = BytesIO()
        # GIF delays are centiseconds. Distribute rounding rather than shorten
        # every frame (for example, 8 FPS alternates 120 and 130 ms).
        durations = [10 * (round((index + 1) * 100 / fps) - round(index * 100 / fps))
                     for index in range(len(decoded))]
        decoded[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=decoded[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )
        payload = buffer.getvalue()
        with Image.open(BytesIO(payload)) as validated:
            frame_count = int(getattr(validated, "n_frames", 1))
            if validated.format != "GIF" or not 1 <= frame_count <= len(decoded):
                raise ValueError("AnimateDiff GIF validation found an invalid frame count")
            width, height = validated.size
            playback_duration = 0
            for index in range(frame_count):
                validated.seek(index)
                validated.load()
                playback_duration += validated.info.get("duration", 0)
            # Identical consecutive frames may merge; their duration must not
            # disappear, and every encoded frame must decode successfully.
            if playback_duration != sum(durations) or (width, height) != decoded[0].size:
                raise ValueError("AnimateDiff GIF validation found incorrect duration or dimensions")
    except (OSError, ValueError) as exc:
        if logger is not None:
            logger.error("AnimateDiff GIF output validation failed: %s", exc)
        raise RuntimeError(f"AnimateDiff GIF output validation failed: {exc}") from exc
    finally:
        for frame in decoded:
            frame.close()

    if logger is not None:
        logger.info(
            "AnimateDiff GIF output validated: format=GIF frames=%d size=%dx%d bytes=%d source_frames=%d duration_ms=%d",
            frame_count,
            width,
            height,
            len(payload),
            len(frames),
            playback_duration,
        )
    return "data:image/gif;base64," + base64.b64encode(payload).decode("ascii"), {
        "format": "gif",
        "frame_count": frame_count,
        "source_frame_count": len(frames),
        "width": width,
        "height": height,
        "bytes": len(payload),
        "duration_ms": playback_duration,
    }
