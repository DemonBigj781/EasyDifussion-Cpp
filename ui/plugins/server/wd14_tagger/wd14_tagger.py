"""Built-in WD14 image tagging support for Easy Diffusion."""

import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading

from pydantic import BaseModel, Field
from easydiffusion.privacy_debug import block_alphabetic_context, report_safe_path


DEFAULT_MODEL = "wd-v1-4-moat-tagger-v2"

_native_lock = threading.Lock()
_native_process = None
_logger = logging.getLogger("easydiffusion.wd14_tagger")


def _censor(value, limit=320):
    return block_alphabetic_context(str(value)[:limit])


def _forward_native_diagnostics(stream):
    model_event = re.compile(r"^model_loaded provider=(CPUExecutionProvider|CUDAExecutionProvider) labels=(\d+)$")
    result_event = re.compile(r"^result matches=(\d+) ratings=(\d+)$")
    for line in stream:
        event = line.strip()
        if event == "WD14_DIAG activated backend=cpp":
            _logger.info("WD14 tagger activated backend=cpp")
        elif event.startswith("WD14_DIAG "):
            detail = event.removeprefix("WD14_DIAG ")
            match = model_event.fullmatch(detail)
            if match:
                _logger.debug("WD14 C++ model loaded provider=%s labels=%s", *match.groups())
                continue
            match = result_event.fullmatch(detail)
            if match:
                _logger.debug("WD14 C++ result matches=%s ratings=%s", *match.groups())
            elif detail == "request_failed":
                _logger.warning("WD14 C++ request failed; details omitted")


def _native_executable():
    configured = os.environ.get("SDKIT_WD14_TAGGER")
    if configured:
        return configured if os.path.isfile(configured) else None
    found = shutil.which("sdkit-wd14-tagger")
    if found:
        return found
    root = Path(__file__).resolve().parents[4]
    candidates = [root / "source/sdkit3-port-source/build/bin/sdkit-wd14-tagger"]
    candidates.extend((root / "source/sdkit3-port-source/build").glob("*/bin/sdkit-wd14-tagger"))
    return next((str(path) for path in candidates if path.is_file()), None)


def _native_tag_image(request, model_name, model_path, csv_path):
    global _native_process
    executable = _native_executable()
    if not executable:
        raise RuntimeError("C++ WD14 tagger is unavailable; build sdkit-wd14-tagger first")
    payload = {"image": request.image, "model": model_name, "model_path": model_path,
               "csv_path": csv_path, "threshold": request.threshold,
               "character_threshold": request.character_threshold,
               "exclude_tags": request.exclude_tags,
               "replace_underscore": request.replace_underscore,
               "trailing_comma": request.trailing_comma}
    with _native_lock:
        try:
            if _native_process is None or _native_process.poll() is not None:
                _logger.debug("WD14 native process starting executable=%s",
                              _censor(report_safe_path(executable)))
                _native_process = subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE, text=True, bufsize=1)
                threading.Thread(target=_forward_native_diagnostics, args=(_native_process.stderr,),
                                 name="WD14 C++ diagnostics", daemon=True).start()
            _native_process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            _native_process.stdin.flush()
            response = json.loads(_native_process.stdout.readline())
        except (OSError, ValueError, BrokenPipeError) as exc:
            if _native_process is not None:
                _native_process.kill()
                _native_process = None
            _logger.warning("WD14 native request failed error_type=%s detail=%s",
                            type(exc).__name__, _censor(exc))
            raise RuntimeError(f"Native WD14 tagger failed: {exc}") from exc
    if "error" in response:
        _logger.warning("WD14 native inference failed detail=%s", _censor(response["error"]))
        raise RuntimeError(response["error"])
    return response


class WD14TagRequest(BaseModel):
    image: str
    model: str = DEFAULT_MODEL
    threshold: float = Field(0.35, ge=0.0, le=1.0)
    character_threshold: float = Field(0.85, ge=0.0, le=1.0)
    exclude_tags: str = ""
    replace_underscore: bool = False
    trailing_comma: bool = False


def _resolve_files(model_name: str):
    from easydiffusion.model_manager import resolve_model_to_use

    model_name = os.path.splitext(os.path.basename(model_name))[0]
    model_path = resolve_model_to_use(model_name, "wd14-tagger")
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"WD14 model not found: {model_name}")
    csv_path = os.path.splitext(model_path)[0] + ".csv"
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"WD14 tag CSV not found: {csv_path}")
    return model_name, model_path, csv_path


def tag_image(request: WD14TagRequest):
    model_name, model_path, csv_path = _resolve_files(request.model)
    return _native_tag_image(request, model_name, model_path, csv_path)
