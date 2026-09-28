"""Lifecycle controller for the separate local SpriteGPT 64px runtime."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(os.environ.get("SPRITEGPT_ROOT", "/mnt/38FEF88DFEF84522/sprite-diffusion")).expanduser().resolve()
PYTHON = Path(os.environ.get("SPRITEGPT_PYTHON", "/home/jack/.pyenv/versions/comfyui/bin/python")).expanduser()
HOST, PORT = "127.0.0.1", 11000
URL = f"http://{HOST}:{PORT}"
_lock = threading.RLock()
_process = None
_log_handle = None


def _available():
    if not (ROOT / "spritegpt_webui.py").is_file() or not (ROOT / "sprite-gpt-text64.pt").is_file():
        return False, f"SpriteGPT files are missing under {ROOT}"
    if not PYTHON.is_file():
        return False, f"Python runtime is missing: {PYTHON}"
    return True, "SpriteGPT files and Python runtime found"


def _probe():
    try:
        with urllib.request.urlopen(URL + "/api/status", timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def status():
    global _process, _log_handle
    with _lock:
        available, detail = _available()
        running = _probe()
        if _process is not None and _process.poll() is not None:
            code = _process.returncode
            _process = None
            if _log_handle is not None:
                _log_handle.close()
                _log_handle = None
            if not running:
                detail = f"SpriteGPT exited with code {code}; inspect training/logs/spritegpt-runtime.log"
        return {"available": available, "running": running, "url": URL if running else None,
                "detail": "SpriteGPT runtime ready" if running else detail}


def start():
    global _process, _log_handle
    with _lock:
        state = status()
        if state["running"]:
            return state
        if not state["available"]:
            raise ValueError(state["detail"])
        logs = Path(__file__).resolve().parent / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        _log_handle = (logs / "spritegpt-runtime.log").open("a", encoding="utf-8", buffering=1)
        env = os.environ.copy()
        env.update({"SPRITEGPT_HOST": HOST, "SPRITEGPT_PORT": str(PORT),
                    "SPRITEGPT_PYTHON": str(PYTHON), "PYTHONUNBUFFERED": "1"})
        try:
            _process = subprocess.Popen([str(PYTHON), str(ROOT / "spritegpt_webui.py")], cwd=ROOT,
                                        env=env, stdin=subprocess.DEVNULL, stdout=_log_handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            _log_handle.close()
            _log_handle = None
            raise ValueError(f"Could not start SpriteGPT: {exc}") from exc
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if _probe():
            return status()
        if _process is not None and _process.poll() is not None:
            break
        time.sleep(.25)
    return status()


def stop():
    global _process, _log_handle
    with _lock:
        if _process is not None:
            if _process.poll() is None:
                try:
                    os.killpg(_process.pid, signal.SIGTERM)
                    _process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(_process.pid, signal.SIGKILL)
                    _process.wait(timeout=5)
                except ProcessLookupError:
                    pass
            _process = None
        if _log_handle is not None:
            _log_handle.close()
            _log_handle = None
    return status()
