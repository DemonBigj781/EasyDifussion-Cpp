"""Local llama.cpp vision backend for the AI Image Critic UI plugin."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
from collections import deque
from glob import glob
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import HTTPException
from pydantic import BaseModel

from easydiffusion.privacy_debug import redact_log_message


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "ai-critic"
TEXT_MODEL_DIR = Path(
    os.environ.get("AI_CRITIC_TEXT_MODEL_DIR", str(DEFAULT_MODEL_DIR / "text"))
).expanduser()
VISION_MODEL_DIR = Path(
    os.environ.get("AI_CRITIC_VISION_MODEL_DIR", str(DEFAULT_MODEL_DIR / "vision"))
).expanduser()
START_TIMEOUT_SECONDS = int(os.environ.get("AI_CRITIC_START_TIMEOUT", "300"))
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("AI_CRITIC_REQUEST_TIMEOUT", "300"))
MAX_IMAGE_CHARACTERS = 32 * 1024 * 1024

SYSTEM_PROMPT = """You are an expert AI image quality analyst. Examine the image carefully for generation artifacts, including anatomy errors, extra or missing body parts, distorted faces or hands, duplicated objects, unreadable text, inconsistent lighting or shadows, broken perspective, melting backgrounds, seams, blur, and unintended object merging.

Return only a valid JSON object with this schema and no markdown:
{
  "severity": "none|minor|moderate|severe",
  "issues": [
    {"area": "short label", "description": "what is wrong", "severity": "minor|moderate|severe"}
  ],
  "positive_prompt_additions": "comma-separated fixes or an empty string",
  "negative_prompt_additions": "comma-separated exclusions or an empty string",
  "parameter_suggestions": "brief generation-setting advice or an empty string",
  "summary": "one or two sentence summary"
}
If the image is clean, use severity "none" and an empty issues array."""


def server_executable() -> Optional[Path]:
    configured = os.environ.get("AI_CRITIC_LLAMA_SERVER") or os.environ.get("LLAMA_CPP_SERVER")
    root = Path(__file__).resolve().parents[3]
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        [
            root / "source" / "llama.cpp" / "build" / "bin" / "llama-server",
            root / "source" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            root / "backends" / "llama.cpp" / "llama-server",
        ]
    )
    candidates.extend(
        Path(path) for path in sorted(glob(str(root / "backends" / "sdkit3" / "*" / "llama-server")))
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


class CriticAnalyzeRequest(BaseModel):
    image: str
    text_model: str = ""
    vision_model: str
    context_length: int = 4096
    gpu_layers: int = 99


def _model_files(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.gguf")
        if path.is_file() and not path.is_symlink()
    )


def _resolve_model(root: Path, relative: str, field: str) -> Path:
    if not relative:
        raise HTTPException(status_code=400, detail=f"Select a {field}.")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HTTPException(status_code=403, detail=f"{field} is outside {root}.") from error
    if not candidate.is_file() or candidate.is_symlink() or candidate.suffix.lower() != ".gguf":
        raise HTTPException(status_code=404, detail=f"{field} was not found: {relative}")
    return candidate


def _looks_like_projector(path: Path) -> bool:
    name = path.name.lower()
    return "mmproj" in name or "projector" in name


def _projectors_beside(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.parent.glob("*.gguf")
        if candidate.is_file() and not candidate.is_symlink() and _looks_like_projector(candidate)
    )


def _resolve_model_pair(text_model: str, vision_model: str) -> tuple[Path, Path]:
    selected_vision = _resolve_model(VISION_MODEL_DIR, vision_model, "vision GGUF")
    if _looks_like_projector(selected_vision):
        main_model = _resolve_model(TEXT_MODEL_DIR, text_model, "text GGUF")
        return main_model, selected_vision

    projectors = _projectors_beside(selected_vision)
    if not projectors:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{selected_vision.name} has no mmproj/projector GGUF beside it. "
                "llama.cpp cannot inspect images with a text-only GGUF."
            ),
        )
    if len(projectors) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{selected_vision.name} has multiple projector GGUFs. "
                "Select the intended mmproj file directly."
            ),
        )
    return selected_vision, projectors[0]


def model_inventory() -> dict[str, Any]:
    executable = server_executable()
    text_models = _model_files(TEXT_MODEL_DIR)
    vision_models = _model_files(VISION_MODEL_DIR)
    entries = []
    for relative in vision_models:
        path = VISION_MODEL_DIR / relative
        is_projector = _looks_like_projector(path)
        companion_count = len(_projectors_beside(path)) if not is_projector else 1
        entries.append(
            {
                "model": relative,
                "kind": "projector" if is_projector else "bundle",
                "usable": (is_projector and bool(text_models)) or (not is_projector and companion_count == 1),
                "detail": (
                    "Uses the selected text GGUF."
                    if is_projector and text_models
                    else "A text GGUF is required in the configured text directory."
                    if is_projector
                    else "Vision model and companion projector found."
                    if companion_count == 1
                    else "Missing a companion mmproj/projector GGUF."
                ),
            }
        )
    return {
        "ready": executable is not None,
        "executable": str(executable) if executable else None,
        "text_directory": str(TEXT_MODEL_DIR),
        "vision_directory": str(VISION_MODEL_DIR),
        "text_models": text_models,
        "vision_models": entries,
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _VisionServer:
    def __init__(self, model: Path, projector: Path, context_length: int, gpu_layers: int):
        executable = server_executable()
        if executable is None:
            raise RuntimeError("llama-server is not built; run ./install.sh --llama-build")
        self.model = model
        self.projector = projector
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_lines: deque[str] = deque(maxlen=120)
        environment = os.environ.copy()
        library_dir = str(executable.parent)
        old_library_path = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            f"{library_dir}{os.pathsep}{old_library_path}" if old_library_path else library_dir
        )
        self.process = subprocess.Popen(
            [
                str(executable),
                "--model",
                str(model),
                "--mmproj",
                str(projector),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--ctx-size",
                str(context_length),
                "--gpu-layers",
                str(gpu_layers),
                "--no-webui",
            ],
            cwd=str(executable.parent),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._drain_thread = threading.Thread(target=self._drain_output, daemon=True)
        self._drain_thread.start()
        self._wait_until_ready()

    def _drain_output(self) -> None:
        if not self.process.stdout:
            return
        for line in self.process.stdout:
            self.log_lines.append(line.rstrip())

    def _safe_failure_detail(self) -> str:
        tail = "\n".join(list(self.log_lines)[-8:])
        if not tail:
            return f"llama-server exited with status {self.process.poll()}"
        return redact_log_message(tail, force=True)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server failed to start: {self._safe_failure_detail()}")
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        self.close()
        raise RuntimeError("Timed out while loading the local vision model.")

    def analyze(self, image: str) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this generated image."},
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": 1800,
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as error:
            raise RuntimeError("The local llama.cpp vision request failed.") from error
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("The local vision model returned no response text.") from error
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The local vision model returned no response text.")
        return content

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


_SERVER_LOCK = threading.Lock()
_CURRENT_KEY: Optional[tuple[str, str, int, int]] = None
_CURRENT_SERVER: Optional[_VisionServer] = None


def _server_for(model: Path, projector: Path, context_length: int, gpu_layers: int) -> _VisionServer:
    global _CURRENT_KEY, _CURRENT_SERVER
    key = (str(model), str(projector), context_length, gpu_layers)
    if _CURRENT_KEY == key and _CURRENT_SERVER is not None:
        return _CURRENT_SERVER
    if _CURRENT_SERVER is not None:
        _CURRENT_SERVER.close()
    _CURRENT_KEY = None
    _CURRENT_SERVER = _VisionServer(model, projector, context_length, gpu_layers)
    _CURRENT_KEY = key
    return _CURRENT_SERVER


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    candidates = [cleaned]
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("The local vision model did not return valid JSON.")


def _normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    severity_values = {"none", "minor", "moderate", "severe"}
    severity = str(report.get("severity", "moderate")).lower()
    if severity not in severity_values:
        severity = "moderate"
    issues = []
    raw_issues = report.get("issues", [])
    if isinstance(raw_issues, list):
        for issue in raw_issues[:50]:
            if not isinstance(issue, dict):
                continue
            issue_severity = str(issue.get("severity", "minor")).lower()
            if issue_severity not in severity_values - {"none"}:
                issue_severity = "minor"
            issues.append(
                {
                    "area": str(issue.get("area", "Image"))[:120],
                    "description": str(issue.get("description", ""))[:2000],
                    "severity": issue_severity,
                }
            )
    return {
        "severity": severity,
        "issues": issues,
        "positive_prompt_additions": str(report.get("positive_prompt_additions", ""))[:4000],
        "negative_prompt_additions": str(report.get("negative_prompt_additions", ""))[:4000],
        "parameter_suggestions": str(report.get("parameter_suggestions", ""))[:4000],
        "summary": str(report.get("summary", ""))[:4000],
    }


def analyze(request: CriticAnalyzeRequest) -> dict[str, Any]:
    if not request.image.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="The critic requires a base64 image data URL.")
    if len(request.image) > MAX_IMAGE_CHARACTERS:
        raise HTTPException(status_code=413, detail="The critic image is too large.")
    context_length = max(2048, min(int(request.context_length), 32768))
    gpu_layers = max(0, min(int(request.gpu_layers), 999))
    model, projector = _resolve_model_pair(request.text_model, request.vision_model)
    with _SERVER_LOCK:
        try:
            raw = _server_for(model, projector, context_length, gpu_layers).analyze(request.image)
            return _normalize_report(_extract_json(raw))
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=502, detail=str(error)) from error


def shutdown() -> None:
    global _CURRENT_KEY, _CURRENT_SERVER
    with _SERVER_LOCK:
        if _CURRENT_SERVER is not None:
            _CURRENT_SERVER.close()
        _CURRENT_SERVER = None
        _CURRENT_KEY = None
