"""Managed local llama.cpp server backend for TIPO prompt generation."""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from glob import glob
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from easydiffusion import app as easy_app
from easydiffusion.utils import log as _logger


_START_TIMEOUT = int(os.environ.get("TIPO_LLAMA_START_TIMEOUT", "300"))
_REQUEST_TIMEOUT = int(os.environ.get("TIPO_LLAMA_REQUEST_TIMEOUT", "600"))


def server_executable() -> Optional[Path]:
    candidates = []
    configured = os.environ.get("TIPO_LLAMA_SERVER") or os.environ.get("LLAMA_CPP_SERVER")
    if configured:
        candidates.append(Path(configured).expanduser())
    llama_root = Path(easy_app.ROOT_DIR) / "source" / "llama.cpp"
    candidates.extend(
        [
            llama_root / "build" / "bin" / "llama-server",
            llama_root / "build" / "bin" / "Release" / "llama-server.exe",
            Path(easy_app.ROOT_DIR) / "backends" / "llama.cpp" / "llama-server",
        ]
    )
    candidates.extend(
        Path(path)
        for path in sorted(glob(str(Path(easy_app.ROOT_DIR) / "backends" / "sdkit3" / "*" / "llama-server")))
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def backend_readiness() -> Dict[str, Any]:
    executable = server_executable()
    if executable is None:
        return {
            "ready": False,
            "backend": "llama.cpp server",
            "detail": "llama-server is not built. Run ./install.sh --llama-build.",
            "executable": None,
        }
    return {
        "ready": True,
        "backend": "llama.cpp server",
        "detail": "The vendored llama.cpp server is ready.",
        "executable": str(executable),
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaServer:
    def __init__(self, model_path: Path, context_length: int, gpu_layers: int):
        executable = server_executable()
        if executable is None:
            raise RuntimeError("llama-server is not built; run ./install.sh --llama-build")
        self.model_path = model_path
        self.context_length = int(context_length)
        self.gpu_layers = int(gpu_layers)
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_lines = deque(maxlen=200)
        self.process = subprocess.Popen(
            [
                str(executable),
                "--model",
                str(model_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--ctx-size",
                str(self.context_length),
                "--gpu-layers",
                str(self.gpu_layers),
                "--no-webui",
            ],
            cwd=str(executable.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_thread = threading.Thread(target=self._drain_log, daemon=True)
        self._log_thread.start()
        self._wait_until_ready()

    def _drain_log(self) -> None:
        if not self.process.stdout:
            return
        for line in self.process.stdout:
            clean = line.rstrip()
            self.log_lines.append(clean)
            if clean:
                _logger.debug(f"TIPO llama.cpp: {clean}")

    def _failure_detail(self) -> str:
        tail = "\n".join(list(self.log_lines)[-12:])
        return tail or f"llama-server exited with status {self.process.poll()}"

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + _START_TIMEOUT
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(f"llama-server failed to start: {self._failure_detail()}")
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    _logger.info(
                        f"TIPO llama.cpp server ready: {self.model_path.name} "
                        f"(context={self.context_length}, gpu_layers={self.gpu_layers})"
                    )
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        self.close()
        raise RuntimeError(f"Timed out waiting for llama-server: {self._failure_detail()}")

    def create_completion(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "temperature": float(kwargs.get("temperature", 0.8)),
            "top_p": float(kwargs.get("top_p", 0.95)),
            "min_p": float(kwargs.get("min_p", 0.05)),
            "top_k": int(kwargs.get("top_k", 40)),
            "n_predict": int(kwargs.get("max_tokens", 256)),
            "repeat_penalty": float(kwargs.get("repeat_penalty", 1.0)),
            "seed": int(kwargs.get("seed", -1)),
            "cache_prompt": False,
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"llama.cpp completion failed: {exc}; {self._failure_detail()}") from exc
        content = result.get("content")
        if not isinstance(content, str):
            raise RuntimeError("llama.cpp completion response did not contain text")
        return {"choices": [{"text": content}], "llama_server": result}

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
