"""Long-running local model conversion jobs used by the Model Tools UI."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from easydiffusion import app
from easydiffusion.utils import log
from fastapi import APIRouter, HTTPException


router = APIRouter()
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_ALLOWED_OUTTYPES = {"auto", "f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0"}
_OUTPUT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _models_root() -> Path:
    return Path(app.MODELS_DIR).expanduser().resolve()


def _llama_root() -> Path:
    return (Path(app.ROOT_DIR) / "source" / "llama.cpp").resolve()


def _inside(candidate: Path, root: Path, message: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(message) from exc
    return resolved


def _python_version(executable: Path) -> Optional[Tuple[int, int]]:
    try:
        result = subprocess.run(
            [str(executable), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        major, minor = result.stdout.strip().split(".", 1)
        return int(major), int(minor)
    except Exception:
        return None


def _converter_python() -> Optional[Path]:
    candidates = []
    configured = os.environ.get("LLAMA_CPP_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path(app.ROOT_DIR) / ".venv-llama-cpp" / "bin" / "python",
            Path(sys.executable),
        ]
    )
    system_python = shutil.which("python3")
    if system_python:
        candidates.append(Path(system_python))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            version = _python_version(candidate)
            if version and (3, 10) <= version < (3, 15):
                # Do not resolve a virtual environment's Python symlink to the
                # base interpreter; its original path is how Python discovers
                # pyvenv.cfg and the environment's installed packages.
                return candidate.absolute()
    return None


def converter_readiness() -> Dict[str, Any]:
    llama_root = _llama_root()
    script = llama_root / "convert_hf_to_gguf.py"
    python = _converter_python()
    result: Dict[str, Any] = {
        "ready": False,
        "source": str(llama_root),
        "script": str(script),
        "python": str(python) if python else None,
        "installCommand": "./install.sh --gguf-tools",
    }
    if not script.is_file():
        result["detail"] = "The vendored llama.cpp converter script is missing."
        return result
    if python is None:
        result["detail"] = "A Python 3.10-3.14 GGUF-tools environment is not installed."
        return result
    try:
        probe = subprocess.run(
            [
                str(python),
                "-c",
                "import numpy, sentencepiece, torch, transformers, google.protobuf",
            ],
            cwd=str(llama_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        result["detail"] = f"Unable to check converter dependencies: {exc}"
        return result
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "missing Python dependencies").strip().splitlines()[-1]
        result["detail"] = f"GGUF converter dependencies are incomplete: {detail}"
        return result
    result["ready"] = True
    result["detail"] = "llama.cpp Hugging Face conversion is ready."
    return result


def _resolve_source(value: str) -> Path:
    raw = Path(str(value or "").strip())
    if not str(raw) or raw.is_absolute():
        raise ValueError("Source must be a folder relative to the configured models directory")
    source = _inside(
        _models_root() / raw,
        _models_root(),
        "Source model folder must be inside the configured models directory",
    )
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Source model folder does not exist or is a symbolic link")
    if not (source / "config.json").is_file():
        raise ValueError("Source model folder must contain config.json")
    return source


def _resolve_output(value: str, outtype: str) -> Path:
    filename = _OUTPUT_RE.sub("-", str(value or "model").strip()).strip("-.") or "model"
    if filename.lower().endswith(".gguf"):
        filename = filename[:-5]
    output_dir = _inside(
        _models_root() / "Image_GGUF",
        _models_root(),
        "GGUF output directory must be inside the configured models directory",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{filename}.{outtype}.gguf"


def _set_job(job_id: str, **fields: Any) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.setdefault(job_id, {})
        job.update(fields)
        if len(_JOBS) > 128:
            completed = [key for key, value in _JOBS.items() if value.get("status") in {"completed", "failed"}]
            for key in completed[: len(_JOBS) - 128]:
                _JOBS.pop(key, None)
        return dict(job)


def _append_log(job_id: str, line: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS.setdefault(job_id, {})
        lines = list(job.get("log") or [])
        lines.append(str(line).rstrip())
        job["log"] = lines[-300:]


def _conversion_worker(job_id: str, source: Path, output: Path, outtype: str, python: Path) -> None:
    start = time.perf_counter()
    temporary_output = output.with_name(f".{output.name}.{job_id}.partial")
    command = [
        str(python),
        "-u",
        str(_llama_root() / "convert_hf_to_gguf.py"),
        "--outfile",
        str(temporary_output),
        "--outtype",
        outtype,
        str(source),
    ]
    _set_job(job_id, status="running", startedAt=time.time())
    try:
        process = subprocess.Popen(
            command,
            cwd=str(_llama_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _set_job(job_id, pid=process.pid)
        if process.stdout:
            for line in process.stdout:
                _append_log(job_id, line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"llama.cpp converter exited with status {return_code}")
        if not temporary_output.is_file():
            raise RuntimeError("llama.cpp reported success but did not create the GGUF output")
        if output.exists():
            raise RuntimeError(f"GGUF output was created by another process: {output.name}")
        temporary_output.replace(output)
        _set_job(
            job_id,
            status="completed",
            output=str(output),
            size=output.stat().st_size,
            completedAt=time.time(),
            elapsedSeconds=round(time.perf_counter() - start, 2),
        )
    except Exception as exc:
        try:
            temporary_output.unlink(missing_ok=True)
        except OSError:
            pass
        log.error(f"GGUF conversion failed: {exc}")
        _append_log(job_id, str(exc))
        _set_job(
            job_id,
            status="failed",
            error=str(exc),
            completedAt=time.time(),
            elapsedSeconds=round(time.perf_counter() - start, 2),
        )


@router.get("/gguf/readiness")
def gguf_readiness():
    return converter_readiness()


@router.post("/gguf/convert")
def gguf_convert(payload: Dict[str, Any]):
    source = _resolve_source(str(payload.get("source") or ""))
    outtype = str(payload.get("outtype") or "auto").lower()
    if outtype not in _ALLOWED_OUTTYPES:
        raise HTTPException(status_code=400, detail="Unsupported GGUF output type")
    output = _resolve_output(str(payload.get("outputName") or source.name), outtype)
    if output.exists():
        raise HTTPException(status_code=409, detail=f"Output already exists: {output.name}")
    readiness = converter_readiness()
    if not readiness["ready"]:
        raise HTTPException(status_code=503, detail=readiness["detail"])
    job_id = uuid4().hex
    with _JOBS_LOCK:
        if any(job.get("status") in {"queued", "running"} for job in _JOBS.values()):
            raise HTTPException(status_code=409, detail="Another GGUF conversion is already running")
        _JOBS[job_id] = {
            "status": "queued",
            "source": str(source),
            "output": str(output),
            "outtype": outtype,
            "log": [],
            "createdAt": time.time(),
        }
    worker = threading.Thread(
        target=_conversion_worker,
        args=(job_id, source, output, outtype, Path(readiness["python"])),
        daemon=True,
    )
    worker.start()
    return {"ok": True, "jobId": job_id, "status": "queued"}


@router.get("/gguf/jobs/{job_id}")
def gguf_job(job_id: str):
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id) or {})
    if not job:
        raise HTTPException(status_code=404, detail="GGUF conversion job not found")
    return {"ok": True, "jobId": job_id, **job}
