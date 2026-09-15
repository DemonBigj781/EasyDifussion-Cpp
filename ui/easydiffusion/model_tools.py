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
from easydiffusion.utils.model_identifier import identify_model_type
from fastapi import APIRouter, HTTPException


router = APIRouter()
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_LLAMA_OUTTYPES = {"auto", "f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0"}
_NATIVE_OUTTYPES = {
    "auto", "f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1",
    "tq1_0", "tq2_0",
}
_CHECKPOINT_EXTENSIONS = {".ckpt", ".safetensors", ".pt", ".pth", ".sft"}
_EMBEDDING_SOURCE_EXTENSIONS = {".pt", ".pth", ".bin", ".ckpt"}
_GGUF_ARCHITECTURE_FOLDERS = {
    "sd15": "1.5",
    "sd3": "3.0",
    "anima": "anima",
    "sdxl": "sdxl",
}
_CONVERTER_SCRIPTS = {
    "huggingface": "convert_hf_to_gguf.py",
    "tokenizer-update": "convert_hf_to_gguf_update.py",
    "legacy-ggml": "convert_llama_ggml_to_gguf.py",
    "lora": "convert_lora_to_gguf.py",
}
_OUTPUT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _models_root() -> Path:
    return Path(app.MODELS_DIR).expanduser().resolve()


def _llama_root() -> Path:
    return (Path(app.ROOT_DIR) / "source" / "llama.cpp").resolve()


def _native_sdkit_binary() -> Optional[Path]:
    executable = "sdkit.exe" if os.name == "nt" else "sdkit"
    candidates = []
    configured = os.environ.get("SDKIT_BINARY")
    if configured:
        candidates.append(Path(configured).expanduser())
    try:
        from easydiffusion.backends.sdkit3 import get_backend_dir

        candidates.append(Path(get_backend_dir()) / executable)
    except Exception:
        pass
    candidates.extend(
        sorted((Path(app.ROOT_DIR) / "backends" / "sdkit3").glob(f"*/{executable}"))
    )
    candidates.extend(
        sorted(
            (Path(app.ROOT_DIR) / "source" / "sdkit3-port-source" / "build").glob(
                f"*/bin/{executable}"
            )
        )
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.absolute()
    return None


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
            Path(sys.executable),
            Path(app.ROOT_DIR) / ".venv" / "bin" / "python",
        ]
    )
    system_python = shutil.which("python3")
    if system_python:
        candidates.append(Path(system_python))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            version = _python_version(candidate)
            if version == (3, 13):
                # Do not resolve a virtual environment's Python symlink to the
                # base interpreter; its original path is how Python discovers
                # pyvenv.cfg and the environment's installed packages.
                return candidate.absolute()
    return None


def converter_readiness() -> Dict[str, Any]:
    llama_root = _llama_root()
    script = llama_root / "convert_hf_to_gguf.py"
    python = _converter_python()
    converters = {name: str(llama_root / filename) for name, filename in _CONVERTER_SCRIPTS.items()}
    native_binary = _native_sdkit_binary()
    result: Dict[str, Any] = {
        "ready": False,
        "source": str(llama_root),
        "script": str(script),
        "converters": converters,
        "python": str(python) if python else None,
        "installCommand": "./install.sh --gguf-tools",
        "llamaReady": False,
        "nativeReady": False,
        "nativeBinary": str(native_binary) if native_binary else None,
    }
    missing = [name for name, path in converters.items() if not Path(path).is_file()]
    if missing:
        llama_detail = f"Vendored llama.cpp converters are missing: {', '.join(missing)}."
    elif python is None:
        llama_detail = "The Python 3.13 Easy Diffusion environment is unavailable."
    else:
        try:
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import gguf, numpy, requests, sentencepiece, torch, transformers, google.protobuf",
                ],
                cwd=str(llama_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if probe.returncode == 0:
                result["llamaReady"] = True
                llama_detail = "llama.cpp Hugging Face conversion is ready."
            else:
                detail = (probe.stderr or probe.stdout or "missing Python dependencies").strip().splitlines()[-1]
                llama_detail = f"GGUF converter dependencies are incomplete: {detail}"
        except Exception as exc:
            llama_detail = f"Unable to check converter dependencies: {exc}"

    if native_binary is None:
        native_detail = "The native sdkit checkpoint converter is not installed."
    else:
        try:
            native_probe = subprocess.run(
                [str(native_binary), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            native_help = f"{native_probe.stdout}\n{native_probe.stderr}"
            if native_probe.returncode == 0 and "--convert-model" in native_help:
                result["nativeReady"] = True
                native_detail = "Native checkpoint and Diffusers conversion is ready."
            else:
                native_detail = "The installed sdkit binary predates native checkpoint conversion; rebuild it."
        except Exception as exc:
            native_detail = f"Unable to check the native checkpoint converter: {exc}"

    result["ready"] = bool(result["llamaReady"] or result["nativeReady"])
    result["llamaDetail"] = llama_detail
    result["nativeDetail"] = native_detail
    result["detail"] = f"{native_detail} {llama_detail}"
    return result


def _resolve_source(value: str) -> Path:
    raw = Path(str(value or "").strip())
    if not str(raw) or raw.is_absolute():
        raise ValueError("Source must be a file or folder relative to the configured models directory")
    source = _inside(
        _models_root() / raw,
        _models_root(),
        "Source model must be inside the configured models directory",
    )
    if source.is_file():
        if source.suffix.lower() not in _CHECKPOINT_EXTENSIONS:
            raise ValueError("Checkpoint source must be .safetensors, .ckpt, .pt, .pth, or .sft")
        return source
    if not source.is_dir():
        raise ValueError("Source model does not exist")
    if not (source / "config.json").is_file() and not (source / "model_index.json").is_file():
        raise ValueError("Model folder must contain config.json or model_index.json")
    return source


def _source_kind(source: Path) -> str:
    if source.is_file() or (source / "model_index.json").is_file():
        return "native"
    return "llama"


def _checkpoint_sources(limit: int = 4000) -> list[Dict[str, Any]]:
    root = _models_root()
    checkpoint_root = root / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    sources = []
    for current, directories, filenames in os.walk(checkpoint_root, followlinks=False):
        directories[:] = sorted(name for name in directories if not name.startswith("."))
        for filename in sorted(filenames):
            candidate = Path(current) / filename
            if candidate.suffix.lower() not in _CHECKPOINT_EXTENSIONS:
                continue
            try:
                resolved = _inside(candidate, root, "Checkpoint is outside the models directory")
                relative = candidate.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            sources.append(
                {
                    "path": relative,
                    "name": candidate.stem,
                    "kind": "native",
                    "format": resolved.suffix.lower().lstrip("."),
                    "size": resolved.stat().st_size,
                    **_gguf_source_classification(resolved),
                }
            )
            if len(sources) >= limit:
                return sources
    return sources


def _gguf_family_from_model_type(model_type: str) -> Optional[str]:
    normalized = str(model_type or "").lower()
    if normalized == "anima":
        return "anima"
    if normalized.startswith("sd_v3"):
        return "sd3"
    if normalized.startswith(("sd_xl", "playground_v2_5")):
        return "sdxl"
    if normalized.startswith(("sd_v1", "sd_v2")) or normalized == "instruct_pix2pix":
        return "sd15"
    return None


def _gguf_family_from_path(source: Path) -> Optional[str]:
    hint = "/" + str(source).replace("\\", "/").lower().strip("/") + "/"
    if "anima" in hint:
        return "anima"
    if any(token in hint for token in ("/sdxl/", "/sd_xl/", "pony", "illustrious", "noobai")):
        return "sdxl"
    if any(token in hint for token in ("/3.0/", "/sd3/", "stable-diffusion-3")):
        return "sd3"
    if any(token in hint for token in ("/1.5/", "/sd1.5/", "/sd15/", "stable-diffusion-1")):
        return "sd15"
    return None


def _detect_gguf_family(source: Path) -> str:
    if source.is_file() and source.suffix.lower() in {".safetensors", ".sft"}:
        try:
            family = _gguf_family_from_model_type(identify_model_type(str(source)))
            if family:
                return family
        except Exception:
            pass

    family = _gguf_family_from_path(source)
    if family:
        return family

    if source.is_dir():
        for config_name in ("model_index.json", "config.json", "unet/config.json", "transformer/config.json"):
            config_path = source / config_name
            if not config_path.is_file():
                continue
            try:
                config_hint = config_path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if "anima" in config_hint:
                return "anima"
            if "stablediffusion3" in config_hint or "stable-diffusion-3" in config_hint:
                return "sd3"
            if "stablediffusionxl" in config_hint or "stable-diffusion-xl" in config_hint:
                return "sdxl"
            if "stablediffusion" in config_hint:
                return "sd15"

    raise ValueError(
        "Could not classify this image model as SD 1.5, SD 3, Anima, or SDXL; "
        "place the source under a matching checkpoints architecture folder"
    )


def _gguf_source_classification(source: Path) -> Dict[str, Any]:
    try:
        family = _detect_gguf_family(source)
        return {
            "architecture": family,
            "targetFolder": f"Image_GGUF/{_GGUF_ARCHITECTURE_FOLDERS[family]}",
        }
    except ValueError as exc:
        return {"architecture": None, "targetFolder": None, "detectionError": str(exc)}


def _resolve_output(value: str, outtype: str, family: str) -> Path:
    filename = _OUTPUT_RE.sub("-", str(value or "model").strip()).strip("-.") or "model"
    if filename.lower().endswith(".gguf"):
        filename = filename[:-5]
    folder_name = _GGUF_ARCHITECTURE_FOLDERS.get(family)
    if folder_name is None:
        raise ValueError("Unsupported GGUF image architecture")
    output_dir = _inside(
        _models_root() / "Image_GGUF" / folder_name,
        _models_root(),
        "GGUF output directory must be inside the configured models directory",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{filename}.{outtype}.gguf"


def _embedding_roots() -> list[Path]:
    from easydiffusion.model_manager import get_model_dirs

    roots = []
    seen = set()
    for value in get_model_dirs("embeddings"):
        root = Path(value).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return roots


def _embedding_display_path(candidate: Path) -> str:
    try:
        return candidate.relative_to(_models_root()).as_posix()
    except ValueError:
        return str(candidate)


def _resolve_embedding_source(value: str) -> Path:
    text_value = str(value or "").strip()
    if not text_value:
        raise ValueError("Choose an embedding source file")
    raw = Path(text_value).expanduser()
    candidates = [raw] if raw.is_absolute() else [_models_root() / raw]
    if not raw.is_absolute():
        candidates.extend(root / raw for root in _embedding_roots())

    roots = _embedding_roots()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_file():
            continue
        inside_root = False
        for root in roots:
            try:
                resolved.relative_to(root)
                inside_root = True
                break
            except ValueError:
                continue
        if not inside_root:
            continue
        if resolved.suffix.lower() not in _EMBEDDING_SOURCE_EXTENSIONS:
            raise ValueError("Embedding source must be .pt, .pth, .bin, or .ckpt")
        return resolved
    raise ValueError("Embedding source must be a file inside a configured embeddings directory")


def _embedding_sources(limit: int = 4000) -> list[Dict[str, Any]]:
    sources = []
    seen = set()
    for root in _embedding_roots():
        if not root.is_dir():
            continue
        for current, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = sorted(name for name in directories if not name.startswith("."))
            for filename in sorted(filenames):
                candidate = Path(current) / filename
                if candidate.suffix.lower() not in _EMBEDDING_SOURCE_EXTENSIONS:
                    continue
                try:
                    resolved = _inside(candidate, root, "Embedding is outside its configured directory")
                except (OSError, ValueError):
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                output = resolved.with_suffix(".safetensors")
                sources.append(
                    {
                        "path": _embedding_display_path(resolved),
                        "name": resolved.stem,
                        "format": resolved.suffix.lower().lstrip("."),
                        "size": resolved.stat().st_size,
                        "output": _embedding_display_path(output),
                        "outputExists": output.exists(),
                    }
                )
                if len(sources) >= limit:
                    return sources
    return sources


def safetensors_readiness_state() -> Dict[str, Any]:
    try:
        import torch  # noqa: F401
        from safetensors.torch import save_file  # noqa: F401

        return {"ready": True, "detail": "Secure embedding conversion is ready."}
    except Exception as exc:
        return {
            "ready": False,
            "detail": f"Embedding conversion needs PyTorch and safetensors in the Easy Diffusion environment: {exc}",
        }


def _embedding_tensor_map(payload: Any, torch_module: Any) -> Dict[str, Any]:
    if torch_module.is_tensor(payload):
        return {"emb_params": payload.detach().cpu().contiguous()}
    if not isinstance(payload, dict):
        raise ValueError("Embedding file does not contain a tensor mapping")

    container = payload
    for wrapper in ("string_to_param", "state_dict"):
        wrapped = container.get(wrapper)
        if isinstance(wrapped, dict) and any(torch_module.is_tensor(value) for value in wrapped.values()):
            container = wrapped
            break

    tensors = {
        str(key): value.detach().cpu().contiguous()
        for key, value in container.items()
        if torch_module.is_tensor(value)
    }
    if not tensors:
        raise ValueError("Embedding file does not contain any tensors")
    if len(tensors) == 1 and not any(key in tensors for key in ("emb_params", "clip_l", "clip_g")):
        return {"emb_params": next(iter(tensors.values()))}
    return tensors


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


def _conversion_worker(
    job_id: str,
    source: Path,
    output: Path,
    outtype: str,
    kind: str,
    runtime: Path,
) -> None:
    start = time.perf_counter()
    temporary_output = output.with_name(f".{output.name}.{job_id}.partial")
    if kind == "native":
        command = [
            str(runtime),
            "--convert-model",
            str(source),
            "--convert-output",
            str(temporary_output),
            "--convert-type",
            outtype,
        ]
        working_directory = runtime.parent
    else:
        command = [
            str(runtime),
            "-u",
            str(_llama_root() / "convert_hf_to_gguf.py"),
            "--outfile",
            str(temporary_output),
            "--outtype",
            outtype,
            str(source),
        ]
        working_directory = _llama_root()
    _set_job(job_id, status="running", startedAt=time.time())
    try:
        process = subprocess.Popen(
            command,
            cwd=str(working_directory),
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
            converter_name = "native checkpoint converter" if kind == "native" else "llama.cpp converter"
            raise RuntimeError(f"{converter_name} exited with status {return_code}")
        if not temporary_output.is_file():
            raise RuntimeError("Converter reported success but did not create the GGUF output")
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


def _safetensors_conversion_worker(job_id: str, source: Path, output: Path) -> None:
    start = time.perf_counter()
    temporary_output = output.with_name(f".{output.stem}.{job_id}.partial.safetensors")
    _set_job(job_id, status="running", startedAt=time.time())
    try:
        import torch
        from safetensors.torch import save_file

        # weights_only is deliberately mandatory. Legacy pickle loading can
        # execute code embedded in a downloaded .pt/.bin file.
        payload = torch.load(str(source), map_location="cpu", weights_only=True)
        tensors = _embedding_tensor_map(payload, torch)
        save_file(
            tensors,
            str(temporary_output),
            metadata={"converted_from": source.name, "converter": "Easy Diffusion Model Tools"},
        )
        if output.exists():
            raise RuntimeError(f"Safetensors output was created by another process: {output.name}")
        temporary_output.replace(output)
        _set_job(
            job_id,
            status="completed",
            output=str(output),
            tensorCount=len(tensors),
            size=output.stat().st_size,
            completedAt=time.time(),
            elapsedSeconds=round(time.perf_counter() - start, 2),
        )
    except Exception as exc:
        try:
            temporary_output.unlink(missing_ok=True)
        except OSError:
            pass
        log.error(f"Embedding safetensors conversion failed: {exc}")
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


@router.get("/gguf/sources")
def gguf_sources():
    sources = _checkpoint_sources()
    return {"ok": True, "sources": sources, "count": len(sources)}


@router.post("/gguf/convert")
def gguf_convert(payload: Dict[str, Any]):
    try:
        source = _resolve_source(str(payload.get("source") or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    kind = _source_kind(source)
    outtype = str(payload.get("outtype") or "auto").lower()
    allowed_outtypes = _NATIVE_OUTTYPES if kind == "native" else _LLAMA_OUTTYPES
    if outtype not in allowed_outtypes:
        raise HTTPException(status_code=400, detail=f"Unsupported {kind} GGUF output type")
    # stable-diffusion.cpp conversion needs a concrete tensor type. For a
    # checkpoint, Auto means the broadly compatible F16 export.
    if kind == "native" and outtype == "auto":
        outtype = "f16"
    try:
        architecture = _detect_gguf_family(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    default_output_name = source.stem if source.is_file() else source.name
    output = _resolve_output(str(payload.get("outputName") or default_output_name), outtype, architecture)
    if output.exists():
        raise HTTPException(status_code=409, detail=f"Output already exists: {output.name}")
    readiness = converter_readiness()
    readiness_key = "nativeReady" if kind == "native" else "llamaReady"
    detail_key = "nativeDetail" if kind == "native" else "llamaDetail"
    if not readiness[readiness_key]:
        raise HTTPException(status_code=503, detail=readiness[detail_key])
    runtime_value = readiness["nativeBinary"] if kind == "native" else readiness["python"]
    if not runtime_value:
        raise HTTPException(status_code=503, detail=readiness[detail_key])
    job_id = uuid4().hex
    with _JOBS_LOCK:
        if any(job.get("status") in {"queued", "running"} for job in _JOBS.values()):
            raise HTTPException(status_code=409, detail="Another GGUF conversion is already running")
        _JOBS[job_id] = {
            "jobType": "gguf",
            "status": "queued",
            "source": str(source),
            "output": str(output),
            "outtype": outtype,
            "kind": kind,
            "architecture": architecture,
            "targetFolder": f"Image_GGUF/{_GGUF_ARCHITECTURE_FOLDERS[architecture]}",
            "log": [],
            "createdAt": time.time(),
        }
    worker = threading.Thread(
        target=_conversion_worker,
        args=(job_id, source, output, outtype, kind, Path(runtime_value)),
        daemon=True,
    )
    worker.start()
    return {"ok": True, "jobId": job_id, "status": "queued"}


@router.get("/gguf/jobs/{job_id}")
def gguf_job(job_id: str):
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id) or {})
    if not job or job.get("jobType") != "gguf":
        raise HTTPException(status_code=404, detail="GGUF conversion job not found")
    return {"ok": True, "jobId": job_id, **job}


@router.get("/safetensors/readiness")
def safetensors_readiness():
    return safetensors_readiness_state()


@router.get("/safetensors/sources")
def safetensors_sources():
    sources = _embedding_sources()
    return {"ok": True, "sources": sources, "count": len(sources)}


@router.post("/safetensors/convert")
def safetensors_convert(payload: Dict[str, Any]):
    try:
        source = _resolve_embedding_source(str(payload.get("source") or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    output = source.with_suffix(".safetensors")
    if output.exists():
        raise HTTPException(status_code=409, detail=f"Output already exists: {output.name}")
    readiness = safetensors_readiness_state()
    if not readiness["ready"]:
        raise HTTPException(status_code=503, detail=readiness["detail"])

    job_id = uuid4().hex
    with _JOBS_LOCK:
        if any(job.get("status") in {"queued", "running"} for job in _JOBS.values()):
            raise HTTPException(status_code=409, detail="Another model conversion is already running")
        _JOBS[job_id] = {
            "jobType": "safetensors",
            "status": "queued",
            "source": str(source),
            "output": str(output),
            "log": [],
            "createdAt": time.time(),
        }
    worker = threading.Thread(
        target=_safetensors_conversion_worker,
        args=(job_id, source, output),
        daemon=True,
    )
    worker.start()
    return {"ok": True, "jobId": job_id, "status": "queued"}


@router.get("/safetensors/jobs/{job_id}")
def safetensors_job(job_id: str):
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id) or {})
    if not job or job.get("jobType") != "safetensors":
        raise HTTPException(status_code=404, detail="Safetensors conversion job not found")
    return {"ok": True, "jobId": job_id, **job}
