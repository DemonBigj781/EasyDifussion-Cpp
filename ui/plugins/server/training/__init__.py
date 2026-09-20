"""Training API. Runtime settings and executable paths are server-owned."""
from contextlib import contextmanager
import threading
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from training.service import Conflict, TrainingService

router = APIRouter()
_service = None
_service_lock = threading.Lock()


def service():
    global _service
    from easydiffusion import app, task_manager
    from ui.plugins.server.model_manager import get_model_dirs
    with _service_lock:
        if _service is None:
            _service = TrainingService(app.ROOT_DIR, get_model_dirs, task_manager.backend_is_idle)
        return _service


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@contextmanager
def generation_guard():
    try:
        with service().generation_guard():
            yield
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc


class DatasetRequest(BaseModel):
    dataset: str = Field(..., min_length=1, max_length=1024)


class CaptionRequest(DatasetRequest):
    image: str
    caption: str = Field(..., max_length=65536)
    previous: str = Field(..., max_length=65536)


class TagRequest(DatasetRequest):
    tagger: str = "wd-v1-4-moat-tagger-v2"
    trigger: str = Field("", max_length=200)
    threshold: float = Field(.35, ge=0, le=1)
    character_threshold: float = Field(.85, ge=0, le=1)
    exclude_tags: str = Field("", max_length=4096)


class TrainRequest(DatasetRequest):
    kind: Literal["lora", "embedding"] = "lora"
    architecture: Literal["sd15", "sdxl"] = "sd15"
    model: str
    output_name: str
    trigger: str = ""
    init_word: str = "object"
    steps: int = 1000
    batch_size: int = 1
    resolution: int = 512
    rank: int = 16
    vectors: int = 4
    save_every: int = 100
    seed: int = 42
    learning_rate: float = .0001
    precision: Literal["fp16", "bf16", "no"] = "fp16"


@router.get("/readiness")
def readiness():
    current = service()
    return {**call(current.readiness), "dataset_root": str(current.dataset_root)}


@router.get("/models")
def models():
    return {"models": call(service().models)}


@router.post("/dataset/scan")
def scan(request: DatasetRequest):
    return call(service().scan, request.dataset)


@router.post("/dataset/caption")
def caption(request: CaptionRequest):
    return call(service().save_caption, **request.model_dump())


@router.post("/autotag")
def autotag(request: TagRequest):
    from easydiffusion import app
    port = int(app.getConfig().get("net", {}).get("listen_port", 9000))
    return call(service().start, {**request.model_dump(), "command": "autotag"}, port=port)


@router.post("/jobs")
def start(request: TrainRequest):
    return call(service().start, request.model_dump())


@router.get("/jobs")
def history():
    return {"jobs": service().history()}


@router.get("/jobs/{job_id}")
def job(job_id: str):
    return call(service().get, job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    return call(service().cancel, job_id)


@router.post("/jobs/{job_id}/resume")
def resume(job_id: str):
    return call(service().start, {}, resume_id=job_id)


def shutdown():
    if _service is not None:
        _service.shutdown()
