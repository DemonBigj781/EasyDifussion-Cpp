"""Training API. Runtime settings and executable paths are server-owned."""
from contextlib import contextmanager
from pathlib import Path
import threading
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from training.service import Conflict, TrainingService
from training import scraper, spritegpt_runtime, trainer

router = APIRouter()
_service = None
_service_lock = threading.Lock()


def service():
    global _service
    from easydiffusion import app, task_manager
    from ui.plugins.server.Browse_files.model_manager import get_model_dirs
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


class ScrapeSearchRequest(DatasetRequest):
    query: str = Field(..., min_length=1, max_length=500)
    sources: list[str] = Field(default_factory=list, max_length=50)
    limit: int = Field(80, ge=1, le=300)


class ScrapeTagSuggestionsRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=100)
    sources: list[str] = Field(default_factory=list, max_length=50)


class ScrapeDownloadRequest(DatasetRequest):
    attempt: str
    keys: list[str] = Field(..., min_length=1, max_length=300)
    threshold: float = Field(.95, ge=.5, le=1)


class ScrapeAttemptRequest(DatasetRequest):
    attempt: str


class ScrapeNamesRequest(ScrapeAttemptRequest):
    names: list[str] = Field(default_factory=list, max_length=300)
    frame_names: list[str] = Field(default_factory=list, max_length=300)


class ScrapeArchiveRequest(ScrapeNamesRequest):
    target_dataset: str = Field(..., min_length=1, max_length=1024)


class ScrapeCaptionRequest(ScrapeAttemptRequest):
    name: str
    caption: str = Field(..., max_length=65536)
    kind: Literal["filtered", "frames"] = "filtered"


class ScrapeTagRequest(ScrapeAttemptRequest):
    replace_existing: bool = True
    tagger: str = "wd-v1-4-moat-tagger-v2"
    threshold: float = Field(.35, ge=0, le=1)
    character_threshold: float = Field(.85, ge=0, le=1)
    exclude_tags: str = ""


class ScrapeImageTagRequest(ScrapeAttemptRequest):
    name: str
    kind: Literal["filtered", "frames"] = "filtered"
    trigger: str = Field("", max_length=200)
    tagger: str = "wd-v1-4-moat-tagger-v2"
    threshold: float = Field(.35, ge=0, le=1)
    character_threshold: float = Field(.85, ge=0, le=1)
    exclude_tags: str = Field("", max_length=4096)


class TrainRequest(DatasetRequest):
    kind: Literal["lora", "embedding"] = "lora"
    architecture: Literal["sd15", "sdxl", "anima"] = "sd15"
    model: str
    qwen3: str = ""
    vae: str = ""
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
    checkpointing: Literal["off", "standard", "cpu_offload", "unsloth"] = "standard"
    blocks_to_swap: int = Field(0, ge=0, le=40)
    optimizer_type: Literal["AdamW", "AdamW8bit"] = "AdamW"
    cache_text_encoder_outputs: bool = True


@router.get("/spritegpt")
def spritegpt_status():
    return spritegpt_runtime.status()


@router.post("/spritegpt/start")
def spritegpt_start():
    current = service()
    try:
        with current.generation_guard():
            if not current.idle():
                raise HTTPException(409, "Wait for generation to finish before starting SpriteGPT")
            return spritegpt_runtime.start()
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/spritegpt/stop")
def spritegpt_stop():
    return spritegpt_runtime.stop()


@router.get("/scrape/sources")
def scrape_sources():
    return {"sources": call(scraper.sources)}


@router.get("/datasets")
def datasets():
    current = service()
    root = current.dataset_root
    root.mkdir(parents=True, exist_ok=True)
    available = []
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            images = trainer.dataset_images(path)
            if all(trainer.read_caption(image) for image in images):
                available.append(path.name)
        except (OSError, UnicodeError, ValueError):
            continue
    return {"datasets": sorted(available)}


@router.post("/scrape/tag-suggestions")
def scrape_tag_suggestions(request: ScrapeTagSuggestionsRequest):
    return {"suggestions": call(scraper.tag_suggestions, request.query, request.sources)}


@router.post("/scrape/attempts")
def scrape_attempts(request: DatasetRequest):
    current = service()
    call(current.dataset, request.dataset)
    return {"attempts": call(scraper.attempts, current.dataset_root, request.dataset)}


@router.get("/scrape/archived-datasets")
def scrape_archived_datasets():
    current = service()
    current.dataset_root.mkdir(parents=True, exist_ok=True)
    return {"datasets": call(scraper.archived_datasets, current.dataset_root)}


@router.post("/dataset/gallery")
def archived_dataset_gallery(request: DatasetRequest):
    current = service()
    call(current.dataset, request.dataset)
    return call(scraper.archived_gallery, current.dataset_root, request.dataset)


@router.get("/dataset/file/{dataset}/{filename}")
def archived_dataset_file(dataset: str, filename: str):
    current = service()
    folder = call(current.dataset, dataset)
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".png":
        raise HTTPException(400, "Invalid archived image path")
    path = folder / filename
    if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) or not path.is_file():
        raise HTTPException(404, "Archived image not found")
    return FileResponse(path)


@router.post("/scrape/search")
def scrape_search(request: ScrapeSearchRequest):
    current = service()
    folder = call(current.dataset, request.dataset)
    folder.mkdir(parents=True, exist_ok=True)
    return call(scraper.search, current.dataset_root, request.dataset, request.query,
                request.sources, request.limit)


@router.post("/scrape/download")
def scrape_download(request: ScrapeDownloadRequest):
    current = service()
    return call(scraper.download, current.dataset_root, request.dataset, request.attempt,
                request.keys, request.threshold)


@router.post("/scrape/gallery")
def scrape_gallery(request: ScrapeAttemptRequest):
    current = service()
    return call(scraper.gallery, current.dataset_root, request.dataset, request.attempt)


@router.get("/scrape/file/{dataset}/{attempt}/{kind}/{filename}")
def scrape_file(dataset: str, attempt: str, kind: str, filename: str):
    current = service()
    folder = call(current.dataset, dataset) / ".scrapes" / attempt
    if kind not in {"before", "filtered", "frames"} or not attempt.isalnum() or Path(filename).name != filename:
        raise HTTPException(400, "Invalid gallery file path")
    path = (folder / kind / filename).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise HTTPException(404, "Gallery image not found")
    return FileResponse(path)


@router.post("/scrape/delete")
def scrape_delete(request: ScrapeNamesRequest):
    current = service()
    try:
        with current.generation_guard():
            return call(scraper.delete, current.dataset_root, request.dataset, request.attempt,
                        request.names, request.frame_names)
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/scrape/caption")
def scrape_caption(request: ScrapeCaptionRequest):
    current = service()
    try:
        with current.generation_guard():
            return call(scraper.save_caption, current.dataset_root, request.dataset, request.attempt,
                        request.name, request.caption, request.kind)
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/scrape/tag")
def scrape_tag(request: ScrapeTagRequest):
    current = service()
    call(current.dataset, request.dataset)
    payload = {"command": "autotag", "dataset": f"{request.dataset}/.scrapes/{request.attempt}/filtered",
               "tagger": request.tagger, "threshold": request.threshold,
               "character_threshold": request.character_threshold,
               "exclude_tags": request.exclude_tags, "trigger": "",
               "replace_existing": request.replace_existing}
    from easydiffusion import app
    port = int(app.getConfig().get("net", {}).get("listen_port", 9000))
    return call(current.start, payload, port=port)


@router.post("/scrape/tag-image")
def scrape_tag_image(request: ScrapeImageTagRequest):
    current = service()
    call(current.dataset, request.dataset)
    gallery = call(scraper.gallery, current.dataset_root, request.dataset, request.attempt)
    collection = "frames" if request.kind == "frames" else "filtered"
    if not any(item["name"] == request.name for item in gallery[collection]):
        raise HTTPException(404, "Image is not in this gallery")
    from easydiffusion import app
    port = int(app.getConfig().get("net", {}).get("listen_port", 9000))
    payload = {"command": "autotag", "dataset": f"{request.dataset}/.scrapes/{request.attempt}/{collection}",
               "images": [request.name], "replace_existing": True, "trigger": request.trigger,
               "tagger": request.tagger, "threshold": request.threshold,
               "character_threshold": request.character_threshold, "exclude_tags": request.exclude_tags}
    return call(current.start, payload, port=port)


@router.post("/scrape/archive")
def scrape_archive(request: ScrapeArchiveRequest):
    current = service()
    try:
        with current.generation_guard():
            destination = call(current.dataset, request.target_dataset)
            destination.mkdir(parents=True, exist_ok=True)
            return call(scraper.archive, current.dataset_root, request.dataset, request.attempt,
                        request.names, request.target_dataset, request.frame_names)
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/readiness")
def readiness():
    current = service()
    return {**call(current.readiness), "dataset_root": str(current.dataset_root)}


@router.get("/models")
def models():
    return {"models": call(service().models)}


@router.get("/assets")
def training_assets():
    return call(service().training_assets)


@router.post("/dataset/scan")
def scan(request: DatasetRequest):
    return call(service().scan, request.dataset)


@router.post("/dataset/create")
def create_dataset(request: DatasetRequest):
    return call(service().create_dataset, request.dataset)


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
