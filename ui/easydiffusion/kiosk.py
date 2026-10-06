"""Persisted kiosk display and generation policy; not an authentication boundary."""

from collections import OrderedDict
import re
import threading
import time

from easydiffusion import app
from fastapi import HTTPException


BASE_MODELS = {
    "1.5/sd-v1-5": "SD1.5 base",
    "sdxl/sd_xl_base_1.0": "SDXL 1.0 base",
    "Anima/anima-base-v1.0": "Anima base v1.0",
}
PERCHANCE_FILTER = "g"  # Perchance has no PG level; G also excludes PG-13 content.
_lock = threading.RLock()
_approvals: OrderedDict[str, float] = OrderedDict()
_approval_ttl = 600
_inline_lora = re.compile(r"<\s*(?:lora|lyco)\s*:", re.IGNORECASE)


def enabled():
    value = app.getConfig().get("kiosk_mode", False)
    if type(value) is not bool:
        raise HTTPException(status_code=503, detail="Invalid kiosk_mode configuration; expected true or false.")
    return value


def status():
    active = enabled()
    return {"enabled": active, "allowed_models": [{"model": key, "label": label} for key, label in BASE_MODELS.items()],
            "gallery_source": "destockd" if active else "local",
            "perchance_content_filter": PERCHANCE_FILTER if active else "none",
            "perchance_generated_images": not active}


def clear_approvals():
    with _lock:
        _approvals.clear()


def set_enabled(value):
    if type(value) is not bool:
        raise HTTPException(status_code=400, detail="enabled must be a boolean.")
    with _lock:
        config = app.getConfig()
        config["kiosk_mode"] = value
        app.setConfig(config)
        if enabled() != value:
            raise HTTPException(status_code=500, detail="Kiosk setting was not saved.")
        clear_approvals()
        return status()


def filter_models(models):
    if not enabled():
        return models
    seen, result = set(), []
    for item in models:
        name = item.get("model")
        if name in BASE_MODELS and name not in seen:
            result.append(item)
            seen.add(name)
    return result


def validate_render(request):
    if not enabled():
        return
    checkpoint = request.get("use_stable_diffusion_model")
    if not isinstance(checkpoint, str) or checkpoint not in BASE_MODELS:
        raise HTTPException(status_code=403, detail="Kiosk mode allows only the installed SD1.5, SDXL and Anima base checkpoints.")
    paths = request.get("model_paths") or {}
    if not isinstance(paths, dict):
        raise HTTPException(status_code=400, detail="model_paths must be an object.")
    if paths.get("ip-adapter") or any(request.get(field) for field in
                                     ("ip_adapter_model", "ip_adapter_image", "ip_adapter_clip_vision")):
        raise HTTPException(status_code=403, detail="IP-Adapters, including their bundled LoRAs, are disabled in kiosk mode.")
    if request.get("use_lora_model") or paths.get("lora") or any(
        _inline_lora.search(str(request.get(field, ""))) for field in ("prompt", "negative_prompt")
    ):
        raise HTTPException(status_code=403, detail="LoRAs are disabled in kiosk mode.")


def blocked_route(path, method):
    path = path.rstrip("/") or "/"
    prefixes = ("/training/", "/files/", "/meta/", "/civitai-api/", "/huggingface-api/",
                "/gallery/file/", "/gallery/thumb/", "/gallery-plugin/file/", "/gallery-plugin/thumb/",
                "/perchance/file/", "/perchance-plugin/file/", "/perchance/generated/file/")
    exact = {"/legacy", "/video", "/model/merge", "/perchance/image", "/perchance-plugin/image",
             "/perchance/images/save", "/get/model/metadata", "/get/lora/metadata",
             "/cpp-ui/training", "/cpp-ui/models/download", "/cpp-ui/gallery/datasets",
             "/cpp-ui/tagging", "/cpp-ui/perchance/image"}
    return (path in exact or any(path == prefix[:-1] or path.startswith(prefix) for prefix in prefixes)
            or (method != "GET" and path in {"/gallery/settings", "/gallery-plugin/settings"}))


def approve_perchance(filename):
    with _lock:
        _approvals[filename] = time.monotonic() + _approval_ttl
        _approvals.move_to_end(filename)
        while len(_approvals) > 1024:
            _approvals.popitem(last=False)


def require_approved_perchance(filename):
    if not enabled():
        return
    with _lock:
        if _approvals.get(filename, 0) <= time.monotonic():
            _approvals.pop(filename, None)
            raise HTTPException(status_code=403, detail="Reload this image through the kiosk's G-filtered Perchance gallery.")


def reject_unrated_perchance():
    if enabled():
        raise HTTPException(status_code=403, detail="Unrated Perchance generated images are unavailable in kiosk mode. Use the G-filtered gallery.")


def perchance_g_entry_is_valid(entry, filename, channel):
    """Accept only identified public entries from the native client's G-filtered result."""
    if not isinstance(entry, dict):
        return False
    return (entry.get("imageId") == filename.split(".", 1)[0]
            and entry.get("channel") == channel and entry.get("subChannel") == "public"
            and all(key not in entry or entry[key] is False for key in ("nsfw", "shocking", "pg13Soft")))
