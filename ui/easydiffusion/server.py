"""server.py: FastAPI SD-UI Web Host.
Notes:
    async endpoints always run on the main thread. Without they run on the thread pool.
"""

import datetime
import functools
import glob
import mimetypes
import os
import shlex
import subprocess
import time
import traceback
from collections import deque
from threading import Lock
from typing import List, Union

from easydiffusion import app, task_manager, kiosk
from easydiffusion.backend_args import parse_backend_commandline_args
from ui.plugins.server import ai_image_critic, gallery, package_manager, perchance
from ui.plugins.server.Browse_files import model_manager
from ui.plugins.server.tasks import RenderTask, FilterTask, VideoTask
from easydiffusion.types import (
    GenerateImageRequest,
    VideoGenerationRequest,
    FilterImageRequest,
    MergeRequest,
    TaskData,
    RenderTaskData,
    ModelsData,
    OutputFormatData,
    SaveToDiskData,
    convert_legacy_render_req_to_new,
    convert_legacy_controlnet_filter_name,
)
from ui.plugins.server.utils import log
from ui.plugins.server.wd14_tagger import WD14TagRequest, tag_image
from ui.plugins.server import training
from ui.plugins.server.native_image_tools import (
    BackgroundRemovalRequest,
    NativeDetectionRequest,
    ObjectRemovalRequest,
    TextMaskRequest,
    detect as native_detect,
    remove_background as native_remove_background,
    remove_object as native_remove_object,
    text_mask as native_text_mask,
)
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Extra
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pycloudflared import try_cloudflare

log.info(f"started in {app.SD_DIR}")
log.info(f"started at {datetime.datetime.now():%x %X}")

server_api = FastAPI()

NOCACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}
PROTECTED_CONFIG_KEYS = ("block_nsfw", "kiosk_mode")  # kiosk changes use the dedicated, validated endpoint


class NoCacheStaticFiles(StaticFiles):
    def __init__(self, directory: str):
        # follow_symlink is only available on fastapi >= 0.92.0
        if os.path.islink(directory):
            super().__init__(directory=os.path.realpath(directory))
        else:
            super().__init__(directory=directory)

    def is_not_modified(self, response_headers, request_headers) -> bool:
        if "content-type" in response_headers and any(
            asset_type in response_headers["content-type"]
            for asset_type in ("javascript", "css", "text/html")
        ):
            response_headers.update(NOCACHE_HEADERS)
            return False

        return super().is_not_modified(response_headers, request_headers)


class SetAppConfigRequest(BaseModel, extra=Extra.allow):
    render_devices: Union[List[str], List[int], str, int] = None
    model_vae: str = None
    ui_open_browser_on_start: bool = None
    listen_to_network: bool = None
    listen_port: int = None
    backend_platform: str = "auto"
    models_dir: str = None
    directories: dict = None
    vram_usage_level: str = "balanced"
    backend_commandline_args: Union[List[str], str] = None
    reload_backend: bool = False


def init():
    mimetypes.init()
    mimetypes.add_type("text/css", ".css")
    gallery.migrate_legacy_settings()

    @server_api.middleware("http")
    async def kiosk_policy(request, call_next):
        try:
            active_at_start = kiosk.enabled()
            if active_at_start and kiosk.blocked_route(request.url.path, request.method):
                return JSONResponse({"detail": "This operation is unavailable in kiosk mode."}, status_code=403, headers=NOCACHE_HEADERS)
        except HTTPException as error:
            return JSONResponse({"detail": error.detail}, status_code=error.status_code, headers=NOCACHE_HEADERS)
        response = await call_next(request)
        if (not active_at_start and kiosk.enabled()
                and request.url.path.startswith(("/gallery", "/perchance", "/get/"))):
            return JSONResponse({"detail": "Kiosk mode changed; reload this request."}, status_code=409, headers=NOCACHE_HEADERS)
        if request.url.path.startswith(("/gallery", "/perchance")):
            response.headers.update(NOCACHE_HEADERS)
        return response

    @server_api.get("/kiosk")
    def kiosk_status():
        return JSONResponse(kiosk.status(), headers=NOCACHE_HEADERS)

    @server_api.post("/kiosk")
    def kiosk_setting(payload: dict):
        value = payload.get("enabled")
        if type(value) is not bool:
            raise HTTPException(status_code=400, detail="enabled must be a boolean.")
        if value:
            try:
                with training.generation_guard(), task_manager.backend_maintenance():
                    if perchance.PERCHANCE_LOCK.locked():
                        raise HTTPException(status_code=409, detail="Wait for Perchance to finish before enabling kiosk mode.")
                    return JSONResponse(kiosk.set_enabled(True), headers=NOCACHE_HEADERS)
            except ConnectionRefusedError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
        return JSONResponse(kiosk.set_enabled(False), headers=NOCACHE_HEADERS)

    from ui.plugins.server.online_model_browser import (
        huggingface_router,
        router as online_model_browser_router,
    )

    # Keep the existing route prefix for compatibility with saved plugin URLs.
    server_api.include_router(online_model_browser_router, prefix="/civitai-api")
    server_api.include_router(huggingface_router, prefix="/huggingface-api")

    from ui.plugins.server.model_tools import router as model_tools_router

    server_api.include_router(model_tools_router, prefix="/model-tools")
    server_api.include_router(training.router, prefix="/training")

    if os.path.isdir(app.CUSTOM_MODIFIERS_DIR):
        server_api.mount(
            "/media/modifier-thumbnails/custom",
            NoCacheStaticFiles(directory=app.CUSTOM_MODIFIERS_DIR),
            name="custom-thumbnails",
        )

    server_api.mount(
        "/media",
        NoCacheStaticFiles(directory=os.path.join(app.SD_UI_DIR, "media")),
        name="media",
    )

    for plugins_dir, dir_prefix in app.UI_PLUGINS_SOURCES:
        server_api.mount(
            f"/plugins/{dir_prefix}",
            NoCacheStaticFiles(directory=plugins_dir),
            name=f"plugins-{dir_prefix}",
        )

    @server_api.post("/app_config")
    async def set_app_config(req: SetAppConfigRequest):
        return set_app_config_internal(req)

    @server_api.post("/clear-vram")
    def clear_vram():
        return clear_vram_internal()

    @server_api.get("/get/{key:path}")
    def read_web_data(key: str = None, scan_for_malicious: bool = True):
        return read_web_data_internal(key, scan_for_malicious=scan_for_malicious)

    @server_api.get("/ping")  # Get server and optionally session status.
    def ping(session_id: str = None):
        return ping_internal(session_id)

    @server_api.post("/render")
    def render(req: dict):
        return render_internal(req)

    @server_api.post("/video")
    def video(req: dict):
        return video_internal(req)

    @server_api.post("/filter")
    def render(req: dict):
        return filter_internal(req)

    @server_api.post("/tag")
    def wd14_tag(req: WD14TagRequest):
        try:
            return JSONResponse(tag_image(req), headers=NOCACHE_HEADERS)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"WD14 tagging failed: {exc}") from exc

    @server_api.get("/ai-critic/models")
    def ai_critic_models():
        return JSONResponse(ai_image_critic.model_inventory(), headers=NOCACHE_HEADERS)

    @server_api.post("/ai-critic/analyze")
    def ai_critic_analyze(req: ai_image_critic.CriticAnalyzeRequest):
        return JSONResponse(ai_image_critic.analyze(req), headers=NOCACHE_HEADERS)

    @server_api.post("/native-vision/detect")
    def detect_with_native_vision(req: NativeDetectionRequest):
        try:
            return JSONResponse(native_detect(req), headers=NOCACHE_HEADERS)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Native detection timed out") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Native detection failed: {exc}") from exc

    @server_api.post("/image-tools/text-mask")
    def create_native_text_mask(req: TextMaskRequest):
        try:
            return JSONResponse(native_text_mask(req), headers=NOCACHE_HEADERS)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Text-mask detection timed out") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Text-mask detection failed: {exc}") from exc

    @server_api.post("/image-tools/remove-background")
    def remove_image_background(req: BackgroundRemovalRequest):
        try:
            return JSONResponse(native_remove_background(req), headers=NOCACHE_HEADERS)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}") from exc

    @server_api.post("/image-tools/remove-object")
    def remove_masked_object(req: ObjectRemovalRequest):
        try:
            return JSONResponse(native_remove_object(req), headers=NOCACHE_HEADERS)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Object removal failed: {exc}") from exc

    from ui.plugins.server.tipo import GenerateRequest as TipoGenerateRequest

    @server_api.get("/tipo/health")
    @server_api.get("/tipo-api/health", include_in_schema=False)
    def tipo_health():
        from ui.plugins.server.tipo import health

        return health()

    @server_api.get("/tipo/models")
    @server_api.get("/tipo-api/models", include_in_schema=False)
    def tipo_models():
        from ui.plugins.server.tipo import list_models

        return list_models()

    @server_api.post("/tipo/generate")
    @server_api.post("/tipo-api/generate", include_in_schema=False)
    def tipo_generate(req: TipoGenerateRequest):
        from ui.plugins.server.tipo import generate

        return generate(req)

    @server_api.get("/gallery/settings")
    @server_api.get("/gallery-plugin/settings", include_in_schema=False)
    def gallery_settings_get():
        return JSONResponse(gallery.get_settings(), headers=NOCACHE_HEADERS)

    @server_api.post("/gallery/settings")
    @server_api.post("/gallery-plugin/settings", include_in_schema=False)
    def gallery_settings_save(payload: dict):
        return JSONResponse(gallery.save_settings(payload), headers=NOCACHE_HEADERS)

    @server_api.get("/gallery/images")
    @server_api.get("/gallery-plugin/images", include_in_schema=False)
    def gallery_images(page: int = 1, page_size: int = gallery.DEFAULT_PAGE_SIZE):
        return JSONResponse(gallery.list_images(page, page_size), headers=NOCACHE_HEADERS)

    @server_api.get("/gallery/file/{relative_path:path}")
    @server_api.get("/gallery-plugin/file/{relative_path:path}", include_in_schema=False)
    def gallery_file(relative_path: str):
        path = gallery.resolve_gallery_file(relative_path)
        return FileResponse(
            path,
            media_type=gallery.file_media_type(path),
            headers={"Cache-Control": "private, max-age=60"},
        )

    @server_api.delete("/gallery/file/{relative_path:path}")
    @server_api.delete("/gallery-plugin/file/{relative_path:path}", include_in_schema=False)
    def gallery_delete_file(relative_path: str):
        return JSONResponse(gallery.delete_file(relative_path), headers=NOCACHE_HEADERS)

    @server_api.get("/gallery/thumb/{relative_path:path}")
    @server_api.get("/gallery-plugin/thumb/{relative_path:path}", include_in_schema=False)
    def gallery_thumbnail(relative_path: str):
        source = gallery.resolve_gallery_file(relative_path)
        thumbnail = gallery.create_thumbnail(source)
        media_type = "image/jpeg" if thumbnail != source else gallery.file_media_type(source)
        return FileResponse(
            thumbnail,
            media_type=media_type,
            headers={"Cache-Control": "private, max-age=86400"},
        )

    @server_api.get("/perchance/status")
    @server_api.get("/perchance-plugin/status", include_in_schema=False)
    def perchance_status():
        return JSONResponse(perchance.status(), headers=NOCACHE_HEADERS)

    @server_api.get("/perchance/settings")
    def perchance_settings_get():
        return JSONResponse(perchance.get_settings(), headers=NOCACHE_HEADERS)

    @server_api.post("/perchance/settings")
    def perchance_settings_save(payload: dict):
        return JSONResponse(perchance.save_settings(payload), headers=NOCACHE_HEADERS)

    @server_api.post("/perchance/image")
    @server_api.post("/perchance-plugin/image", include_in_schema=False)
    async def perchance_image(payload: dict):
        return JSONResponse(await perchance.generate_image(payload), headers=NOCACHE_HEADERS)

    @server_api.get("/perchance/images/recent")
    def perchance_recent_images(limit: int = 8):
        return JSONResponse(perchance.recent_images(limit), headers=NOCACHE_HEADERS)

    @server_api.post("/perchance/images/save")
    def perchance_save_generated_image(payload: dict):
        return JSONResponse(perchance.save_generated_image(payload), headers=NOCACHE_HEADERS)

    @server_api.post("/perchance/text")
    @server_api.post("/perchance-plugin/text", include_in_schema=False)
    async def perchance_text(payload: dict):
        return JSONResponse(await perchance.generate_text(payload), headers=NOCACHE_HEADERS)

    @server_api.post("/perchance/gallery/list")
    async def perchance_gallery_list(payload: dict):
        return JSONResponse(await perchance.gallery_list(payload), headers=NOCACHE_HEADERS)

    @server_api.post("/perchance/gallery/get")
    async def perchance_gallery_get(payload: dict):
        return JSONResponse(await perchance.gallery_get(payload), headers=NOCACHE_HEADERS)

    @server_api.get("/perchance/file/{relative_path:path}")
    @server_api.get(
        "/perchance-plugin/file/{relative_path:path}",
        include_in_schema=False,
    )
    def perchance_file(relative_path: str):
        return FileResponse(
            perchance.resolve_output_file(relative_path),
            headers={"Cache-Control": "private, max-age=60"},
        )

    @server_api.get("/perchance/generated/file/{relative_path:path}")
    def perchance_generated_file(relative_path: str):
        return FileResponse(
            perchance.resolve_generated_file(relative_path),
            headers={"Cache-Control": "private, max-age=60"},
        )

    @server_api.get("/perchance/gallery/cache/{relative_path:path}")
    def perchance_gallery_cache_file(relative_path: str):
        return FileResponse(
            perchance.resolve_gallery_cache_file(relative_path),
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @server_api.get("/files/health")
    def fileparser_health():
        from ui.plugins.server.Browse_files import lora_dir

        return {"status": "ok", "lora_dir": str(lora_dir())}

    @server_api.post("/files/list_lora")
    def fileparser_list_lora():
        from ui.plugins.server.Browse_files import list_lora_files
        from ui.plugins.server.Metadata import scan_lora_metadata

        files = list_lora_files()
        return {"files": files, "meta": scan_lora_metadata(), "count": len(files)}

    @server_api.post("/files/list_model")
    def fileparser_list_model():
        from ui.plugins.server.Browse_files import list_checkpoint_files
        from ui.plugins.server.Metadata import scan_checkpoint_metadata

        files = list_checkpoint_files()
        return {"files": files, "meta": scan_checkpoint_metadata(), "count": len(files)}

    @server_api.post("/files/list_vae")
    def fileparser_list_vae():
        from ui.plugins.server.Browse_files import list_vae_files
        from ui.plugins.server.Metadata import scan_vae_metadata

        files = list_vae_files()
        return {"files": files, "meta": scan_vae_metadata(), "count": len(files)}

    @server_api.post("/files/list_tipo")
    def fileparser_list_tipo():
        from easydiffusion.tipo import list_model_files, list_model_metadata

        files = list_model_files()
        return {"files": files, "meta": list_model_metadata(), "count": len(files)}

    @server_api.post("/files/list_checkpoints")
    def fileparser_list_checkpoints():
        from ui.plugins.server.Browse_files import list_checkpoint_files
        from ui.plugins.server.Metadata import scan_checkpoint_metadata

        files = list_checkpoint_files()
        return {"files": files, "meta": scan_checkpoint_metadata(), "count": len(files)}

    @server_api.post("/meta/get_triggers")
    def fileparser_get_triggers(payload: dict):
        from ui.plugins.server.Metadata import extract_lora_metadata

        try:
            return {"meta": extract_lora_metadata(payload.get("filepath"), bool(payload.get("include_metadata")))}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @server_api.post("/meta/scan_loras")
    def fileparser_scan_loras():
        from ui.plugins.server.Metadata import scan_lora_metadata

        items = scan_lora_metadata()
        return {"meta": items, "count": len(items)}

    @server_api.post("/meta/scan_checkpoints")
    def fileparser_scan_checkpoints():
        from ui.plugins.server.Metadata import scan_checkpoint_metadata

        items = scan_checkpoint_metadata()
        return {"meta": items, "count": len(items)}

    @server_api.post("/model/merge")
    def model_merge(req: dict):
        print(req)
        return model_merge_internal(req)

    @server_api.get("/image/stream/{task_id:int}")
    def stream(task_id: int):
        return stream_internal(task_id)

    @server_api.get("/image/stop")
    def stop(task: int):
        return stop_internal(task)

    @server_api.get("/image/tmp/{task_id:int}/{img_id:int}")
    def get_image(task_id: int, img_id: int):
        return get_image_internal(task_id, img_id)

    @server_api.post("/tunnel/cloudflare/start")
    def start_cloudflare_tunnel(req: dict):
        return start_cloudflare_tunnel_internal(req)

    @server_api.post("/tunnel/cloudflare/stop")
    def stop_cloudflare_tunnel(req: dict):
        return stop_cloudflare_tunnel_internal(req)

    @server_api.post("/package/{package_name:str}")
    def modify_package(package_name: str, req: dict):
        return modify_package_internal(package_name, req)

    @server_api.get("/sha256/{obj_path:path}")
    def get_sha256(obj_path: str):
        return get_sha256_internal(obj_path)

    @functools.lru_cache(maxsize=1)
    def find_cpp_ui_renderer():
        configured = os.environ.get("SD_UI_CPP_RENDERER")
        if configured and os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        build_root = os.path.join(app.ROOT_DIR, "source", "sdkit3-port-source", "build")
        candidates = glob.glob(os.path.join(build_root, "*", "bin", "easy-diffusion-ui-render"))
        candidates.append(os.path.join(build_root, "bin", "easy-diffusion-ui-render"))
        for candidate in sorted(candidates):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        standalone = os.path.join(app.ROOT_DIR, "source", "UI.cpp", "build", "bin", "easy-diffusion-ui-render")
        if os.path.isfile(standalone) and os.access(standalone, os.X_OK):
            return standalone
        return None

    cpp_ui_assets = os.path.join(app.ROOT_DIR, "source", "UI.cpp", "Pages", "assets")
    if not os.path.isdir(cpp_ui_assets):
        renderer = find_cpp_ui_renderer()
        if renderer:
            cpp_ui_assets = os.path.join(os.path.dirname(renderer), "cpp-ui-assets")
    if os.path.isdir(cpp_ui_assets):
        server_api.mount(
            "/cpp-ui/assets",
            NoCacheStaticFiles(directory=cpp_ui_assets),
            name="cpp-ui-assets",
        )
    cpp_ui_scripts = os.path.join(app.ROOT_DIR, "source", "UI.cpp", "Pages", "src", "Plugin", "plugin_scripts")
    if os.path.isdir(cpp_ui_scripts):
        server_api.mount(
            "/cpp-ui/scripts",
            NoCacheStaticFiles(directory=cpp_ui_scripts),
            name="cpp-ui-scripts",
        )
    else:
        renderer = find_cpp_ui_renderer()
        if renderer:
            bundled_scripts = os.path.join(os.path.dirname(renderer), "cpp-ui-scripts")
            if os.path.isdir(bundled_scripts):
                server_api.mount(
                    "/cpp-ui/scripts",
                    NoCacheStaticFiles(directory=bundled_scripts),
                    name="cpp-ui-scripts",
                )
    cpp_ui_events = deque(maxlen=200)
    cpp_ui_events_lock = Lock()

    def record_cpp_ui_event(level: str, message: str):
        event = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": message,
        }
        with cpp_ui_events_lock:
            cpp_ui_events.append(event)
        getattr(log, level)(f"C++ UI {message}")

    renderer_state = find_cpp_ui_renderer()
    asset_state = os.path.isfile(os.path.join(cpp_ui_assets, "ui.css"))
    if renderer_state:
        record_cpp_ui_event(
            "info",
            f"serving enabled routes=16 renderer=ready assets={'ready' if asset_state else 'missing'} path=/cpp-ui",
        )
    else:
        record_cpp_ui_event(
            "warning",
            f"serving unavailable routes=16 renderer=missing assets={'ready' if asset_state else 'missing'} path=/cpp-ui",
        )

    def render_cpp_ui_page(page_path: str):
        renderer = find_cpp_ui_renderer()
        if renderer is None:
            record_cpp_ui_event("warning", f"request unavailable page={page_path} status=503 renderer=missing")
            raise HTTPException(status_code=503, detail="C++ UI renderer is not built; set SD_UI_CPP_RENDERER or build the SDKit targets")
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [renderer, page_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            record_cpp_ui_event("error", f"render failed page={page_path} error={type(error).__name__} elapsed_ms={elapsed_ms:.1f}")
            raise HTTPException(status_code=500, detail="C++ UI rendering failed") from error
        elapsed_ms = (time.perf_counter() - started) * 1000
        record_cpp_ui_event("info", f"page served page={page_path} status=200 elapsed_ms={elapsed_ms:.1f}")
        return HTMLResponse(result.stdout, headers=NOCACHE_HEADERS)

    @server_api.get("/cpp-ui/api/events", include_in_schema=False)
    def read_cpp_ui_events():
        with cpp_ui_events_lock:
            events = list(cpp_ui_events)
        return JSONResponse({"events": events, "count": len(events)})

    @server_api.get("/cpp-ui/api/logs", include_in_schema=False)
    def read_cpp_ui_logs(lines: int = 500):
        from easydiffusion.diagnostics import read_log_tail
        try:
            return JSONResponse(read_log_tail(app.DEBUG_LOG_PATH, lines), headers=NOCACHE_HEADERS)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except OSError as error:
            raise HTTPException(status_code=503, detail="Application log is unavailable") from error

    @server_api.get("/cpp-ui/api/files", include_in_schema=False)
    def read_cpp_ui_files(path: str = ".", offset: int = 0):
        from easydiffusion.diagnostics import inspect_path
        try:
            return JSONResponse(inspect_path(app.ROOT_DIR, path, offset), headers=NOCACHE_HEADERS)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Path does not exist") from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail="Permission denied") from error
        except OSError as error:
            raise HTTPException(status_code=400, detail="Could not inspect this path") from error

    cpp_ui_routes = {
        "/cpp-ui": "/",
        "/cpp-ui/canvas": "/canvas",
        "/cpp-ui/perchance/text": "/perchance/text",
        "/cpp-ui/perchance/image": "/perchance/image",
        "/cpp-ui/perchance/gallery": "/perchance/gallery",
        "/cpp-ui/gallery/images": "/gallery/images",
        "/cpp-ui/gallery/datasets": "/gallery/datasets",
        "/cpp-ui/tagging": "/tagging",
        "/cpp-ui/training": "/training",
        "/cpp-ui/models/download": "/models/download",
        "/cpp-ui/settings": "/settings",
        "/cpp-ui/settings/gpu": "/settings/gpu",
        "/cpp-ui/settings/plugins": "/settings/plugins",
        "/cpp-ui/stats": "/stats",
        "/cpp-ui/logs": "/logs",
        "/cpp-ui/console": "/console",
    }
    for route_path, page_path in cpp_ui_routes.items():
        route_name = route_path.strip("/").replace("/", "-") or "main"
        server_api.add_api_route(
            route_path,
            lambda page_path=page_path: render_cpp_ui_page(page_path),
            methods=["GET"],
            include_in_schema=False,
            name="cpp-ui-" + route_name,
        )

    @server_api.get("/")
    def read_root():
        if find_cpp_ui_renderer() is not None:
            return render_cpp_ui_page("/")
        record_cpp_ui_event("warning", "root page using legacy UI because C++ renderer is unavailable")
        return FileResponse(os.path.join(app.SD_UI_DIR, "index.html"), headers=NOCACHE_HEADERS)

    @server_api.get("/legacy")
    def read_legacy_ui():
        return FileResponse(os.path.join(app.SD_UI_DIR, "index.html"), headers=NOCACHE_HEADERS)

    @server_api.on_event("shutdown")
    def shutdown_event():  # Signal render thread to close on shutdown
        task_manager.current_state_error = SystemExit("Application shutting down.")
        from easydiffusion.tipo import shutdown as shutdown_tipo

        shutdown_tipo()
        ai_image_critic.shutdown()
        training.shutdown()

    @server_api.on_event("startup")
    def start_event():
        from easydiffusion.app import open_browser

        open_browser()


# API implementations
def set_app_config_internal(req: SetAppConfigRequest):
    config = app.getConfig()
    backend_commandline_args = None
    if req.backend_commandline_args is not None:
        try:
            backend_commandline_args = parse_backend_commandline_args(req.backend_commandline_args)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=f"Invalid backend command-line arguments: {error}")

    if req.reload_backend:
        if not task_manager.backend_is_idle():
            raise HTTPException(
                status_code=409,
                detail="The backend can only reload while generation is idle and the queue is empty.",
            )
    if req.render_devices is not None:
        update_render_devices_in_config(config, req.render_devices)
    if req.ui_open_browser_on_start is not None:
        if "ui" not in config:
            config["ui"] = {}
        config["ui"]["open_browser_on_start"] = req.ui_open_browser_on_start
    if req.listen_to_network is not None:
        if "net" not in config:
            config["net"] = {}
        config["net"]["listen_to_network"] = bool(req.listen_to_network)
    if req.listen_port is not None:
        if "net" not in config:
            config["net"] = {}
        config["net"]["listen_port"] = int(req.listen_port)

    config["models_dir"] = req.models_dir
    if req.directories is not None:
        try:
            config["directories"] = model_manager.normalize_directory_config(req.directories)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    config["vram_usage_level"] = req.vram_usage_level

    config["backend_config"] = config.get("backend_config") or {}
    if backend_commandline_args is not None:
        config["backend_config"]["COMMANDLINE_ARGS"] = backend_commandline_args
    config["backend_config"]["platform"] = None
    if req.backend_platform == "auto":
        del config["backend_config"]["platform"]
    else:
        config["backend_config"]["platform"] = req.backend_platform

    for property, property_value in req.dict().items():
        if property_value is not None and property not in req.__fields__ and property not in PROTECTED_CONFIG_KEYS:
            config[property] = property_value

    try:
        app.setConfig(config)

        if req.render_devices:
            app.update_render_threads()

        if req.reload_backend:
            from easydiffusion import backend_manager

            backend_manager.restart_backend()

        return JSONResponse(
            {"status": "OK", "backend_restarted": bool(req.reload_backend)},
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def update_render_devices_in_config(config, render_devices):
    from easydiffusion.device_manager import validate_render_devices

    try:
        render_devices = render_devices.split(",")
        validate_render_devices(render_devices)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config["render_devices"] = render_devices


def read_web_data_internal(key: str = None, **kwargs):
    if not key:  # /get without parameters, stable-diffusion easter egg.
        raise HTTPException(status_code=418, detail="StableDiffusion is drawing a teapot!")  # HTTP418 I'm a teapot
    elif key == "app_config":
        config = dict(app.getConfig())

        if "models_dir" not in config:
            config["models_dir"] = app.MODELS_DIR

        configured_directories = config.get("directories") or {}
        config["directories"] = {
            **model_manager.effective_directory_config(config["models_dir"]),
            **configured_directories,
        }

        commandline_args = (config.get("backend_config") or {}).get("COMMANDLINE_ARGS", [])
        if isinstance(commandline_args, (list, tuple)):
            config["backend_commandline_args"] = shlex.join(map(str, commandline_args))
        else:
            config["backend_commandline_args"] = str(commandline_args or "")

        return JSONResponse(config, headers=NOCACHE_HEADERS)
    elif key == "system_info":
        from easydiffusion.backend_manager import backend

        config = app.getConfig()

        output_dir = config.get("force_save_path", os.path.join(os.path.expanduser("~"), app.OUTPUT_DIRNAME))

        system_info = {
            "devices": task_manager.get_devices(),
            "hosts": app.getIPConfig(),
            "default_output_dir": output_dir,
            "enforce_output_dir": ("force_save_path" in config),
            "enforce_output_metadata": ("force_save_metadata" in config),
            "backend_url": backend.get_url(),
        }
        system_info["devices"]["config"] = config.get("render_devices", "auto")
        return JSONResponse(system_info, headers=NOCACHE_HEADERS)
    elif key in ("model", "lora", "vae"):
        selector_types = {
            "model": "stable-diffusion",
            "lora": "lora",
            "vae": "vae",
        }
        if key == "model":
            from easydiffusion.backend_manager import backend

            backend.refresh_models()
        models = [] if key == "lora" and kiosk.enabled() else model_manager.list_models([selector_types[key]])
        if key == "model":
            models = kiosk.filter_models(models)
        return JSONResponse(
            {"models": models},
            headers=NOCACHE_HEADERS,
        )
    elif key == "tipo":
        from easydiffusion.tipo import selector_models

        return JSONResponse({"models": selector_models()}, headers=NOCACHE_HEADERS)
    elif key == "other":
        if kiosk.enabled():
            return JSONResponse({"models": []}, headers=NOCACHE_HEADERS)
        selector_types = {"stable-diffusion", "lora", "vae"}
        return JSONResponse(
            {"models": model_manager.list_models(set(model_manager.LISTABLE_MODEL_TYPES) - selector_types)},
            headers=NOCACHE_HEADERS,
        )
    elif key in ("model/metadata", "lora/metadata", "vae/metadata", "tipo/metadata"):
        from ui.plugins.server.Metadata import (
            scan_checkpoint_metadata,
            scan_lora_metadata,
            scan_vae_metadata,
        )

        from easydiffusion.tipo import list_model_metadata

        scanners = {
            "model/metadata": scan_checkpoint_metadata,
            "lora/metadata": scan_lora_metadata,
            "vae/metadata": scan_vae_metadata,
            "tipo/metadata": list_model_metadata,
        }
        metadata = scanners[key]()
        return JSONResponse({"metadata": metadata, "count": len(metadata)}, headers=NOCACHE_HEADERS)
    elif key == "models":
        from easydiffusion.backend_manager import backend

        backend.refresh_models()

        return JSONResponse({"models": kiosk.filter_models(model_manager.list_models())}, headers=NOCACHE_HEADERS)
    elif key == "modifiers":
        return JSONResponse(app.get_image_modifiers(), headers=NOCACHE_HEADERS)
    elif key == "ui_plugins":
        return JSONResponse(app.getUIPlugins(), headers=NOCACHE_HEADERS)
    elif key == "backend_devices":
        from easydiffusion.backend_manager import backend

        if not hasattr(backend, "get_backend_devices"):
            return JSONResponse({"devices": []}, headers=NOCACHE_HEADERS)
        try:
            devices = backend.get_backend_devices()
        except Exception as error:
            log.warning(f"Could not query native backend devices: {error}")
            devices = []
        return JSONResponse({"devices": devices}, headers=NOCACHE_HEADERS)
    else:
        raise HTTPException(status_code=404, detail=f"Request for unknown {key}")  # HTTP404 Not Found


def ping_internal(session_id: str = None):
    if task_manager.is_alive() <= 0:  # Check that render threads are alive.
        if task_manager.current_state_error:
            raise HTTPException(status_code=500, detail=str(task_manager.current_state_error))
        raise HTTPException(status_code=500, detail="Render thread is dead.")

    if task_manager.current_state_error and not isinstance(task_manager.current_state_error, StopAsyncIteration):
        raise HTTPException(status_code=500, detail=str(task_manager.current_state_error))

    # Alive
    response = {"status": str(task_manager.current_state)}

    if session_id:
        # Merely opening or reloading the UI must not allocate a server-side
        # session that remains cached until TASK_TTL. Sessions are created when
        # the client actually queues work.
        session = task_manager.get_cached_session(session_id, update_ttl=True, create=False)
        response["tasks"] = {id(t): t.status for t in session.tasks} if session else {}

    response["devices"] = task_manager.get_devices()
    response["packages_installed"] = package_manager.get_installed_packages()
    response["packages_installing"] = package_manager.installing

    if cloudflare.address != None:
        response["cloudflare"] = cloudflare.address

    return JSONResponse(response, headers=NOCACHE_HEADERS)


def clear_vram_internal():
    from easydiffusion import backend_manager

    try:
        # This also excludes training while the native process releases its models.
        with training.generation_guard(), task_manager.backend_maintenance():
            backend_manager.restart_backend()
        log.info("VRAM cleared: native generation models unloaded.")
        return JSONResponse({
            "status": "OK",
            "message": "Generation models unloaded. They will reload for the next image or video.",
        }, headers=NOCACHE_HEADERS)
    except ConnectionRefusedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Failed to clear VRAM")
        raise HTTPException(status_code=500, detail=f"Could not clear VRAM: {exc}") from exc


def render_internal(req: dict):
    try:
        kiosk.validate_render(req)
        req = convert_legacy_render_req_to_new(req)

        # separate out the request data into rendering and task-specific data
        render_req: GenerateImageRequest = GenerateImageRequest.parse_obj(req)
        task_data: RenderTaskData = RenderTaskData.parse_obj(req)
        models_data: ModelsData = ModelsData.parse_obj(req)
        output_format: OutputFormatData = OutputFormatData.parse_obj(req)
        save_data: SaveToDiskData = SaveToDiskData.parse_obj(req)

        # Overwrite user specified save path
        config = app.getConfig()
        if "force_save_path" in config:
            save_data.save_to_disk_path = config["force_save_path"]

        render_req.init_image_mask = req.get("mask")  # hack: will rename this in the HTTP API in a future revision

        app.save_to_config(
            models_data.model_paths.get("stable-diffusion"),
            models_data.model_paths.get("vae"),
            models_data.model_paths.get("hypernetwork"),
            task_data.vram_usage_level,
        )

        # enqueue the task
        task = RenderTask(render_req, task_data, models_data, output_format, save_data)
        return enqueue_task(task)
    except HTTPException as e:
        raise e
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def video_internal(req: dict):
    try:
        req = convert_legacy_render_req_to_new(req)
        video_req = VideoGenerationRequest.parse_obj(req)
        task_data = TaskData.parse_obj(req)
        models_data = ModelsData.parse_obj(req)
        output_format = OutputFormatData.parse_obj(req)

        if not models_data.model_paths.get("stable-diffusion"):
            raise ValueError("Select a native SVD, Wan, or LTX video checkpoint")

        task = VideoTask(video_req, task_data, models_data, output_format)
        return enqueue_task(task)
    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def filter_internal(req: dict):
    try:
        filter_req: FilterImageRequest = FilterImageRequest.parse_obj(req)
        task_data: TaskData = TaskData.parse_obj(req)
        models_data: ModelsData = ModelsData.parse_obj(req)
        output_format: OutputFormatData = OutputFormatData.parse_obj(req)
        save_data: SaveToDiskData = SaveToDiskData.parse_obj(req)

        filter_req.filter = convert_legacy_controlnet_filter_name(filter_req.filter)

        for model_name in ("realesrgan", "esrgan_4x", "lanczos", "nearest", "scunet", "swinir"):
            if models_data.model_paths.get(model_name):
                if model_name not in filter_req.filter_params:
                    filter_req.filter_params[model_name] = {}

                filter_req.filter_params[model_name]["upscaler"] = models_data.model_paths[model_name]

        # enqueue the task
        task = FilterTask(filter_req, task_data, models_data, output_format, save_data)
        return enqueue_task(task)
    except HTTPException as e:
        raise e
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def enqueue_task(task):
    try:
        with training.generation_guard():
            task_manager.enqueue_task(task)
        response = {
            "status": str(task_manager.current_state),
            "queue": len(task_manager.tasks_queue),
            "stream": f"/image/stream/{task.id}",
            "task": task.id,
        }
        return JSONResponse(response, headers=NOCACHE_HEADERS)
    except ChildProcessError as e:  # Render thread is dead
        raise HTTPException(status_code=500, detail=f"Rendering thread has died.")  # HTTP500 Internal Server Error
    except ConnectionRefusedError as e:  # Unstarted task pending limit reached, deny queueing too many.
        raise HTTPException(status_code=503, detail=str(e))  # HTTP503 Service Unavailable


def model_merge_internal(req: dict):
    try:
        from easydiffusion.utils.save_utils import filename_regex
        from sdkit.train import merge_models

        mergeReq: MergeRequest = MergeRequest.parse_obj(req)

        sd_model_dir = model_manager.get_model_dirs("stable-diffusion")[0]

        merge_models(
            model_manager.resolve_model_to_use(mergeReq.model0, "stable-diffusion"),
            model_manager.resolve_model_to_use(mergeReq.model1, "stable-diffusion"),
            mergeReq.ratio,
            os.path.join(sd_model_dir, filename_regex.sub("_", mergeReq.out_path)),
            mergeReq.use_fp16,
        )
        return JSONResponse({"status": "OK"}, headers=NOCACHE_HEADERS)
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def stream_internal(task_id: int):
    # TODO Move to WebSockets ??
    task = task_manager.get_cached_task(task_id, update_ttl=True)
    if not task:
        raise HTTPException(status_code=404, detail=f"Request {task_id} not found.")  # HTTP404 NotFound
    # if (id(task) != task_id): raise HTTPException(status_code=409, detail=f'Wrong task id received. Expected:{id(task)}, Received:{task_id}') # HTTP409 Conflict
    if task.buffer_queue.empty() and not task.lock.locked():
        if task.response:
            # log.info(f'Session {session_id} sending cached response')
            return JSONResponse(task.response, headers=NOCACHE_HEADERS)
        raise HTTPException(status_code=425, detail="Too Early, task not started yet.")  # HTTP425 Too Early
    # log.info(f'Session {session_id} opened live render stream {id(task.buffer_queue)}')
    return StreamingResponse(task.read_buffer_generator(), media_type="application/json")


def _interrupt_active_backend():
    """Forward a UI stop immediately instead of waiting for a progress poll."""
    try:
        from easydiffusion import backend_manager, runtime

        backend_manager.backend.stop_rendering(runtime.context)
    except Exception as e:
        # The task's StopAsyncIteration flag remains authoritative, so a
        # backend that has already finished must not turn Stop into an error.
        log.warning(f"Could not forward generation interrupt to the backend: {e}")


def stop_internal(task: int):
    if not task:
        if (
            task_manager.current_state == task_manager.ServerStates.Online
            or task_manager.current_state == task_manager.ServerStates.Unavailable
        ):
            raise HTTPException(status_code=409, detail="Not currently running any tasks.")  # HTTP409 Conflict
        task_manager.current_state_error = StopAsyncIteration("")
        _interrupt_active_backend()
        return {"OK"}
    task_id = task
    task = task_manager.get_cached_task(task_id, update_ttl=False)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} was not found.")  # HTTP404 Not Found
    if isinstance(task.error, StopAsyncIteration):
        raise HTTPException(status_code=409, detail=f"Task {task_id} is already stopped.")  # HTTP409 Conflict
    task.error = StopAsyncIteration(f"Task {task_id} stop requested.")
    if task.lock.locked():
        _interrupt_active_backend()
    return {"OK"}


def get_image_internal(task_id: int, img_id: int):
    task = task_manager.get_cached_task(task_id, update_ttl=True)
    if not task:
        raise HTTPException(status_code=410, detail=f"Task {task_id} could not be found.")  # HTTP404 NotFound
    if not task.temp_images[img_id]:
        raise HTTPException(status_code=425, detail="Too Early, task data is not available yet.")  # HTTP425 Too Early
    try:
        img_data = task.temp_images[img_id]
        img_data.seek(0)
        return StreamingResponse(img_data, media_type="image/jpeg")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Cloudflare Tunnel ----
class CloudflareTunnel:
    def __init__(self):
        config = app.getConfig()
        self.urls = None
        self.port = config.get("net", {}).get("listen_port")

    def start(self):
        if self.port:
            self.urls = try_cloudflare(self.port)

    def stop(self):
        if self.urls:
            try_cloudflare.terminate(self.port)
            self.urls = None

    @property
    def address(self):
        if self.urls:
            return self.urls.tunnel
        else:
            return None


cloudflare = CloudflareTunnel()


def start_cloudflare_tunnel_internal(req: dict):
    try:
        cloudflare.start()
        log.info(f"- Started cloudflare tunnel. Using address: {cloudflare.address}")
        return JSONResponse({"address": cloudflare.address})
    except Exception as e:
        log.error(str(e))
        log.error(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))


def stop_cloudflare_tunnel_internal(req: dict):
    try:
        cloudflare.stop()
    except Exception as e:
        log.error(str(e))
        log.error(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))


def modify_package_internal(package_name: str, req: dict):
    try:
        cmd = req["command"]
        if cmd not in ("install", "uninstall"):
            raise RuntimeError(f"Unknown command: {cmd}")

        cmd = getattr(package_manager, cmd)
        cmd(package_name)

        return JSONResponse({"status": "OK"}, headers=NOCACHE_HEADERS)
    except Exception as e:
        log.error(str(e))
        log.error(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))


def get_sha256_internal(obj_path):
    from easydiffusion.utils import sha256sum

    path = obj_path.split("/")
    type = path.pop(0)

    try:
        model_path = model_manager.resolve_model_to_use("/".join(path), type)
    except Exception as e:
        log.error(str(e))
        log.error(traceback.format_exc())

        return HTTPException(status_code=404)
    try:
        digest = sha256sum(model_path)
        return {"digest": digest}
    except Exception as e:
        log.error(str(e))
        log.error(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))
