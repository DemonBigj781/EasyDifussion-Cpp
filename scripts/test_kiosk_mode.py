import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "ui")]

from fastapi import HTTPException


class ASGIClient:
    """Exercise the real ASGI stack without requiring optional httpx test packages."""
    def __init__(self, application):
        self.application = application

    def request(self, method, path, payload=None):
        async def run():
            messages, delivered = [], False
            body = json.dumps(payload).encode() if payload is not None else b""

            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                messages.append(message)

            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                     "http_version": "1.1", "method": method, "scheme": "http", "path": path,
                     "raw_path": path.encode(), "query_string": b"", "root_path": "",
                     "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
                     "client": ("127.0.0.1", 12345), "server": ("test", 80)}
            await self.application(scope, receive, send)
            status = next(message["status"] for message in messages if message["type"] == "http.response.start")
            output = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
            return SimpleNamespace(status_code=status, json=lambda: json.loads(output))
        return asyncio.run(run())

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, json):
        return self.request("POST", path, json)


class KioskTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("easydiffusion.kiosk"), "Kiosk policy is missing")
        from easydiffusion import kiosk
        self.kiosk = kiosk
        self.config = {"models_dir": "keep-this", "backend_config": {"platform": "cuda"}, "kiosk_mode": False}
        self.read = patch.object(kiosk.app, "getConfig", side_effect=lambda: copy.deepcopy(self.config))
        self.write = patch.object(kiosk.app, "setConfig", side_effect=lambda value: self.config.update(copy.deepcopy(value)))
        self.read.start(); self.write.start()
        self.addCleanup(self.read.stop); self.addCleanup(self.write.stop)
        kiosk.clear_approvals()

    def test_setting_roundtrip_preserves_other_settings(self):
        original = copy.deepcopy(self.config)
        self.assertFalse(self.kiosk.status()["enabled"])
        self.kiosk.set_enabled(True)
        self.assertTrue(self.kiosk.status()["enabled"])
        self.kiosk.set_enabled(False)
        self.assertEqual(self.config, original)
        for invalid in ("false", "true", 1, None):
            with self.assertRaises(HTTPException): self.kiosk.set_enabled(invalid)

    def test_exact_base_model_allowlist_not_architecture_finetunes(self):
        self.config["kiosk_mode"] = True
        good = [{"model": model, "tags": ["stable-diffusion"]} for model in self.kiosk.BASE_MODELS]
        bad = [{"model": "sdxl/custom-finetune", "tags": ["stable-diffusion", "sd_xl_base"]},
               {"model": "unlisted/anima-base-v1.0", "tags": ["anima"]},
               {"model": "lora/test", "tags": ["lora"]}]
        self.assertEqual(self.kiosk.filter_models(good + bad + good), good)
        self.config["kiosk_mode"] = False
        self.assertEqual(self.kiosk.filter_models(good + bad), good + bad)

    def test_render_rejects_loras_aliases_and_unapproved_checkpoints(self):
        self.config["kiosk_mode"] = True
        for model in self.kiosk.BASE_MODELS:
            self.kiosk.validate_render({"use_stable_diffusion_model": model, "prompt": "a forest"})
        good = {"use_stable_diffusion_model": next(iter(self.kiosk.BASE_MODELS)), "prompt": "a forest"}
        for extra in ({"use_lora_model": "test"}, {"use_lora_model": ["test"]},
                      {"model_paths": {"lora": "test"}}, {"prompt": "<LoRA:test:1>"},
                      {"negative_prompt": "<lyco:test:1>"}, {"use_stable_diffusion_model": "sdxl/fine-tune"},
                      {"use_stable_diffusion_model": ""}, {"use_stable_diffusion_model": [good["use_stable_diffusion_model"]]}):
            with self.subTest(extra=extra), self.assertRaises(HTTPException): self.kiosk.validate_render({**good, **extra})
        self.config["kiosk_mode"] = False
        self.kiosk.validate_render({**good, "use_lora_model": "test"})

    def test_sensitive_gallery_and_model_routes_blocked_but_settings_remain(self):
        for url in ("/gallery/file/test.png", "/gallery-plugin/thumb/test.png", "/files/list_lora",
                    "/meta/scan_loras", "/files/list_checkpoints", "/perchance/image",
                    "/perchance-plugin/image", "/perchance/generated/file/test.png",
                    "/training/models", "/cpp-ui/models/download", "/legacy"):
            self.assertTrue(self.kiosk.blocked_route(url, "GET"), url)
        self.assertTrue(self.kiosk.blocked_route("/gallery/settings", "POST"))
        for url in ("/kiosk", "/cpp-ui/settings", "/get/model", "/gallery/images", "/gallery/settings", "/perchance/gallery/list"):
            self.assertFalse(self.kiosk.blocked_route(url, "GET"), url)

    def test_pg_cache_approval_is_expiring_and_reset_on_mode_change(self):
        self.config["kiosk_mode"] = True
        name = "a" * 64 + ".jpeg"
        with self.assertRaises(HTTPException): self.kiosk.require_approved_perchance(name)
        with patch.object(self.kiosk.time, "monotonic", return_value=10): self.kiosk.approve_perchance(name)
        with patch.object(self.kiosk.time, "monotonic", return_value=11): self.kiosk.require_approved_perchance(name)
        with patch.object(self.kiosk.time, "monotonic", return_value=10000):
            with self.assertRaises(HTTPException): self.kiosk.require_approved_perchance(name)
        self.kiosk.approve_perchance(name)
        self.kiosk.set_enabled(False); self.kiosk.set_enabled(True)
        with self.assertRaises(HTTPException): self.kiosk.require_approved_perchance(name)

    def test_perchance_forces_g_and_disables_unrated_generated_images(self):
        from ui.plugins.server.perchance import perchance
        self.config["kiosk_mode"] = True
        _, rating, download, visible = perchance._gallery_common({"content_filter": "none", "download": True, "visible": True})
        self.assertEqual(rating, "g"); self.assertFalse(download); self.assertFalse(visible)
        with patch.object(perchance, "_run_locked", new=AsyncMock()) as launch:
            with self.assertRaises(HTTPException): asyncio.run(perchance.generate_image({"prompt": "forest"}))
            launch.assert_not_called()
        self.assertEqual(perchance.recent_images()["images"], [])
        self.config["kiosk_mode"] = False
        self.assertEqual(perchance._gallery_common({})[1], "none")

    def test_perchance_g_list_approves_only_filtered_results(self):
        from ui.plugins.server.perchance import perchance
        self.config["kiosk_mode"] = True
        name = "a" * 64 + ".jpeg"
        item = {"imageId": "a" * 64, "imageUrl": "https://aigc.uploads.dev/image/" + name, "prompt": "forest",
                "channel": "ai-text-to-image-generator", "subChannel": "public"}
        result = {"stdout": json.dumps({"entries": [item]}), "stderr": "", "returncode": 0}
        with patch.object(perchance, "_run_gallery_locked", new=AsyncMock(return_value=result)) as launch, \
             patch.object(perchance, "_attach_gallery_previews", new=AsyncMock(side_effect=lambda entries: entries)):
            page = asyncio.run(perchance.gallery_list({"content_filter": "none", "limit": 1}))
        args = launch.call_args.args[0]
        self.assertEqual(args[args.index("--content-filter") + 1], "g")
        self.assertEqual(len(page["entries"]), 1)
        self.kiosk.require_approved_perchance(name)

    def test_g_result_requires_matching_public_identity_and_rejects_flagged_entries(self):
        safe = {"imageId": "a" * 64, "channel": "demo", "subChannel": "public"}
        self.assertTrue(self.kiosk.perchance_g_entry_is_valid(safe, "a" * 64 + ".jpeg", "demo"))
        for field in ("nsfw", "shocking", "pg13Soft"):
            for value in (True, None, 0, "false"):
                self.assertFalse(self.kiosk.perchance_g_entry_is_valid({**safe, field: value}, "a" * 64 + ".jpeg", "demo"))
        self.assertFalse(self.kiosk.perchance_g_entry_is_valid(safe, "b" * 64 + ".jpeg", "demo"))
        self.assertFalse(self.kiosk.perchance_g_entry_is_valid(safe, "a" * 64 + ".jpeg", "other"))
        self.assertFalse(self.kiosk.perchance_g_entry_is_valid({**safe, "subChannel": "private"}, "a" * 64 + ".jpeg", "demo"))

    def test_normal_mode_does_not_change_existing_gallery_rating_behavior(self):
        from ui.plugins.server.perchance import perchance
        self.config["kiosk_mode"] = False
        entries, verified = asyncio.run(perchance._filter_g_entries([{"test": "unchanged"}], "g", "demo"))
        self.assertEqual(entries, [{"test": "unchanged"}])
        self.assertFalse(verified)

    def test_unverified_feed_entry_is_suppressed_before_preview_download(self):
        from ui.plugins.server.perchance import perchance
        self.config["kiosk_mode"] = True
        item = {"imageId": "b" * 64, "imageUrl": "https://aigc.uploads.dev/image/" + "b" * 64 + ".jpeg", "prompt": "unknown"}
        result = {"stdout": json.dumps({"entries": [item]}), "stderr": "", "returncode": 0}
        with patch.object(perchance, "_run_gallery_locked", new=AsyncMock(return_value=result)), \
             patch.object(perchance, "_attach_gallery_previews", new=AsyncMock(side_effect=lambda entries: entries)) as previews:
            page = asyncio.run(perchance.gallery_list({"limit": 1}))
        self.assertEqual(page["entries"], [])
        self.assertEqual(page["suppressed_count"], 1)
        self.assertEqual(previews.call_args.args[0], [])

    def test_unfiltered_request_cannot_be_approved_after_kiosk_activates(self):
        from ui.plugins.server.perchance import perchance
        self.config["kiosk_mode"] = True
        with self.assertRaises(HTTPException):
            asyncio.run(perchance._filter_g_entries([], "none", "demo"))

    def test_gallery_uses_destockd_without_scanning_local_directory(self):
        from ui.plugins.server.gallery import gallery
        from easydiffusion import destockd
        self.config["kiosk_mode"] = True
        expected = {"source": "destockd", "images": []}
        with patch.object(destockd, "list_images", return_value=expected), \
             patch.object(gallery, "configured_directory", side_effect=AssertionError("must not inspect local images")):
            self.assertEqual(gallery.list_images(), expected)
            self.assertEqual(gallery.get_settings()["source"], "destockd")


class DestockdTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("easydiffusion.destockd"), "Destockd integration is missing")
        from easydiffusion import destockd
        self.module = destockd

    def test_response_maps_only_destockd_keyframes_and_hides_flagged_footage(self):
        data = {"results": [
            {"film": "Forests", "shot": "shot_001", "keyframe": "/keyframes/Forests/shot_001.jpg"},
            {"film": "Excluded", "shot": "shot_002", "keyframe": "/keyframes/x.jpg", "homepage_excluded": True},
        ]}
        with patch.object(self.module, "fetch_catalog", return_value=data): result = self.module.list_images()
        self.assertEqual(result["source"], "destockd")
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0]["url"], "https://destockd.com/keyframes/Forests/shot_001.jpg")
        self.assertNotIn("/gallery/file/", json.dumps(result))
        self.assertFalse(result["has_next"])

    def test_untrusted_urls_and_invalid_catalog_fail_without_local_fallback(self):
        for url in ("https://evil.invalid/image.jpg", "/keyframes/../admin", "/keyframes/%2e%2e/admin", "file:///tmp/x"):
            payload = {"results": [{"film": "test", "shot": "shot_1", "keyframe": url}]}
            with self.subTest(url=url), patch.object(self.module, "fetch_catalog", return_value=payload):
                with self.assertRaises(HTTPException): self.module.list_images()
        with patch.object(self.module, "fetch_catalog", return_value={"bad": "shape"}):
            with self.assertRaises(HTTPException): self.module.list_images()


class KioskHTTPTests(unittest.TestCase):
    def test_http_toggle_selectors_gallery_and_render_enforcement(self):
        from fastapi import FastAPI
        from easydiffusion import app, backend_manager, kiosk, server
        from ui.plugins.server.gallery import gallery
        config = {"kiosk_mode": False, "models_dir": "untouched", "gallery": {"directory": "/tmp"},
                  "backend_config": {"platform": "cuda"}}
        bases = [{"model": name, "tags": ["stable-diffusion"]} for name in kiosk.BASE_MODELS]
        extra = {"model": "custom/finetune", "tags": ["stable-diffusion", "sd_xl_base"]}
        lora = {"model": "private/lora", "tags": ["lora"]}
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, "getConfig", side_effect=lambda: copy.deepcopy(config)))
            stack.enter_context(patch.object(app, "setConfig", side_effect=lambda value: config.update(value)))
            stack.enter_context(patch.object(app, "save_to_config"))
            stack.enter_context(patch.object(server, "server_api", FastAPI()))
            stack.enter_context(patch.object(server.training, "generation_guard", side_effect=nullcontext))
            stack.enter_context(patch.object(server.task_manager, "backend_maintenance", side_effect=nullcontext))
            stack.enter_context(patch.object(backend_manager, "backend", SimpleNamespace(refresh_models=lambda: None)))
            stack.enter_context(patch.object(server.model_manager, "list_models", return_value=bases + [extra, lora]))
            enqueue = stack.enter_context(patch.object(server, "enqueue_task", return_value={"task": "intercepted"}))
            stack.enter_context(patch.object(server, "RenderTask", return_value=object()))
            server.init()
            client = ASGIClient(server.server_api)
            self.assertEqual(len(client.get("/get/model").json()["models"]), 5)
            self.assertEqual(client.post("/kiosk", json={"enabled": "true"}).status_code, 400)
            self.assertEqual(client.post("/kiosk", json={"enabled": True}).status_code, 200)
            self.assertEqual(config["models_dir"], "untouched")
            self.assertEqual(config["backend_config"], {"platform": "cuda"})
            self.assertEqual(client.get("/get/model").json()["models"], bases)
            self.assertEqual(client.get("/get/models").json()["models"], bases)
            self.assertEqual(client.get("/get/lora").json()["models"], [])
            for url in ("/gallery/file/private.png", "/gallery-plugin/thumb/private.png", "/files/list_lora",
                        "/perchance/generated/file/private.png", "/get/lora/metadata", "/legacy"):
                self.assertEqual(client.get(url).status_code, 403, url)
            request = {"prompt": "forest", "use_stable_diffusion_model": bases[0]["model"]}
            self.assertEqual(client.post("/render", json={**request, "use_lora_model": "private/lora"}).status_code, 403)
            self.assertEqual(client.post("/render", json={**request, "use_stable_diffusion_model": "custom/finetune"}).status_code, 403)
            enqueue.assert_not_called()
            self.assertEqual(client.post("/render", json=request).status_code, 200)
            enqueue.assert_called_once()
            with patch.object(gallery.destockd, "list_images", return_value={"source": "destockd", "images": []}), \
                 patch.object(gallery, "configured_directory", side_effect=AssertionError("local scan")):
                self.assertEqual(client.get("/gallery/images").json()["source"], "destockd")
            self.assertEqual(client.post("/kiosk", json={"enabled": False}).status_code, 200)
            self.assertEqual(len(client.get("/get/model").json()["models"]), 5)


if __name__ == "__main__":
    unittest.main()
