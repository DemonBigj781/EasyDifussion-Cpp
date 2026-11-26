import base64
import os
from pathlib import Path
import tempfile
import json
import importlib.util
import sys
import socket
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import unittest
from unittest import mock

from fastapi import FastAPI
import uvicorn

from import_run_history import import_saved
from easydiffusion import run_history as history

TOKEN = "ab" * 32
IMAGE = b"GIF89a-fixture"


class TestPrivateRunHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.archive = root / "archive"
        self.archive.mkdir(mode=0o700)
        self.key = root / "token"
        self.key.write_text(TOKEN + "\n")
        self.key.chmod(0o600)
        env = mock.patch.dict(os.environ, {
            "EASY_DIFFUSION_RUN_HISTORY": "1",
            "EASY_DIFFUSION_RUN_HISTORY_DIR": str(self.archive),
            "EASY_DIFFUSION_RUN_HISTORY_TOKEN_FILE": str(self.key),
        })
        env.start()
        self.addCleanup(env.stop)
        self.app = FastAPI()
        self.app.include_router(history.router)
        config = {"gallery": {"directory": str(root)}}
        self.gallery_config = config
        stub = SimpleNamespace(
            app=SimpleNamespace(BUCKET_DIR=root, ROOT_DIR=root,
                                getConfig=lambda: config, setConfig=lambda new: None),
            kiosk=SimpleNamespace(enabled=lambda: False),
            destockd=SimpleNamespace(),
            run_history=history,
        )
        path = Path(__file__).resolve().parents[1] / "ui/plugins/server/gallery/gallery.py"
        spec = importlib.util.spec_from_file_location("test_real_gallery_routes", path)
        gallery = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"easydiffusion": stub}):
            spec.loader.exec_module(gallery)
        self.app.include_router(gallery.router)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.addCleanup(sock.close)
        self.url = "http://127.0.0.1:" + str(sock.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(self.app, lifespan="off", log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        def stop():
            server.should_exit = True
            thread.join(timeout=5)
        self.addCleanup(stop)
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.02)
        self.assertTrue(server.started, "Fixture server failed to start")

    def get(self, path, token=TOKEN):
        req = Request(self.url + path, headers={"Authorization": "Bearer " + token} if token else {})
        try:
            response = urlopen(req, timeout=5)
        except HTTPError as exc:
            response = exc
        with response:
            data = response.read()
            return SimpleNamespace(status_code=response.code, headers=response.headers,
                                   content=data, text=data.decode(errors="replace"),
                                   json=lambda: json.loads(data))

    def capture(self):
        return history.capture_video(
            [{"data": "data:image/gif;base64," + base64.b64encode(IMAGE).decode()}],
            {"width": 256, "frame_count": 25, "prompt": "keep this private",
             "init_image": "also private", "sampler_name": "Euler"})

    def test_feature_requires_explicit_enable_and_private_token_before_access(self):
        with mock.patch.dict(os.environ, {"EASY_DIFFUSION_RUN_HISTORY": ""}):
            self.assertEqual(self.get("/v1/runs").status_code, 404)
            self.assertIsNone(self.capture())
        self.key.write_text("")
        self.assertEqual(self.get("/v1/runs?limit=garbage").status_code, 404)
        self.assertIsNone(self.capture())
        self.key.write_text(TOKEN)
        self.key.chmod(0o644)
        self.assertEqual(self.get("/v1/runs").status_code, 404)
        self.assertFalse(list(self.archive.iterdir()))

    def test_list_detail_artifact_permissions_and_redaction(self):
        id_ = self.capture()
        self.assertIsNotNone(id_)
        asset = f"/v1/runs/{id_}/artifacts/output-000.gif"
        for path in ("/v1/runs", f"/v1/runs/{id_}", asset):
            for token in (None, "cd" * 32):
                response = self.get(path, token)
                self.assertEqual(response.status_code, 401)
                self.assertIn("no-store", response.headers["Cache-Control"])
        row = self.get("/v1/runs").json()["runs"][0]
        self.assertNotIn("artifacts", row)
        self.assertEqual(row["settings"]["width"], 256)
        detail = self.get(f"/v1/runs/{id_}")
        self.assertNotIn("keep this private", detail.text)
        self.assertNotIn("also private", detail.text)
        self.assertEqual(detail.json()["artifacts"][0]["url"], asset)
        response = self.get(asset)
        self.assertEqual(response.content, IMAGE)
        self.assertEqual(response.headers["content-type"], "image/gif")
        self.assertIn("no-store", response.headers["cache-control"])
        self.assertEqual(self.get(asset.replace("output-000.gif", "manifest.json")).status_code, 404)
        self.key.write_text("cd" * 32)
        self.assertEqual(self.get(asset).status_code, 401)
        self.assertEqual(self.get(asset, "cd" * 32).status_code, 200)
        self.key.unlink()
        self.assertEqual(self.get(asset).status_code, 404)

    def test_operator_import_and_symlink_rejected(self):
        source = Path(self.tmp.name) / "selected.gif"
        source.write_bytes(IMAGE)
        id_ = import_saved(self.archive, [source], {"fps": 8, "seed": 42})
        path = f"/v1/runs/{id_}"
        self.assertEqual(self.get(path).json()["settings"], {"fps": 8, "seed": 42})
        asset = self.archive / id_ / "output-000.gif"
        asset.unlink()
        asset.symlink_to(self.key)
        self.assertEqual(self.get(path + "/artifacts/output-000.gif").status_code, 404)

    def test_gallery_aliases_cannot_list_serve_or_thumbnail_private_archive(self):
        id_ = self.capture()
        visible = self.archive.parent / "visible.gif"
        visible.write_bytes(IMAGE)
        now = time.time() - 10
        os.utime(visible, (now, now))
        secret = self.archive / id_ / "output-000.gif"
        os.utime(secret, (now, now))
        relative = secret.relative_to(self.archive.parent).as_posix()
        for prefix in ("/gallery", "/gallery-plugin"):
            url = prefix + "/file/" + relative
            self.assertEqual(self.get(url, token=None).status_code, 404)
            self.assertEqual(self.get(prefix + "/thumb/" + relative, token=None).status_code, 404)
            gallery = self.get(prefix + "/images", token=None).json()
            self.assertEqual([i["filename"] for i in gallery["images"]], ["visible.gif"])
        with mock.patch.dict(os.environ, {"EASY_DIFFUSION_RUN_HISTORY": ""}):
            self.assertEqual(self.get("/gallery/file/" + relative, token=None).status_code, 404)
        self.gallery_config["gallery"]["directory"] = str(self.archive)
        self.assertEqual(self.get("/gallery/images", token=None).json()["total"], 0)

    def test_no_public_static_route_or_schema_entry(self):
        self.assertNotIn("/v1/runs", self.get("/openapi.json").json()["paths"])
        self.assertEqual(self.get("/media/private/manifest.json").status_code, 404)


if __name__ == "__main__":
    unittest.main()
