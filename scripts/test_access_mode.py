"""Loopback HTTP regressions for secured website and explicit dev mode."""
import http.cookiejar
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener

import uvicorn
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
from easydiffusion.access_mode import install
from easydiffusion.run_history import router as history_router

KEY = "ac" * 32
AUTH = {"Authorization": "Bearer " + KEY}


class AccessModeTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {
            "EASY_DIFFUSION_DEV_MODE": "",
            "EASY_DIFFUSION_BEARER_TOKEN_FILE": "",
            "EASY_DIFFUSION_RUN_HISTORY_TOKEN_FILE": "",
            "EASY_DIFFUSION_BEARER_TOKEN": "",
            "EASY_DIFFUSION_RUN_HISTORY": "",
        })
        patch.start()
        self.addCleanup(patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.token = Path(self.tmp.name) / "token"
        self.token.write_text(KEY + "\n")
        self.token.chmod(0o600)
        os.environ["EASY_DIFFUSION_BEARER_TOKEN_FILE"] = str(self.token)
        app = FastAPI()
        for route in ("/", "/legacy", "/get/app_config", "/media/private.gif", "/image/stream/demo"):
            app.add_api_route(route, lambda: {"private": True})
        app.add_api_route("/render", lambda: {"queued": True}, methods=["POST"])
        app.include_router(history_router)
        install(app)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.addCleanup(sock.close)
        self.url = "http://127.0.0.1:" + str(sock.getsockname()[1])
        self.client = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error", proxy_headers=False))
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
        self.assertTrue(server.started)

    def call(self, path, method="GET", headers=None):
        r = Request(self.url + path, headers=headers or {}, method=method)
        try:
            response = self.client.open(r, timeout=5)
        except HTTPError as exc:
            response = exc
        with response:
            return response.code, response.headers, response.read()

    def login(self):
        return self.call("/auth/login", "POST", {**AUTH, "Origin": self.url})

    def test_production_requires_bearer_on_every_plain_rest_call(self):
        for path in ("/get/app_config", "/media/private.gif", "/image/stream/demo", "/docs"):
            self.assertEqual(self.call(path)[0], 401)
            self.assertEqual(self.call(path, headers=AUTH)[0], 200)
        self.assertEqual(self.call("/render", "POST")[0], 401)
        self.assertEqual(self.call("/render", "POST", AUTH)[0], 200)

    def test_browser_login_protects_navigation_assets_and_same_origin_api(self):
        status, headers, body = self.call("/")
        self.assertEqual(status, 401)
        self.assertIn(b"Bearer key", body)
        self.assertNotIn(b'"private"', body)
        self.assertIn("no-store", headers["Cache-Control"])
        status, headers, body = self.login()
        self.assertEqual(status, 200)
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=strict", headers["Set-Cookie"])
        self.assertNotIn(KEY, headers["Set-Cookie"])
        self.assertEqual(self.call("/")[0], 200)
        self.assertEqual(self.call("/media/private.gif", headers={"Sec-Fetch-Site": "same-origin"})[0], 200)
        self.assertEqual(self.call("/get/app_config")[0], 401)
        self.assertEqual(self.call("/render", "POST", {"Origin": self.url})[0], 200)
        self.assertEqual(self.call("/render", "POST", {"Origin": "https://other.example"})[0], 401)
        self.assertEqual(self.call("/auth/logout", "POST", {"Origin": self.url})[0], 200)
        self.assertEqual(self.call("/")[0], 401)

    def test_no_token_no_access_and_rotation_invalidates_browser_session(self):
        self.assertEqual(self.login()[0], 200)
        self.token.write_text("ef" * 32)
        self.assertEqual(self.call("/")[0], 401)
        self.assertEqual(self.call("/get/app_config", headers=AUTH)[0], 401)
        self.token.unlink()
        self.assertEqual(self.call("/")[0], 503)
        self.assertEqual(self.call("/get/app_config", headers=AUTH)[0], 503)

    def test_dev_is_explicit_direct_loopback_only(self):
        os.environ["EASY_DIFFUSION_DEV_MODE"] = "true"
        self.assertEqual(self.call("/")[0], 401)
        os.environ["EASY_DIFFUSION_DEV_MODE"] = "1"
        self.token.unlink()
        self.assertEqual(self.call("/")[0], 200)
        self.assertEqual(self.call("/render", "POST")[0], 200)
        self.assertEqual(self.call("/v1/runs")[0], 404)
        self.assertEqual(self.call("/render", "POST", {"Host": "example.org"})[0], 403)
        self.assertEqual(self.call("/render", "POST", {"X-Forwarded-For": "192.0.2.2"})[0], 403)
        self.assertEqual(self.call("/render", "POST", {"Origin": "https://other.example"})[0], 403)

    def test_cross_origin_login_is_rejected(self):
        self.assertEqual(self.call("/auth/login", "POST", {**AUTH, "Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.call("/")[0], 401)


if __name__ == "__main__":
    unittest.main()
