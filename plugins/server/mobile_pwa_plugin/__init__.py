"""Root-scoped PWA assets and optional HTTPS for the mobile UI plugin."""

from pathlib import Path
import threading

import uvicorn
from easydiffusion import app as easy_app
from easydiffusion.server import server_api
from starlette.responses import FileResponse, JSONResponse


ASSET_DIR = Path(__file__).resolve().parent
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _https_settings() -> dict:
    config = easy_app.getConfig()
    net = config.get("net", {}) or {}
    https = net.get("https", {}) or {}
    return {
        "enabled": https.get("enabled", False) is True,
        "port": int(https.get("port", 10000)),
        "host": net.get("bind_ip", "0.0.0.0") if net.get("listen_to_network", False) else "127.0.0.1",
        "certfile": str(Path(https.get("certfile", "")).expanduser()),
        "keyfile": str(Path(https.get("keyfile", "")).expanduser()),
        "ca_certfile": str(Path(https.get("ca_certfile", "")).expanduser()),
        "http_port": int(net.get("listen_port", 9000)),
    }


def _route_exists(path: str) -> bool:
    return any(getattr(route, "path", None) == path for route in server_api.routes)


if not _route_exists("/mobile-pwa-sw.js"):

    @server_api.get("/mobile-pwa-sw.js", include_in_schema=False)
    def mobile_pwa_service_worker():
        headers = dict(NO_CACHE_HEADERS)
        headers["Service-Worker-Allowed"] = "/"
        return FileResponse(
            ASSET_DIR / "mobile-pwa-sw.js",
            media_type="application/javascript",
            headers=headers,
        )


if not _route_exists("/mobile-pwa.webmanifest"):

    @server_api.get("/mobile-pwa.webmanifest", include_in_schema=False)
    def mobile_pwa_manifest():
        return FileResponse(
            ASSET_DIR / "mobile-pwa.webmanifest",
            media_type="application/manifest+json",
            headers=NO_CACHE_HEADERS,
        )


if not _route_exists("/mobile-pwa-icon-192.png"):

    @server_api.get("/mobile-pwa-icon-192.png", include_in_schema=False)
    def mobile_pwa_icon_192():
        return FileResponse(
            ASSET_DIR / "mobile-pwa-icon-192.png",
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )


if not _route_exists("/mobile-pwa-ca.crt"):

    @server_api.get("/mobile-pwa-ca.crt", include_in_schema=False)
    def mobile_pwa_ca_certificate():
        ca_certfile = Path(_https_settings()["ca_certfile"])
        if not ca_certfile.is_file():
            return JSONResponse(
                {"error": "The local CA certificate is not configured."},
                status_code=404,
            )
        return FileResponse(
            ca_certfile,
            media_type="application/x-x509-ca-cert",
            filename="Easy-Diffusion-Local-CA.crt",
            headers={"Cache-Control": "no-store"},
        )


if not _route_exists("/mobile-pwa-https"):

    @server_api.get("/mobile-pwa-https", include_in_schema=False)
    def mobile_pwa_https_status():
        https = _https_settings()
        return JSONResponse(
            {
                "enabled": https["enabled"],
                "port": https["port"],
                "http_port": https["http_port"],
                "ca_available": Path(https["ca_certfile"]).is_file(),
            },
            headers=NO_CACHE_HEADERS,
        )


def _start_https_companion() -> None:
    https = _https_settings()
    if not https["enabled"]:
        return
    if https["port"] == https["http_port"]:
        print(
            "[Mobile PWA] HTTPS port must differ from net.listen_port; "
            "the HTTPS companion was not started."
        )
        return

    certfile = Path(https["certfile"])
    keyfile = Path(https["keyfile"])
    if not certfile.is_file() or not keyfile.is_file():
        print(
            "[Mobile PWA] HTTPS is enabled but the certificate or private key "
            "is missing; the HTTPS companion was not started."
        )
        return

    state = server_api.state
    if getattr(state, "mobile_pwa_https_started", False):
        return
    state.mobile_pwa_https_started = True

    def run_https() -> None:
        try:
            config = uvicorn.Config(
                server_api,
                host=https["host"],
                port=https["port"],
                log_level="warning",
                access_log=False,
                ssl_certfile=str(certfile),
                ssl_keyfile=str(keyfile),
            )
            https_server = uvicorn.Server(config)
            state.mobile_pwa_https_server = https_server
            print(
                f"[Mobile PWA] HTTPS ready on "
                f"https://{https['host']}:{https['port']}"
            )
            https_server.run()
        except Exception as error:
            state.mobile_pwa_https_started = False
            print(f"[Mobile PWA] HTTPS companion failed: {error}")

    thread = threading.Thread(
        target=run_https,
        name="mobile-pwa-https",
        daemon=True,
    )
    state.mobile_pwa_https_thread = thread
    thread.start()


_start_https_companion()
