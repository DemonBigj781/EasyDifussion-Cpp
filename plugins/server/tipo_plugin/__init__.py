from easydiffusion.server import server_api

from .app import app  # re-export for uvicorn


# Serve the helper through Easy Diffusion's own HTTP/HTTPS origin. This avoids
# sending TLS traffic to the standalone, HTTP-only helper port.
if not any(getattr(route, "path", None) == "/tipo-api" for route in server_api.routes):
    server_api.mount("/tipo-api", app)

try:
    from . import server_runner

    server_runner.ensure_server_running()
except Exception:
    pass
