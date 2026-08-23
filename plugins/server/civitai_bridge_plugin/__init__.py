"""Expose the CivitAI helper through Easy Diffusion's current origin."""

from easydiffusion.server import server_api
from civitai.app import app


if not any(getattr(route, "path", None) == "/civitai-api" for route in server_api.routes):
    server_api.mount("/civitai-api", app)
