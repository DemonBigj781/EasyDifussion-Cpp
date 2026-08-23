from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware


def apply_cors(app: FastAPI) -> None:
    if getattr(app.state, "_cors_installed", False):
        return
    app.state._cors_installed = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.options("/{path:path}")
    def _options_handler(path: str):
        return Response(status_code=200)

    @app.middleware("http")
    async def _cors_fallback(request: Request, call_next):
        if request.method == "OPTIONS":
            response = Response(status_code=200)
        else:
            response = await call_next(request)

        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-Civitai-Key, "
                "X-Civitai-Search-Key"
            )
            response.headers["Access-Control-Max-Age"] = "86400"
            response.headers["Vary"] = "Origin"

        return response
