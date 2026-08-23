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
    async def _cors_preflight(request: Request, call_next):
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        return await call_next(request)
