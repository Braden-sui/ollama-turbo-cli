from __future__ import annotations

import os
import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from .router_v1 import router as router_v1
# Import core.config for global logging/warning silencers
from ..core import config as _core_config  # noqa: F401  (import for side effects)


def create_app() -> FastAPI:
    # Load environment variables from the closest .env.local then .env (searching upward)
    try:
        path_local = find_dotenv('.env.local', usecwd=True)
        if path_local:
            load_dotenv(path_local, override=True)
        path_default = find_dotenv('.env', usecwd=True)
        if path_default:
            # Do not override values already loaded from .env.local or process env
            load_dotenv(path_default, override=False)
    except Exception:
        # Fail-closed: continue without env file if lookup fails
        pass
    app = FastAPI(title="Ollama Turbo CLI API", version="1.1.0")

    logger = logging.getLogger("ollama_turbo_api")
    logger.setLevel(logging.INFO)

    # Allow dev frontend origins (Vite/Tauri) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "tauri://localhost",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            dur_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "%s %s -> %s in %.1fms",
                request.method,
                request.url.path,
                getattr(locals().get('response', None), 'status_code', 'NA'),
                dur_ms,
            )

    # Health (root)
    @app.get("/health")
    async def root_health() -> dict:
        return {"status": "ok"}

    # Feature flag to enable/disable API
    if os.getenv("API_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}:
        app.include_router(router_v1)
    return app


app = create_app()
