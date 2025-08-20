from __future__ import annotations

import os
import time
import logging
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from .router_v1 import router as router_v1


def create_app() -> FastAPI:
    # Ensure environment variables from .env are loaded when running the API server
    load_dotenv()
    app = FastAPI(title="Ollama Turbo CLI API", version="1.1.0")

    logger = logging.getLogger("ollama_turbo_api")
    logger.setLevel(logging.INFO)

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
