"""AskDocs AI backend entrypoint (FastAPI + Uvicorn).

Zero-login, zero-API-key. Configures localhost CORS, mounts the v1 API,
exposes a health check, and serves generated exports statically.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router, router_status
from app.core.config import get_settings
from app.models.schemas import HealthResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="100% local-first document Q&A, study, and export backend.",
)

# Localhost-only CORS (zero external dependency).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness probe; reports optional local service reachability."""
    ollama_ok = False
    redis_ok = False
    # Best-effort, non-blocking checks (no hard failure if absent).
    try:
        import httpx

        with httpx.Client(timeout=0.5) as client:
            r = client.get(f"{settings.ollama_host}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    try:
        import redis

        r = redis.Redis.from_url(settings.redis_url, socket_timeout=0.5)
        redis_ok = bool(r.ping())
    except Exception:
        redis_ok = False
    return HealthResponse(
        status="ok",
        ollama_reachable=ollama_ok,
        redis_reachable=redis_ok,
        version=settings.app_version,
    )


@app.get("/api/v1/router-status", tags=["meta"])
async def router_status_endpoint() -> dict:
    return router_status()


app.include_router(api_router, prefix="/api/v1")

# Serve generated exports statically (read-only). Created on demand.
exports_path = Path(settings.exports_dir)
exports_path.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(exports_path)), name="exports")

# In production the backend also serves the built frontend (frontend/dist),
# so one Render service exposes the UI and the API on the same $PORT.
# Registered last, so /api/v1 and /exports match first; unmatched GETs fall
# through to the SPA, which returns index.html for client-side routes.
frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if (frontend_dist / "index.html").exists():

    @app.get("/{path:path}")
    async def spa(path: str) -> FileResponse:
        requested = frontend_dist / path
        if path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
