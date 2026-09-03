"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.database import init_db
from app.core.database import check_db_connection
from app.core.database import SessionLocal
from app.core.ffmpeg import check_ffmpeg
from app.core.funasr_loader import model_manager
from app.core.minio_client import minio_service
from app.core.redis_client import close_redis_pool
from app.routers import health, meetings, voiceprint
from app.services.voiceprint_service import voice_library_service


def startup_preload() -> None:
    """Synchronously complete all heavyweight startup work before serving."""
    # Keep this as an environment check only.  Existing upload behavior and
    # its error handling remain unchanged when FFmpeg is unavailable.
    check_ffmpeg()
    # Database schema and object storage must be reachable before model/index
    # readiness is advertised.
    if settings.APP_DEBUG:
        init_db()
    elif not check_db_connection():
        raise RuntimeError("MySQL 不可用，无法完成启动预热")
    minio_service.ensure_bucket()
    model_manager.preload_all(strict=True)
    db = SessionLocal()
    try:
        voice_library_service.preload_index(db)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Do this once, before the first request.  Running in a worker thread keeps
    # the event loop responsive while the lifespan still waits for completion.
    await asyncio.to_thread(startup_preload)
    yield
    await close_redis_pool()


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(voiceprint.router)
app.include_router(meetings.router)
app.include_router(health.router)

# Compatibility aliases matching the standalone FunASR demo protocol.
app.add_api_route("/meeting/offline", meetings.offline_page, methods=["GET"])
app.add_api_route("/meeting/offline2", meetings.offline2_page, methods=["GET"])
app.add_api_route("/meeting/realtime", meetings.realtime_page, methods=["GET"])
app.add_api_route("/upload", meetings.legacy_upload, methods=["POST"])
app.add_api_websocket_route("/ws", meetings.legacy_websocket)
