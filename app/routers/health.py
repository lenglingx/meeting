from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.database import check_db_connection
from app.core.funasr_loader import model_manager
from app.core.minio_client import check_minio_connection
from app.core.redis_client import check_redis_connection
from app.services.voiceprint_service import voice_library_service

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/health/simple", response_class=HTMLResponse)
@router.get("/health_simple.html", response_class=HTMLResponse)
def simple_page(request: Request):
    return templates.TemplateResponse(request, "health_simple.html", {})


@router.get("/health/live")
def live():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready():
    db_ok = check_db_connection()
    redis_ok = await check_redis_connection()
    minio_ok = check_minio_connection()
    models = model_manager.status()
    models_ready = bool(models) and all(item["loaded"] and not item["error"] for item in models.values())
    voiceprint_ready = voice_library_service.model_loaded and voice_library_service.index_loaded and not voice_library_service.index_error
    ready = all([db_ok, redis_ok, minio_ok, models_ready, voiceprint_ready])
    return {
        "status": "ok" if ready else "degraded",
        "mysql": db_ok,
        "redis": redis_ok,
        "minio": minio_ok,
        "models_ready": models_ready,
        "voiceprint_index_ready": voiceprint_ready,
        "voiceprint_model_loaded": voice_library_service.model_loaded,
        "voiceprint_model_error": voice_library_service.model_error,
        "voiceprint": {
            "index_version": voice_library_service.index_version,
            "loaded_speakers": voice_library_service.loaded_speakers,
            "loaded_samples": voice_library_service.loaded_samples,
            "error": voice_library_service.index_error,
        },
        "models": models,
    }


@router.get("/health")
async def health():
    return await ready()
