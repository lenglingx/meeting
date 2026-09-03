from fastapi import HTTPException

from app.config import settings


def require_admin() -> None:
    if settings.ADMIN_AUTH_ENABLED:
        raise HTTPException(status_code=401, detail="管理员鉴权尚未接入")
