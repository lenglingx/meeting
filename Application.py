"""
项目启动入口。
python Application.py
"""

import uvicorn

from app.config import settings

if __name__ == "__main__":
    ssl_kwargs = {}

    if settings.APP_SSL_ENABLED and settings.APP_SSL_KEYFILE and settings.APP_SSL_CERTFILE:
        ssl_kwargs = {
            "ssl_keyfile": settings.APP_SSL_KEYFILE,
            "ssl_certfile": settings.APP_SSL_CERTFILE,
        }
        print(f"🔒 HTTPS 已启用，证书: {settings.APP_SSL_CERTFILE}")
    else:
        print("🌐 以 HTTP 方式启动（未启用 SSL）")

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        workers=1,
        reload=settings.APP_DEBUG,
        **ssl_kwargs,
    )
