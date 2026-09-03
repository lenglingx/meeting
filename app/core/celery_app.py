from app.config import settings

try:
    from celery import Celery
except ImportError:  # keep voiceprint/readiness endpoints usable before worker deps are installed
    class _UnavailableTask:
        def delay(self, *_args, **_kwargs):
            raise RuntimeError("Celery 未安装，请先安装 requirements.txt")

    class _UnavailableCelery:
        control = type("Control", (), {"revoke": lambda *_args, **_kwargs: None})()

        def task(self, *args, **kwargs):
            def decorator(func):
                func.delay = _UnavailableTask().delay
                return func
            return decorator

    celery_app = _UnavailableCelery()
else:
    celery_app = Celery("meeting", broker=settings.CELERY_BROKER, backend=settings.CELERY_BACKEND)
    celery_app.conf.update(task_track_started=True, result_expires=86400, task_serializer="json", accept_content=["json"])
