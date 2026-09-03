from __future__ import annotations

from pathlib import Path
import threading

from app.config import settings
from app.constants import MODELS


class ModelManager:
    def __init__(self) -> None:
        self.models = {}
        self.errors: dict[str, str] = {}
        self._preloaded = False
        # FunASR instances are shared by all WebSocket/Celery callers.  A
        # process-local lock prevents concurrent GPU calls from corrupting
        # streaming caches or exhausting VRAM.
        self.inference_lock = threading.RLock()

    def path_for(self, name: str) -> Path:
        return Path(settings.MODELS_CACHE_DIR).resolve() / name

    def _resolve_path(self, name: str) -> Path:
        direct = self.path_for(name)
        if direct.exists() and any(direct.iterdir()):
            return direct
        token = MODELS[name].replace("/", "--")
        candidates = list(Path(settings.MODELS_CACHE_DIR).resolve().rglob("config.yaml"))
        for candidate in candidates:
            if token.lower() in str(candidate.parent).lower() or name in str(candidate.parent).lower():
                return candidate.parent
        return direct

    def resolve_path(self, name: str) -> Path:
        return self._resolve_path(name)

    def status(self) -> dict:
        result = {}
        for name in MODELS:
            path = self._resolve_path(name)
            incomplete = any(p.name.endswith(".incomplete") for p in path.rglob("*") if p.is_file()) if path.exists() else False
            result[name] = {"available": path.exists() and any(path.iterdir()) and not incomplete, "path": str(path), "loaded": name in self.models, "error": self.errors.get(name)}
        return result

    @property
    def preloaded(self) -> bool:
        return self._preloaded

    def load(self, name: str, **kwargs):
        if name in self.models:
            return self.models[name]
        path = self._resolve_path(name)
        if not path.exists() or not any(path.iterdir()):
            raise RuntimeError(f"模型 {name} 不存在: {path}")
        if any(p.name.endswith(".incomplete") for p in path.rglob("*") if p.is_file()):
            raise RuntimeError(f"模型 {name} 含未完成权重文件: {path}")
        from funasr import AutoModel
        kwargs.setdefault("disable_update", True)
        model = AutoModel(model=str(path), device=settings.DEVICE, **kwargs)
        self.models[name] = model
        self.errors.pop(name, None)
        return model

    def preload_all(self, *, strict: bool = True) -> dict:
        """Load every configured model once during application startup.

        This is intentionally separate from ``get``: request handlers must never
        trigger a heavyweight model construction or a ModelScope download.
        """
        errors: dict[str, str] = {}
        # Load component models first so the offline recognizer can use local
        # paths and all model states are visible in readiness checks.
        for name in ("fsmn-vad", "ct-punc", "campplus", "paraformer-zh-streaming"):
            try:
                self.load(name)
            except Exception as exc:  # noqa: BLE001
                errors[name] = str(exc)
                self.errors[name] = str(exc)
        try:
            self.load(
                "paraformer-zh",
                vad_model=str(self.resolve_path("fsmn-vad")),
                vad_kwargs={
                    "max_single_segment_time": settings.VAD_MAX_SINGLE_SEGMENT_TIME_MS,
                },
                punc_model=str(self.resolve_path("ct-punc")),
                spk_model=str(self.resolve_path("campplus")),
            )
        except Exception as exc:  # noqa: BLE001
            errors["paraformer-zh"] = str(exc)
            self.errors["paraformer-zh"] = str(exc)
        self._preloaded = not errors
        if errors and strict:
            details = "; ".join(f"{name}: {message}" for name, message in errors.items())
            raise RuntimeError(f"模型预加载失败: {details}")
        return {"loaded": sorted(self.models), "errors": errors}

    def require_loaded(self, name: str):
        """Return a startup-loaded model; never construct one on demand."""
        model = self.models.get(name)
        if model is None:
            error = self.errors.get(name) or "模型尚未在启动阶段加载"
            raise RuntimeError(f"模型 {name} 不可用: {error}")
        return model

    def get(self, name: str, **kwargs):
        """Compatibility alias with strict no-lazy-loading semantics."""
        if kwargs:
            # kwargs used to configure lazy construction.  They are no longer
            # accepted at runtime because model instances are fixed at startup.
            return self.require_loaded(name)
        return self.require_loaded(name)


model_manager = ModelManager()
