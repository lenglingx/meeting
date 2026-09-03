"""Database-backed voiceprint management and CAM++ matching."""
from __future__ import annotations

import hashlib
import io
import logging
import re
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydub import AudioSegment
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.config import settings
from app.core.database import SessionLocal
from app.core.funasr_loader import model_manager
from app.core.minio_client import minio_service
from app.models.business import Speaker, VoiceprintSample

logger = logging.getLogger("voiceprint_service")
NAME_PATTERN = re.compile(r"^[\w\- ]+$", re.UNICODE)


def validate_name(value: str) -> str:
    value = (value or "").strip()
    if not value or len(value) > 128 or not NAME_PATTERN.fullmatch(value):
        raise ValueError("姓名只能包含中文、字母、数字、空格、下划线和短横线，且不能为空")
    return value


def normalize_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    arr = arr.reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValueError("声纹向量为空")
    return arr / norm


class VoiceLibraryService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._index: dict[str, np.ndarray] = {}
        self._vectors: dict[str, list[np.ndarray]] = {}
        self._loaded = False
        self._model = None
        self._model_error: str | None = None
        self._index_error: str | None = None
        self.index_version = 0
        self._speaker_meta: dict[str, dict] = {}

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_error(self) -> str | None:
        return self._model_error

    @property
    def model_version(self) -> str:
        return "cam++/speech_campplus_sv_zh-cn_16k-common"

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            self._model = model_manager.require_loaded("campplus")
        except Exception as exc:  # noqa: BLE001
            self._model_error = str(exc)
            logger.warning("CAM++ 加载失败: %s", exc)
        return self._model

    def preload_index(self, db: Session) -> dict:
        """Build the in-memory index during application startup."""
        self._model = model_manager.require_loaded("campplus")
        self._model_error = None
        return self.reload_all(db)

    @staticmethod
    def _embedding_from_result(result: Any) -> Any:
        if isinstance(result, (list, tuple)):
            for item in result:
                found = VoiceLibraryService._embedding_from_result(item)
                if found is not None:
                    return found
        if isinstance(result, dict):
            for key in ("spk_embedding", "speaker_embedding", "embedding", "embeddings", "vector"):
                if result.get(key) is not None:
                    return result[key]
            for key in ("result", "results", "output", "outputs"):
                if key in result:
                    found = VoiceLibraryService._embedding_from_result(result[key])
                    if found is not None:
                        return found
        if hasattr(result, "shape"):
            return result
        return None

    def extract_embedding(self, wav_path: Path) -> np.ndarray:
        model = self._get_model()
        if model is None:
            raise RuntimeError(self._model_error or "CAM++ 模型不可用")
        result = model.generate(input=str(wav_path))
        value = self._embedding_from_result(result)
        if value is None:
            raise RuntimeError("CAM++ 返回结果中没有 embedding")
        return normalize_vector(value)

    def extract_embedding_bytes(self, wav_bytes: bytes) -> np.ndarray:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp.write(wav_bytes)
                temp_path = Path(temp.name)
            return self.extract_embedding(temp_path)
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)

    def speaker_info(self, speaker_id: str | None) -> dict | None:
        if not speaker_id:
            return None
        with self._lock:
            return self._speaker_meta.get(speaker_id)

    @staticmethod
    def normalize_audio(raw_bytes: bytes, src_format: str | None = None) -> tuple[bytes, int]:
        try:
            audio = AudioSegment.from_file(io.BytesIO(raw_bytes), format=src_format)
        except Exception:
            audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
        audio = audio.set_frame_rate(settings.SAMPLE_RATE).set_channels(settings.CHANNELS)
        duration = len(audio)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return buffer.getvalue(), duration

    @staticmethod
    def object_key(speaker_id: str, sample_id: str) -> str:
        return f"{settings.MINIO_VOICEPRINT_PREFIX}/{speaker_id}/{sample_id}.wav"

    def list_speakers(self, db: Session) -> list[dict]:
        rows = db.scalars(select(Speaker).options(joinedload(Speaker.samples)).where(Speaker.status != "deleted").order_by(Speaker.display_name)).unique().all()
        return [{
            "id": s.id,
            "name": s.display_name,
            "display_name": s.display_name,
            "department": s.department,
            "display_label": f"{s.display_name}（{s.department}）" if s.department else s.display_name,
            "load_enabled": s.load_enabled,
            "status": s.status,
            "samples": [{"id": x.id, "object_key": x.object_key, "review_status": x.review_status, "embedding_status": x.embedding_status, "source_type": x.source_type, "duration_ms": x.duration_ms} for x in s.samples if x.review_status != "deleted"],
        } for s in rows]

    def create_speaker(self, db: Session, name: str, department: str | None = None, load_enabled: bool = True) -> Speaker:
        speaker = Speaker(display_name=validate_name(name), department=(department or "").strip() or None, load_enabled=load_enabled)
        db.add(speaker)
        db.commit()
        db.refresh(speaker)
        return speaker

    def save_sample(self, db: Session, speaker_id: str, raw_bytes: bytes, src_format: str | None, source_type: str = "manual_upload", review_status: str = "pending") -> dict:
        speaker = db.get(Speaker, speaker_id)
        if not speaker or speaker.status == "deleted":
            raise ValueError("人员不存在")
        wav_bytes, duration = self.normalize_audio(raw_bytes, src_format)
        if duration < settings.MIN_ENROLL_DURATION_MS:
            raise ValueError("声纹录音至少需要 2 秒")
        if duration > settings.MAX_ENROLL_DURATION_MS:
            raise ValueError("声纹录音最长不能超过 120 秒")
        digest = hashlib.sha256(wav_bytes).hexdigest()
        duplicate = db.scalar(select(VoiceprintSample).where(VoiceprintSample.sha256 == digest, VoiceprintSample.review_status != "deleted"))
        if duplicate:
            raise ValueError("该音频已经存在于声纹库")
        sample_id = str(uuid.uuid4())
        key = self.object_key(speaker_id, sample_id)
        minio_service.upload_bytes(key, wav_bytes, "audio/wav")
        sample = VoiceprintSample(id=sample_id, speaker_id=speaker_id, object_key=key, sha256=digest, duration_ms=duration, sample_rate=settings.SAMPLE_RATE, source_type=source_type, review_status=review_status)
        db.add(sample)
        db.commit()
        return {"id": sample_id, "speaker_id": speaker_id, "object_key": key, "duration_ms": duration, "review_status": review_status}

    def delete_sample(self, db: Session, speaker_id: str, sample_id: str) -> None:
        sample = db.scalar(select(VoiceprintSample).where(VoiceprintSample.id == sample_id, VoiceprintSample.speaker_id == speaker_id))
        if not sample:
            raise FileNotFoundError("声纹样本不存在")
        sample.review_status = "deleted"
        db.commit()
        try:
            minio_service.delete_object(sample.object_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除 MinIO 声纹对象失败: %s", exc)

    def reload_all(self, db: Session) -> dict:
        new_vectors: dict[str, list[np.ndarray]] = {}
        samples = db.scalars(select(VoiceprintSample).join(Speaker).where(Speaker.status == "active", Speaker.load_enabled.is_(True), VoiceprintSample.review_status == "approved")).all()
        speaker_meta = {
            row.id: {"display_name": row.display_name, "department": row.department, "label": f"{row.display_name}（{row.department}）" if row.department else row.display_name}
            for row in db.scalars(select(Speaker).where(Speaker.status == "active", Speaker.load_enabled.is_(True))).all()
        }
        errors = []
        with tempfile.TemporaryDirectory(prefix="voiceprint-") as temp:
            for sample in samples:
                path = Path(temp) / f"{sample.id}.wav"
                try:
                    minio_service.download_to_file(sample.object_key, str(path))
                    vector = self.extract_embedding(path)
                    new_vectors.setdefault(sample.speaker_id, []).append(vector)
                    sample.embedding_status = "ready"
                    sample.embedding_model_version = self.model_version
                    sample.error_message = None
                except Exception as exc:  # noqa: BLE001
                    sample.embedding_status = "failed"
                    sample.error_message = str(exc)
                    errors.append({"sample_id": sample.id, "error": str(exc)})
        db.commit()
        new_index = {sid: normalize_vector(np.mean(vectors, axis=0)) for sid, vectors in new_vectors.items() if vectors}
        with self._lock:
            self._vectors, self._index = new_vectors, new_index
            self._speaker_meta = {sid: speaker_meta[sid] for sid in new_index if sid in speaker_meta}
            self._loaded = True
            self._index_error = "; ".join(f"{item['sample_id']}: {item['error']}" for item in errors) or None
            self.index_version += 1
        return {"index_version": self.index_version, "speakers": len(new_index), "samples": sum(len(v) for v in new_vectors.values()), "errors": errors}

    @property
    def index_loaded(self) -> bool:
        with self._lock:
            return self._loaded

    @property
    def loaded_speakers(self) -> int:
        with self._lock:
            return len(self._index)

    @property
    def loaded_samples(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._vectors.values())

    @property
    def index_error(self) -> str | None:
        with self._lock:
            return self._index_error

    def identify(self, embedding: np.ndarray) -> dict:
        embedding = normalize_vector(embedding)
        with self._lock:
            scores = sorted(((sid, float(np.dot(embedding, centroid))) for sid, centroid in self._index.items()), key=lambda item: item[1], reverse=True)
        if not scores:
            return {"speaker_id": None, "score": 0.0, "status": "empty_library"}
        sid, score = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.0
        margin = score - second
        if score < settings.VOICEPRINT_MATCH_THRESHOLD or margin < settings.VOICEPRINT_MIN_MARGIN:
            return {"speaker_id": None, "candidate_id": sid, "score": round(score, 4), "second_score": round(second, 4), "margin": round(margin, 4), "status": "unknown"}
        return {"speaker_id": sid, "score": round(score, 4), "second_score": round(second, 4), "margin": round(margin, 4), "status": "matched"}


voice_library_service = VoiceLibraryService()
