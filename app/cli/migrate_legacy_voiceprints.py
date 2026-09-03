"""Import legacy voicelibrary/{name}/*.wav files into MySQL and MinIO."""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import settings
from app.core.database import SessionLocal, init_db
from app.models.business import Speaker, VoiceprintSample
from app.services.voiceprint_service import validate_name, voice_library_service


def main() -> None:
    init_db()
    root = Path(settings.VOICELIB_DIR)
    db = SessionLocal()
    success = failed = 0
    try:
        for person_dir in sorted(root.iterdir() if root.exists() else []):
            if not person_dir.is_dir():
                continue
            try:
                name = validate_name(person_dir.name)
                speaker = db.scalar(__import__("sqlalchemy").select(Speaker).where(Speaker.display_name == name, Speaker.status != "deleted"))
                if not speaker:
                    speaker = Speaker(display_name=name, department=None, load_enabled=True)
                    db.add(speaker)
                    db.flush()
                for source in sorted(person_dir.glob("*.wav")):
                    raw = source.read_bytes()
                    wav, duration = voice_library_service.normalize_audio(raw, "wav")
                    digest = hashlib.sha256(wav).hexdigest()
                    if db.scalar(__import__("sqlalchemy").select(VoiceprintSample).where(VoiceprintSample.sha256 == digest)):
                        continue
                    sample_id = __import__("uuid").uuid4().hex
                    key = voice_library_service.object_key(speaker.id, sample_id)
                    from app.core.minio_client import minio_service
                    minio_service.upload_bytes(key, wav, "audio/wav")
                    db.add(VoiceprintSample(id=sample_id, speaker_id=speaker.id, object_key=key, sha256=digest, duration_ms=duration, sample_rate=settings.SAMPLE_RATE, source_type="legacy_import", review_status="approved"))
                    success += 1
                db.commit()
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                failed += 1
                print(f"迁移失败 {person_dir}: {exc}")
        print(f"迁移完成: success={success}, failed={failed}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
