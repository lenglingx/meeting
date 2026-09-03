from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile

from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.core.minio_client import minio_service
from app.models.business import AnalysisTask, Meeting, Speaker, TranscriptSegment
from app.core.funasr_loader import model_manager
from app.services.meeting_pipeline import enrich_segments, identify_meeting_speakers
from app.services.voiceprint_service import voice_library_service


@celery_app.task(bind=True, max_retries=3, name="meeting.process_offline")
def process_offline(self, task_id: str) -> dict:
    db = SessionLocal()
    try:
        task = db.get(AnalysisTask, task_id)
        if not task:
            return {"status": "missing"}
        meeting = db.get(Meeting, task.meeting_id)
        if not meeting or not meeting.audio_object_key:
            raise RuntimeError("会议音频不存在")
        task.status = "running"
        meeting.status = "processing"
        task.celery_task_id = self.request.id
        db.commit()
        with tempfile.TemporaryDirectory(prefix="meeting-task-") as temp:
            source = Path(temp) / "input"
            normalized = Path(temp) / "audio.wav"
            minio_service.download_to_file(meeting.audio_object_key, str(source))
            from app.services.meeting_pipeline import run_offline_pipeline
            segments = run_offline_pipeline(source, normalized, model_manager)
            speaker_rows = [{"id": row.id, "display_name": row.display_name, "department": row.department} for row in db.query(Speaker).filter(Speaker.status != "deleted").all()]
            speaker_matches = identify_meeting_speakers(normalized, segments, voice_library_service, speaker_rows)
            segments = enrich_segments(segments, speaker_matches)
            db.query(TranscriptSegment).filter(TranscriptSegment.meeting_id == meeting.id, TranscriptSegment.version == 1).delete()
            for index, item in enumerate(segments):
                match = speaker_matches.get(str(item.get("speaker", 0)), {})
                db.add(TranscriptSegment(meeting_id=meeting.id, sequence=index, start_ms=item.get("start_ms", 0), end_ms=item.get("end_ms", 0), text=item.get("text", ""), confidence=item.get("confidence"), speaker_id=match.get("speaker_id"), speaker_name_snapshot=item.get("speaker_name"), speaker_department_snapshot=item.get("speaker_department"), source="offline", version=1))
        task.status, task.progress = "succeeded", 100.0
        meeting.status, meeting.ended_at = "completed", datetime.utcnow()
        db.commit()
        return {"status": "succeeded", "segments": len(segments)}
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        task = db.get(AnalysisTask, task_id)
        if task:
            task.status, task.error_message = "failed", str(exc)
            db.commit()
        raise
    finally:
        db.close()
