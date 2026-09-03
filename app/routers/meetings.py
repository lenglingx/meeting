from __future__ import annotations

import uuid
import asyncio
import json
import tempfile
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from starlette.concurrency import run_in_threadpool

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app
from app.core.database import get_db
from app.core.database import SessionLocal
from app.core.minio_client import minio_service
from app.core.funasr_loader import model_manager
from app.config import settings
from app.models.business import AnalysisTask, Meeting, Speaker, TranscriptSegment, RealtimeTranscriptEvent
from app.tasks.meeting_tasks import process_offline
from app.services.meeting_pipeline import enrich_segments, identify_meeting_speakers, run_offline_pipeline, wav_duration_ms
from app.services.voiceprint_service import voice_library_service
from app.services.realtime_engine import RealtimeEngine
from app.services.realtime_session import RealtimeSession

router = APIRouter(prefix="/api/v1")
templates = Jinja2Templates(directory="app/templates")


@router.get("/meeting/offline", response_class=HTMLResponse)
def offline_page(request: Request):
    return templates.TemplateResponse(request, "meeting_offline.html", {})


@router.get("/meeting/offline2", response_class=HTMLResponse)
def offline2_page(request: Request):
    return templates.TemplateResponse(request, "meeting_offline2.html", {})


@router.get("/meeting/realtime", response_class=HTMLResponse)
def realtime_page(request: Request):
    return templates.TemplateResponse(request, "meeting_realtime.html", {})


@router.websocket("/realtime/meetings/{meeting_id}/stream")
async def realtime_stream(websocket: WebSocket, meeting_id: str):
    await websocket.accept()
    try:
        engine = RealtimeEngine(model_manager, RealtimeSession(meeting_id=meeting_id))
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"event": "error", "code": "MODEL_NOT_READY", "message": str(exc)})
        await websocket.close(code=1011)
        return
    session = engine.session
    completed = False
    await websocket.send_json({
        "event": "ready",
        "sample_rate": settings.SAMPLE_RATE,
        "encoding": "pcm_s16le",
        "protocol": "meeting-v2",
        "mode": session.mode,
        "meeting_id": meeting_id,
    })

    def finish_meeting() -> dict:
        raw = b"".join(session.recorded_chunks)
        key = f"meetings/{meeting_id}/original/{uuid.uuid4().hex}.pcm"
        if raw:
            minio_service.upload_bytes(key, raw, "audio/pcm")
        db = SessionLocal()
        try:
            meeting = db.get(Meeting, meeting_id)
            if not meeting:
                meeting = Meeting(id=meeting_id, mode="realtime", status="queued", audio_object_key=key)
                db.add(meeting)
            else:
                meeting.audio_object_key, meeting.status = key, "queued"
            task = AnalysisTask(meeting_id=meeting_id, status="queued")
            db.add(task)
            for event in session.realtime_events:
                db.add(RealtimeTranscriptEvent(
                    meeting_id=meeting_id,
                    sequence=int(event.get("sequence", 0)),
                    event_type=str(event.get("event", "partial")),
                    text=event.get("text"),
                    payload=event,
                ))
            db.commit()
            try:
                async_result = process_offline.delay(task.id)
                task.celery_task_id, meeting.task_id = async_result.id, task.id
                db.commit()
            except Exception as exc:  # noqa: BLE001
                task.status, task.error_message = "failed", str(exc)
                db.commit()
                async_result = None
            return {"meeting_id": meeting_id, "task_id": task.id if async_result else None, "audio_object_key": key}
        finally:
            db.close()

    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                data = message["bytes"]
                if not data:
                    continue
                if len(data) % settings.SAMPLE_WIDTH:
                    await websocket.send_json({"event": "error", "code": "INVALID_PCM", "message": "音频帧必须是 16-bit PCM"})
                    continue
                if len(data) > 2 * settings.SAMPLE_RATE * settings.SAMPLE_WIDTH:
                    await websocket.send_json({"event": "error", "code": "FRAME_TOO_LARGE", "message": "音频帧过大"})
                    continue
                if session.total_duration_seconds + len(data) / settings.SAMPLE_WIDTH / settings.SAMPLE_RATE > settings.MAX_REALTIME_MEETING_SECONDS:
                    await websocket.send_json({"event": "error", "code": "MEETING_TOO_LONG", "message": "已达到在线会议最大时长"})
                    break
                if session.buffer_size_bytes + len(data) > settings.STREAM_BUFFER_MAX_MB * 1024 * 1024:
                    await websocket.send_json({"event": "error", "code": "BUFFER_OVERFLOW", "message": "在线音频缓冲区已满"})
                    break
                try:
                    events = await run_in_threadpool(engine.process_pcm, data)
                except Exception as exc:  # noqa: BLE001
                    await websocket.send_json({"event": "error", "code": "INFERENCE_ERROR", "message": str(exc)})
                    continue
                for event in events:
                    session.realtime_events.append(event)
                    await websocket.send_json(event)
                continue
            if message.get("text") is None:
                continue
            try:
                command = json.loads(message["text"])
            except json.JSONDecodeError:
                await websocket.send_json({"event": "error", "code": "INVALID_COMMAND", "message": "无效的 JSON 控制消息"})
                continue
            # meeting2 sends a FunASR initialization object; the old page sends
            # {type: ...}. Both are accepted on the same endpoint.
            if any(key in command for key in ("chunk_size", "mode", "audio_fs", "is_speaking", "hotwords")):
                try:
                    session.update_from_command(command)
                    await websocket.send_json({"event": "configured", "mode": session.mode, "sample_rate": settings.SAMPLE_RATE})
                except (TypeError, ValueError) as exc:
                    await websocket.send_json({"event": "error", "code": "INVALID_CONFIG", "message": str(exc)})
            if command.get("type") == "ping" or command.get("action") == "ping":
                await websocket.send_json({"event": "pong"})
            elif command.get("type") == "stop":
                final_events = await run_in_threadpool(engine.finalize)
                for event in final_events:
                    session.realtime_events.append(event)
                    await websocket.send_json(event)
                result = await run_in_threadpool(finish_meeting)
                await websocket.send_json({"event": "offline_processing", **result})
                await websocket.send_json({"event": "closed"})
                completed = True
                await websocket.close()
                return
    except (WebSocketDisconnect, json.JSONDecodeError):
        # A browser refresh or network drop should not discard an already
        # recorded meeting. Persist it without attempting to send more frames.
        if not completed and session.recorded_chunks:
            try:
                final_events = await run_in_threadpool(engine.finalize)
                session.realtime_events.extend(final_events)
                await run_in_threadpool(finish_meeting)
            except Exception:
                pass


@router.websocket("/ws")
async def legacy_websocket(websocket: WebSocket):
    await realtime_stream(websocket, str(uuid.uuid4()))


@router.post("/meetings/offline", status_code=202)
async def create_offline_meeting(file: UploadFile = File(...), title: str = Form(""), db: Session = Depends(get_db)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "会议音频为空")
    meeting_id, task_id = str(uuid.uuid4()), str(uuid.uuid4())
    key = f"meetings/{meeting_id}/original/{uuid.uuid4().hex}"
    minio_service.upload_bytes(key, raw, file.content_type or "application/octet-stream")
    meeting = Meeting(id=meeting_id, title=title or file.filename, mode="offline", status="queued", audio_object_key=key, task_id=task_id)
    task = AnalysisTask(id=task_id, meeting_id=meeting_id, status="queued")
    db.add_all([meeting, task])
    db.commit()
    async_result = process_offline.delay(task_id)
    task.celery_task_id = async_result.id
    db.commit()
    return {"meeting_id": meeting_id, "task_id": task_id, "celery_task_id": async_result.id}


@router.post("/upload")
async def legacy_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "会议音频为空")
    meeting_id = str(uuid.uuid4())
    key = f"meetings/{meeting_id}/original/{uuid.uuid4().hex}"
    minio_service.upload_bytes(key, raw, file.content_type or "application/octet-stream")
    meeting = Meeting(id=meeting_id, title=file.filename, mode="offline", status="processing", audio_object_key=key)
    db.add(meeting)
    db.commit()
    try:
        speaker_rows = [{"id": row.id, "display_name": row.display_name, "department": row.department} for row in db.query(Speaker).filter(Speaker.status != "deleted").all()]
        with tempfile.TemporaryDirectory(prefix="meeting-upload-") as temp:
            source, normalized = Path(temp) / "input", Path(temp) / "audio.wav"
            source.write_bytes(raw)
            segments = await run_in_threadpool(run_offline_pipeline, source, normalized, model_manager)
            duration_ms = wav_duration_ms(normalized)
            speaker_matches = await run_in_threadpool(identify_meeting_speakers, normalized, segments, voice_library_service, speaker_rows)
            segments = enrich_segments(segments, speaker_matches)
        for index, item in enumerate(segments):
            db.add(TranscriptSegment(meeting_id=meeting_id, sequence=index, start_ms=item.get("start_ms", 0), end_ms=item.get("end_ms", 0), text=item.get("text", ""), confidence=item.get("confidence"), speaker_id=(speaker_matches.get(str(item.get("speaker", 0))) or {}).get("speaker_id"), speaker_name_snapshot=item.get("speaker_name"), speaker_department_snapshot=item.get("speaker_department"), source="offline", version=1))
        meeting.status = "completed"
        db.commit()
        return {"success": True, "meeting_id": meeting_id, "filename": file.filename, "duration_ms": duration_ms, "text": "".join(item.get("text", "") for item in segments), "sentences": segments, "speakers": speaker_matches, "voice_match_config": {"similarity_threshold": settings.VOICEPRINT_MATCH_THRESHOLD, "min_margin": settings.VOICEPRINT_MIN_MARGIN}}
    except Exception as exc:  # noqa: BLE001
        meeting.status = "failed"
        db.commit()
        return {"success": False, "meeting_id": meeting_id, "error": str(exc)}


@router.get("/tasks/{task_id}")
def task_status(task_id: str, db: Session = Depends(get_db)):
    task = db.get(AnalysisTask, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return {"id": task.id, "meeting_id": task.meeting_id, "status": task.status, "progress": task.progress, "error": task.error_message}


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str, db: Session = Depends(get_db)):
    task = db.get(AnalysisTask, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    task.status = "cancelled"
    db.commit()
    if task.celery_task_id:
        celery_app.control.revoke(task.celery_task_id, terminate=False)
    return {"cancelled": True}


@router.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str, db: Session = Depends(get_db)):
    meeting = db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(404, "会议不存在")
    return {"id": meeting.id, "title": meeting.title, "mode": meeting.mode, "status": meeting.status, "audio_object_key": meeting.audio_object_key, "task_id": meeting.task_id}


@router.get("/meetings/{meeting_id}/transcript")
def get_transcript(meeting_id: str, db: Session = Depends(get_db)):
    rows = db.scalars(select(TranscriptSegment).where(TranscriptSegment.meeting_id == meeting_id).order_by(TranscriptSegment.sequence)).all()
    return {"segments": [{"sequence": x.sequence, "start_ms": x.start_ms, "end_ms": x.end_ms, "text": x.text, "speaker_id": x.speaker_id, "speaker_name": x.speaker_name_snapshot, "department": x.speaker_department_snapshot, "source": x.source, "version": x.version} for x in rows]}


@router.get("/meetings/{meeting_id}/realtime-events")
def get_realtime_events(meeting_id: str, db: Session = Depends(get_db)):
    rows = db.scalars(select(RealtimeTranscriptEvent).where(RealtimeTranscriptEvent.meeting_id == meeting_id).order_by(RealtimeTranscriptEvent.sequence, RealtimeTranscriptEvent.created_at)).all()
    records = []
    for row in rows:
        payload = row.payload or {}
        if row.event_type != "final":
            continue
        speaker = payload.get("speaker") or {}
        records.append({
            "sequence": row.sequence,
            "utterance_id": payload.get("utterance_id") or f"event-{row.id}",
            "start_ms": payload.get("start_ms", 0),
            "end_ms": payload.get("end_ms", 0),
            "text": row.text or "",
            "speaker_id": speaker.get("id"),
            "speaker_name": speaker.get("name"),
            "department": speaker.get("department"),
            "speaker_label": speaker.get("label") or speaker.get("name") or "说话人待匹配",
            "speaker_status": speaker.get("status", "pending"),
            "speaker_score": speaker.get("score", 0.0),
            "speaker_second_score": speaker.get("second_score", 0.0),
            "speaker_margin": speaker.get("margin", 0.0),
            "source": "streaming",
        })
    return {"meeting_id": meeting_id, "records": records, "committed_text": "".join(item["text"] for item in records)}
