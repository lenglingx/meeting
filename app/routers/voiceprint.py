from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import require_admin
from app.core.database import get_db
from app.core.minio_client import minio_service
from app.models.business import Speaker, VoiceprintSample
from app.services.voiceprint_service import validate_name, voice_library_service

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/voiceprint/", response_class=HTMLResponse)
def voiceprint_page(request: Request):
    return templates.TemplateResponse(request, "voiceprint.html", {})


@router.get("/api/v1/voiceprints")
def list_voiceprints(db: Session = Depends(get_db)):
    return {"speakers": voice_library_service.list_speakers(db), "model_loaded": voice_library_service.model_loaded, "model_error": voice_library_service.model_error, "index_version": voice_library_service.index_version}


@router.post("/api/v1/voiceprints")
def create_voiceprint(name: str = Form(...), department: str = Form(""), load_enabled: bool = Form(True), db: Session = Depends(get_db), _: None = Depends(require_admin)):
    try:
        speaker = voice_library_service.create_speaker(db, name, department, load_enabled)
        return {"id": speaker.id, "name": speaker.display_name, "department": speaker.department, "load_enabled": speaker.load_enabled}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/api/v1/voiceprints/reload")
def reload_voiceprints(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    return voice_library_service.reload_all(db)


@router.patch("/api/v1/voiceprints/{speaker_id}")
def update_voiceprint(speaker_id: str, name: str | None = Form(None), department: str | None = Form(None), load_enabled: bool | None = Form(None), db: Session = Depends(get_db), _: None = Depends(require_admin)):
    speaker = db.get(Speaker, speaker_id)
    if not speaker or speaker.status == "deleted":
        raise HTTPException(404, "人员不存在")
    if name is not None:
        speaker.display_name = validate_name(name)
    if department is not None:
        speaker.department = department.strip() or None
    if load_enabled is not None:
        speaker.load_enabled = load_enabled
    db.commit()
    return {"id": speaker.id, "name": speaker.display_name, "department": speaker.department, "load_enabled": speaker.load_enabled}


@router.delete("/api/v1/voiceprints/{speaker_id}")
def delete_voiceprint(speaker_id: str, db: Session = Depends(get_db), _: None = Depends(require_admin)):
    speaker = db.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(404, "人员不存在")
    speaker.status, speaker.load_enabled = "deleted", False
    db.commit()
    return {"deleted": True, "id": speaker_id}


@router.post("/api/v1/voiceprints/{speaker_id}/samples")
async def upload_sample(speaker_id: str, file: UploadFile = File(...), source_type: str = Form("manual_upload"), db: Session = Depends(get_db), _: None = Depends(require_admin)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "音频文件为空")
    suffix = Path(file.filename or "").suffix.lower().lstrip(".") or None
    try:
        return voice_library_service.save_sample(db, speaker_id, raw, suffix, source_type)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/api/v1/voiceprints/{speaker_id}/samples/{sample_id}")
def delete_sample(speaker_id: str, sample_id: str, db: Session = Depends(get_db), _: None = Depends(require_admin)):
    try:
        voice_library_service.delete_sample(db, speaker_id, sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": True, "id": sample_id}


@router.get("/api/v1/voiceprints/{speaker_id}/samples/{sample_id}/url")
def sample_url(speaker_id: str, sample_id: str, db: Session = Depends(get_db)):
    sample = db.scalar(select(VoiceprintSample).where(VoiceprintSample.id == sample_id, VoiceprintSample.speaker_id == speaker_id))
    if not sample or sample.review_status == "deleted":
        raise HTTPException(404, "样本不存在")
    return {"url": minio_service.get_presigned_url(sample.object_key)}


@router.post("/api/v1/voiceprint-samples/{sample_id}/approve")
def approve_sample(sample_id: str, db: Session = Depends(get_db), _: None = Depends(require_admin)):
    sample = db.get(VoiceprintSample, sample_id)
    if not sample:
        raise HTTPException(404, "样本不存在")
    sample.review_status, sample.embedding_status = "approved", "pending"
    db.commit()
    return {"id": sample_id, "review_status": sample.review_status}


@router.post("/api/v1/voiceprint-samples/{sample_id}/pending")
def pending_sample(sample_id: str, db: Session = Depends(get_db), _: None = Depends(require_admin)):
    sample = db.get(VoiceprintSample, sample_id)
    if not sample or sample.review_status == "deleted":
        raise HTTPException(404, "样本不存在")
    sample.review_status = "pending"
    sample.embedding_status = "pending"
    db.commit()
    return {"id": sample_id, "review_status": sample.review_status}


@router.post("/api/v1/voiceprint-samples/{sample_id}/reject")
def reject_sample(sample_id: str, db: Session = Depends(get_db), _: None = Depends(require_admin)):
    sample = db.get(VoiceprintSample, sample_id)
    if not sample:
        raise HTTPException(404, "样本不存在")
    sample.review_status = "rejected"
    db.commit()
    return {"id": sample_id, "review_status": sample.review_status}


@router.get("/api/voiceprint/list")
def legacy_list(db: Session = Depends(get_db)):
    return list_voiceprints(db)


@router.post("/api/voiceprint/reload")
def legacy_reload(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    return reload_voiceprints(db)


@router.post("/api/voiceprint/upload")
async def legacy_upload(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), _: None = Depends(require_admin)):
    speaker = db.scalar(select(Speaker).where(Speaker.display_name == validate_name(name), Speaker.status != "deleted"))
    if not speaker:
        speaker = voice_library_service.create_speaker(db, name)
    raw = await file.read()
    suffix = Path(file.filename or "").suffix.lower().lstrip(".") or None
    return voice_library_service.save_sample(db, speaker.id, raw, suffix)


@router.post("/api/voiceprint/record")
async def legacy_record(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), _: None = Depends(require_admin)):
    return await legacy_upload(name, file, db, None)
