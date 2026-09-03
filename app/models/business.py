from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import BIGINT, DATETIME
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def uuid_str() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Speaker(TimestampMixin, Base):
    __tablename__ = "speakers"
    __table_args__ = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    department: Mapped[str | None] = mapped_column(String(128))
    load_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False, index=True)
    embedding_model_version: Mapped[str | None] = mapped_column(String(128))
    samples: Mapped[list["VoiceprintSample"]] = relationship(back_populates="speaker", cascade="all, delete-orphan")


class VoiceprintSample(TimestampMixin, Base):
    __tablename__ = "voiceprint_samples"
    __table_args__ = (
        Index("ix_voiceprint_samples_speaker", "speaker_id"),
        Index("ix_voiceprint_samples_sha256", "sha256", unique=True),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    speaker_id: Mapped[str] = mapped_column(String(36), ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False)
    object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(BIGINT)
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    source_type: Mapped[str] = mapped_column(String(32), default="manual_upload", nullable=False)
    source_meeting_id: Mapped[str | None] = mapped_column(String(36))
    source_segment_id: Mapped[str | None] = mapped_column(String(36))
    review_status: Mapped[str] = mapped_column(String(32), default="approved", nullable=False, index=True)
    embedding_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    embedding_model_version: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    speaker: Mapped[Speaker] = relationship(back_populates="samples")


class Meeting(TimestampMixin, Base):
    __tablename__ = "meetings"
    __table_args__ = (Index("ix_meetings_status_created", "status", "created_at"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    title: Mapped[str | None] = mapped_column(String(255))
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created", nullable=False)
    audio_object_key: Mapped[str | None] = mapped_column(String(512))
    task_id: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6))
    ended_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6))


class TranscriptSegment(TimestampMixin, Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint("meeting_id", "sequence", "version", name="uq_transcript_meeting_sequence_version"),
        Index("ix_transcript_meeting_start", "meeting_id", "start_ms"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    meeting_id: Mapped[str] = mapped_column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(BIGINT, nullable=False)
    start_ms: Mapped[int] = mapped_column(BIGINT, default=0, nullable=False)
    end_ms: Mapped[int] = mapped_column(BIGINT, default=0, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    speaker_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("speakers.id", ondelete="SET NULL"))
    speaker_name_snapshot: Mapped[str | None] = mapped_column(String(128))
    speaker_department_snapshot: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(32), default="offline", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class RealtimeTranscriptEvent(TimestampMixin, Base):
    __tablename__ = "realtime_transcript_events"
    __table_args__ = (Index("ix_realtime_meeting_sequence", "meeting_id", "sequence"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    meeting_id: Mapped[str] = mapped_column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(BIGINT, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JSON)


class AnalysisTask(TimestampMixin, Base):
    __tablename__ = "analysis_tasks"
    __table_args__ = (Index("ix_analysis_status_created", "status", "created_at"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    meeting_id: Mapped[str] = mapped_column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON)
