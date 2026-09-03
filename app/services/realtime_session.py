from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class RealtimeSession:
    """State owned by one WebSocket connection.

    Model instances are shared by the process, but all FunASR caches and audio
    buffers are deliberately kept here so concurrent meetings cannot mix state.
    """

    meeting_id: str
    wav_name: str = "microphone"
    mode: str = "2pass"
    sample_rate: int = settings.SAMPLE_RATE
    chunk_size: list[int] = field(default_factory=lambda: list(settings.STREAM_CHUNK_SIZE))
    chunk_interval: int = settings.STREAM_CHUNK_INTERVAL
    encoder_chunk_look_back: int = settings.STREAM_ENCODER_LOOK_BACK
    decoder_chunk_look_back: int = settings.STREAM_DECODER_LOOK_BACK
    hotwords: list[str] = field(default_factory=list)
    is_speaking: bool = True
    is_file_upload: bool = False
    asr_cache: dict = field(default_factory=dict)
    vad_cache: dict = field(default_factory=dict)
    punc_cache: dict = field(default_factory=dict)
    recorded_chunks: list[bytes] = field(default_factory=list)
    online_chunks: list[bytes] = field(default_factory=list)
    realtime_events: list[dict] = field(default_factory=list)
    sequence: int = 0
    frame_count: int = 0
    total_samples: int = 0
    total_bytes: int = 0
    accumulated_samples: int = 0
    speech_active: bool = False
    pending_final: bool = False
    last_recognized_text: str = ""
    last_emitted_final: bool = False
    stopped: bool = False
    utterance_id: str = field(default_factory=lambda: f"u-{uuid.uuid4().hex[:12]}")
    utterance_chunks: list[bytes] = field(default_factory=list)
    utterance_start_ms: int | None = None
    last_audio_time: float = field(default_factory=time.monotonic)
    last_recognition_time: float = field(default_factory=time.monotonic)

    @property
    def total_duration_seconds(self) -> float:
        return self.total_samples / float(self.sample_rate)

    @property
    def buffer_size_bytes(self) -> int:
        return self.total_bytes

    def update_from_command(self, command: dict) -> None:
        if "wav_name" in command:
            self.wav_name = str(command["wav_name"] or "microphone")[:128]
        if "mode" in command and str(command["mode"]) in {"online", "2pass", "offline"}:
            self.mode = str(command["mode"])
        if "is_speaking" in command:
            new_speaking = bool(command["is_speaking"])
            if self.is_speaking and not new_speaking:
                self.pending_final = True
            self.is_speaking = new_speaking
        if "is_file_upload" in command:
            self.is_file_upload = bool(command["is_file_upload"])
        if "chunk_interval" in command:
            self.chunk_interval = max(1, min(100, int(command["chunk_interval"])))
        if "chunk_size" in command:
            value = command["chunk_size"]
            if isinstance(value, str):
                value = value.split(",")
            if isinstance(value, (list, tuple)) and len(value) == 3:
                self.chunk_size = [max(1, int(item)) for item in value]
        if "encoder_chunk_look_back" in command:
            self.encoder_chunk_look_back = max(0, int(command["encoder_chunk_look_back"]))
        if "decoder_chunk_look_back" in command:
            self.decoder_chunk_look_back = max(0, int(command["decoder_chunk_look_back"]))
        if "hotwords" in command:
            words = command["hotwords"]
            if isinstance(words, str):
                words = [item.strip() for item in words.split(",") if item.strip()]
            self.hotwords = [str(item)[:64] for item in (words or [])[:100]]
        if "audio_fs" in command and int(command["audio_fs"]) != settings.SAMPLE_RATE:
            raise ValueError(f"实时音频必须是 {settings.SAMPLE_RATE}Hz PCM")

    def append_audio(self, pcm: bytes) -> None:
        if self.utterance_start_ms is None:
            self.utterance_start_ms = round(self.total_duration_seconds * 1000)
        self.recorded_chunks.append(pcm)
        self.online_chunks.append(pcm)
        self.utterance_chunks.append(pcm)
        self.total_samples += len(pcm) // settings.SAMPLE_WIDTH
        self.total_bytes += len(pcm)
        self.accumulated_samples += len(pcm) // settings.SAMPLE_WIDTH
        self.frame_count += 1
        self.last_audio_time = time.monotonic()

    def take_online_audio(self) -> bytes:
        data = b"".join(self.online_chunks)
        self.online_chunks.clear()
        self.accumulated_samples = 0
        self.last_recognition_time = time.monotonic()
        return data

    def finish_utterance(self) -> tuple[str, bytes, int, int]:
        start_ms = self.utterance_start_ms if self.utterance_start_ms is not None else 0
        end_ms = round(self.total_duration_seconds * 1000)
        data = b"".join(self.utterance_chunks)
        utterance_id = self.utterance_id
        self.utterance_chunks.clear()
        self.utterance_start_ms = None
        self.utterance_id = f"u-{uuid.uuid4().hex[:12]}"
        return utterance_id, data, start_ms, end_ms
