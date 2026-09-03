from __future__ import annotations

import time
import io
import wave
from typing import Any

import numpy as np

from app.config import settings
from app.core.funasr_loader import ModelManager
from app.services.realtime_session import RealtimeSession
from app.services.voiceprint_service import voice_library_service


class RealtimeEngine:
    """FunASR VAD + streaming ASR adapter for one realtime session."""

    def __init__(self, manager: ModelManager, session: RealtimeSession) -> None:
        self.manager = manager
        self.session = session
        self.stream_model = manager.require_loaded("paraformer-zh-streaming")
        self.vad_model = manager.require_loaded("fsmn-vad")

    @staticmethod
    def _first_result(result: Any) -> dict:
        item = result[0] if isinstance(result, list) and result else result
        return item if isinstance(item, dict) else {}

    def _vad(self, pcm: bytes) -> tuple[bool, bool]:
        vad_chunk_size = max(1, int(self.session.chunk_size[1] * 60 / self.session.chunk_interval))
        kwargs = {
            "cache": self.session.vad_cache,
            "is_final": False,
            "chunk_size": vad_chunk_size,
            "speech_threshold": settings.VAD_SPEECH_THRESHOLD,
            "silence_threshold": settings.VAD_SILENCE_THRESHOLD,
        }
        with self.manager.inference_lock:
            try:
                result = self.vad_model.generate(input=pcm, **kwargs)
            except TypeError:
                # Older FunASR releases do not expose threshold keywords.
                kwargs.pop("speech_threshold", None)
                kwargs.pop("silence_threshold", None)
                result = self.vad_model.generate(input=pcm, **kwargs)
        value = self._first_result(result).get("value", [])
        if not value:
            return False, False
        started = ended = False
        for segment in value if isinstance(value, list) else []:
            if not isinstance(segment, (list, tuple)) or len(segment) < 2:
                continue
            started = started or segment[0] != -1
            ended = ended or segment[1] != -1
        return started, ended

    def _recognize(self, pcm: bytes, *, is_final: bool) -> dict | None:
        if not pcm:
            return None
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0 or float(np.mean(np.abs(samples))) < settings.STREAM_MIN_AUDIO_ENERGY:
            return None
        kwargs = {
            "cache": self.session.asr_cache,
            "is_final": is_final,
            "chunk_size": self.session.chunk_size,
            "encoder_chunk_look_back": self.session.encoder_chunk_look_back,
            "decoder_chunk_look_back": self.session.decoder_chunk_look_back,
        }
        if self.session.hotwords:
            kwargs["hotwords"] = self.session.hotwords
        with self.manager.inference_lock:
            try:
                result = self.stream_model.generate(input=pcm, **kwargs)
            except TypeError:
                # Keep compatibility with older streaming wrappers that only
                # accept the core cache/final/chunk arguments.
                for optional in ("hotwords", "decoder_chunk_look_back", "encoder_chunk_look_back"):
                    kwargs.pop(optional, None)
                result = self.stream_model.generate(input=pcm, **kwargs)
        item = self._first_result(result)
        text = str(item.get("text", "") or "").strip()
        if text:
            self.session.last_recognized_text = text
        return {"text": text, "raw": item} if text else None

    def _match_speaker(self, pcm: bytes, duration_ms: int) -> dict:
        """Best-effort low-latency CAM++ match for a completed utterance."""
        pending = {"id": None, "name": None, "department": None, "label": None, "status": "pending", "score": 0.0, "second_score": 0.0, "margin": 0.0}
        if duration_ms < settings.MIN_MEETING_SEGMENT_MS or len(pcm) < settings.SAMPLE_RATE * settings.SAMPLE_WIDTH:
            pending["status"] = "insufficient_audio"
            return pending
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as writer:
                writer.setnchannels(settings.CHANNELS)
                writer.setsampwidth(settings.SAMPLE_WIDTH)
                writer.setframerate(settings.SAMPLE_RATE)
                writer.writeframes(pcm)
            with self.manager.inference_lock:
                embedding = voice_library_service.extract_embedding_bytes(buffer.getvalue())
            result = voice_library_service.identify(embedding)
            speaker_id = result.get("speaker_id") or result.get("candidate_id")
            metadata = voice_library_service.speaker_info(speaker_id) if speaker_id else None
            matched = result.get("speaker_id") is not None
            pending.update({
                "id": speaker_id,
                "name": metadata.get("display_name") if metadata else None,
                "department": metadata.get("department") if metadata else None,
                "label": metadata.get("label") if metadata else None,
                "status": "matched" if matched else result.get("status", "unknown"),
                "score": result.get("score", 0.0),
                "second_score": result.get("second_score", 0.0),
                "margin": result.get("margin", 0.0),
            })
        except Exception:
            # Real-time matching must never break ASR or the meeting record.
            pending["status"] = "unavailable"
        return pending

    def process_pcm(self, pcm: bytes) -> list[dict]:
        if len(pcm) % settings.SAMPLE_WIDTH:
            raise ValueError("PCM 音频帧必须按 16-bit 样本对齐")
        self.session.append_audio(pcm)
        started, ended = self._vad(pcm)
        if self.session.pending_final:
            ended = True
            self.session.pending_final = False
        if started:
            self.session.speech_active = True
        if ended:
            self.session.speech_active = False
        elapsed = time.monotonic() - self.session.last_recognition_time
        force_send = self.session.accumulated_samples >= int(settings.STREAM_FORCE_SEND_SECONDS * settings.SAMPLE_RATE)
        interval_due = self.session.frame_count % self.session.chunk_interval == 0
        active_audio = float(np.mean(np.abs(np.frombuffer(pcm, dtype=np.int16)))) >= settings.STREAM_MIN_AUDIO_ENERGY
        should_process = bool(self.session.online_chunks) and (
            ended or (interval_due and (self.session.speech_active or started)) or (force_send and active_audio) or (elapsed >= settings.STREAM_FORCE_SEND_SECONDS and active_audio)
        )
        if not should_process:
            return []
        audio = self.session.take_online_audio()
        recognized = self._recognize(audio, is_final=ended)
        if not recognized:
            if ended:
                self.session.finish_utterance()
                self.session.asr_cache = {}
                self.session.last_emitted_final = True
            return []
        self.session.sequence += 1
        event = "final" if ended else "partial"
        self.session.last_emitted_final = ended
        if ended:
            # Keep the meeting alive for the next utterance, but do not carry
            # the previous utterance's streaming context into it.
            self.session.asr_cache = {}
        utterance_id = self.session.utterance_id
        start_ms = self.session.utterance_start_ms if self.session.utterance_start_ms is not None else 0
        end_ms = round(self.session.total_duration_seconds * 1000)
        speaker = {"id": None, "name": None, "department": None, "label": None, "status": "pending", "score": 0.0, "second_score": 0.0, "margin": 0.0}
        if ended:
            _, utterance_audio, start_ms, end_ms = self.session.finish_utterance()
            speaker = self._match_speaker(utterance_audio, end_ms - start_ms)
        return [{
            "event": event,
            "sequence": self.session.sequence,
            "utterance_id": utterance_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": recognized["text"],
            "is_final": ended,
            "source": "streaming",
            "mode": "2pass-online" if self.session.mode == "2pass" else self.session.mode,
            "wav_name": self.session.wav_name,
            "speaker": speaker,
        }]

    def finalize(self) -> list[dict]:
        if self.session.stopped:
            return []
        self.session.stopped = True
        audio = self.session.take_online_audio()
        recognized = self._recognize(audio, is_final=True)
        if not recognized and not self.session.last_emitted_final and self.session.last_recognized_text:
            recognized = {"text": self.session.last_recognized_text, "raw": {}}
        if not recognized:
            return []
        self.session.sequence += 1
        utterance_id, utterance_audio, start_ms, end_ms = self.session.finish_utterance()
        speaker = self._match_speaker(utterance_audio, end_ms - start_ms)
        return [{
            "event": "final",
            "sequence": self.session.sequence,
            "utterance_id": utterance_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": recognized["text"],
            "is_final": True,
            "source": "streaming",
            "mode": "2pass-online",
            "wav_name": self.session.wav_name,
            "speaker": speaker,
        }]
