from __future__ import annotations

import subprocess
import tempfile
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.config import settings
from app.core.ffmpeg import find_ffmpeg
from app.services.voiceprint_service import normalize_vector


def normalize_with_ffmpeg(source: Path, target: Path) -> None:
    ffmpeg = find_ffmpeg() or "ffmpeg"
    completed = subprocess.run([ffmpeg, "-y", "-i", str(source), "-vn", "-ac", str(settings.CHANNELS), "-ar", str(settings.SAMPLE_RATE), "-c:a", "pcm_s16le", str(target)], capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="ignore")[-1000:])


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        if rate <= 0:
            raise ValueError("WAV 采样率无效")
        return round(reader.getnframes() * 1000 / rate)


def run_offline_pipeline(source: Path, normalized: Path, manager) -> list[dict]:
    normalize_with_ffmpeg(source, normalized)
    model = manager.require_loaded("paraformer-zh")
    result = model.generate(input=str(normalized), batch_size_s=300)
    item = result[0] if isinstance(result, list) and result else result
    if not isinstance(item, dict):
        return []
    rows = item.get("sentence_info") or []
    return [{"start_ms": int(row.get("start", 0)), "end_ms": int(row.get("end", 0)), "text": row.get("text", row.get("sentence", "")), "confidence": row.get("confidence"), "speaker": _speaker_id(row)} for row in rows]


def _speaker_id(row: dict) -> int:
    try:
        return int(row.get("spk", 0))
    except (TypeError, ValueError):
        return 0


def _write_wav_segment(source: Path, target: Path, start_ms: int, end_ms: int) -> None:
    with wave.open(str(source), "rb") as reader:
        rate, channels, width, total = reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getnframes()
        start = max(0, round(start_ms * rate / 1000))
        end = min(total, round(end_ms * rate / 1000))
        if end <= start:
            raise ValueError("音频片段为空")
        reader.setpos(start)
        frames = reader.readframes(end - start)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)


def identify_meeting_speakers(normalized_wav: Path, segments: list[dict], voice_service, speaker_rows) -> dict[str, dict]:
    """Match FunASR anonymous speaker clusters against the loaded CAM++ index."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    all_ids: set[int] = set()
    for row in segments:
        sid = _speaker_id(row)
        all_ids.add(sid)
        duration = int(row.get("end_ms", 0)) - int(row.get("start_ms", 0))
        if duration >= settings.MIN_MEETING_SEGMENT_MS:
            grouped[sid].append({"start_ms": int(row.get("start_ms", 0)), "end_ms": int(row.get("end_ms", 0)), "duration_ms": duration})

    speakers = {speaker["id"]: speaker for speaker in speaker_rows}
    matches: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="meeting-speakers-") as temp_dir:
        temp = Path(temp_dir)
        for sid in sorted(all_ids):
            vectors, weights = [], []
            candidates = sorted(grouped.get(sid, []), key=lambda item: item["duration_ms"], reverse=True)[: settings.MAX_SEGMENTS_PER_SPEAKER]
            for index, item in enumerate(candidates):
                start = item["start_ms"] + 100
                end = item["end_ms"] - 100
                if end - start < 800:
                    continue
                try:
                    path = temp / f"speaker_{sid}_{index}.wav"
                    _write_wav_segment(normalized_wav, path, start, end)
                    vectors.append(voice_service.extract_embedding(path))
                    weights.append(end - start)
                except Exception:
                    continue
            if not vectors:
                matches[str(sid)] = {"speaker": sid, "speaker_id": None, "name": None, "department": None, "label": f"说话人 {sid}", "candidate_id": None, "score": 0.0, "second_score": 0.0, "margin": 0.0, "status": "insufficient_audio", "segment_count": 0}
                continue
            cluster = normalize_vector(np.average(np.stack(vectors), axis=0, weights=np.asarray(weights, dtype=np.float32)))
            result = voice_service.identify(cluster)
            matched_id = result.get("speaker_id")
            candidate_id = result.get("candidate_id") or matched_id
            person = speakers.get(matched_id) if matched_id else None
            candidate = speakers.get(candidate_id) if candidate_id else None
            display = person.get("display_name") if person else None
            department = person.get("department") if person else None
            label = f"{display}（{department}）" if display and department else (display or f"说话人 {sid}")
            matches[str(sid)] = {"speaker": sid, "speaker_id": matched_id, "name": display, "department": department, "label": label, "candidate_id": candidate_id, "candidate_name": candidate.get("display_name") if candidate else None, "score": result.get("score", 0.0), "second_score": result.get("second_score", 0.0), "margin": result.get("margin", 0.0), "status": result.get("status", "unknown"), "segment_count": len(vectors)}
    return matches


def enrich_segments(segments: list[dict], matches: dict[str, dict]) -> list[dict]:
    enriched = []
    for row in segments:
        match = matches.get(str(row.get("speaker", 0)), {})
        enriched.append({**row, "speaker_name": match.get("name"), "speaker_department": match.get("department"), "speaker_label": match.get("label", f"说话人 {row.get('speaker', 0)}"), "speaker_status": match.get("status", "not_processed"), "speaker_score": match.get("score", 0.0), "speaker_second_score": match.get("second_score", 0.0), "speaker_margin": match.get("margin", 0.0), "speaker_segment_count": match.get("segment_count", 0)})
    return enriched
