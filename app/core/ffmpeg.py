"""FFmpeg discovery used by startup checks and audio normalization."""

from __future__ import annotations

import logging
import shutil

logger = logging.getLogger("ffmpeg")


def find_ffmpeg() -> str | None:
    """Return the FFmpeg executable resolved from the current process PATH."""
    return shutil.which("ffmpeg")


def check_ffmpeg() -> str | None:
    """Check FFmpeg during application startup without changing startup policy."""
    path = find_ffmpeg()
    if path is None:
        logger.warning(
            "未检测到 FFmpeg。上传 WebM、MP3、M4A 等格式以及离线音频转码可能失败；"
            "请安装 FFmpeg 并确保 ffmpeg 已加入 PATH。"
        )
    else:
        logger.info("检测到 FFmpeg: %s", path)
    return path
