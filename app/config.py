"""
全局配置。
其他模块统一 from app.config import settings 使用，
避免各处散落 os.getenv 调用。
"""
import os
from pathlib import Path

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "meeting"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    SECRET_KEY: str = "change_me"

    # --- SSL ---
    APP_SSL_ENABLED: bool = True
    APP_SSL_KEYFILE: str = "ssl_key/server.key"
    APP_SSL_CERTFILE: str = "ssl_key/server.crt"

    # --- Hardware (硬件设备，通过 .env 配置，如 cuda:0, cpu, mps) ---
    DEVICE: str = "cuda:0"
    
    # --- Redis ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # --- MySQL ---
    MYSQL_HOST: str = "192.168.202.61"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "daixw"
    MYSQL_PASSWORD: str = "T8m#qL!9@vZ$"
    MYSQL_DATABASE: str = "meeting"
    MYSQL_POOL_SIZE: int = 10
    MYSQL_MAX_OVERFLOW: int = 20
    MYSQL_POOL_RECYCLE: int = 1800

    # --- MinIO ---
    MINIO_ENDPOINT: str = "192.168.202.24:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "Sym@2025@#__"
    MINIO_BUCKET: str = "meeting"
    MINIO_SECURE: bool = False
    MINIO_VOICEPRINT_PREFIX: str = "voiceprints"
    MINIO_AUDIO_PREFIX: str = "meetings"

    # --- 声纹库 ---
    VOICELIB_DIR: str = "./voicelibrary"

    # --- 音频处理参数 ---
    # 统一使用 16 kHz、单声道、16-bit PCM，供声纹、实时和离线管线共享。
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2
    STREAM_CHUNK_MS: int = 600
    # 实时 FunASR 协议参数。前端最终发送 16kHz PCM16，每 60ms 一帧。
    STREAM_CHUNK_SIZE: list[int] = [5, 10, 5]
    STREAM_CHUNK_INTERVAL: int = 10
    STREAM_ENCODER_LOOK_BACK: int = 4
    STREAM_DECODER_LOOK_BACK: int = 1
    STREAM_FORCE_SEND_SECONDS: float = 5.0
    STREAM_BUFFER_MAX_MB: int = 100
    STREAM_MIN_AUDIO_ENERGY: float = 50.0
    VAD_SPEECH_THRESHOLD: float = 0.3
    VAD_SILENCE_THRESHOLD: float = 0.1
    REALTIME_IDLE_TIMEOUT_SECONDS: int = 30
    MAX_REALTIME_MEETING_SECONDS: int = 4 * 60 * 60

    # --- 模型缓存目录 ---
    MODELS_CACHE_DIR: str = "./models_cache"

    # 声纹匹配初始阈值。
    # 注意：必须根据实际会议环境调试。
    # VOICE_SIMILARITY_THRESHOLD = 0.68
    # 声纹库和说话人匹配度
    VOICEPRINT_MATCH_THRESHOLD: float = 0.38
    # 第一名和第二名相似度至少相差多少，才认为结果较稳定。
    VOICEPRINT_MIN_MARGIN: float = 0.05
    # FSMN-VAD 单个语音片段的最大时长（毫秒），与旧版 Demo 保持一致。
    VAD_MAX_SINGLE_SEGMENT_TIME_MS: int = 60000

    # 每个匿名说话人参与声纹识别的最多会议片段数，以及最短片段时长。
    MAX_SEGMENTS_PER_SPEAKER: int = 8
    # 会议片段短于该值时，不用于声纹识别
    MIN_MEETING_SEGMENT_MS: int = 1000
    # 声纹录音上传最大 30MB
    MAX_VOICE_FILE_SIZE: int = 30 * 1024 * 1024
    # 会议录音上传最大 500MB
    MAX_MEETING_FILE_SIZE: int = 500 * 1024 * 1024
    # 单条注册录音建议至少 2 秒
    MIN_ENROLL_DURATION_MS: int = 2000
    # 单条注册录音最大 120 秒
    MAX_ENROLL_DURATION_MS: int = 120000

    
    ADMIN_AUTH_ENABLED: bool = False
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    

    # ------------------------------------------------------------
    # 组装好的连接串，其他模块直接用，不用自己拼字符串
    # ------------------------------------------------------------
    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def MYSQL_DSN(self) -> str:
        """
        注意：pymysql 是同步驱动。
        如果你用 SQLAlchemy 异步 (AsyncSession)，请将 pymysql 改为 asyncmy，
        并安装 pip install asyncmy
        """
        # 修复了原来 POSTGRES_ 前缀的 Bug，统一使用 MYSQL_
        from urllib.parse import quote_plus
        pwd = quote_plus(self.MYSQL_PASSWORD) # 防止密码中的特殊字符(#@等)导致连接串解析失败
        
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{pwd}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    @property
    def CELERY_BROKER(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def CELERY_BACKEND(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    def setup_model_cache(self):
        """初始化模型缓存环境变量"""
        # ------------------------------------------------------------
        # 统一模型缓存目录：必须在 funasr/modelscope 被 import 之前设置好，
        # 否则它们已经用默认路径初始化过一次，这里设置就晚了。
        # 所以要保证 app.config 是整个项目里"最早被 import"的模块。
        # ------------------------------------------------------------
        _models_cache_path = Path(self.MODELS_CACHE_DIR).resolve()
        _models_cache_path.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("MODELSCOPE_CACHE", str(_models_cache_path))
        os.environ.setdefault("HF_HOME", str(_models_cache_path / "huggingface"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_models_cache_path / "huggingface"))





@lru_cache
def get_settings() -> Settings:
    return Settings()

# 全局单例
settings = get_settings()

# 项目启动时立即执行环境设置
settings.setup_model_cache()





