"""MinIO object storage helpers with lazy availability checks."""
import io
import logging
from datetime import timedelta

from minio import Minio

from app.config import settings

logger = logging.getLogger("minio_client")


class MinioService:
    def __init__(self) -> None:
        self.bucket = settings.MINIO_BUCKET
        self.client = Minio(settings.MINIO_ENDPOINT, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY, secure=settings.MINIO_SECURE)

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload_bytes(self, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.ensure_bucket()
        self.client.put_object(self.bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type)
        return object_name

    def upload_file(self, object_name: str, file_path: str, content_type: str = "application/octet-stream") -> str:
        self.ensure_bucket()
        self.client.fput_object(self.bucket, object_name, file_path, content_type=content_type)
        return object_name

    def download_to_file(self, object_name: str, dest_path: str) -> None:
        self.client.fget_object(self.bucket, object_name, dest_path)

    def download_bytes(self, object_name: str) -> bytes:
        response = self.client.get_object(self.bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_presigned_url(self, object_name: str, expires_minutes: int = 60) -> str:
        return self.client.presigned_get_object(self.bucket, object_name, expires=timedelta(minutes=expires_minutes))

    def delete_object(self, object_name: str) -> None:
        self.client.remove_object(self.bucket, object_name)

    def object_exists(self, object_name: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except Exception:  # noqa: BLE001
            return False


minio_service = MinioService()


def check_minio_connection() -> bool:
    try:
        minio_service.ensure_bucket()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("MinIO 连接检查失败: %s", exc)
        return False
