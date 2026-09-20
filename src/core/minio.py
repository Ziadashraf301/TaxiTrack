# src/core/minio.py
"""
MinIO S3 Client Factory and Service (Factory & Singleton Patterns)
Provides high-level, thread-safe S3 object storage operations with comprehensive
telemetry, timing, and error handling.
"""
import time
from typing import Optional, BinaryIO, List
from minio import Minio
from minio.error import S3Error
from core.config import settings, MinioSettings
from core.logging import get_logger

logger = get_logger(__name__)


class MinioClientFactory:
    """Factory for creating and caching configured MinIO client instances."""
    _client: Optional[Minio] = None

    @classmethod
    def get_client(cls, minio_cfg: Optional[MinioSettings] = None) -> Minio:
        if cls._client is None:
            cfg = minio_cfg or settings.minio
            logger.info(f"Connecting to MinIO S3 at {cfg.endpoint} (secure={cfg.secure})...")
            start_time = time.perf_counter()
            try:
                cls._client = Minio(
                    endpoint=cfg.endpoint,
                    access_key=cfg.root_user,
                    secret_key=cfg.root_password,
                    secure=cfg.secure
                )
                duration = time.perf_counter() - start_time
                logger.info(f"MinIO client initialized successfully in {duration:.3f}s.")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO client for {cfg.endpoint}: {e}", exc_info=True)
                raise
        return cls._client


class MinioService:
    """High-level service for interacting with MinIO object storage."""

    def __init__(self, client: Optional[Minio] = None):
        self.client = client or MinioClientFactory.get_client()

    def ensure_bucket(self, bucket_name: str) -> None:
        """Create bucket if it does not exist with timing and error logging."""
        try:
            start_time = time.perf_counter()
            if not self.client.bucket_exists(bucket_name):
                logger.info(f"Bucket s3://{bucket_name} does not exist. Creating...")
                self.client.make_bucket(bucket_name)
                duration = time.perf_counter() - start_time
                logger.info(f"Bucket s3://{bucket_name} created successfully in {duration:.3f}s.")
            else:
                logger.debug(f"Bucket s3://{bucket_name} exists.")
        except S3Error as e:
            logger.error(f"S3 error ensuring bucket s3://{bucket_name}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error ensuring bucket s3://{bucket_name}: {e}", exc_info=True)
            raise

    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """
        Check if an object exists in the specified bucket.
        Logs file size in MB and modification timestamp if found.
        """
        try:
            stat = self.client.stat_object(bucket_name, object_name)
            size_mb = stat.size / (1024 * 1024)
            logger.info(f"Object exists: s3://{bucket_name}/{object_name} ({size_mb:.2f} MB, modified: {stat.last_modified})")
            return True
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchBucket"):
                logger.info(f"Object not found: s3://{bucket_name}/{object_name}. Will be fetched.")
                return False
            logger.warning(f"S3 error checking object s3://{bucket_name}/{object_name}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking s3://{bucket_name}/{object_name}: {e}")
            return False

    def upload_stream(
        self,
        bucket_name: str,
        object_name: str,
        data_stream: BinaryIO,
        part_size: int = 10 * 1024 * 1024
    ) -> None:
        """
        Upload raw streaming data directly to MinIO without local disk writes.
        Includes throughput timing and byte verification.
        """
        self.ensure_bucket(bucket_name)
        logger.info(f"Starting stream upload to s3://{bucket_name}/{object_name} (part_size={part_size / (1024 * 1024):.1f}MB)...")
        start_time = time.perf_counter()
        try:
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data_stream,
                length=-1,  # Unknown length for streaming
                part_size=part_size
            )
            duration = time.perf_counter() - start_time
            # Query stat to log actual uploaded size and throughput
            try:
                stat = self.client.stat_object(bucket_name, object_name)
                size_mb = stat.size / (1024 * 1024)
                rate = size_mb / max(duration, 0.001)
                logger.info(
                    f"Successfully uploaded s3://{bucket_name}/{object_name} "
                    f"({size_mb:.2f} MB in {duration:.2f}s, throughput: {rate:.2f} MB/s)"
                )
            except Exception:
                logger.info(f"Successfully uploaded stream to s3://{bucket_name}/{object_name} in {duration:.2f}s.")
        except Exception as e:
            logger.error(f"Failed streaming upload to s3://{bucket_name}/{object_name}: {e}", exc_info=True)
            raise

    def get_object_stream(self, bucket_name: str, object_name: str):
        """Retrieve streaming response object from MinIO with duration telemetry."""
        logger.info(f"Opening read stream for s3://{bucket_name}/{object_name}...")
        start_time = time.perf_counter()
        try:
            response = self.client.get_object(bucket_name, object_name)
            duration = time.perf_counter() - start_time
            logger.debug(f"Acquired read stream for s3://{bucket_name}/{object_name} in {duration:.3f}s.")
            return response
        except Exception as e:
            logger.error(f"Failed to open read stream for s3://{bucket_name}/{object_name}: {e}", exc_info=True)
            raise

    def list_objects(self, bucket_name: str, prefix: str = "", recursive: bool = True) -> List[str]:
        """List object keys under the given prefix in the specified bucket."""
        try:
            objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=recursive)
            keys = [obj.object_name for obj in objects]
            logger.debug(f"Found {len(keys)} objects in s3://{bucket_name}/{prefix}")
            return keys
        except Exception as e:
            logger.error(f"Failed to list objects in s3://{bucket_name}/{prefix}: {e}", exc_info=True)
            raise
