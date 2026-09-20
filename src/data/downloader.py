# src/data/downloader.py
"""
Resilient Streaming Downloader (Single Responsibility Principle)
Streams Parquet data directly from NYC TLC CloudFront into MinIO S3 without local disk writes.
"""
import time
import requests
from core.minio import MinioService
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class StreamingDownloader:
    """Streams data from remote HTTP endpoints directly into MinIO S3."""

    def __init__(self, minio_service: MinioService, base_url: str = settings.base_url):
        self.minio = minio_service
        self.base_url = base_url

    def download_to_minio(
        self,
        bucket_name: str,
        file_name: str,
        max_retries: int = 3,
        backoff_seconds: int = 2
    ) -> None:
        """Stream remote Parquet file to MinIO with retry mechanism and comprehensive logging."""
        url = f"{self.base_url}/{file_name}"
        attempt = 0
        total_start_time = time.perf_counter()

        while attempt < max_retries:
            attempt += 1
            attempt_start = time.perf_counter()
            try:
                logger.info(f"Connecting to remote endpoint: {url} (Attempt {attempt}/{max_retries})...")
                response = requests.get(url, stream=True, timeout=90)
                response.raise_for_status()

                content_len = response.headers.get("Content-Length")
                size_str = f"{int(content_len) / (1024 * 1024):.2f} MB" if content_len else "chunked/unknown"
                logger.info(f"CloudFront connection established: HTTP {response.status_code}, Remote Size: {size_str}")

                # Upload directly from raw response stream to MinIO
                self.minio.upload_stream(
                    bucket_name=bucket_name,
                    object_name=file_name,
                    data_stream=response.raw
                )

                duration = time.perf_counter() - attempt_start
                logger.info(f"Successfully streamed {url} ➔ s3://{bucket_name}/{file_name} in {duration:.2f}s")
                return

            except (requests.RequestException, Exception) as e:
                attempt_duration = time.perf_counter() - attempt_start
                if attempt < max_retries:
                    sleep_time = backoff_seconds * attempt
                    logger.warning(
                        f"Download failed after {attempt_duration:.2f}s (Attempt {attempt}/{max_retries}) for {url}: {e}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    total_duration = time.perf_counter() - total_start_time
                    logger.error(
                        f"Exhausted all {max_retries} download attempts for {url} after {total_duration:.2f}s. "
                        f"Final error: {e}",
                        exc_info=True
                    )
                    raise
