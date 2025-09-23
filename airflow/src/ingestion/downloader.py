import logging
import requests
import time
from minio import Minio
from minio.error import S3Error


def download_file_to_minio(
    url: str,
    bucket_name: str,
    object_name: str,
    minio_client: Minio,
    retries: int = 3,
    wait_time: int = 2
) -> None:
    """
    Download a file from a URL and upload it directly to MinIO.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    bucket_name : str
        The MinIO bucket where the file will be stored.
    object_name : str
        The object name (path) inside the bucket.
    minio_client : Minio
        An initialized Minio client.
    retries : int, optional
        Number of retries in case of failure (default is 3).
    wait_time : int, optional
        Time in seconds to wait between retries (default is 2).
    """

    attempt = 0
    while attempt < retries:
        try:
            logging.info(f"Downloading file from {url} (Attempt {attempt + 1}/{retries})")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Ensure bucket exists
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
                logging.info(f"Created bucket: {bucket_name}")

            # Upload directly from response stream
            minio_client.put_object(
                bucket_name,
                object_name,
                data=response.raw,
                length=-1,  # unknown size, stream mode
                part_size=10 * 1024 * 1024,  # 10MB chunks
            )

            logging.info(f"File successfully saved to MinIO: {bucket_name}/{object_name}")
            return

        except (requests.RequestException, S3Error, Exception) as e:
            logging.error(f"Error processing file {url} → {bucket_name}/{object_name}: {e}")
            attempt += 1
            if attempt < retries:
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
