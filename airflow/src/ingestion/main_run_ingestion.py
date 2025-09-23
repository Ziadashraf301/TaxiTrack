from datetime import datetime
import logging
from types import SimpleNamespace
from ingestion.logger import setup_logging
from ingestion.ingestor import process_file
from minio import Minio  # ✅ import Minio client


def run_ingestion_for_date(file_name: str, ingestion_date_obj: datetime, log_file: str = "ingestion.log") -> None:
    """
    Run ingestion for a specific data file and store it in the database.
    """

    # Setup logging (writes to file + console)
    setup_logging(log_file)

    # Initialize MinIO client
    minio_client = Minio(
        "minio:9000",               # host:port
        access_key="ziadashraf98765",    # from your config
        secret_key="x5x6x7x8",    # from your config
        secure=False                # set True if using HTTPS
    )

    # Define ingestion + MinIO parameters
    params = SimpleNamespace(
        # Database parameters
        user="ziadashraf98765",
        password="x5x6x7x8",
        host="clickhouse",
        port=8123,
        db="data_warehouse",

        # Source dataset
        base_url="https://d37ci6vzurychx.cloudfront.net/trip-data",
        processed_urls=set(),

        # Pass initialized MinIO client ✅
        minio_client=minio_client
    )

    # Run ingestion
    process_file(params, dataset_type=file_name.split('_')[0], file_date=ingestion_date_obj)

    logging.info(f"Ingestion completed for file: {file_name}")
