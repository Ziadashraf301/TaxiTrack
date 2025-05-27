import logging
from types import SimpleNamespace
from ingestion.logger import setup_logging  # ✅ updated import
from ingestion.ingestor import ingest_data_for_file  # ✅ updated import

def run_ingestion_for_date(file_name: str):
    setup_logging()

    params = SimpleNamespace()
    params.user = "ingest_user"
    params.password = "ingest_password"
    params.host = "postgres_ingest"
    params.port = 5432
    params.db = "ingest_db"
    params.base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    params.processed_urls = set()

    ingest_data_for_file(params, file_name)
    logging.info(f"Ingestion completed for file: {file_name}")
