# src/data/pipeline.py
"""
Monthly Ingestion Pipeline (Orchestrator Pattern / Dependency Inversion)
Coordinates the idempotent flow: check MinIO -> download if missing -> check ClickHouse -> insert batch.
"""
import time
from datetime import datetime
from typing import Dict, Any, Optional
from core.clickhouse import ClickHouseService
from core.minio import MinioService
from core.logging import get_logger
from data.factory import IngestorFactory
from data.downloader import StreamingDownloader
from data.loader import ClickHouseBatchLoader

logger = get_logger(__name__)


class MonthlyIngestionPipeline:
    """Orchestrates idempotent monthly parquet ingestion into MinIO and ClickHouse."""

    def __init__(
        self,
        minio_service: Optional[MinioService] = None,
        clickhouse_service: Optional[ClickHouseService] = None
    ):
        self.minio = minio_service or MinioService()
        self.clickhouse = clickhouse_service or ClickHouseService()
        self.downloader = StreamingDownloader(self.minio)
        self.loader = ClickHouseBatchLoader(self.minio, self.clickhouse)

    def execute(self, dataset_type: str, execution_date: datetime) -> Dict[str, Any]:
        """Execute ingestion pipeline for a given dataset and execution month with rich observability."""
        pipeline_start = time.perf_counter()
        month_str = execution_date.strftime("%Y-%m")

        ingestor = IngestorFactory.get_ingestor(
            dataset_type,
            minio_service=self.minio,
            clickhouse_service=self.clickhouse
        )

        file_name = ingestor.format_filename(execution_date)
        bucket_name = ingestor.bucket_name
        batch_table = ingestor.batch_table

        logger.info(
            f"=== [START PIPELINE] Dataset: '{dataset_type.upper()}' | Month: {month_str} | Target: '{batch_table}' ==="
        )

        try:
            # Step 0: Ensure target database and batch tables exist
            self.clickhouse.ensure_batch_tables()

            # Step 1: Verify presence in MinIO S3 Lake (download only if missing)
            logger.info(f"--- [Step 1/3] Verifying S3 Lake: s3://{bucket_name}/{file_name} ---")
            if not self.minio.object_exists(bucket_name, file_name):
                logger.info(f"Object s3://{bucket_name}/{file_name} not found. Initiating remote stream download...")
                self.downloader.download_to_minio(bucket_name, file_name)
            else:
                logger.info(f"Object confirmed in S3 lake: s3://{bucket_name}/{file_name}")

            # Step 2: Verify presence in ClickHouse batch table (idempotent skip if already ingested)
            logger.info(f"--- [Step 2/3] Checking ClickHouse deduplication: table='{batch_table}', file='{file_name}' ---")
            if self.clickhouse.is_file_ingested(batch_table, file_name):
                elapsed = time.perf_counter() - pipeline_start
                logger.info(
                    f"=== [PIPELINE SKIPPED] '{file_name}' already ingested into '{batch_table}'. "
                    f"Zero redundant work performed. (Elapsed: {elapsed:.2f}s) ==="
                )
                return {"status": "skipped", "file_name": file_name, "rows": 0, "duration_seconds": elapsed}

            # Step 3: Stream and ingest into ClickHouse
            logger.info(f"--- [Step 3/3] Streaming Parquet into ClickHouse: '{batch_table}' ---")
            rows_inserted = self.loader.load_parquet_to_clickhouse(
                bucket_name=bucket_name,
                file_name=file_name,
                table_name=batch_table,
                preprocessor=ingestor.preprocess,
                batch_size=500_000
            )

            total_elapsed = time.perf_counter() - pipeline_start
            logger.info(
                f"=== [PIPELINE COMPLETED] {dataset_type.upper()} {month_str} | Ingested {rows_inserted:,} rows "
                f"into '{batch_table}' in {total_elapsed:.2f}s ==="
            )
            return {
                "status": "success",
                "file_name": file_name,
                "rows": rows_inserted,
                "duration_seconds": total_elapsed
            }

        except Exception as e:
            total_elapsed = time.perf_counter() - pipeline_start
            logger.error(
                f"=== [PIPELINE FAILED] Ingestion failed for {dataset_type} {month_str} after {total_elapsed:.2f}s: {e} ===",
                exc_info=True
            )
            raise


def run_monthly_ingestion(dataset_type: str, execution_date: datetime) -> Dict[str, Any]:
    """Top-level functional entrypoint for Airflow PythonOperator."""
    pipeline = MonthlyIngestionPipeline()
    return pipeline.execute(dataset_type=dataset_type, execution_date=execution_date)

