# src/data/storage/loader.py
"""
ClickHouse Batch Loader (Single Responsibility Principle)
Loads Parquet from MinIO and inserts into ClickHouse with robust schema alignment.
"""
import time
from io import BytesIO
from typing import Callable, Optional
import pandas as pd
from core.minio import MinioService
from core.clickhouse import ClickHouseService
from core.logging import get_logger

logger = get_logger(__name__)


class ClickHouseBatchLoader:
    """Streams Parquet from MinIO and executes chunked inserts into ClickHouse."""

    def __init__(self, minio_service: MinioService, clickhouse_service: ClickHouseService):
        self.minio = minio_service
        self.clickhouse = clickhouse_service

    def load_parquet_to_clickhouse(
        self,
        bucket_name: str,
        file_name: str,
        table_name: str,
        preprocessor: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        batch_size: int = 500_000
    ) -> int:
        """
        Stream Parquet from MinIO, preprocess, and insert into ClickHouse batch table.
        Ensures column alignment to prevent dimension mismatch errors across different schema years.
        """
        logger.info(f"Initiating batch load: s3://{bucket_name}/{file_name} ➔ ClickHouse '{table_name}'")
        total_start = time.perf_counter()

        # Step 1: Stream bytes from MinIO into memory
        read_start = time.perf_counter()
        response = self.minio.get_object_stream(bucket_name, file_name)
        try:
            raw_bytes = response.read()
            read_duration = time.perf_counter() - read_start
            size_mb = len(raw_bytes) / (1024 * 1024)
            logger.info(f"Streamed {size_mb:.2f} MB from s3://{bucket_name}/{file_name} into memory in {read_duration:.2f}s")

            decode_start = time.perf_counter()
            parquet_buffer = BytesIO(raw_bytes)
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_buffer, use_threads=True)
            df = table.to_pandas()
            del raw_bytes, parquet_buffer, table
            logger.info(
                f"Decoded Parquet (parallel PyArrow): {len(df):,} rows, {len(df.columns)} columns in {time.perf_counter() - decode_start:.2f}s"
            )
        except Exception as e:
            logger.error(f"Failed reading or parsing Parquet from s3://{bucket_name}/{file_name}: {e}", exc_info=True)
            raise
        finally:
            response.close()
            response.release_conn()

        # Step 2: Apply dataset-specific preprocessing (e.g. datetime parsing)
        if preprocessor:
            prep_start = time.perf_counter()
            df = preprocessor(df)
            logger.debug(f"Preprocessing completed in {time.perf_counter() - prep_start:.3f}s")

        # Step 3: Append tracking metadata columns
        df["file_name"] = file_name
        df["ingest_time"] = pd.Timestamp.now()

        # Step 4: Schema introspection: match DataFrame columns with ClickHouse table schema
        target_columns = self.clickhouse.get_table_columns(table_name)
        matched_columns = [col for col in target_columns if col in df.columns]
        missing_in_df = set(target_columns) - set(matched_columns)
        if missing_in_df:
            logger.debug(f"Target table columns omitted (not present in raw data): {sorted(missing_in_df)}")

        logger.info(
            f"Schema alignment for '{file_name}': {len(matched_columns)} of {len(target_columns)} table columns mapped."
        )

        df = df[matched_columns]
        total_rows = len(df)

        if total_rows == 0:
            logger.warning(f"File '{file_name}' contains 0 rows after processing. Nothing to insert.")
            return 0

        # Step 5: Chunked insertion
        insert_start = time.perf_counter()
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]
            records = chunk.values.tolist()
            pct = (end_idx / total_rows) * 100
            logger.info(f"[{file_name}] Inserting batch rows {start_idx:,} to {end_idx:,} of {total_rows:,} ({pct:.1f}%)...")
            self.clickhouse.insert_records(table_name, records, column_names=matched_columns)

        insert_duration = time.perf_counter() - insert_start
        total_duration = time.perf_counter() - total_start
        velocity = total_rows / max(insert_duration, 0.001)

        logger.info(
            f"Successfully ingested {total_rows:,} rows from '{file_name}' into '{table_name}' "
            f"in {total_duration:.2f}s (insert velocity: {velocity:,.0f} rows/s)"
        )
        return total_rows
