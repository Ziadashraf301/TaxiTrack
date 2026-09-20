# src/core/clickhouse.py
"""
ClickHouse Client Factory and Service (Factory & Singleton Patterns)
Provides connection management, schema initialization, high-performance querying,
and granular telemetry logging.
"""
import time
from typing import Optional, List, Any
import clickhouse_connect
from clickhouse_connect.driver.client import Client
from core.config import settings, ClickHouseSettings
from core.logging import get_logger

logger = get_logger(__name__)


class ClickHouseClientFactory:
    """Factory for creating and caching configured ClickHouse client instances."""
    _client: Optional[Client] = None

    @classmethod
    def get_client(cls, ch_cfg: Optional[ClickHouseSettings] = None) -> Client:
        if cls._client is None:
            cfg = ch_cfg or settings.clickhouse
            logger.info(f"Connecting to ClickHouse at {cfg.host}:{cfg.http_port} (database='{cfg.db}', user='{cfg.user}')...")
            try:
                start_time = time.perf_counter()
                cls._client = clickhouse_connect.get_client(
                    host=cfg.host,
                    port=cfg.http_port,
                    username=cfg.user,
                    password=cfg.password,
                    database=cfg.db,
                    compress=True
                )
                duration = time.perf_counter() - start_time
                logger.info(f"Connected to ClickHouse successfully in {duration:.3f}s.")
            except Exception as e:
                logger.error(f"Failed to connect to ClickHouse at {cfg.host}:{cfg.http_port}: {e}", exc_info=True)
                raise
        return cls._client


class ClickHouseService:
    """High-level service for interacting with ClickHouse OLAP warehouse."""

    def __init__(self, client: Optional[Client] = None):
        self.client = client or ClickHouseClientFactory.get_client()

    def ensure_database(self, db_name: str = "data_warehouse") -> None:
        """Ensure the target database exists in ClickHouse."""
        logger.info(f"Ensuring ClickHouse database '{db_name}' exists...")
        try:
            self.client.command(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            logger.info(f"Database '{db_name}' verified.")
        except Exception as e:
            logger.error(f"Failed to ensure database '{db_name}': {e}", exc_info=True)
            raise


    def is_file_ingested(self, batch_table: str, file_name: str) -> bool:
        """Check if rows for the given file_name already exist in ClickHouse."""
        query = f"SELECT count(*) FROM {batch_table} WHERE file_name = '{file_name}'"
        try:
            result = self.client.query(query).result_rows
            row_count = result[0][0] if result else 0
            if row_count > 0:
                logger.info(f"File '{file_name}' already ingested ({row_count:,} rows in '{batch_table}'). Skipping.")
                return True
            logger.info(f"File '{file_name}' not found in '{batch_table}' (0 rows). Proceeding with ingestion.")
            return False
        except Exception as e:
            logger.warning(f"Could not check ingestion status for '{file_name}' in '{batch_table}': {e}. Assuming not ingested.")
            return False

    def get_table_columns(self, table_name: str) -> List[str]:
        """Fetch column names for a given table."""
        try:
            table_desc = self.client.query(f"DESCRIBE TABLE {table_name}").result_rows
            cols = [col[0] for col in table_desc]
            logger.debug(f"Retrieved {len(cols)} columns for table '{table_name}'.")
            return cols
        except Exception as e:
            logger.error(f"Failed to describe table '{table_name}': {e}", exc_info=True)
            raise

    def insert_records(self, table_name: str, records: List[List[Any]], column_names: List[str]) -> None:
        """Insert records with explicit column list, timing telemetry, and error logging."""
        count = len(records)
        start_time = time.perf_counter()
        try:
            self.client.insert(table_name, records, column_names=column_names)
            duration = time.perf_counter() - start_time
            rate = count / max(duration, 0.001)
            logger.info(f"Successfully inserted {count:,} rows into '{table_name}' in {duration:.3f}s ({rate:,.0f} rows/sec).")
        except Exception as e:
            logger.error(
                f"Failed to insert {count:,} records into '{table_name}' with columns {column_names}: {e}",
                exc_info=True
            )
            raise
