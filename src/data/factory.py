# src/data/factory.py
"""
Ingestor Factory (Factory Pattern)
Dynamically instantiates dataset-specific ingestors.
"""
from typing import Optional
from core.minio import MinioService
from core.clickhouse import ClickHouseService
from core.logging import get_logger
from data.base import BaseIngestor
from data.green_ingestor import GreenTaxiIngestor
from data.yellow_ingestor import YellowTaxiIngestor

logger = get_logger(__name__)


class IngestorFactory:
    """Factory creating configured BaseIngestor instances based on dataset type."""

    @staticmethod
    def get_ingestor(
        dataset_type: str,
        minio_service: Optional[MinioService] = None,
        clickhouse_service: Optional[ClickHouseService] = None
    ) -> BaseIngestor:
        minio_svc = minio_service or MinioService()
        ch_svc = clickhouse_service or ClickHouseService()

        normalized = dataset_type.lower().strip()
        if "green" in normalized:
            ingestor = GreenTaxiIngestor(minio_service=minio_svc, clickhouse_service=ch_svc)
        elif "yellow" in normalized:
            ingestor = YellowTaxiIngestor(minio_service=minio_svc, clickhouse_service=ch_svc)
        else:
            logger.error(f"Failed to resolve ingestor: unsupported dataset type '{dataset_type}'.")
            raise ValueError(f"Unsupported dataset type: '{dataset_type}'. Expected 'green' or 'yellow'.")

        logger.debug(f"Factory created {ingestor.__class__.__name__} for dataset_type='{dataset_type}'")
        return ingestor

