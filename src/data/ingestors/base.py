# src/data/ingestors/base.py
"""
Abstract Ingestor Contract (Open/Closed & Template Method Patterns)
"""
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from core.logging import get_logger

logger = get_logger(__name__)


class BaseIngestor(ABC):
    """
    Abstract Base Ingestor.
    Subclasses define dataset-specific metadata and preprocessing logic.
    """

    def __init__(self):
        logger.debug(
            f"Initialized {self.__class__.__name__} "
            f"(dataset={self.dataset_type}, bucket={self.bucket_name}, batch_table={self.batch_table})"
        )

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Name of the dataset (e.g., 'green', 'yellow')."""
        pass

    @property
    @abstractmethod
    def bucket_name(self) -> str:
        """MinIO bucket name for this dataset."""
        pass

    @property
    @abstractmethod
    def batch_table(self) -> str:
        """ClickHouse destination batch table."""
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dataset-specific transformations (e.g. datetime conversion)."""
        pass

    def format_filename(self, execution_date: datetime) -> str:
        """Format standardized monthly filename: {dataset_type}_tripdata_YYYY-MM.parquet."""
        filename = f"{self.dataset_type}_tripdata_{execution_date.strftime('%Y-%m')}.parquet"
        logger.debug(f"Resolved filename for {execution_date:%Y-%m}: {filename}")
        return filename
