# src/data/ingestors/green.py
"""
Green Taxi Dataset Ingestor Strategy
"""
import pandas as pd
from data.ingestors.base import BaseIngestor
from core.logging import get_logger

logger = get_logger(__name__)


class GreenTaxiIngestor(BaseIngestor):
    """Concrete ingestor for NYC Green Taxi dataset."""

    @property
    def dataset_type(self) -> str:
        return "green"

    @property
    def bucket_name(self) -> str:
        return "taxi-green"

    @property
    def batch_table(self) -> str:
        return "green_trips_batch"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert known green taxi datetime columns to pandas datetime."""
        converted = []
        for col in ["lpep_pickup_datetime", "lpep_dropoff_datetime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                converted.append(col)
        logger.debug(f"Preprocessed green taxi data: converted {converted} to datetime across {len(df):,} rows.")
        return df
