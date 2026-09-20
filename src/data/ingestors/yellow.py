# src/data/ingestors/yellow.py
"""
Yellow Taxi Dataset Ingestor Strategy
"""
import pandas as pd
from data.ingestors.base import BaseIngestor
from core.logging import get_logger

logger = get_logger(__name__)


class YellowTaxiIngestor(BaseIngestor):
    """Concrete ingestor for NYC Yellow Taxi dataset."""

    @property
    def dataset_type(self) -> str:
        return "yellow"

    @property
    def bucket_name(self) -> str:
        return "taxi-yellow"

    @property
    def batch_table(self) -> str:
        return "yellow_trips_batch"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert known yellow taxi datetime columns to pandas datetime."""
        converted = []
        for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                converted.append(col)
        logger.debug(f"Preprocessed yellow taxi data: converted {converted} to datetime across {len(df):,} rows.")
        return df
