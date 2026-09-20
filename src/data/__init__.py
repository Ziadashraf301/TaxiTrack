# src/data/__init__.py
"""Data Engineering package for TaxiTrack platform."""
from data.base import BaseIngestor
from data.downloader import StreamingDownloader
from data.loader import ClickHouseBatchLoader
from data.green_ingestor import GreenTaxiIngestor
from data.yellow_ingestor import YellowTaxiIngestor
from data.factory import IngestorFactory
from data.pipeline import MonthlyIngestionPipeline, run_monthly_ingestion

__all__ = [
    "BaseIngestor",
    "StreamingDownloader",
    "ClickHouseBatchLoader",
    "GreenTaxiIngestor",
    "YellowTaxiIngestor",
    "IngestorFactory",
    "MonthlyIngestionPipeline",
    "run_monthly_ingestion",
]
