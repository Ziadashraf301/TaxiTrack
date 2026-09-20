# src/data/__init__.py
"""
Data Engineering package for TaxiTrack platform (Modular Architecture).
Provides clean modular components for streaming data ingestion, ClickHouse batch loading,
and dataset preprocessing.
"""
from data.ingestors.base import BaseIngestor
from data.ingestors.green import GreenTaxiIngestor
from data.ingestors.yellow import YellowTaxiIngestor
from data.ingestors.factory import IngestorFactory
from data.storage.downloader import StreamingDownloader
from data.storage.loader import ClickHouseBatchLoader
from data.schemas.tables import (
    GREEN_TRIPS_BATCH_SCHEMA,
    YELLOW_TRIPS_BATCH_SCHEMA,
    ensure_batch_tables,
)
from data.pipeline import MonthlyIngestionPipeline, run_monthly_ingestion

__all__ = [
    # Ingestors
    "BaseIngestor",
    "GreenTaxiIngestor",
    "YellowTaxiIngestor",
    "IngestorFactory",
    # Storage & Streaming
    "StreamingDownloader",
    "ClickHouseBatchLoader",
    # Schemas
    "GREEN_TRIPS_BATCH_SCHEMA",
    "YELLOW_TRIPS_BATCH_SCHEMA",
    "ensure_batch_tables",
    # Pipeline
    "MonthlyIngestionPipeline",
    "run_monthly_ingestion",
]
