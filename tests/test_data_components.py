# tests/test_data_components.py
"""
Unit tests for Data Engineering modular components:
- IngestorFactory and concrete strategy ingestors (GreenTaxiIngestor, YellowTaxiIngestor)
- Standardized monthly filename formatting & datetime preprocessing
- Batch table DDL definitions and provisioning logic
"""
from datetime import datetime
from unittest.mock import MagicMock
import pandas as pd
import pytest

from data.ingestors.factory import IngestorFactory
from data.ingestors.green import GreenTaxiIngestor
from data.ingestors.yellow import YellowTaxiIngestor
from data.schemas.tables import (
    GREEN_TRIPS_BATCH_SCHEMA,
    YELLOW_TRIPS_BATCH_SCHEMA,
    ensure_batch_tables,
)


def test_ingestor_factory_resolution():
    """Verify IngestorFactory returns appropriate strategy instance or raises ValueError."""
    green = IngestorFactory.get_ingestor("green")
    assert isinstance(green, GreenTaxiIngestor)
    assert green.dataset_type == "green"
    assert green.bucket_name == "taxi-green"
    assert green.batch_table == "green_trips_batch"

    yellow = IngestorFactory.get_ingestor("yellow")
    assert isinstance(yellow, YellowTaxiIngestor)
    assert yellow.dataset_type == "yellow"
    assert yellow.bucket_name == "taxi-yellow"
    assert yellow.batch_table == "yellow_trips_batch"

    with pytest.raises(ValueError, match="Unsupported dataset type"):
        IngestorFactory.get_ingestor("uber")


def test_green_taxi_filename_and_preprocessing():
    """Verify filename generation and datetime conversions for Green Taxi dataset."""
    ingestor = GreenTaxiIngestor()
    filename = ingestor.format_filename(datetime(2024, 3, 15))
    assert filename == "green_tripdata_2024-03.parquet"

    raw_df = pd.DataFrame({
        "VendorID": [1, 2],
        "lpep_pickup_datetime": ["2024-03-01 10:00:00", "2024-03-01 11:00:00"],
        "lpep_dropoff_datetime": ["2024-03-01 10:15:00", "2024-03-01 11:20:00"],
        "trip_distance": [2.5, 4.0],
    })
    processed = ingestor.preprocess(raw_df)
    assert pd.api.types.is_datetime64_any_dtype(processed["lpep_pickup_datetime"])
    assert pd.api.types.is_datetime64_any_dtype(processed["lpep_dropoff_datetime"])


def test_yellow_taxi_filename_and_preprocessing():
    """Verify filename generation and datetime conversions for Yellow Taxi dataset."""
    ingestor = YellowTaxiIngestor()
    filename = ingestor.format_filename(datetime(2024, 7, 1))
    assert filename == "yellow_tripdata_2024-07.parquet"

    raw_df = pd.DataFrame({
        "VendorID": [1, 2],
        "tpep_pickup_datetime": ["2024-07-01 08:30:00", "2024-07-01 09:15:00"],
        "tpep_dropoff_datetime": ["2024-07-01 08:45:00", "2024-07-01 09:30:00"],
        "trip_distance": [1.8, 3.2],
    })
    processed = ingestor.preprocess(raw_df)
    assert pd.api.types.is_datetime64_any_dtype(processed["tpep_pickup_datetime"])
    assert pd.api.types.is_datetime64_any_dtype(processed["tpep_dropoff_datetime"])


def test_schema_ddl_and_ensure_batch_tables():
    """Verify DDL integrity and table provisioning calls against ClickHouse service."""
    assert "CREATE TABLE IF NOT EXISTS green_trips_batch" in GREEN_TRIPS_BATCH_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS yellow_trips_batch" in YELLOW_TRIPS_BATCH_SCHEMA

    mock_clickhouse = MagicMock()
    ensure_batch_tables(mock_clickhouse)

    assert mock_clickhouse.client.command.call_count == 2
    mock_clickhouse.client.command.assert_any_call(GREEN_TRIPS_BATCH_SCHEMA)
    mock_clickhouse.client.command.assert_any_call(YELLOW_TRIPS_BATCH_SCHEMA)
