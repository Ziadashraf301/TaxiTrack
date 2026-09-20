# src/data/schemas/tables.py
"""
ClickHouse Raw Batch Table Schemas & Provisioning (Data Engineering Layer)
"""
from core.clickhouse import ClickHouseService
from core.logging import get_logger

logger = get_logger(__name__)

GREEN_TRIPS_BATCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS green_trips_batch (
    VendorID UInt8,
    lpep_pickup_datetime DateTime,
    lpep_dropoff_datetime DateTime,
    store_and_fwd_flag Nullable(String),
    RatecodeID UInt8,
    PULocationID UInt16,
    DOLocationID UInt16,
    passenger_count UInt8,
    trip_distance Float32,
    fare_amount Float32,
    extra Float32,
    mta_tax Float32,
    tip_amount Float32,
    tolls_amount Float32,
    ehail_fee Nullable(Float32),
    improvement_surcharge Float32,
    total_amount Float32,
    payment_type UInt8,
    trip_type Nullable(UInt8),
    congestion_surcharge Nullable(Float32),
    file_name String,
    ingest_time DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(lpep_pickup_datetime)
ORDER BY (lpep_pickup_datetime, DOLocationID)
"""

YELLOW_TRIPS_BATCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS yellow_trips_batch (
    VendorID UInt8,
    tpep_pickup_datetime DateTime,
    tpep_dropoff_datetime DateTime,
    passenger_count UInt8,
    trip_distance Float32,
    RatecodeID UInt8,
    store_and_fwd_flag Nullable(String),
    PULocationID UInt16,
    DOLocationID UInt16,
    payment_type UInt8,
    fare_amount Float32,
    extra Float32,
    mta_tax Float32,
    tip_amount Float32,
    tolls_amount Float32,
    improvement_surcharge Float32,
    total_amount Float32,
    congestion_surcharge Nullable(Float32),
    Airport_fee Nullable(Float32),
    file_name String,
    ingest_time DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(tpep_pickup_datetime)
ORDER BY (tpep_pickup_datetime, DOLocationID)
"""


def ensure_batch_tables(clickhouse_service: ClickHouseService) -> None:
    """Provision raw batch ingestion tables in ClickHouse if they do not exist."""
    logger.info("Verifying batch table schemas in ClickHouse...")
    try:
        clickhouse_service.client.command(GREEN_TRIPS_BATCH_SCHEMA)
        clickhouse_service.client.command(YELLOW_TRIPS_BATCH_SCHEMA)
        logger.info("Batch tables verified: green_trips_batch, yellow_trips_batch")
    except Exception as e:
        logger.error(f"Error provisioning batch tables: {e}", exc_info=True)
        raise
