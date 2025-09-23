# ingestion/db_utils.py
from clickhouse_connect import get_client
from ingestion.logger import setup_logging
from datetime import datetime

# Initialize custom logger
logger = setup_logging("logs/ingestion.log")


def get_clickhouse_client(host: str, port: int, user: str, password: str, database: str):
    """
    Initialize and return a ClickHouse client.

    Args:
        host (str): ClickHouse server host.
        port (int): ClickHouse server port.
        user (str): ClickHouse username.
        password (str): ClickHouse password.
        database (str): Target database name.

    Returns:
        clickhouse_connect.driver.Client: Connected ClickHouse client.
    """
    client = get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        database=database
    )
    logger.info(f"Connected to ClickHouse at {host}:{port}, database={database}")
    return client


def ensure_batch_tables(client):
    """
    Ensure green_trips_batch and yellow_trips_batch tables exist in ClickHouse.
    Create them if missing.
    """

    # Green Taxi table schema
    green_schema = """
    CREATE TABLE IF NOT EXISTS green_trips_batch
    (
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
        ehail_fee Float32,
        improvement_surcharge Float32,
        total_amount Float32,
        payment_type UInt8,
        trip_type UInt8,
        congestion_surcharge Float32,
        file_name String,
        ingest_time DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(lpep_pickup_datetime)
    ORDER BY (lpep_pickup_datetime, DOLocationID)
    """

    # Yellow Taxi table schema
    yellow_schema = """
    CREATE TABLE IF NOT EXISTS yellow_trips_batch
    (
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
        congestion_surcharge Float32,
        file_name String,
        ingest_time DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(tpep_pickup_datetime)
    ORDER BY (tpep_pickup_datetime, DOLocationID)
    """

    logger.info("Ensuring ClickHouse batch tables exist...")

    client.command(green_schema)
    logger.info("✅ Ensured table exists: green_trips_batch")

    client.command(yellow_schema)
    logger.info("✅ Ensured table exists: yellow_trips_batch")


def format_monthly_filename(dataset_type: str, execution_date) -> str:
    """
    Format the filename for the given execution date.

    Args:
        dataset_type (str): 'green' or 'yellow'
        execution_date (datetime): Date to generate filename for

    Returns:
        str: Formatted filename like 'green_2019-01.parquet'
    """
    return f"{dataset_type}_tripdata_{execution_date.strftime('%Y-%m')}.parquet"
