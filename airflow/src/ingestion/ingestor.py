# ingestion/ingestor.py
import logging
import pandas as pd
from io import BytesIO
from ingestion.db_utils import get_clickhouse_client, ensure_batch_tables, format_monthly_filename
from ingestion.logger import setup_logging
from ingestion.downloader import download_file_to_minio 

logger = setup_logging("logs/ingestion.log")


def preprocess_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known datetime columns to pandas datetime objects if present."""
    for col in ['lpep_pickup_datetime', 'lpep_dropoff_datetime',
                'tpep_pickup_datetime', 'tpep_dropoff_datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def check_file_already_ingested(client, batch_table: str, file_name: str) -> bool:
    """Check if the data for this file is already ingested in the batch table."""
    query = f"SELECT count(*) FROM {batch_table} WHERE file_name = '{file_name}'"
    result = client.query(query).result_rows
    if result and result[0][0] > 0:
        logger.info(f"Data from file '{file_name}' already exists in {batch_table}. Skipping ingestion.")
        return True
    return False


def ingest_parquet_to_batch(bucket: str, object_name: str, dataset_type: str, client, minio_client, batch_size=100_000):
    """Load parquet from MinIO and ingest into the corresponding batch table with ingest_time tracking."""
    batch_table = f"{dataset_type}_trips_batch"
    logger.info(f"Loading parquet from MinIO: s3://{bucket}/{object_name}")

    # Read parquet from MinIO
    response = minio_client.get_object(bucket, object_name)
    df = pd.read_parquet(BytesIO(response.read()), engine="pyarrow")
    response.close()
    response.release_conn()

    # Preprocess datetime columns
    df = preprocess_datetime_columns(df)

    # Track which file was ingested
    df['file_name'] = object_name
    # Add ingestion timestamp
    df['ingest_time'] = pd.Timestamp.now()

    # Fetch ClickHouse table columns
    table_desc = client.query(f"DESCRIBE TABLE {batch_table}").result_rows
    clickhouse_columns = [col[0] for col in table_desc]  # column names only

    # Keep only columns that exist in ClickHouse and ensure the correct order
    df = df[[c for c in clickhouse_columns if c in df.columns]]

    # Insert in batches
    total_rows = len(df)
    for start in range(0, total_rows, batch_size):
        chunk = df.iloc[start:start + batch_size]
        # Convert the DataFrame chunk to a list of lists, preserving the column order
        records = chunk.values.tolist()
        logger.info(f"[{object_name}] Inserting rows {start}-{start + len(chunk)} / {total_rows}")
        # Insert with explicit columns
        client.insert(batch_table, records, column_names=clickhouse_columns)

    logger.info(f"Completed ingestion for file: {object_name} ({total_rows} rows)")



def process_file(params, dataset_type: str, file_date):
    """
    Process a monthly parquet file:
    - Format filename and MinIO object name
    - Ensure batch tables exist
    - Download from source if missing in MinIO
    - Ingest into batch table only if data not present
    """
    client = get_clickhouse_client(params.host, params.port, params.user, params.password, params.db)
    ensure_batch_tables(client)

    file_name = format_monthly_filename(dataset_type, file_date)
    bucket_name = f"taxi-{dataset_type}"

    # Download file if not in MinIO
    try:
        minio_client = params.minio_client
        minio_client.stat_object(bucket_name, file_name)
        logger.info(f"File exists in MinIO: s3://{bucket_name}/{file_name}")
    except Exception:
        url = f"{params.base_url}/{file_name}"
        logger.info(f"File not found in MinIO. Downloading: {url}")
        download_file_to_minio(url, bucket_name, file_name, minio_client)

    # Skip ingestion if already exists in batch table
    batch_table = f"{dataset_type}_trips_batch"
    if check_file_already_ingested(client, batch_table, file_name):
        return

    # Ingest file into batch table
    ingest_parquet_to_batch(bucket_name, file_name, dataset_type, client, minio_client)
