import os
import pandas as pd
import logging
from tqdm import tqdm
from time import time
from ingestion.downloader import download_file
from ingestion.db_utils import get_engine, table_exists

# Predefined dtypes
DTYPE_MAPPING = {
    "VendorID": "Int64",
    "passenger_count": "Int64",
    "RatecodeID": "Int64",
    "payment_type": "Int64",
    "trip_distance": "float32",
    "fare_amount": "float32",
    "extra": "float32",
    "mta_tax": "float32",
    "tip_amount": "float32",
    "tolls_amount": "float32",
    "total_amount": "float32"
}


def construct_url_and_filename(base_url, file_line):
    file_name = file_line.strip()
    url = base_url + file_name
    output_file = os.path.basename(url)
    table_name = file_name.split('.')[0]
    return url, output_file, table_name


def should_skip(url, table_name, engine, processed_urls):
    if url in processed_urls:
        logging.info(f"Skipping already processed URL: {url}")
        return True
    if table_exists(engine, table_name):
        logging.info(f"Table '{table_name}' already exists. Skipping.")
        return True
    return False


def preprocess_datetime_columns(df):
    for col in ['lpep_pickup_datetime', 'lpep_dropoff_datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def ingest_chunks(df_iter, table_name, engine):
    with tqdm(total=None, desc=f"Ingesting {table_name}", unit="chunk") as pbar:
        while True:
            try:
                t_start = time()
                df = next(df_iter)
                df = preprocess_datetime_columns(df)
                df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
                logging.info(f"Inserted chunk in {time() - t_start:.3f}s")
                pbar.update(1)
            except StopIteration:
                break


def load_file(output_file, file_type, dtype_mapping):
    if file_type == 'csv':
        df_iter = pd.read_csv(output_file, iterator=True, chunksize=100_000, dtype=dtype_mapping)
        return df_iter
    elif file_type == 'parquet':
        df = pd.read_parquet(output_file)
        return iter([df])  # Wrap in iterator for uniformity
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def ingest_data_for_file(params, file_line):
    url, raw_output_file, table_name = construct_url_and_filename(params.base_url, file_line)
    output_file = os.path.join('data', raw_output_file)

    engine = get_engine(params.user, params.password, params.host, params.port, params.db)

    if should_skip(url, table_name, engine, params.processed_urls):
        return

    params.processed_urls.add(url)
    download_file(url, output_file)

    file_type = 'parquet' if url.endswith('.parquet') else 'csv'

    try:
        df_iter = load_file(output_file, file_type, DTYPE_MAPPING if file_type == 'csv' else None)
        df = next(df_iter)
        df = preprocess_datetime_columns(df)
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        logging.info(f"Table '{table_name}' created.")

        ingest_chunks(df_iter, table_name, engine)

        logging.info(f"Finished ingesting table '{table_name}'")

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise
