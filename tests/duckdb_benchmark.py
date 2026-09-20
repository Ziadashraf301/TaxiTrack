import duckdb
import logging

# -------------------- SETUP LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("duckdb_minio.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting DuckDB-MinIO integration script...")

try:
    # # -------------------- CONNECT TO DUCKDB --------------------
    # logging.info("Connecting to DuckDB database...")
    # con = duckdb.connect("taxi_data.duckdb")
    # logging.info("Connected successfully!")

    # # -------------------- INSTALL & LOAD EXTENSIONS --------------------
    # logging.info("Installing and loading httpfs extension...")
    # con.execute("INSTALL httpfs")
    # con.execute("LOAD httpfs")
    # logging.info("httpfs extension loaded successfully!")

    # # -------------------- CONFIGURE MINIO/S3 SETTINGS --------------------
    # logging.info("Configuring S3/MinIO credentials...")
    # con.execute("""
    #     SET s3_endpoint='localhost:9000';
    #     SET s3_access_key_id='ziadashraf98765';
    #     SET s3_secret_access_key='x5x6x7x8';
    #     SET s3_use_ssl=false;
    #     SET s3_url_style='path';
    # """)
    # logging.info("S3/MinIO configuration complete.")

    # logging.info("Reading parquet files from s3://taxi-green/ ...")
    # con.execute("""
    #     CREATE OR REPLACE TABLE green_trips AS
    #     SELECT * FROM read_parquet('s3://taxi-green/*.parquet');
    # """)
    # green_count = con.execute("SELECT COUNT(*) FROM green_trips").fetchone()[0]
    # logging.info(f"Green trips table created successfully with {green_count:,} rows.")

    # logging.info("Reading parquet files from s3://taxi-yellow/ (with schema merge)...")
    # con.execute("""
    #     CREATE OR REPLACE TABLE yellow_trips AS
    #     SELECT * FROM read_parquet('s3://taxi-yellow/*.parquet', union_by_name=true);
    # """)
    # yellow_count = con.execute("SELECT COUNT(*) FROM yellow_trips").fetchone()[0]
    # logging.info(f"Yellow trips table created successfully with {yellow_count:,} rows.")

    # print("\nTables created successfully!")
    # print(f"Green trips count: {green_count:,}")
    # print(f"Yellow trips count: {yellow_count:,}")
    # Make sure the tables exist
    import time
    con = duckdb.connect("taxi_data.duckdb")
    
    for table in ["green_trips", "yellow_trips"]:
        if con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0:
            raise ValueError(f"Table {table} is empty!")

    # Benchmark function
    def benchmark(name, query):
        start = time.time()
        con.execute(query)
        elapsed = time.time() - start
        print(f"[{name}] Time taken: {elapsed:.3f} seconds")
        return elapsed

    # Heavy query with multiple unions and aggregations
    heavy_query = """
    SELECT PULocationID AS location, COUNT(*) AS trips, SUM(total_amount) AS revenue
    FROM green_trips
    GROUP BY PULocationID
    UNION ALL
    SELECT PULocationID AS location, COUNT(*) AS trips, SUM(total_amount) AS revenue
    FROM yellow_trips
    GROUP BY PULocationID
    UNION ALL
    SELECT DOLocationID AS location, COUNT(*) AS trips, SUM(total_amount) AS revenue
    FROM green_trips
    GROUP BY DOLocationID
    UNION ALL
    SELECT DOLocationID AS location, COUNT(*) AS trips, SUM(total_amount) AS revenue
    FROM yellow_trips
    GROUP BY DOLocationID
    """

    # Run benchmark
    elapsed = benchmark("Heavy UNION + Aggregation", heavy_query)

    # Optional: fetch a small sample to check results
    result_sample = con.execute(heavy_query + " LIMIT 10").fetchall()
    print("Sample results:", result_sample)

except Exception as e:
    logging.error(f"❌ Error occurred: {e}", exc_info=True)
    print(f"An error occurred: {e}")

finally:
    logging.info("Script execution completed.")
