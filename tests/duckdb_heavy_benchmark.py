import duckdb
import logging
import time

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
    # -------------------- CONNECT TO DUCKDB --------------------
    logging.info("Connecting to DuckDB database...")
    con = duckdb.connect("taxi_data.duckdb")
    # -------------------- AGGRESSIVE MEMORY OPTIMIZATION FOR FULL DATA --------------------
    logging.info("Configuring memory settings for full dataset processing...")
    con.execute("SET memory_limit='16GB'")
    con.execute("SET threads=12")  # Use more threads for parallelism
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='./duckdb_temp'")
    con.execute("SET max_temp_directory_size='50GB'")  # Allow disk spillover
    con.execute("SET enable_object_cache=true")
    logging.info("Memory settings configured for full dataset.")

    # Verify tables exist
    for table in ["green_trips", "yellow_trips"]:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count == 0:
            raise ValueError(f"Table {table} is empty!")
        logging.info(f"Table {table} has {count:,} rows")

    # -------------------- BENCHMARK FUNCTION --------------------
    def benchmark(name, query, fetch_results=False):
        logging.info(f"Starting benchmark: {name}")
        start = time.time()
        result = con.execute(query)
        if fetch_results:
            data = result.fetchall()
            elapsed = time.time() - start
            logging.info(f"✓ [{name}] Completed in {elapsed:.3f}s | Rows returned: {len(data):,}")
            return elapsed, data
        else:
            data = result.fetchdf()
            elapsed = time.time() - start
            logging.info(f"✓ [{name}] Completed in {elapsed:.3f}s | Rows returned: {len(data):,}")
            return elapsed, data

    print("\n" + "="*80)
    print("STARTING PERFORMANCE BENCHMARKS")
    print("="*80 + "\n")

    # -------------------- QUERY 1: COMPLEX AGGREGATION WITH UNIONS --------------------
    q1 = """
    WITH combined_trips AS (
        SELECT 
            'green' AS taxi_type,
            PULocationID,
            DOLocationID,
            lpep_pickup_datetime AS pickup_time,
            lpep_dropoff_datetime AS dropoff_time,
            trip_distance,
            fare_amount,
            total_amount,
            passenger_count
        FROM green_trips
        UNION ALL
        SELECT 
            'yellow' AS taxi_type,
            PULocationID,
            DOLocationID,
            tpep_pickup_datetime AS pickup_time,
            tpep_dropoff_datetime AS dropoff_time,
            trip_distance,
            fare_amount,
            total_amount,
            passenger_count
        FROM yellow_trips
    )
    SELECT 
        taxi_type,
        PULocationID,
        COUNT(*) AS total_trips,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_fare,
        AVG(trip_distance) AS avg_distance,
        SUM(passenger_count) AS total_passengers,
        MIN(total_amount) AS min_fare,
        MAX(total_amount) AS max_fare,
        STDDEV(total_amount) AS stddev_fare
    FROM combined_trips
    WHERE total_amount > 0 AND trip_distance > 0
    GROUP BY taxi_type, PULocationID
    ORDER BY total_revenue DESC
    LIMIT 100
    """
    t1, r1 = benchmark("Q1: Complex Union + Aggregation", q1, fetch_results=True)
    print(f"Sample results: {r1[:3]}\n")

    # -------------------- QUERY 2: WINDOW FUNCTIONS WITH PARTITIONING --------------------
    q2 = """
    WITH ranked_trips AS (
        SELECT 
            PULocationID,
            DOLocationID,
            total_amount,
            trip_distance,
            tpep_pickup_datetime,
            ROW_NUMBER() OVER (PARTITION BY PULocationID ORDER BY total_amount DESC) AS revenue_rank,
            RANK() OVER (PARTITION BY DOLocationID ORDER BY trip_distance DESC) AS distance_rank,
            AVG(total_amount) OVER (PARTITION BY PULocationID) AS avg_location_fare,
            SUM(total_amount) OVER (PARTITION BY PULocationID ORDER BY tpep_pickup_datetime 
                                     ROWS BETWEEN 100 PRECEDING AND CURRENT ROW) AS rolling_revenue,
            LAG(total_amount, 1) OVER (PARTITION BY PULocationID ORDER BY tpep_pickup_datetime) AS prev_fare,
            LEAD(total_amount, 1) OVER (PARTITION BY PULocationID ORDER BY tpep_pickup_datetime) AS next_fare
        FROM yellow_trips
        WHERE total_amount > 0 AND tpep_pickup_datetime >= '2019-01-01'
    )
    SELECT 
        PULocationID,
        COUNT(*) AS trips,
        AVG(revenue_rank) AS avg_revenue_rank,
        AVG(distance_rank) AS avg_distance_rank,
        MAX(rolling_revenue) AS max_rolling_revenue,
        AVG(prev_fare) AS avg_prev_fare
    FROM ranked_trips
    WHERE revenue_rank <= 10
    GROUP BY PULocationID
    ORDER BY trips DESC
    LIMIT 50
    """
    t2, r2 = benchmark("Q2: Window Functions (ROW_NUMBER, RANK, LAG, LEAD)", q2, fetch_results=True)
    print(f"Sample results: {r2[:3]}\n")

    # -------------------- QUERY 3: COMPLEX JOIN WITH MULTIPLE CONDITIONS --------------------
    q3 = """
    SELECT 
        g.PULocationID AS green_pickup,
        y.PULocationID AS yellow_pickup,
        COUNT(DISTINCT g.lpep_pickup_datetime) AS green_count,
        COUNT(DISTINCT y.tpep_pickup_datetime) AS yellow_count,
        AVG(g.total_amount) AS avg_green_fare,
        AVG(y.total_amount) AS avg_yellow_fare,
        SUM(g.trip_distance + y.trip_distance) AS total_distance,
        CORR(g.total_amount, y.total_amount) AS fare_correlation
    FROM green_trips g
    INNER JOIN yellow_trips y 
        ON g.PULocationID = y.PULocationID 
        AND DATE_TRUNC('hour', g.lpep_pickup_datetime) = DATE_TRUNC('hour', y.tpep_pickup_datetime)
        AND g.passenger_count = y.passenger_count
    WHERE g.total_amount > 5 AND y.total_amount > 5
        AND g.trip_distance > 0 AND y.trip_distance > 0
    GROUP BY g.PULocationID, y.PULocationID
    HAVING COUNT(*) > 100
    ORDER BY fare_correlation DESC
    LIMIT 100
    """
    t3, r3 = benchmark("Q3: Complex JOIN with Multiple Conditions", q3, fetch_results=True)
    print(f"Sample results: {r3[:3]}\n")

    # -------------------- QUERY 4: ADVANCED WINDOW FUNCTIONS + PERCENTILES --------------------
    q4 = """
    WITH trip_stats AS (
        SELECT 
            PULocationID,
            DOLocationID,
            EXTRACT(hour FROM tpep_pickup_datetime) AS hour_of_day,
            EXTRACT(dow FROM tpep_pickup_datetime) AS day_of_week,
            total_amount,
            trip_distance,
            NTILE(10) OVER (PARTITION BY PULocationID ORDER BY total_amount) AS fare_decile,
            PERCENT_RANK() OVER (PARTITION BY DOLocationID ORDER BY trip_distance) AS distance_percentile,
            CUME_DIST() OVER (PARTITION BY PULocationID ORDER BY total_amount) AS cumulative_fare_dist
        FROM yellow_trips
        WHERE total_amount > 0 AND trip_distance > 0
    )
    SELECT 
        PULocationID,
        hour_of_day,
        day_of_week,
        fare_decile,
        COUNT(*) AS trips_in_decile,
        AVG(total_amount) AS avg_fare,
        AVG(distance_percentile) AS avg_distance_percentile,
        MAX(cumulative_fare_dist) AS max_cumulative_dist,
        MEDIAN(total_amount) AS median_fare
    FROM trip_stats
    GROUP BY PULocationID, hour_of_day, day_of_week, fare_decile
    HAVING trips_in_decile > 50
    ORDER BY PULocationID, fare_decile
    LIMIT 500
    """
    t4, r4 = benchmark("Q4: NTILE, PERCENT_RANK, CUME_DIST + MEDIAN", q4, fetch_results=True)
    print(f"Sample results: {r4[:3]}\n")

    # -------------------- QUERY 5: SELF-JOIN WITH SUBQUERIES --------------------
    q5 = """
    WITH high_value_trips AS (
        SELECT 
            PULocationID,
            DOLocationID,
            tpep_pickup_datetime,
            total_amount,
            trip_distance
        FROM yellow_trips
        WHERE total_amount > (SELECT AVG(total_amount) * 2 FROM yellow_trips)
    ),
    location_pairs AS (
        SELECT 
            t1.PULocationID AS loc1,
            t2.PULocationID AS loc2,
            COUNT(*) AS pair_count,
            AVG(t1.total_amount + t2.total_amount) AS avg_combined_fare,
            SUM(t1.trip_distance + t2.trip_distance) AS total_combined_distance
        FROM high_value_trips t1
        INNER JOIN high_value_trips t2 
            ON t1.DOLocationID = t2.PULocationID
            AND t1.tpep_pickup_datetime < t2.tpep_pickup_datetime
            AND t2.tpep_pickup_datetime - t1.tpep_pickup_datetime < INTERVAL '1 hour'
        GROUP BY t1.PULocationID, t2.PULocationID
    )
    SELECT * FROM location_pairs
    WHERE pair_count > 10
    ORDER BY avg_combined_fare DESC
    LIMIT 100
    """
    t5, r5 = benchmark("Q5: Self-JOIN with Subquery Filter", q5, fetch_results=True)
    print(f"Sample results: {r5[:3]}\n")

    # -------------------- QUERY 6: MULTI-LEVEL GROUPING SETS --------------------
    q6 = """
    SELECT 
        EXTRACT(year FROM tpep_pickup_datetime) AS year,
        EXTRACT(month FROM tpep_pickup_datetime) AS month,
        EXTRACT(dow FROM tpep_pickup_datetime) AS day_of_week,
        PULocationID,
        COUNT(*) AS trip_count,
        SUM(total_amount) AS total_revenue,
        AVG(trip_distance) AS avg_distance
    FROM yellow_trips
    WHERE tpep_pickup_datetime >= '2019-01-01'
    GROUP BY GROUPING SETS (
        (year, month, day_of_week, PULocationID),
        (year, month, PULocationID),
        (year, PULocationID),
        (PULocationID),
        ()
    )
    ORDER BY year, month, day_of_week, PULocationID
    LIMIT 1000
    """
    t6, r6 = benchmark("Q6: GROUPING SETS Multi-Level Aggregation", q6, fetch_results=True)
    print(f"Sample results: {r6[:3]}\n")

    # -------------------- SUMMARY --------------------
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    total_time = t1 + t2 + t3 + t4 + t5 + t6
    print(f"Q1 - Complex Union + Aggregation:           {t1:>8.3f}s")
    print(f"Q2 - Window Functions (Multiple):           {t2:>8.3f}s")
    print(f"Q3 - Complex JOIN:                          {t3:>8.3f}s")
    print(f"Q4 - Advanced Window + Percentiles:         {t4:>8.3f}s")
    print(f"Q5 - Self-JOIN with Subquery:               {t5:>8.3f}s")
    print(f"Q6 - GROUPING SETS:                         {t6:>8.3f}s")
    print("-"*80)
    print(f"TOTAL TIME:                                 {total_time:>8.3f}s")
    print("="*80 + "\n")

except Exception as e:
    logging.error(f"❌ Error occurred: {e}", exc_info=True)
    print(f"An error occurred: {e}")

finally:
    logging.info("Script execution completed.")