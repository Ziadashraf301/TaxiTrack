{{  
  config(
    materialized='incremental',
    alias='trip_location_network_metrics',
    unique_key=['source_location', 'target_location', 'trip_year_month']
  )  
}}

WITH base AS (
    SELECT
        t."PULocationID" AS source_location,
        t."DOLocationID" AS target_location,
        TO_CHAR(DATE_TRUNC('month', t.lpep_pickup_datetime), 'YYYY-MM') AS trip_year_month,
        COUNT(*) AS trip_count,
        ROUND(AVG(t.trip_distance)::numeric, 2) AS avg_distance,
        ROUND(SUM(t.fare_amount)::numeric, 2) AS total_fare,
        ROUND(SUM(t.tip_amount)::numeric, 2) AS total_tip,
        ROUND(AVG(t.total_amount)::numeric, 2) AS avg_total,
        ROUND(AVG(EXTRACT(EPOCH FROM (t.lpep_dropoff_datetime - t.lpep_pickup_datetime)) / 60)::numeric, 2) AS avg_duration_minutes
    FROM {{ ref('staging_green_tripdata_current_month') }} t
    WHERE {{ is_valid_trip() }}
    GROUP BY source_location, target_location, trip_year_month
),

pickup_lookup AS (
    SELECT
        locationid AS source_location,
        borough AS pickup_borough,
        zone AS pickup_zone,
        REPLACE(service_zone, 'Boro', 'Green') AS pickup_service_zone
    FROM {{ ref('taxi_zone_lookup') }}
),

dropoff_lookup AS (
    SELECT
        locationid AS target_location,
        borough AS dropoff_borough,
        zone AS dropoff_zone,
        REPLACE(service_zone, 'Boro', 'Green') AS dropoff_service_zone
    FROM {{ ref('taxi_zone_lookup') }}
)

SELECT
    -- Key fields
    b.trip_year_month,

    -- Pickup location metadata
    b.source_location,
    p.pickup_borough,
    p.pickup_zone,
    p.pickup_service_zone,

    -- Dropoff location metadata
    b.target_location,
    d.dropoff_borough,
    d.dropoff_zone,
    d.dropoff_service_zone,

    -- Trip metrics
    b.trip_count,
    b.avg_distance,
    b.total_fare,
    b.total_tip,
    b.avg_total,
    b.avg_duration_minutes


FROM base b
LEFT JOIN pickup_lookup p ON b.source_location = p.source_location
LEFT JOIN dropoff_lookup d ON b.target_location = d.target_location

