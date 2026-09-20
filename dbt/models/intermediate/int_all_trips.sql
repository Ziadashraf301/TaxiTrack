{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key='trip_id'
    )
}}

-- ====================================================================
-- Intermediate model: combine green and yellow taxi trips
-- ====================================================================

with green as (
    select
        trip_id,
        vendor_id,
        pickup_datetime,
        dropoff_datetime,
        store_and_forward_flag,
        rate_code_id,
        pickup_location_id,
        dropoff_location_id,
        passenger_count,
        trip_distance,
        fare_amount,
        extra,
        mta_tax,
        tip_amount,
        tolls_amount,
        ehail_fee,
        improvement_surcharge,
        total_amount,
        payment_type,
        trip_type,
        congestion_surcharge,
        file_name,
        ingest_time,
        dbt_loaded_at,
        vendor_name,
        payment_type_name,
        trip_type_name,
        rate_code_description,
        store_and_forward_flag_description,
        service_type,
        pickup_hour,
        pickup_day_of_week,
        trip_duration_minutes
    from {{ ref('stg_green_trips') }}
    {% if is_incremental() %}
    where dbt_loaded_at > (select max(dbt_loaded_at) from {{ this }})
    {% endif %}
),

yellow as (
    select
        trip_id,
        vendor_id,
        pickup_datetime,
        dropoff_datetime,
        store_and_forward_flag,
        rate_code_id,
        pickup_location_id,
        dropoff_location_id,
        passenger_count,
        trip_distance,
        fare_amount,
        extra,
        mta_tax,
        tip_amount,
        tolls_amount,
        NULL as ehail_fee,
        improvement_surcharge,
        total_amount,
        payment_type,
        NULL as trip_type,
        congestion_surcharge,
        file_name,
        ingest_time,
        dbt_loaded_at,
        vendor_name,
        payment_type_name,
        NULL as trip_type_name,
        rate_code_description,
        store_and_forward_flag_description,
        service_type,
        pickup_hour,
        pickup_day_of_week,
        trip_duration_minutes
    from {{ ref('stg_yellow_trips') }}
    {% if is_incremental() %}
    where dbt_loaded_at > (select max(dbt_loaded_at) from {{ this }})
    {% endif %}
),

all_trips as (
    select * from green
    union all
    select * from yellow
),

pickup_zones as (
    select
        locationid,
        zone as pickup_zone,
        borough as pickup_borough
    from {{ ref('taxi_zone_lookup') }} 
),

dropoff_zones as (
    select
        locationid,
        zone as dropoff_zone,
        borough as dropoff_borough
    from {{ ref('taxi_zone_lookup') }}
)

select
    t.trip_id,
    t.vendor_id,
    t.pickup_datetime,
    t.dropoff_datetime,
    t.store_and_forward_flag,
    t.rate_code_id,
    t.pickup_location_id,
    t.dropoff_location_id,
    t.passenger_count,
    t.trip_distance,
    t.fare_amount,
    t.extra,
    t.mta_tax,
    t.tip_amount,
    t.tolls_amount,
    t.ehail_fee,
    t.improvement_surcharge,
    t.total_amount,
    t.payment_type,
    t.trip_type,
    t.congestion_surcharge,
    t.file_name,
    t.ingest_time,
    t.dbt_loaded_at,
    t.vendor_name,
    t.payment_type_name,
    t.trip_type_name,
    t.rate_code_description,
    t.store_and_forward_flag_description,
    t.service_type,
    t.pickup_hour,
    t.pickup_day_of_week,
    t.trip_duration_minutes,
    p.pickup_zone,
    p.pickup_borough,
    d.dropoff_zone,
    d.dropoff_borough
from all_trips t
left join pickup_zones p on t.pickup_location_id = p.locationid
left join dropoff_zones d on t.dropoff_location_id = d.locationid
