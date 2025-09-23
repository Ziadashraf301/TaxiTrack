{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key=['pickup_datetime', 'dropoff_location_id', 'service_type']
    )
}}

-- ====================================================================
-- Intermediate model: combine green and yellow taxi trips
-- ====================================================================

with green as (
    select
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
),

yellow as (
    select
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
),

all_trips as (
    select * from green
    union all
    select * from yellow
),

pickup_zones as (
    select
        locationid as pickup_location_id,
        zone as pickup_zone,
        borough as pickup_borough
    from {{ ref('taxi_zone_lookup') }} 
),

dropoff_zones as (
    select
        locationid as dropoff_location_id,
        zone as dropoff_zone,
        borough as dropoff_borough
    from {{ ref('taxi_zone_lookup') }}
)

select
    t.*,
    t.pickup_location_id as `pickup_location_id`,
    t.dropoff_location_id as `dropoff_location_id`,
    p.pickup_zone,
    p.pickup_borough,
    d.dropoff_zone,
    d.dropoff_borough
from all_trips t
left join pickup_zones p on t.pickup_location_id = p.pickup_location_id
left join dropoff_zones d on t.dropoff_location_id = d.dropoff_location_id

{% if is_incremental() %}
where dbt_loaded_at > (select max(dbt_loaded_at) from {{ this }})
{% endif %}