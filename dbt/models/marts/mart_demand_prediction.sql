{{
    config(
        materialized='table'
    )
}}

with base as (
    select
        pickup_datetime,
        pickup_location_id,
        service_type,
        passenger_count,
        trip_distance,
        fare_amount,
        tip_amount,
        congestion_surcharge,
        pickup_zone,
        pickup_borough,
        dbt_loaded_at
    from {{ ref('stg_all_trips') }}
),

aggregated as (
    select
        toDate(pickup_datetime) as pickup_date,
        toHour(pickup_datetime) as pickup_hour,
        pickup_location_id,
        pickup_zone,
        pickup_borough,
        service_type,
        count(*) as total_trips,
        sum(passenger_count) as total_passengers,
        avg(trip_distance) as avg_trip_distance,
        avg(fare_amount) as avg_fare_amount,
        avg(tip_amount) as avg_tip_amount,
        sum(congestion_surcharge) as total_congestion_surcharge
    from base
    group by
        pickup_date,
        pickup_hour,
        pickup_location_id,
        pickup_zone,
        pickup_borough,
        service_type
)

select * from aggregated
