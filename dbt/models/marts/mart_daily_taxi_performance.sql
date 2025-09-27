{{ config(
    materialized='table'
) }}

with base as (
    select
        pickup_datetime,
        service_type,
        fare_amount,
        tip_amount,
        congestion_surcharge,
        passenger_count,
        trip_distance,
        pickup_borough as borough
    from {{ ref('stg_all_trips') }}
    where fare_amount >= 0
      and tip_amount >= 0
      and congestion_surcharge >= 0
      and passenger_count >= 0
      and trip_distance > 0
),

aggregated as (
    select
        toDate(pickup_datetime) as pickup_date,
        borough,
        service_type,
        count(*) as num_trips,
        sum(passenger_count) as total_passengers,
        sum(fare_amount) as total_fare,
        sum(tip_amount) as total_tips,
        sum(congestion_surcharge) as total_congestion_fees,
        sum(fare_amount + tip_amount + congestion_surcharge) as total_revenue
    from base
    group by
        pickup_date,
        borough,
        service_type
)

select * from aggregated
