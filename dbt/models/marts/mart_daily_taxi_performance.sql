{{ config(
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=['pickup_date', 'borough', 'service_type']
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
        pickup_borough as borough,
        dbt_loaded_at
    from {{ ref('int_all_trips') }}
    where fare_amount >= 0
      and tip_amount >= 0
      and congestion_surcharge >= 0
      and passenger_count >= 0
      and trip_distance > 0
      and pickup_borough != ''
    {% if is_incremental() %}
      and dbt_loaded_at > (select coalesce(max(dbt_loaded_at), toDateTime('1970-01-01 00:00:00')) from {{ this }})
    {% endif %}
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
        sum(fare_amount + tip_amount + congestion_surcharge) as total_revenue,
        max(dbt_loaded_at) as dbt_loaded_at
    from base
    group by
        pickup_date,
        borough,
        service_type
)

select * from aggregated
