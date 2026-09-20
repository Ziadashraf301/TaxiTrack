{{ config(
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=['pickup_month', 'source_location', 'target_location'],
    alias='trip_location_network_metrics'
) }}

with base as (
    select
        pickup_datetime,
        pickup_zone as source_location,
        dropoff_zone as target_location,
        trip_distance,
        total_amount,
        trip_duration_minutes,
        dbt_loaded_at
    from {{ ref('int_all_trips') }}
    where trip_distance > 0
      and total_amount >= 0
      and pickup_zone != ''
      and dropoff_zone != ''
    {% if is_incremental() %}
      and dbt_loaded_at > (select coalesce(max(dbt_loaded_at), toDateTime('1970-01-01 00:00:00')) from {{ this }})
    {% endif %}
),

aggregated as (
    select
        toYYYYMM(pickup_datetime) as pickup_month,
        source_location,
        target_location,
        count(*) as trip_count,
        round(avg(trip_distance), 2) as avg_distance,
        round(sum(total_amount), 2) as sum_total_amounts,
        round(avg(trip_duration_minutes), 2) as avg_duration_minutes,
        max(dbt_loaded_at) as dbt_loaded_at
    from base
    group by
        pickup_month,
        source_location,
        target_location
)

select *
from aggregated
