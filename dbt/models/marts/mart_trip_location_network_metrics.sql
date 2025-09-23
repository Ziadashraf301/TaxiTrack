{{
    config(
        materialized='table',
        alias='trip_location_network_metrics'
    )
}}

with base as (
    select
    pickup_zone as source_location,
    dropoff_zone as target_location,
        count(*) as trip_count,
        round(avg(trip_distance), 2) as avg_distance,
        round(sum(total_amount), 2) as sum_total_amounts,
        round(avg(trip_duration_minutes), 2) as avg_duration_minutes
    from {{ ref('stg_all_trips') }}
    group by pickup_zone, dropoff_zone
)

select
    *
from base
