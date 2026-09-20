{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key='trip_id'
    )
}}

with raw_yellow_trips as (
    select *
    from {{ source('staging', 'yellow_trips_batch') }}
    {% if is_incremental() %}
    where ingest_time > (select max(ingest_time) from {{ this }})
    {% endif %}
),

standardized_trips as (
    {{ yellow_trips_standardize('raw_yellow_trips') }}
),

categorized_trips as (
    {{ yellow_trips_categorize('standardized_trips') }}
),

featured_trips as (
    {{ yellow_trips_features('categorized_trips') }}
),

final_trips as (
    select
        hex(MD5(concat(
            'yellow_',
            toString(vendor_id),
            toString(pickup_datetime),
            toString(dropoff_datetime),
            toString(pickup_location_id),
            toString(dropoff_location_id),
            toString(fare_amount),
            toString(trip_distance)
        ))) as trip_id,
        *
    from featured_trips
    limit 1 by trip_id
)

select *
from final_trips
