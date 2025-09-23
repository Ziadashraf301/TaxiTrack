{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key=['pickup_datetime', 'dropoff_location_id']
    )
}}

with raw_yellow_trips as (
    select *
    from {{ source('staging', 'yellow_trips_batch') }}
),

standardized_trips as (
    {{ yellow_trips_standardize('raw_yellow_trips') }}
),

categorized_trips as (
    {{ yellow_trips_categorize('standardized_trips') }}
),

final_trips as (
    {{ yellow_trips_features('categorized_trips') }}
)

select *
from final_trips

{% if is_incremental() %}
where ingest_time > (select max(ingest_time) from {{ this }})
{% endif %}
