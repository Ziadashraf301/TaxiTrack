{{ 
  config(
    alias='green_tripdata_summary_all',
    materialized='incremental',
    unique_key='trip_uid'
  ) 
}}

-- 1. Run the summary macro into a CTE
with monthly_summary as (
  {{ generate_tripdata_summary('staging_green_tripdata_current_month') }}
),

-- 2. Add trip_uid column in a second CTE
with_uid as (
  select
    {{ generate_trip_uid([
      'first_trip',
      'last_trip',
      'unique_pickup_locations',
    ]) }} as trip_uid,
    *
  from monthly_summary
)

-- 3. Final output with optional incremental filtering
select *
from with_uid

{% if is_incremental() %}
where trip_uid not in (
    select trip_uid from {{ this }}
)
{% endif %}
