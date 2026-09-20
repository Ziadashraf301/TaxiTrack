{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key=['pickup_date', 'pickup_hour', 'pickup_zone', 'pickup_borough', 'service_type']
    )
}}

with base as (
    select
        pickup_datetime,
        service_type,
        pickup_zone,
        pickup_borough,
        dbt_loaded_at
    from {{ ref('int_all_trips') }}
    where pickup_zone != ''
      and pickup_borough != ''
    {% if is_incremental() %}
      and dbt_loaded_at > (select coalesce(max(dbt_loaded_at), toDateTime('1970-01-01 00:00:00')) from {{ this }})
    {% endif %}
),

aggregated as (
    select
        toDate(pickup_datetime) as pickup_date,
        toHour(pickup_datetime) as pickup_hour,
        pickup_zone,
        pickup_borough,
        service_type,
        count(*) as total_trips,
        max(dbt_loaded_at) as dbt_loaded_at
    from base
    group by
        pickup_date,
        pickup_hour,
        pickup_zone,
        pickup_borough,
        service_type
)

select * from aggregated
