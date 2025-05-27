-- models/staging/staging_green_tripdata_all.sql
-- (This was the file you uploaded with fullContent)

{% set execution_date = var("execution_date") %}
{% set month_suffix = execution_date[0:7] %}

with source as (

    select *
    from {{ source('raw', 'green_tripdata_' ~ month_suffix) }}

),

with_uid as (

    select
        {{ generate_trip_uid([
            'lpep_pickup_datetime',
            'lpep_dropoff_datetime',
            'total_amount'
        ]) }} as trip_uid,
        *
    from source

)

select *
from with_uid