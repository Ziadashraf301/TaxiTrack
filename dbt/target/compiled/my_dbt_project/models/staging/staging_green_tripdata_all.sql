-- models/staging/staging_green_tripdata_all.sql
-- (This was the file you uploaded with fullContent)




with source as (

    select *
    from "ingest_db"."public"."green_tripdata_2020-01"

),

with_uid as (

    select
        
    md5(
        concat_ws(
            '||',
            cast(lpep_pickup_datetime as text), cast(lpep_dropoff_datetime as text), cast(total_amount as text)
        )
    )
 as trip_uid,
        *
    from source

)

select *
from with_uid