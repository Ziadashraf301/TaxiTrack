-- tests/test_is_valid_trip.sql

with test_data as (
    select 1 as trip_distance, 10 as fare_amount, 1 as passenger_count,
           TIMESTAMP '2021-01-01 12:00:00' as pickup_datetime,
           TIMESTAMP '2021-01-01 12:02:00' as dropoff_datetime,
           true as expected_is_valid
    union all
    select 0, 10, 1, TIMESTAMP '2021-01-01 12:00:00', TIMESTAMP '2021-01-01 12:02:00', false
),

actual as (
    select
        *,
        
    trip_distance > 0
    AND fare_amount > 0
    AND passenger_count > 0
    AND EXTRACT(EPOCH FROM lpep_dropoff_datetime - lpep_pickup_datetime) > 60
 as actual_is_valid
    from test_data
)

select *
from actual
where actual_is_valid != expected_is_valid