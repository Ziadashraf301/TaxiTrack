-- tests/test_calculate_tip_rate.sql

with test_data as (
    select 5 as tip_amount, 10 as fare_amount, 0.5 as expected_tip_rate
    union all
    select 3, 0, NULL
    union all
    select NULL, 15, NULL
),

actual as (
    select
        *,
        {{ calculate_tip_rate('tip_amount', 'fare_amount') }} as actual_tip_rate
    from test_data
)

select *
from actual
where not (
    (actual_tip_rate = expected_tip_rate)
    or (actual_tip_rate is null and expected_tip_rate is null)
)
